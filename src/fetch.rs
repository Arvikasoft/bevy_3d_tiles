//! Byte-source IO for 3D Tiles content (BEVY-3D-TILES-PLAN T0).
//!
//! Adapts the basemap fetch layer's discipline to tile streaming:
//!
//! * **Never block the executor** — every wasm operation `.await`s a JS future
//!   inside a `spawn_local` task (the `run_background_task` grey-screen lesson;
//!   see `bevy-client/CLAUDE.md`). Results drain into the ECS per frame over a
//!   crossbeam channel.
//! * **Range requests against a single blob URL** — the `.3tz` reader
//!   ([`super::archive`]) issues byte-range reads ([`ByteSource::read`]);
//!   exploded tilesets fetch whole entries ([`ByteSource::read_all`] /
//!   [`TilesetSource::Exploded`]).
//! * Native gets a real implementation (filesystem + blocking reqwest on a
//!   worker thread) instead of basemap's fail-fast stub, because the T0 gate
//!   requires the fixture to render natively too.
//!
//! T1 additions:
//! * **Cache-Storage CAS** for whole-entry reads ([`TilesetSource::read_entry_cached`]):
//!   keyed by the SAS-stripped archive/base URL + entry path (asset blobs are
//!   hash-named and immutable, mirroring `asset_loader::remote_source`). The
//!   ranged *open* path (index/suffix reads) is never cached — only complete
//!   entries are.
//! * **Abort plumbing** ([`AbortHandle`] + the generation-keyed registry): the
//!   scheduler cancels the actual network transfer of a request that fell out
//!   of the cut, not just its slot state. wasm wires a real `AbortController`
//!   into every fetch; native checks a flag between range requests (blocking
//!   reqwest can't be interrupted mid-transfer — tiles are MBs, good enough).

use std::path::PathBuf;
use std::sync::Arc;

/// IO failure for a tile/tileset byte source.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FetchError {
    #[error("http error: {0}")]
    Http(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("range out of bounds: {start}+{len} > {size}")]
    OutOfRange { start: u64, len: u64, size: u64 },
    #[error("request aborted (fell out of the cut)")]
    Aborted,
}

// ── Cancellation ─────────────────────────────────────────────────────────────

/// Cross-platform cancellation token for one tile request. Cheap to clone —
/// clones share the same underlying controller/flag.
#[derive(Debug, Clone)]
pub struct AbortHandle {
    #[cfg(target_arch = "wasm32")]
    controller: web_sys::AbortController,
    #[cfg(not(target_arch = "wasm32"))]
    flag: Arc<std::sync::atomic::AtomicBool>,
}

impl AbortHandle {
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            // AbortController::new only fails in pathological environments;
            // degrade to a dummy that can never be constructed — unwrap is the
            // pragmatic choice (the basemap fetch layer makes the same call).
            Self { controller: web_sys::AbortController::new().expect("AbortController") }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self { flag: Arc::new(std::sync::atomic::AtomicBool::new(false)) }
        }
    }

    pub fn trigger(&self) {
        #[cfg(target_arch = "wasm32")]
        self.controller.abort();
        #[cfg(not(target_arch = "wasm32"))]
        self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_triggered(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.controller.signal().aborted()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.flag.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn signal(&self) -> web_sys::AbortSignal {
        self.controller.signal()
    }
}

impl Default for AbortHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Bail out early when the request was cancelled.
fn check_abort(abort: Option<&AbortHandle>) -> Result<(), FetchError> {
    match abort {
        Some(a) if a.is_triggered() => Err(FetchError::Aborted),
        _ => Ok(()),
    }
}

// Generation-keyed abort registry. Lives OUTSIDE the ECS: on wasm an
// `AbortController` is a JS object (not `Send`), so it can't sit in a
// `Resource` slot — the scheduler talks to in-flight tasks through this map
// instead. wasm is single-threaded (`thread_local` suffices); native tasks run
// on worker threads (mutexed map).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static ABORTS: std::cell::RefCell<std::collections::HashMap<u64, AbortHandle>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}
#[cfg(not(target_arch = "wasm32"))]
static ABORTS: std::sync::Mutex<
    Option<std::collections::HashMap<u64, AbortHandle>>,
> = std::sync::Mutex::new(None);

/// Create + register the abort handle for a request generation.
pub fn register_abort(generation: u64) -> AbortHandle {
    let handle = AbortHandle::new();
    #[cfg(target_arch = "wasm32")]
    ABORTS.with(|m| m.borrow_mut().insert(generation, handle.clone()));
    #[cfg(not(target_arch = "wasm32"))]
    ABORTS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(generation, handle.clone());
    handle
}

/// Abort the in-flight request of `generation` (no-op when already finished).
pub fn trigger_abort(generation: u64) {
    #[cfg(target_arch = "wasm32")]
    ABORTS.with(|m| {
        if let Some(h) = m.borrow().get(&generation) {
            h.trigger();
        }
    });
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(map) = ABORTS.lock().unwrap().as_ref()
        && let Some(h) = map.get(&generation)
    {
        h.trigger();
    }
}

/// Drop a finished request's registry entry (called by the task on completion).
pub fn unregister_abort(generation: u64) {
    #[cfg(target_arch = "wasm32")]
    ABORTS.with(|m| {
        m.borrow_mut().remove(&generation);
    });
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(map) = ABORTS.lock().unwrap().as_mut() {
        map.remove(&generation);
    }
}

/// A random-access byte source: an in-memory buffer (tests), a local file
/// (native dev + unit tests over the committed fixture), or an HTTP URL
/// (range-GETs on wasm via `gloo-net`, blocking reqwest on native).
#[derive(Debug, Clone)]
pub enum ByteSource {
    Mem(Arc<Vec<u8>>),
    File(PathBuf),
    Http(String),
}

impl ByteSource {
    /// Total size in bytes. For HTTP this costs one HEAD (or suffix-range)
    /// request; callers should prefer [`ByteSource::read_suffix`] when they
    /// want the tail anyway.
    pub async fn size(&self) -> Result<u64, FetchError> {
        match self {
            ByteSource::Mem(buf) => Ok(buf.len() as u64),
            ByteSource::File(path) => std::fs::metadata(path)
                .map(|m| m.len())
                .map_err(|e| FetchError::Io(format!("{}: {e}", path.display()))),
            ByteSource::Http(url) => http_size(url).await,
        }
    }

    /// Read exactly `len` bytes at absolute offset `start`.
    pub async fn read(&self, start: u64, len: u64) -> Result<Vec<u8>, FetchError> {
        self.read_abortable(start, len, None).await
    }

    /// [`Self::read`] with an optional cancellation token.
    pub async fn read_abortable(
        &self,
        start: u64,
        len: u64,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<u8>, FetchError> {
        check_abort(abort)?;
        match self {
            ByteSource::Mem(buf) => {
                let size = buf.len() as u64;
                let end = start.checked_add(len).filter(|&e| e <= size).ok_or(
                    FetchError::OutOfRange { start, len, size },
                )?;
                Ok(buf[start as usize..end as usize].to_vec())
            }
            ByteSource::File(path) => {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = std::fs::File::open(path)
                    .map_err(|e| FetchError::Io(format!("{}: {e}", path.display())))?;
                f.seek(SeekFrom::Start(start))
                    .map_err(|e| FetchError::Io(e.to_string()))?;
                let mut out = vec![0u8; len as usize];
                f.read_exact(&mut out).map_err(|e| {
                    FetchError::Io(format!("short read at {start}+{len}: {e}"))
                })?;
                Ok(out)
            }
            ByteSource::Http(url) => http_range(url, start, len, abort).await,
        }
    }

    /// Read the last `n` bytes, returning `(tail_bytes, total_size)`. One
    /// request on HTTP (suffix range); the tail may be shorter than `n` when
    /// the source itself is.
    pub async fn read_suffix(&self, n: u64) -> Result<(Vec<u8>, u64), FetchError> {
        match self {
            ByteSource::Http(url) => http_suffix(url, n).await,
            _ => {
                let size = self.size().await?;
                let n = n.min(size);
                let bytes = self.read(size - n, n).await?;
                Ok((bytes, size))
            }
        }
    }

    /// Read the whole source (exploded-tileset entries; plain GET on HTTP).
    pub async fn read_all(&self) -> Result<Vec<u8>, FetchError> {
        self.read_all_abortable(None).await
    }

    /// [`Self::read_all`] with an optional cancellation token.
    pub async fn read_all_abortable(
        &self,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<u8>, FetchError> {
        check_abort(abort)?;
        match self {
            ByteSource::Mem(buf) => Ok(buf.as_ref().clone()),
            ByteSource::File(path) => std::fs::read(path)
                .map_err(|e| FetchError::Io(format!("{}: {e}", path.display()))),
            ByteSource::Http(url) => http_get_all(url, abort).await,
        }
    }
}

/// Where a tileset's entries come from: a directory/base-URL of loose files
/// (`tileset.json` + relative content URIs) or a packed `.3tz` archive
/// (range-streamed; see [`super::archive::Archive3tz`]).
#[derive(Debug, Clone)]
pub enum TilesetSource {
    /// Base location; entry `uri`s resolve relative to it.
    Exploded(ExplodedBase),
    Archive(Arc<super::archive::Archive3tz>),
}

/// Base for an exploded tileset: a native directory or an HTTP base URL.
#[derive(Debug, Clone)]
pub enum ExplodedBase {
    Dir(PathBuf),
    /// Kept WITHOUT a trailing slash; `join` inserts it.
    Url(String),
}

impl ExplodedBase {
    fn join(&self, rel: &str) -> ByteSource {
        let rel = rel.trim_start_matches('/');
        match self {
            ExplodedBase::Dir(dir) => ByteSource::File(dir.join(rel)),
            ExplodedBase::Url(base) => {
                ByteSource::Http(format!("{}/{rel}", base.trim_end_matches('/')))
            }
        }
    }
}

impl TilesetSource {
    /// Fetch one entry (e.g. `"tileset.json"`, `"content/3/2/1.glb"`) by its
    /// tileset-relative URI.
    pub async fn read_entry(&self, uri: &str) -> Result<Vec<u8>, FetchError> {
        self.read_entry_raw(uri, None).await
    }

    async fn read_entry_raw(
        &self,
        uri: &str,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<u8>, FetchError> {
        match self {
            TilesetSource::Exploded(base) => base.join(uri).read_all_abortable(abort).await,
            TilesetSource::Archive(ar) => {
                ar.read_entry_abortable(uri, abort).await.map_err(|e| match e {
                    super::archive::ArchiveError::Fetch(FetchError::Aborted) => {
                        FetchError::Aborted
                    }
                    e => FetchError::Io(format!("3tz entry {uri}: {e}")),
                })
            }
        }
    }

    /// [`Self::read_entry`] through the content-addressed Cache Storage layer
    /// (wasm; a pass-through elsewhere), with optional cancellation.
    ///
    /// Cache key = SAS-stripped source URL + `/` + entry path. Asset blobs are
    /// hash-named (`…/whole/<hash>.3tz`) and mirror prefixes are
    /// version-scoped, so the key is content-addressed and survives SAS
    /// rotation — same scheme as `asset_loader::remote_source`.
    pub async fn read_entry_cached(
        &self,
        uri: &str,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<u8>, FetchError> {
        let key = self.entry_cache_key(uri);
        #[cfg(target_arch = "wasm32")]
        if let Some(key) = &key
            && let Some(bytes) = cache_get(key).await
        {
            return Ok(bytes);
        }
        let bytes = self.read_entry_raw(uri, abort).await?;
        #[cfg(target_arch = "wasm32")]
        if let Some(key) = key {
            cache_store_bytes(key, &bytes);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = key;
        Ok(bytes)
    }

    /// Stable cache key for one entry, or `None` when the source isn't an
    /// absolute HTTP URL (the Cache Storage API requires URL keys; local
    /// fixtures and `Mem` sources don't need caching).
    fn entry_cache_key(&self, uri: &str) -> Option<String> {
        let base = match self {
            TilesetSource::Archive(ar) => match ar.source() {
                ByteSource::Http(url) => url.as_str(),
                _ => return None,
            },
            TilesetSource::Exploded(ExplodedBase::Url(base)) => base.as_str(),
            TilesetSource::Exploded(ExplodedBase::Dir(_)) => return None,
        };
        if !(base.starts_with("https://") || base.starts_with("http://")) {
            return None;
        }
        let stripped = base.split('?').next().unwrap_or(base).trim_end_matches('/');
        Some(format!("{stripped}/{}", uri.trim_start_matches('/')))
    }
}

/// Cache Storage bucket shared with the whole-file asset path
/// (`asset_loader::remote_source`) — one CAS, one invalidation knob.
#[cfg(target_arch = "wasm32")]
const CONTENT_CACHE: &str = "tt-asset-cas-v1";

/// Look up `key` in the CAS bucket; `None` on miss or unavailable storage.
#[cfg(target_arch = "wasm32")]
async fn cache_get(key: &str) -> Option<Vec<u8>> {
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let caches = web_sys::window()?.caches().ok()?;
    let cache: web_sys::Cache =
        JsFuture::from(caches.open(CONTENT_CACHE)).await.ok()?.dyn_into().ok()?;
    let matched = JsFuture::from(cache.match_with_str(key)).await.ok()?;
    if matched.is_undefined() {
        return None;
    }
    let resp: web_sys::Response = matched.dyn_into().ok()?;
    let buf = JsFuture::from(resp.array_buffer().ok()?).await.ok()?;
    Some(Uint8Array::new(&buf).to_vec())
}

/// Persist decoded entry bytes under `key`, fire-and-forget (best-effort —
/// quota/denied storage only costs future-session speed).
#[cfg(target_arch = "wasm32")]
fn cache_store_bytes(key: String, bytes: &[u8]) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::{spawn_local, JsFuture};

    let mut owned = bytes.to_vec();
    spawn_local(async move {
        let Some(caches) = web_sys::window().and_then(|w| w.caches().ok()) else {
            return;
        };
        let Ok(opened) = JsFuture::from(caches.open(CONTENT_CACHE)).await else {
            return;
        };
        let Ok(cache) = opened.dyn_into::<web_sys::Cache>() else {
            return;
        };
        // The constructor copies into the JS heap; `owned` frees at task end.
        let Ok(response) = web_sys::Response::new_with_opt_u8_array(Some(owned.as_mut_slice()))
        else {
            return;
        };
        if JsFuture::from(cache.put_with_str(&key, &response)).await.is_err() {
            bevy::log::debug!("tiles3d: cache store failed for {key} (non-fatal)");
        }
    });
}

/// Spawn a fire-and-forget IO task. wasm: `spawn_local` on the single-threaded
/// executor (the future yields at every `.await` — never blocks). Native: a
/// worker thread driving the future to completion, so file/HTTP IO stays off
/// the frame loop just like on wasm.
#[cfg(target_arch = "wasm32")]
pub fn spawn_io<F: std::future::Future<Output = ()> + 'static>(fut: F) {
    wasm_bindgen_futures::spawn_local(fut);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_io<F: std::future::Future<Output = ()> + Send + 'static>(fut: F) {
    std::thread::spawn(move || bevy::tasks::block_on(fut));
}

// ── HTTP backends ────────────────────────────────────────────────────────────

/// Parse the total from a `Content-Range: bytes <start>-<end>/<total>` header.
fn parse_content_range_total(value: &str) -> Option<u64> {
    value.trim().strip_prefix("bytes")?.trim().rsplit_once('/')?.1.trim().parse().ok()
}

#[cfg(target_arch = "wasm32")]
async fn http_size(url: &str) -> Result<u64, FetchError> {
    // Suffix-range probe instead of HEAD: some CDN/CORS configs omit
    // Content-Length on HEAD but must send Content-Range on a 206.
    let (_, total) = http_suffix(url, 1).await?;
    Ok(total)
}

/// Map a gloo fetch error, recognizing user-triggered aborts.
#[cfg(target_arch = "wasm32")]
fn map_gloo_error(e: gloo_net::Error, abort: Option<&AbortHandle>) -> FetchError {
    if abort.is_some_and(|a| a.is_triggered()) {
        FetchError::Aborted
    } else {
        FetchError::Http(e.to_string())
    }
}

#[cfg(target_arch = "wasm32")]
async fn http_range(
    url: &str,
    start: u64,
    len: u64,
    abort: Option<&AbortHandle>,
) -> Result<Vec<u8>, FetchError> {
    let end = start + len - 1;
    let resp = gloo_net::http::Request::get(url)
        .header("Range", &format!("bytes={start}-{end}"))
        .abort_signal(abort.map(|a| a.signal()).as_ref())
        .send()
        .await
        .map_err(|e| map_gloo_error(e, abort))?;
    match resp.status() {
        206 => resp.binary().await.map_err(|e| FetchError::Http(e.to_string())),
        // Server ignored the Range header (no range support): take the whole
        // body and slice. Degraded but correct — matters only for dev servers.
        200 => {
            let body = resp.binary().await.map_err(|e| FetchError::Http(e.to_string()))?;
            let size = body.len() as u64;
            let end = start.checked_add(len).filter(|&e| e <= size).ok_or(
                FetchError::OutOfRange { start, len, size },
            )?;
            Ok(body[start as usize..end as usize].to_vec())
        }
        s => Err(FetchError::Http(format!("status {s} for ranged GET {url}"))),
    }
}

#[cfg(target_arch = "wasm32")]
async fn http_suffix(url: &str, n: u64) -> Result<(Vec<u8>, u64), FetchError> {
    // The probe gets its own AbortController: a server that doesn't support
    // suffix ranges answers 200 + FULL body (Azure Blob does exactly this —
    // verified live; only explicit ranges get a 206), and the transfer must
    // be cancelled after the headers, not drained.
    let probe_abort = AbortHandle::new();
    let resp = gloo_net::http::Request::get(url)
        .header("Range", &format!("bytes=-{n}"))
        .abort_signal(Some(&probe_abort.signal()))
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;
    match resp.status() {
        206 => {
            let total = resp
                .headers()
                .get("content-range")
                .as_deref()
                .and_then(parse_content_range_total)
                .ok_or_else(|| {
                    FetchError::Http(format!(
                        "206 without a parseable Content-Range for {url} — \
                         is Content-Range in the CORS ExposedHeaders?"
                    ))
                })?;
            let bytes = resp.binary().await.map_err(|e| FetchError::Http(e.to_string()))?;
            Ok((bytes, total))
        }
        200 => {
            // Suffix range unsupported. With a Content-Length we cancel the
            // full-body transfer and re-ask for the explicit tail range; only
            // a length-less response (dev servers) gets drained whole.
            let total = resp
                .headers()
                .get("content-length")
                .and_then(|v| v.parse::<u64>().ok());
            match total {
                Some(total) if total > 0 => {
                    probe_abort.trigger();
                    drop(resp);
                    let n = n.min(total);
                    let bytes = http_range(url, total - n, n, None).await?;
                    Ok((bytes, total))
                }
                _ => {
                    let body =
                        resp.binary().await.map_err(|e| FetchError::Http(e.to_string()))?;
                    let total = body.len() as u64;
                    let n = n.min(total) as usize;
                    Ok((body[body.len() - n..].to_vec(), total))
                }
            }
        }
        s => Err(FetchError::Http(format!("status {s} for suffix GET {url}"))),
    }
}

#[cfg(target_arch = "wasm32")]
async fn http_get_all(url: &str, abort: Option<&AbortHandle>) -> Result<Vec<u8>, FetchError> {
    let resp = gloo_net::http::Request::get(url)
        .abort_signal(abort.map(|a| a.signal()).as_ref())
        .send()
        .await
        .map_err(|e| map_gloo_error(e, abort))?;
    if !resp.ok() {
        return Err(FetchError::Http(format!("status {} for GET {url}", resp.status())));
    }
    resp.binary().await.map_err(|e| map_gloo_error(e, abort))
}

// Native HTTP: blocking reqwest on the worker thread `spawn_io` already runs
// us on (dev/test convenience — production tile streaming is the wasm path).
#[cfg(not(target_arch = "wasm32"))]
async fn http_size(url: &str) -> Result<u64, FetchError> {
    let (_, total) = http_suffix(url, 1).await?;
    Ok(total)
}

#[cfg(not(target_arch = "wasm32"))]
async fn http_range(
    url: &str,
    start: u64,
    len: u64,
    abort: Option<&AbortHandle>,
) -> Result<Vec<u8>, FetchError> {
    check_abort(abort)?;
    let end = start + len - 1;
    let resp = reqwest::blocking::Client::new()
        .get(url)
        .header("Range", format!("bytes={start}-{end}"))
        .send()
        .map_err(|e| FetchError::Http(e.to_string()))?;
    let status = resp.status().as_u16();
    let body = resp.bytes().map_err(|e| FetchError::Http(e.to_string()))?;
    match status {
        206 => Ok(body.to_vec()),
        200 => {
            let size = body.len() as u64;
            let end = start.checked_add(len).filter(|&e| e <= size).ok_or(
                FetchError::OutOfRange { start, len, size },
            )?;
            Ok(body[start as usize..end as usize].to_vec())
        }
        s => Err(FetchError::Http(format!("status {s} for ranged GET {url}"))),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn http_suffix(url: &str, n: u64) -> Result<(Vec<u8>, u64), FetchError> {
    let resp = reqwest::blocking::Client::new()
        .get(url)
        .header("Range", format!("bytes=-{n}"))
        .send()
        .map_err(|e| FetchError::Http(e.to_string()))?;
    let status = resp.status().as_u16();
    let content_range = resp
        .headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    // Suffix range unsupported (Azure Blob: 200 + full body) → drop the
    // response after the headers (closes the connection mid-transfer) and
    // re-ask for the explicit tail range. Only a length-less 200 is drained.
    if status == 200 {
        let total = resp.content_length().filter(|&t| t > 0);
        if let Some(total) = total {
            drop(resp);
            let n = n.min(total);
            let bytes = http_range(url, total - n, n, None).await?;
            return Ok((bytes, total));
        }
    }
    let body = resp.bytes().map_err(|e| FetchError::Http(e.to_string()))?;
    match status {
        206 => {
            let total = content_range
                .as_deref()
                .and_then(parse_content_range_total)
                .ok_or_else(|| FetchError::Http(format!("206 without Content-Range for {url}")))?;
            Ok((body.to_vec(), total))
        }
        200 => {
            let total = body.len() as u64;
            let n = n.min(total) as usize;
            Ok((body[body.len() - n..].to_vec(), total))
        }
        s => Err(FetchError::Http(format!("status {s} for suffix GET {url}"))),
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn http_get_all(url: &str, abort: Option<&AbortHandle>) -> Result<Vec<u8>, FetchError> {
    check_abort(abort)?;
    let resp = reqwest::blocking::get(url).map_err(|e| FetchError::Http(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(FetchError::Http(format!("status {} for GET {url}", resp.status())));
    }
    resp.bytes()
        .map(|b| b.to_vec())
        .map_err(|e| FetchError::Http(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_total_parses() {
        assert_eq!(parse_content_range_total("bytes 100-199/4096"), Some(4096));
        assert_eq!(parse_content_range_total(" bytes 0-0/12"), Some(12));
        assert_eq!(parse_content_range_total("bytes */512"), Some(512));
        assert_eq!(parse_content_range_total("garbage"), None);
    }

    #[test]
    fn mem_source_range_and_suffix() {
        let src = ByteSource::Mem(Arc::new((0u8..=99).collect()));
        let bytes = bevy::tasks::block_on(src.read(10, 5)).unwrap();
        assert_eq!(bytes, vec![10, 11, 12, 13, 14]);
        let (tail, total) = bevy::tasks::block_on(src.read_suffix(3)).unwrap();
        assert_eq!(total, 100);
        assert_eq!(tail, vec![97, 98, 99]);
        assert!(bevy::tasks::block_on(src.read(99, 2)).is_err());
    }
}
