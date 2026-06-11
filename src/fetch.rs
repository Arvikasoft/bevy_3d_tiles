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
//! Cache-Storage CAS is deliberately NOT here yet: range reads need a
//! range-aware key scheme, which lands with T1's real blob tilesets (whole-tile
//! entries keyed by archive hash + entry path). The basemap CAS pattern carries
//! over then.

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
            ByteSource::Http(url) => http_range(url, start, len).await,
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
        match self {
            ByteSource::Mem(buf) => Ok(buf.as_ref().clone()),
            ByteSource::File(path) => std::fs::read(path)
                .map_err(|e| FetchError::Io(format!("{}: {e}", path.display()))),
            ByteSource::Http(url) => http_get_all(url).await,
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
        match self {
            TilesetSource::Exploded(base) => base.join(uri).read_all().await,
            TilesetSource::Archive(ar) => ar
                .read_entry(uri)
                .await
                .map_err(|e| FetchError::Io(format!("3tz entry {uri}: {e}"))),
        }
    }
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

#[cfg(target_arch = "wasm32")]
async fn http_range(url: &str, start: u64, len: u64) -> Result<Vec<u8>, FetchError> {
    let end = start + len - 1;
    let resp = gloo_net::http::Request::get(url)
        .header("Range", &format!("bytes={start}-{end}"))
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;
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
    let resp = gloo_net::http::Request::get(url)
        .header("Range", &format!("bytes=-{n}"))
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
            let body = resp.binary().await.map_err(|e| FetchError::Http(e.to_string()))?;
            let total = body.len() as u64;
            let n = n.min(total) as usize;
            Ok((body[body.len() - n..].to_vec(), total))
        }
        s => Err(FetchError::Http(format!("status {s} for suffix GET {url}"))),
    }
}

#[cfg(target_arch = "wasm32")]
async fn http_get_all(url: &str) -> Result<Vec<u8>, FetchError> {
    let resp = gloo_net::http::Request::get(url)
        .send()
        .await
        .map_err(|e| FetchError::Http(e.to_string()))?;
    if !resp.ok() {
        return Err(FetchError::Http(format!("status {} for GET {url}", resp.status())));
    }
    resp.binary().await.map_err(|e| FetchError::Http(e.to_string()))
}

// Native HTTP: blocking reqwest on the worker thread `spawn_io` already runs
// us on (dev/test convenience — production tile streaming is the wasm path).
#[cfg(not(target_arch = "wasm32"))]
async fn http_size(url: &str) -> Result<u64, FetchError> {
    let (_, total) = http_suffix(url, 1).await?;
    Ok(total)
}

#[cfg(not(target_arch = "wasm32"))]
async fn http_range(url: &str, start: u64, len: u64) -> Result<Vec<u8>, FetchError> {
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
async fn http_get_all(url: &str) -> Result<Vec<u8>, FetchError> {
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
