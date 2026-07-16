//! `.3tz` ranged reader (BEVY-3D-TILES-PLAN T0, decision D2).
//!
//! A `.3tz` is a ZIP archive whose **last** entry is an uncompressed index
//! file `@3dtilesIndex1@`: 24-byte records (`MD5(path)` 16 bytes + u64 LE
//! offset of the entry's Local File Header), sorted ascending by the MD5
//! interpreted as two **little-endian u64s** — bytes `[0..8]` compared first,
//! then `[8..16]` (Maxar 3tz spec v1.4). That design makes remote reading
//! possible without a zip library or a full central-directory download:
//!
//! 1. **Open** — ONE parallel round-trip pair: a suffix range-GET (EOCD +
//!    ZIP64 + central directory + the index payload, all inside the 512 KiB
//!    tail for multi-thousand-tile sets) and a speculative head range-GET
//!    (our tilers front-pack `tileset.json` + the root tile, so the first
//!    rendered cut is usually already in memory when open returns).
//! 2. **Per entry** — MD5 the normalized path (backslashes → `/`, leading `/`
//!    stripped), binary-search the in-memory index, then serve from the open
//!    head (zero requests), else **one range-GET** covering header + data via
//!    the exact span derived from the next entry's offset, else the legacy
//!    header-then-data pair. Equal-hash collisions are adjacent in the index
//!    and disambiguated by the filename in the Local File Header.
//!
//! Supported entry compression: stored (0) and DEFLATE (8, via `miniz_oxide`).
//! Zstandard (93) is rejected with a clear error — our tilers (D3) emit
//! stored/deflate only. Inner files are capped at 4 GB by the format (no
//! per-entry ZIP64 sizes are written by conforming writers); tile content is
//! MBs, and the tiler asserts the cap at pack time (plan §8).

use md5::{Digest, Md5};

use super::fetch::{AbortHandle, ByteSource, FetchError};

/// The mandated index filename (and its fixed 15-byte length).
const INDEX_NAME: &[u8] = b"@3dtilesIndex1@";

/// Suffix fetched on open. Covers the EOCD (+comment), ZIP64 records, the
/// central-directory, AND the whole index payload for multi-thousand-tile
/// sets (~24 B/record index + ~66 B/entry CD ≈ 90 KiB per 1000 tiles), so the
/// open completes with no follow-up requests. Larger sets degrade gracefully
/// to one extra ranged read.
const OPEN_TAIL_BYTES: u64 = 512 * 1024;

/// Speculative head fetched IN PARALLEL with the tail on open. Our tilers
/// front-pack the archive — `tileset.json` is the first entry and the root
/// tile the second (pack3tz writes preorder) — so this window usually holds
/// the tileset (and on smaller sets the root too): the open costs one parallel
/// round-trip pair instead of five serial round trips (measured 3.2 s to
/// tileset-parsed on a 2074-tile set over Azure Blob).
///
/// SIZED FOR BANDWIDTH, not just RTTs: speculation is a transfer tax on every
/// cold open. The first cut of this at 2 MiB made a 13-tile / 4.4 MiB archive
/// open SLOWER than the old serial reader on a ~15 Mbps path (2.8 s vs 1.6 s —
/// the bytes cost more than the round trips saved). At 512 KiB small sets keep
/// the win and a big set's root merely costs one extra span-bounded GET.
const OPEN_HEAD_BYTES: u64 = 512 * 1024;

/// Local-File-Header over-fetch: one range-GET grabs the 30-byte header, the
/// name/extra fields, and — for small entries (tileset.json, coarse tiles) —
/// often the entire payload, so the second GET is skipped. Fallback path only;
/// entries with a known span (see `span_of`) fetch header + data in one GET.
const LFH_OVERFETCH: u64 = 4096;

/// Cap on the single-GET entry span (`next entry offset − this offset`). Spans
/// are exact for gap-free archives (ours); a foreign archive with padding
/// between entries would over-read, so bound the damage. Entries larger than
/// the cap fall back to header-then-data reads.
const MAX_ENTRY_SPAN: u64 = 64 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    #[error(transparent)]
    Fetch(#[from] FetchError),
    #[error("not a zip archive (no end-of-central-directory record found)")]
    NotZip,
    #[error("no @3dtilesIndex1@ entry — not a 3tz (or index not last in the central directory)")]
    NoIndex,
    #[error("malformed 3tz: {0}")]
    Corrupt(String),
    #[error("unsupported 3tz feature: {0}")]
    Unsupported(String),
    #[error("entry not found in 3tz index: {0}")]
    EntryNotFound(String),
}

/// One parsed index record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexRecord {
    pub hash: [u8; 16],
    /// Absolute archive offset of the entry's Local File Header.
    pub offset: u64,
}

/// The in-memory `@3dtilesIndex1@` table.
#[derive(Debug, Clone, Default)]
pub struct Index3tz {
    /// Sorted by [`hash_sort_key`].
    records: Vec<IndexRecord>,
}

/// Spec sort/compare key: the MD5 as two little-endian u64s, bytes `[0..8]`
/// compared first, then `[8..16]`.
pub fn hash_sort_key(hash: &[u8; 16]) -> (u64, u64) {
    (
        u64::from_le_bytes(hash[0..8].try_into().unwrap()),
        u64::from_le_bytes(hash[8..16].try_into().unwrap()),
    )
}

/// Normalize an archive path per the 3tz spec before hashing: backslashes
/// become forward slashes, leading slashes are dropped.
pub fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches('/').to_string()
}

/// MD5 of a normalized archive path.
pub fn md5_of_path(normalized: &str) -> [u8; 16] {
    Md5::digest(normalized.as_bytes()).into()
}

impl Index3tz {
    /// Parse the raw index payload (`len % 24 == 0`). Records out of spec
    /// order are tolerated by re-sorting (the lookup only needs *a* sorted
    /// order consistent with [`hash_sort_key`]).
    pub fn parse(bytes: &[u8]) -> Result<Self, ArchiveError> {
        if !bytes.len().is_multiple_of(24) {
            return Err(ArchiveError::Corrupt(format!(
                "index payload length {} not a multiple of 24",
                bytes.len()
            )));
        }
        let mut records: Vec<IndexRecord> = bytes
            .chunks_exact(24)
            .map(|rec| IndexRecord {
                hash: rec[0..16].try_into().unwrap(),
                offset: u64::from_le_bytes(rec[16..24].try_into().unwrap()),
            })
            .collect();
        if !records.is_sorted_by_key(|r| hash_sort_key(&r.hash)) {
            records.sort_by_key(|r| hash_sort_key(&r.hash));
        }
        Ok(Self { records })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// All candidate Local-File-Header offsets whose record hash equals
    /// `MD5(normalized_path)` — adjacent in the table; almost always 0 or 1.
    /// The caller verifies the filename in the Local File Header.
    pub fn lookup(&self, normalized_path: &str) -> Vec<u64> {
        let hash = md5_of_path(normalized_path);
        let key = hash_sort_key(&hash);
        let start = self
            .records
            .partition_point(|r| hash_sort_key(&r.hash) < key);
        self.records[start..]
            .iter()
            .take_while(|r| r.hash == hash)
            .map(|r| r.offset)
            .collect()
    }
}

// ── Little-endian slice readers ──────────────────────────────────────────────

fn le16(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(buf[at..at + 2].try_into().unwrap())
}
fn le32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
}
fn le64(buf: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(buf[at..at + 8].try_into().unwrap())
}

// ── ZIP structure parsing (pure, slice-based) ────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Eocd {
    /// Absolute offset of the EOCD record itself.
    abs: u64,
    cd_offset: u32,
    cd_size: u32,
    /// Absolute offset of the ZIP64 EOCD *record* from the locator directly
    /// preceding the EOCD, when present.
    zip64_record_abs: Option<u64>,
}

/// Find the End-Of-Central-Directory record in the archive tail. Scans
/// backwards for `PK\x05\x06` and validates self-consistency: the record +
/// its comment must end exactly at end-of-file.
fn find_eocd(tail: &[u8], tail_abs: u64) -> Result<Eocd, ArchiveError> {
    if tail.len() < 22 {
        return Err(ArchiveError::NotZip);
    }
    let mut pos = tail.len() - 22;
    loop {
        if tail[pos..pos + 4] == [0x50, 0x4B, 0x05, 0x06] {
            let comment_len = le16(tail, pos + 20) as usize;
            if pos + 22 + comment_len == tail.len() {
                let zip64_record_abs = (pos >= 20
                    && tail[pos - 20..pos - 16] == [0x50, 0x4B, 0x06, 0x07])
                .then(|| le64(tail, pos - 12));
                return Ok(Eocd {
                    abs: tail_abs + pos as u64,
                    cd_offset: le32(tail, pos + 16),
                    cd_size: le32(tail, pos + 12),
                    zip64_record_abs,
                });
            }
        }
        if pos == 0 {
            return Err(ArchiveError::NotZip);
        }
        pos -= 1;
    }
}

/// Extract a u64 field from a ZIP64 extended-information extra field
/// (header id `0x0001`). `field_rank` is the position of the wanted field
/// among the fields PRESENT in the block — per APPNOTE the block contains, in
/// order, only the values whose 32-bit counterparts are saturated:
/// uncompressed size, compressed size, local-header offset, disk number.
fn zip64_extra_field(extra: &[u8], field_rank: usize) -> Option<u64> {
    let mut at = 0;
    while at + 4 <= extra.len() {
        let id = le16(extra, at);
        let len = le16(extra, at + 2) as usize;
        let body = extra.get(at + 4..at + 4 + len)?;
        if id == 0x0001 {
            let off = field_rank * 8;
            return (off + 8 <= body.len()).then(|| le64(body, off));
        }
        at += 4 + len;
    }
    None
}

/// The index's central-directory entry, located by scanning backwards from
/// the end of the central directory (the spec mandates the index is the LAST
/// entry, with no comment).
#[derive(Debug, Clone, Copy)]
struct IndexCdEntry {
    method: u16,
    comp_size: u64,
    local_offset: u64,
}

/// Scan `buf` (absolute start `buf_abs`, covering the central directory's
/// tail up to `cd_end_abs`) backwards for the `@3dtilesIndex1@` entry. The
/// consistency requirement — the entry must END exactly at `cd_end_abs` —
/// makes the backwards scan unambiguous.
fn find_index_cd_entry(
    buf: &[u8],
    buf_abs: u64,
    cd_end_abs: u64,
) -> Result<IndexCdEntry, ArchiveError> {
    let in_buf_end = (cd_end_abs.saturating_sub(buf_abs)).min(buf.len() as u64) as usize;
    // Smallest possible entry: 46-byte fixed header + 15-byte name.
    let min_len = 46 + INDEX_NAME.len();
    if in_buf_end < min_len {
        return Err(ArchiveError::NoIndex);
    }
    let mut pos = in_buf_end - min_len;
    loop {
        if buf[pos..pos + 4] == [0x50, 0x4B, 0x01, 0x02] {
            let name_len = le16(buf, pos + 28) as usize;
            let extra_len = le16(buf, pos + 30) as usize;
            let comment_len = le16(buf, pos + 32) as usize;
            let total = 46 + name_len + extra_len + comment_len;
            if name_len == INDEX_NAME.len()
                && comment_len == 0
                && buf_abs + (pos + total) as u64 == cd_end_abs
                && buf.get(pos + 46..pos + 46 + name_len) == Some(INDEX_NAME)
            {
                let extra = &buf[pos + 46 + name_len..pos + 46 + name_len + extra_len];
                let comp_size32 = le32(buf, pos + 20);
                let local_offset32 = le32(buf, pos + 42);
                // ZIP64 extra fields appear in saturation order; track the
                // rank of each saturated field to index the right u64.
                let uncomp_saturated = (le32(buf, pos + 24) == u32::MAX) as usize;
                let comp_saturated = (comp_size32 == u32::MAX) as usize;
                let comp_size = if comp_size32 == u32::MAX {
                    zip64_extra_field(extra, uncomp_saturated).ok_or_else(|| {
                        ArchiveError::Corrupt("saturated comp size without zip64 extra".into())
                    })?
                } else {
                    comp_size32 as u64
                };
                let local_offset = if local_offset32 == u32::MAX {
                    zip64_extra_field(extra, uncomp_saturated + comp_saturated).ok_or_else(
                        || ArchiveError::Corrupt("saturated offset without zip64 extra".into()),
                    )?
                } else {
                    local_offset32 as u64
                };
                return Ok(IndexCdEntry {
                    method: le16(buf, pos + 10),
                    comp_size,
                    local_offset,
                });
            }
        }
        if pos == 0 {
            return Err(ArchiveError::NoIndex);
        }
        pos -= 1;
    }
}

/// Parsed Local File Header.
#[derive(Debug)]
struct LocalHeader {
    method: u16,
    /// `None` when the writer used streaming mode (flag bit 3 with zeroed
    /// sizes) — unsupported for ranged reading.
    comp_size: Option<u64>,
    name: Vec<u8>,
    /// Offset of the entry DATA relative to the Local File Header start.
    data_rel: u64,
}

fn parse_local_header(buf: &[u8]) -> Result<LocalHeader, ArchiveError> {
    if buf.len() < 30 || buf[0..4] != [0x50, 0x4B, 0x03, 0x04] {
        return Err(ArchiveError::Corrupt(
            "bad local file header signature".into(),
        ));
    }
    let flags = le16(buf, 6);
    let method = le16(buf, 8);
    let comp_size32 = le32(buf, 18);
    let name_len = le16(buf, 26) as usize;
    let extra_len = le16(buf, 28) as usize;
    let header_len = 30 + name_len + extra_len;
    let Some(name) = buf.get(30..30 + name_len) else {
        return Err(ArchiveError::Corrupt(
            "local header name/extra exceed over-fetch window".into(),
        ));
    };
    let extra = buf.get(30 + name_len..header_len).unwrap_or(&[]);
    let comp_size = if comp_size32 == u32::MAX {
        // Local-header ZIP64 extra carries uncompressed then compressed size.
        Some(zip64_extra_field(extra, 1).ok_or_else(|| {
            ArchiveError::Corrupt("saturated local comp size without zip64 extra".into())
        })?)
    } else if comp_size32 == 0 && flags & 0x0008 != 0 {
        None
    } else {
        Some(comp_size32 as u64)
    };
    Ok(LocalHeader {
        method,
        comp_size,
        name: name.to_vec(),
        data_rel: header_len as u64,
    })
}

/// Decompress entry data per its compression method.
fn decode_entry(method: u16, data: Vec<u8>) -> Result<Vec<u8>, ArchiveError> {
    match method {
        0 => Ok(data),
        8 => miniz_oxide::inflate::decompress_to_vec(&data)
            .map_err(|e| ArchiveError::Corrupt(format!("deflate decode failed: {e}"))),
        93 => Err(ArchiveError::Unsupported(
            "zstandard-compressed 3tz entries (method 93)".into(),
        )),
        m => Err(ArchiveError::Unsupported(format!(
            "zip compression method {m}"
        ))),
    }
}

/// An opened `.3tz`: the byte source plus the parsed index. Cheap to clone
/// behind an `Arc` (see [`super::fetch::TilesetSource::Archive`]).
#[derive(Debug)]
pub struct Archive3tz {
    source: ByteSource,
    size: u64,
    index: Index3tz,
    /// Speculative first [`OPEN_HEAD_BYTES`] of the archive, fetched in
    /// parallel with the tail on open. Front-packed entries (tileset.json, the
    /// root tile) are served straight from this buffer with zero requests.
    head: Vec<u8>,
    /// Every Local-File-Header offset in the archive (all index records plus
    /// the index entry itself), sorted ascending — `span_of` derives each
    /// entry's exact byte span from its successor, enabling single-GET reads.
    offsets: Vec<u64>,
}

impl Archive3tz {
    /// Open an archive with ONE parallel round-trip pair: a suffix read (the
    /// EOCD, ZIP64 records, central directory and index payload) plus a
    /// speculative head read (front-packed tileset.json and root tile). Extra
    /// ranged reads happen only when a section outruns its window.
    pub async fn open(source: ByteSource) -> Result<Self, ArchiveError> {
        let (suffix, head) = bevy::tasks::futures_lite::future::zip(
            source.read_suffix(OPEN_TAIL_BYTES),
            source.read_prefix(OPEN_HEAD_BYTES),
        )
        .await;
        let (tail, size) = suffix?;
        // The head is a pure optimization — degrade to ranged reads on error.
        let head = head.unwrap_or_default();
        let tail_abs = size - tail.len() as u64;
        let eocd = find_eocd(&tail, tail_abs)?;

        // Resolve the central directory bounds, following ZIP64 indirection
        // when the 32-bit EOCD fields are saturated.
        let (cd_offset, cd_end_abs) = if eocd.cd_offset == u32::MAX || eocd.cd_size == u32::MAX {
            let rec_abs = eocd.zip64_record_abs.ok_or_else(|| {
                ArchiveError::Corrupt("saturated EOCD without a ZIP64 locator".into())
            })?;
            let rec = if rec_abs >= tail_abs {
                let at = (rec_abs - tail_abs) as usize;
                tail.get(at..at + 56)
                    .ok_or_else(|| ArchiveError::Corrupt("ZIP64 EOCD out of tail".into()))?
                    .to_vec()
            } else {
                self_read(&source, rec_abs, 56, size).await?
            };
            if rec[0..4] != [0x50, 0x4B, 0x06, 0x06] {
                return Err(ArchiveError::Corrupt("bad ZIP64 EOCD signature".into()));
            }
            (le64(&rec, 48), rec_abs)
        } else {
            (eocd.cd_offset as u64, eocd.abs)
        };

        // The index entry is the LAST central-directory entry, so scanning the
        // CD's tail suffices. Use the already-fetched tail when it covers it.
        let entry = if cd_offset >= tail_abs {
            find_index_cd_entry(&tail, tail_abs, cd_end_abs)?
        } else {
            let scan_start = cd_offset.max(cd_end_abs.saturating_sub(OPEN_TAIL_BYTES));
            let buf = self_read(&source, scan_start, cd_end_abs - scan_start, size).await?;
            find_index_cd_entry(&buf, scan_start, cd_end_abs)?
        };
        if entry.method != 0 {
            return Err(ArchiveError::Corrupt(
                "@3dtilesIndex1@ must be stored uncompressed".into(),
            ));
        }

        // The tail window usually already contains the index entry whole
        // (payload sits right before the central directory) — parse in place
        // and skip the two ranged reads read_entry_at would issue.
        let data = if entry.local_offset >= tail_abs {
            let at = (entry.local_offset - tail_abs) as usize;
            let header = parse_local_header(&tail[at..])?;
            let start = at + header.data_rel as usize;
            let end = start + entry.comp_size as usize;
            (header.name == INDEX_NAME && end <= tail.len()).then(|| tail[start..end].to_vec())
        } else {
            None
        };
        let data = match data {
            Some(d) => d,
            None => {
                let raw = read_entry_at(&source, size, entry.local_offset, INDEX_NAME, None)
                    .await?
                    .ok_or(ArchiveError::NoIndex)?;
                raw.data
            }
        };
        if data.len() as u64 != entry.comp_size {
            return Err(ArchiveError::Corrupt(format!(
                "index payload size {} != central directory size {}",
                data.len(),
                entry.comp_size
            )));
        }
        let index = Index3tz::parse(&data)?;
        // Sorted LFH offsets (every entry + the index itself as the terminal
        // bound) — `span_of` turns a lookup hit into an exact single-GET range.
        let mut offsets: Vec<u64> = index.records.iter().map(|r| r.offset).collect();
        offsets.push(entry.local_offset);
        offsets.sort_unstable();
        Ok(Self {
            source,
            size,
            index,
            head,
            offsets,
        })
    }

    /// Exact byte span of the entry whose Local File Header sits at `offset`
    /// (header + data), derived from the next entry's offset. `None` when the
    /// offset is unknown or the span exceeds [`MAX_ENTRY_SPAN`].
    fn span_of(&self, offset: u64) -> Option<u64> {
        let ix = self.offsets.partition_point(|&o| o <= offset);
        let next = *self.offsets.get(ix)?;
        let span = next.checked_sub(offset)?;
        (span > 0 && span <= MAX_ENTRY_SPAN).then_some(span)
    }

    pub fn index(&self) -> &Index3tz {
        &self.index
    }

    /// The archive's underlying byte source (URL identity for cache keying).
    pub fn source(&self) -> &ByteSource {
        &self.source
    }

    /// Read + decompress one entry by archive path. Two range-GETs (one when
    /// the over-fetched header window already contains the whole payload).
    pub async fn read_entry(&self, path: &str) -> Result<Vec<u8>, ArchiveError> {
        self.read_entry_abortable(path, None).await
    }

    /// [`Self::read_entry`] with an optional cancellation token threaded
    /// through every range request.
    pub async fn read_entry_abortable(
        &self,
        path: &str,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<u8>, ArchiveError> {
        let normalized = normalize_path(path);
        let candidates = self.index.lookup(&normalized);
        for offset in &candidates {
            if let Some(raw) = self
                .read_raw_at(*offset, normalized.as_bytes(), abort)
                .await?
            {
                return decode_entry(raw.method, raw.data);
            }
        }
        Err(ArchiveError::EntryNotFound(normalized))
    }

    /// Fetch + parse the entry at a Local-File-Header offset; `None` when the
    /// header's filename doesn't match (an MD5-collision candidate to skip).
    ///
    /// Request budget: **zero** range-GETs when the entry sits inside the
    /// speculative open head (front-packed tileset.json / root tile), else
    /// **one** covering header + data via the index-derived exact span, else
    /// the two-GET header-then-data fallback.
    async fn read_raw_at(
        &self,
        offset: u64,
        expected_name: &[u8],
        abort: Option<&AbortHandle>,
    ) -> Result<Option<RawEntry>, ArchiveError> {
        // Head fast path — served from the open()'s speculative buffer.
        if (offset as usize) < self.head.len() {
            let window = &self.head[offset as usize..];
            if let Ok(header) = parse_local_header(window) {
                if header.name != expected_name {
                    return Ok(None);
                }
                if let Some(comp_size) = header.comp_size {
                    let start = header.data_rel as usize;
                    let end = start + comp_size as usize;
                    if end <= window.len() {
                        return Ok(Some(RawEntry {
                            method: header.method,
                            data: window[start..end].to_vec(),
                        }));
                    }
                }
            }
        }
        // Exact-span path: one GET for header + data.
        if let Some(span) = self.span_of(offset) {
            let window = self_read_abortable(&self.source, offset, span, self.size, abort).await?;
            let header = parse_local_header(&window)?;
            if header.name != expected_name {
                return Ok(None);
            }
            if let Some(comp_size) = header.comp_size {
                let start = header.data_rel as usize;
                let end = start + comp_size as usize;
                if end <= window.len() {
                    return Ok(Some(RawEntry {
                        method: header.method,
                        data: window[start..end].to_vec(),
                    }));
                }
                // Span was capped or the archive has inter-entry gaps that
                // lied about the size — fetch the remainder of the data.
                let data = self_read_abortable(
                    &self.source,
                    offset + header.data_rel,
                    comp_size,
                    self.size,
                    abort,
                )
                .await?;
                if data.len() as u64 == comp_size {
                    return Ok(Some(RawEntry {
                        method: header.method,
                        data,
                    }));
                }
            }
        }
        // Legacy fallback: header over-fetch, then data.
        read_entry_at(&self.source, self.size, offset, expected_name, abort).await
    }
}

struct RawEntry {
    method: u16,
    data: Vec<u8>,
}

/// Clamped range read (never asks past EOF).
async fn self_read(
    source: &ByteSource,
    start: u64,
    len: u64,
    size: u64,
) -> Result<Vec<u8>, ArchiveError> {
    self_read_abortable(source, start, len, size, None).await
}

/// Clamped range read with an optional cancellation token.
async fn self_read_abortable(
    source: &ByteSource,
    start: u64,
    len: u64,
    size: u64,
    abort: Option<&AbortHandle>,
) -> Result<Vec<u8>, ArchiveError> {
    let len = len.min(size.saturating_sub(start));
    Ok(source.read_abortable(start, len, abort).await?)
}

/// Fetch + parse the entry at a Local-File-Header offset; `None` when the
/// header's filename doesn't match (an MD5-collision candidate to skip).
async fn read_entry_at(
    source: &ByteSource,
    size: u64,
    offset: u64,
    expected_name: &[u8],
    abort: Option<&AbortHandle>,
) -> Result<Option<RawEntry>, ArchiveError> {
    let window = self_read_abortable(source, offset, LFH_OVERFETCH, size, abort).await?;
    let header = parse_local_header(&window)?;
    if header.name != expected_name {
        return Ok(None);
    }
    let comp_size = header.comp_size.ok_or_else(|| {
        ArchiveError::Unsupported(
            "3tz entry written in zip streaming mode (sizes only in data descriptor)".into(),
        )
    })?;
    let in_window_start = header.data_rel as usize;
    let in_window_end = in_window_start + comp_size as usize;
    let data = if in_window_end <= window.len() {
        window[in_window_start..in_window_end].to_vec()
    } else {
        self_read_abortable(source, offset + header.data_rel, comp_size, size, abort).await?
    };
    if data.len() as u64 != comp_size {
        return Err(ArchiveError::Corrupt(format!(
            "entry data short read: {} of {comp_size} bytes",
            data.len()
        )));
    }
    Ok(Some(RawEntry {
        method: header.method,
        data,
    }))
}

// ── Test support: a minimal deterministic 3tz writer ────────────────────────
// Also used by `examples/gen_tiles3d_fixture.rs` to pack the committed demo
// archive, so reader and fixture stay in lock-step.

/// Build a `.3tz` byte buffer from `(path, data, deflate?)` entries. Writes
/// fixed timestamps + zero CRCs (our reader validates neither) so output is
/// deterministic. Appends the spec-mandated `@3dtilesIndex1@` as the final,
/// stored entry. `comment` lands in the EOCD (exercises the tail scan).
pub fn write_3tz(files: &[(&str, &[u8], bool)], comment: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut central: Vec<(Vec<u8>, u64, u64, u16)> = Vec::new(); // (name, offset, comp_size, method)
    let mut records: Vec<IndexRecord> = Vec::new();

    let write_entry =
        |out: &mut Vec<u8>, name: &[u8], data: &[u8], deflate: bool| -> (u64, u64, u16) {
            let offset = out.len() as u64;
            let (payload, method) = if deflate {
                (miniz_oxide::deflate::compress_to_vec(data, 6), 8u16)
            } else {
                (data.to_vec(), 0u16)
            };
            out.extend_from_slice(&[0x50, 0x4B, 0x03, 0x04]); // LFH signature
            out.extend_from_slice(&20u16.to_le_bytes()); // version needed
            out.extend_from_slice(&0u16.to_le_bytes()); // flags
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // mod time (fixed)
            out.extend_from_slice(&0x2199u16.to_le_bytes()); // mod date (fixed)
            out.extend_from_slice(&0u32.to_le_bytes()); // crc32 (unvalidated)
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(name);
            out.extend_from_slice(&payload);
            (offset, payload.len() as u64, method)
        };

    for (path, data, deflate) in files {
        let name = normalize_path(path);
        let (offset, comp_size, method) = write_entry(&mut out, name.as_bytes(), data, *deflate);
        records.push(IndexRecord {
            hash: md5_of_path(&name),
            offset,
        });
        central.push((name.into_bytes(), offset, comp_size, method));
    }

    // Index payload: spec-sorted 24-byte records, stored, last entry.
    records.sort_by_key(|r| hash_sort_key(&r.hash));
    let mut payload = Vec::with_capacity(records.len() * 24);
    for r in &records {
        payload.extend_from_slice(&r.hash);
        payload.extend_from_slice(&r.offset.to_le_bytes());
    }
    let (offset, comp_size, method) = write_entry(&mut out, INDEX_NAME, &payload, false);
    central.push((INDEX_NAME.to_vec(), offset, comp_size, method));

    let cd_offset = out.len() as u64;
    for (name, offset, comp_size, method) in &central {
        out.extend_from_slice(&[0x50, 0x4B, 0x01, 0x02]); // CD signature
        out.extend_from_slice(&20u16.to_le_bytes()); // version made by
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x2199u16.to_le_bytes()); // mod date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc32
        out.extend_from_slice(&(*comp_size as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // uncomp size (unused by reader)
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out.extend_from_slice(&0u16.to_le_bytes()); // disk start
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        out.extend_from_slice(&(*offset as u32).to_le_bytes());
        out.extend_from_slice(name);
    }
    let cd_size = out.len() as u64 - cd_offset;

    out.extend_from_slice(&[0x50, 0x4B, 0x05, 0x06]); // EOCD signature
    out.extend_from_slice(&0u16.to_le_bytes()); // disk
    out.extend_from_slice(&0u16.to_le_bytes()); // cd start disk
    out.extend_from_slice(&(central.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u16).to_le_bytes());
    out.extend_from_slice(&(cd_size as u32).to_le_bytes());
    out.extend_from_slice(&(cd_offset as u32).to_le_bytes());
    out.extend_from_slice(&(comment.len() as u16).to_le_bytes());
    out.extend_from_slice(comment);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::tasks::block_on;
    use std::sync::Arc;

    fn mem(bytes: Vec<u8>) -> ByteSource {
        ByteSource::Mem(Arc::new(bytes))
    }

    #[test]
    fn path_normalization() {
        assert_eq!(normalize_path("/tileset.json"), "tileset.json");
        assert_eq!(normalize_path("content\\0\\a.glb"), "content/0/a.glb");
        assert_eq!(normalize_path("//x/y"), "x/y");
    }

    #[test]
    fn sort_key_compares_first_eight_bytes_first() {
        // h1: bytes[0..8] = 1 LE, rest 0 → key (1, 0)
        let mut h1 = [0u8; 16];
        h1[0] = 1;
        // h2: bytes[8..16] = 1 LE → key (0, 1)
        let mut h2 = [0u8; 16];
        h2[8] = 1;
        assert!(hash_sort_key(&h2) < hash_sort_key(&h1));
    }

    #[test]
    fn index_lookup_finds_adjacent_collisions() {
        // Craft a table with two records sharing a hash (synthetic collision)
        // plus a third distinct one; lookup must return both candidates.
        let twin = [7u8; 16];
        let mut other = [9u8; 16];
        other[0] = 1;
        let mut payload = Vec::new();
        let mut recs = vec![
            IndexRecord {
                hash: twin,
                offset: 100,
            },
            IndexRecord {
                hash: twin,
                offset: 200,
            },
            IndexRecord {
                hash: other,
                offset: 300,
            },
        ];
        recs.sort_by_key(|r| hash_sort_key(&r.hash));
        for r in &recs {
            payload.extend_from_slice(&r.hash);
            payload.extend_from_slice(&r.offset.to_le_bytes());
        }
        let index = Index3tz::parse(&payload).unwrap();
        // No path maps to [7;16] — drive the search by hash directly via a
        // record-level scan equivalence: both twin offsets are adjacent.
        let key = hash_sort_key(&twin);
        let start = index
            .records
            .partition_point(|r| hash_sort_key(&r.hash) < key);
        let hits: Vec<u64> = index.records[start..]
            .iter()
            .take_while(|r| r.hash == twin)
            .map(|r| r.offset)
            .collect();
        assert_eq!(hits, vec![100, 200]);
    }

    #[test]
    fn roundtrip_stored_and_deflated_entries() {
        let tileset = br#"{"asset":{"version":"1.1"}}"#;
        let glb: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();
        let archive = write_3tz(
            &[
                ("tileset.json", tileset.as_slice(), true),
                ("content/0/a.glb", glb.as_slice(), false),
                ("content/0/b.glb", b"tiny", false),
            ],
            b"",
        );
        let ar = block_on(Archive3tz::open(mem(archive))).expect("open");
        assert_eq!(ar.index().len(), 3);
        assert_eq!(block_on(ar.read_entry("tileset.json")).unwrap(), tileset);
        assert_eq!(block_on(ar.read_entry("content/0/a.glb")).unwrap(), glb);
        // Path normalization on lookup: backslashes + leading slash.
        assert_eq!(
            block_on(ar.read_entry("\\content\\0\\b.glb")).unwrap(),
            b"tiny"
        );
        assert!(matches!(
            block_on(ar.read_entry("missing.glb")),
            Err(ArchiveError::EntryNotFound(_))
        ));
    }

    #[test]
    fn eocd_found_behind_archive_comment() {
        let archive = write_3tz(&[("a.bin", b"data", false)], b"trailing zip comment here");
        let ar = block_on(Archive3tz::open(mem(archive))).expect("open with comment");
        assert_eq!(block_on(ar.read_entry("a.bin")).unwrap(), b"data");
    }

    #[test]
    fn many_entries_binary_search() {
        let blobs: Vec<(String, Vec<u8>)> = (0..200)
            .map(|i| {
                (
                    format!("content/{}/{}.glb", i % 7, i),
                    format!("payload-{i}").into_bytes(),
                )
            })
            .collect();
        let files: Vec<(&str, &[u8], bool)> = blobs
            .iter()
            .map(|(p, d)| (p.as_str(), d.as_slice(), false))
            .collect();
        let archive = write_3tz(&files, b"");
        let ar = block_on(Archive3tz::open(mem(archive))).expect("open");
        assert_eq!(ar.index().len(), 200);
        for (path, data) in &blobs {
            assert_eq!(&block_on(ar.read_entry(path)).unwrap(), data, "path {path}");
        }
    }

    #[test]
    fn rejects_archive_without_index() {
        // A plain zip whose last entry is NOT the index: write one file and
        // strip the index by writing entries manually — simplest is to write
        // an empty archive (EOCD only), which must error NoIndex/NotZip.
        let empty = write_3tz(&[], b"");
        // Empty still writes the index (zero records) — that's a VALID empty
        // 3tz. Corrupt the index name to simulate a foreign zip.
        let mut foreign = empty.clone();
        let pos = foreign
            .windows(INDEX_NAME.len())
            .rposition(|w| w == INDEX_NAME)
            .unwrap();
        foreign[pos] = b'#';
        assert!(matches!(
            block_on(Archive3tz::open(mem(foreign))),
            Err(ArchiveError::NoIndex)
        ));
        // And the pristine empty archive opens with an empty index.
        let ar = block_on(Archive3tz::open(mem(empty))).expect("open empty");
        assert!(ar.index().is_empty());
    }

    /// The fast-open request-budget contract: open = exactly the parallel
    /// suffix+prefix pair; front-packed entries (tileset.json, root tile)
    /// serve from the head with ZERO further requests; an entry outside both
    /// windows costs exactly ONE span-bounded request.
    #[test]
    fn open_and_first_paint_request_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let tileset = br#"{"asset":{"version":"1.1"},"root":{}}"#;
        let root_glb: Vec<u8> = (0u8..=255).cycle().take(300_000).collect();
        // Fillers push later entries beyond the 2 MiB head AND keep them clear
        // of the 512 KiB tail (index+CD are tiny here, so the tail reaches
        // 512 KiB into the archive body — the far entry must sit before that).
        let filler: Vec<u8> = vec![0xAB; 3_000_000];
        let far: Vec<u8> = b"far-tile-payload".repeat(100);
        let tail_guard: Vec<u8> = vec![0xCD; 1_000_000];
        let archive = write_3tz(
            &[
                ("tileset.json", tileset.as_slice(), false),
                ("content/r.glb", root_glb.as_slice(), false),
                ("content/filler.bin", filler.as_slice(), false),
                ("content/far.glb", far.as_slice(), false),
                ("content/tail_guard.bin", tail_guard.as_slice(), false),
            ],
            b"",
        );
        let hits = Arc::new(AtomicUsize::new(0));
        let src = ByteSource::Counting(Arc::new(archive), hits.clone());

        let ar = block_on(Archive3tz::open(src)).expect("open");
        assert_eq!(hits.load(Ordering::Relaxed), 2, "open = suffix + prefix");

        assert_eq!(
            block_on(ar.read_entry("tileset.json")).unwrap(),
            tileset,
            "tileset content"
        );
        assert_eq!(block_on(ar.read_entry("content/r.glb")).unwrap(), root_glb);
        assert_eq!(
            hits.load(Ordering::Relaxed),
            2,
            "front-packed entries serve from the open head — no new requests"
        );

        assert_eq!(block_on(ar.read_entry("content/far.glb")).unwrap(), far);
        assert_eq!(
            hits.load(Ordering::Relaxed),
            3,
            "an entry outside the head costs exactly one span-bounded request"
        );
    }

    #[test]
    fn zip64_extra_field_rank_addressing() {
        // Block: id 0x0001, len 16, two u64s (e.g. comp size then offset when
        // both saturated and uncomp was not).
        let mut extra = Vec::new();
        extra.extend_from_slice(&0x0001u16.to_le_bytes());
        extra.extend_from_slice(&16u16.to_le_bytes());
        extra.extend_from_slice(&111u64.to_le_bytes());
        extra.extend_from_slice(&222u64.to_le_bytes());
        assert_eq!(zip64_extra_field(&extra, 0), Some(111));
        assert_eq!(zip64_extra_field(&extra, 1), Some(222));
        assert_eq!(zip64_extra_field(&extra, 2), None);
        assert_eq!(zip64_extra_field(&[], 0), None);
    }
}
