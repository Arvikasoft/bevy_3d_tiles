//! CPU-side tile preparation for `bevy_3d_tiles` — the S4 seam of the
//! offthread-decode plan: everything between "3D Tiles GLB bytes" and "plain
//! glTF bytes" that is pure CPU work with **no bevy dependency**, so a host
//! can run it inside a Web Worker's own wasm module (or any other thread) and
//! hand the result back through [`prepare_tile`]'s [`PreparedTile`].
//!
//! `bevy_3d_tiles` depends on this crate and re-exports everything, so the
//! split is invisible downstream; its inline decode path is built from these
//! same functions (moved here, never copied — the meshopt codec in
//! [`meshopt`] is documented byte-lossless and must exist exactly once).
//!
//! What deliberately does NOT live here: the platform decoders themselves
//! (Draco's JS shim, splat renderers) — and anything producing bevy types
//! (`Mesh`/`Image` assembly, KTX2 transcode). Draco *around* the decoder DOES
//! live here since 0.2.1: a host that owns a decoder in its own realm (a Web
//! Worker's JS shim) slices the compressed payloads out with
//! [`draco_requests`], decodes them itself, and hands the results to
//! [`prepare_tile_extracting_with_draco`], which splices them in
//! ([`splice_draco`]) and continues the normal pipeline. A host without one
//! keeps the old behaviour: [`prepare_tile`] returns `Ok(None)` and the
//! caller decodes inline.

use std::collections::HashMap;

mod extract;
pub mod meshopt;
mod normals;

pub use extract::{
    ExtractOptions, ExtractedMaterial, ExtractedMeshes, ExtractedPrimitive, ExtractedTexture,
    TextureWrap, TileImage, extract_tile_meshes,
};
pub use normals::compute_normals;

/// Typed failure surface of tile-content decoding — the error of
/// `decode_tile` / `decode_glb` (in `bevy_3d_tiles`), [`prepare_tile`], and
/// the draco/ktx2 shim modules.
///
/// [`DecodeStage`] carries the one distinction a caller can act on:
/// [`DecodeStage::Content`] is a permanent parse/structure failure for these
/// bytes (retrying cannot succeed), while the shim stages (`Draco`/`Ktx2`/
/// `Meshopt`) are transcoder paths whose availability is environmental
/// (missing JS shim, no GPU block format). Internal helpers keep plain
/// `String` messages; the type is applied at the public boundaries — via
/// `From<String>`/`From<&str>` (stage = `Content`) or the per-stage
/// constructors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{stage:?} decode: {message}")]
pub struct DecodeError {
    pub stage: DecodeStage,
    pub message: String,
}

/// Which decode stage a [`DecodeError`] came from. See [`DecodeError`] for the
/// permanent-vs-environmental reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStage {
    /// GLB/glTF/JSON structure or unsupported content — permanent for these bytes.
    Content,
    /// The Draco decoder shim (`__tt_draco_decode`) or its output shape.
    Draco,
    /// KTX2/Basis transcode (JS shim on wasm; bevy's transcoder on native).
    Ktx2,
    /// `EXT_meshopt_compression` CPU decode.
    Meshopt,
}

impl DecodeError {
    pub fn new(stage: DecodeStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
        }
    }

    pub fn draco(message: impl Into<String>) -> Self {
        Self::new(DecodeStage::Draco, message)
    }

    pub fn ktx2(message: impl Into<String>) -> Self {
        Self::new(DecodeStage::Ktx2, message)
    }

    pub fn meshopt(message: impl Into<String>) -> Self {
        Self::new(DecodeStage::Meshopt, message)
    }
}

impl From<String> for DecodeError {
    fn from(message: String) -> Self {
        Self::new(DecodeStage::Content, message)
    }
}

impl From<&str> for DecodeError {
    fn from(message: &str) -> Self {
        Self::new(DecodeStage::Content, message)
    }
}

/// Which decode passes a tile needs, from ONE marker scan of its JSON chunk.
/// Threaded through the whole decode so no pass re-scans and no pass re-parses
/// (`decode_glb` used to re-enter itself per extension). Deliberately naive
/// substring scans of the raw chunk rather than a read of
/// `extensionsUsed`/`extensionsRequired`: content that uses an extension
/// without declaring it still has to route correctly.
// ponytail: O(json_len × needle) × 7. The JSON chunk is kilobytes next to a
// multi-MB BIN; if a producer ever ships a huge JSON chunk, scan once for the
// shared `"EXT_`/`"KHR_` prefixes instead.
#[derive(Default, Clone, Copy)]
pub struct Marks {
    pub splat: bool,
    pub draco: bool,
    pub rtc: bool,
    pub copyright: bool,
    pub meshopt: bool,
    pub basisu: bool,
    pub features: bool,
}

impl Marks {
    pub fn scan(json: &[u8]) -> Self {
        Self {
            splat: memmem(json, b"KHR_gaussian_splatting"),
            draco: memmem(json, b"KHR_draco_mesh_compression"),
            rtc: memmem(json, b"CESIUM_RTC"),
            copyright: memmem(json, b"copyright"),
            meshopt: memmem(json, b"EXT_meshopt_compression"),
            basisu: memmem(json, b"KHR_texture_basisu"),
            features: memmem(json, b"EXT_mesh_features"),
        }
    }

    /// Nothing to rewrite and no side-band data to extract — the bytes go
    /// straight to the `gltf` crate with no JSON parse of our own.
    pub fn vanilla(&self) -> bool {
        !(self.splat
            || self.draco
            || self.rtc
            || self.copyright
            || self.meshopt
            || self.basisu
            || self.features)
    }
}

/// A legacy 3D Tiles 1.0 `b3dm` container, unwrapped to its embedded GLB
/// (the format the open-data fleets — swisstopo, PLATEAU — still serve).
///
/// `rtc_center` is the feature table's `RTC_CENTER`, in the TILE frame
/// (z-up tileset axes). The tile compose applies rtc offsets innermost — in
/// the glTF y-up content frame — so callers rotate it (and any `CESIUM_RTC`
/// in the embedded glb, which the b3dm pipeline defines in the same tile
/// frame) through [`tile_rtc_to_content_frame`] before storing it. A bare-glb
/// `CESIUM_RTC` (offline Google P3DT content) is already content-frame and
/// must NOT be rotated — the container is what decides.
#[derive(Debug)]
pub struct B3dm<'a> {
    pub glb: &'a [u8],
    pub rtc_center: Option<[f64; 3]>,
}

/// Unwrap a `b3dm` container. `Ok(None)` = not a b3dm — hand the bytes to
/// [`split_glb`] as before. The other 1.0 containers error by name (`cmpt`
/// composites and point/instanced tiles have no decoder here).
pub fn unwrap_b3dm(bytes: &[u8]) -> Result<Option<B3dm<'_>>, String> {
    match bytes.get(0..4) {
        Some(b"b3dm") => {}
        Some(m @ (b"i3dm" | b"pnts" | b"cmpt")) => {
            return Err(format!(
                "legacy 3D Tiles 1.0 '{}' content unsupported (only b3dm)",
                String::from_utf8_lossy(m)
            ));
        }
        _ => return Ok(None),
    }
    if bytes.len() < 28 {
        return Err("b3dm truncated before header end".into());
    }
    let u = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let (byte_len, ftj, ftb, btj, btb) = (u(8), u(12), u(16), u(20), u(24));
    let glb = bytes
        .get(28 + ftj + ftb + btj + btb..byte_len.min(bytes.len()))
        .filter(|g| !g.is_empty())
        .ok_or("b3dm tables overrun the buffer")?;
    // Feature table: only RTC_CENTER matters to placement (BATCH_LENGTH and
    // the per-feature semantics have no consumer here).
    let mut rtc_center = None;
    if ftj > 0 {
        let ft: serde_json::Value = serde_json::from_slice(&bytes[28..28 + ftj])
            .map_err(|e| format!("b3dm feature table: {e}"))?;
        rtc_center = ft["RTC_CENTER"].as_array().and_then(|c| {
            let v: Vec<f64> = c.iter().filter_map(|x| x.as_f64()).collect();
            <[f64; 3]>::try_from(v).ok()
        });
    }
    Ok(Some(B3dm { glb, rtc_center }))
}

/// Rotate a tile-frame (z-up) rtc offset into the glTF content frame the tile
/// compose applies rtc in: the inverse of the tiles-spec y-up→z-up content
/// rotation. Pinned against the real `YUP_TO_ZUP` constant by a test in the
/// main crate.
pub fn tile_rtc_to_content_frame(c: [f64; 3]) -> [f64; 3] {
    [c[0], c[2], -c[1]]
}

/// Split a GLB container into its JSON chunk and optional BIN chunk. Bytes
/// without the `glTF` magic are treated as a bare JSON glTF (no buffer).
pub fn split_glb(bytes: &[u8]) -> Result<(&[u8], Option<&[u8]>), String> {
    if bytes.len() < 4 || &bytes[0..4] != b"glTF" {
        return Ok((bytes, None));
    }
    if bytes.len() < 12 {
        return Err("glb truncated before header end".into());
    }
    let mut at = 12; // skip magic + version + length
    let mut json: Option<&[u8]> = None;
    let mut bin: Option<&[u8]> = None;
    while at + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let kind = &bytes[at + 4..at + 8];
        let body = bytes
            .get(at + 8..at + 8 + len)
            .ok_or_else(|| format!("glb chunk at {at} overruns the buffer"))?;
        match kind {
            b"JSON" => json = Some(body),
            b"BIN\0" => bin = Some(body),
            _ => {}
        }
        at += 8 + len;
    }
    Ok((json.ok_or("glb has no JSON chunk")?, bin))
}

/// Naive substring scan (the JSON chunk is small; no memmem dependency).
pub fn memmem(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Assemble a GLB container from JSON + BIN chunks (4-byte padded).
pub fn assemble_glb(json_bytes: &[u8], bin: &[u8]) -> Vec<u8> {
    let mut json_bytes = json_bytes.to_vec();
    let mut bin = bin.to_vec();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let mut glb = Vec::with_capacity(28 + json_bytes.len() + bin.len());
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    let total = 12 + 8 + json_bytes.len() + if bin.is_empty() { 0 } else { 8 + bin.len() };
    glb.extend_from_slice(&(total as u32).to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);
    if !bin.is_empty() {
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
    }
    glb
}

/// When any scene-root node sits at planetary magnitude (Google P3DT bakes
/// ECEF into node matrices), pick the first such translation as the tile's
/// offset and subtract it from EVERY root node **in f64**, so the f32 glTF
/// decode only ever sees tile-local values. Returns the extracted offset
/// (ECEF metres). The spawn transform re-applies it: `world_from_content ×
/// T(offset) × node'` ≡ `world_from_content × node` exactly.
pub fn extract_planetary_root_offset(json: &mut serde_json::Value) -> Option<[f64; 3]> {
    const PLANETARY_M: f64 = 1.0e6;

    let scene_ix = json["scene"].as_u64().unwrap_or(0) as usize;
    let roots: Vec<usize> = json["scenes"][scene_ix]["nodes"]
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();

    let translation_of = |node: &serde_json::Value| -> [f64; 3] {
        if let Some(m) = node["matrix"].as_array()
            && m.len() == 16
        {
            return [
                m[12].as_f64().unwrap_or(0.0),
                m[13].as_f64().unwrap_or(0.0),
                m[14].as_f64().unwrap_or(0.0),
            ];
        }
        node["translation"]
            .as_array()
            .map(|t| {
                [
                    t.first().and_then(|v| v.as_f64()).unwrap_or(0.0),
                    t.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
                    t.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0),
                ]
            })
            .unwrap_or([0.0; 3])
    };

    let center = roots.iter().find_map(|&ix| {
        let t = translation_of(&json["nodes"][ix]);
        (t[0] * t[0] + t[1] * t[1] + t[2] * t[2] > PLANETARY_M * PLANETARY_M).then_some(t)
    })?;

    // Subtract from EVERY root (small-translation roots become -center —
    // mixing untouched and rebased roots under the re-applied offset would
    // shift the untouched ones).
    for &ix in &roots {
        let t = translation_of(&json["nodes"][ix]);
        let new = [t[0] - center[0], t[1] - center[1], t[2] - center[2]];
        let node = &mut json["nodes"][ix];
        if node["matrix"].is_array() {
            let m = node["matrix"].as_array_mut().unwrap();
            for (k, v) in new.iter().enumerate() {
                m[12 + k] = serde_json::json!(v);
            }
        } else {
            node["translation"] = serde_json::json!(new);
        }
    }
    Some(center)
}

/// Drop the extensions the decoder handles itself (`KHR_draco_mesh_compression`
/// spliced out by the caller, `CESIUM_RTC` extracted as side-band data) from
/// the document, so the strict `gltf` crate — which hard-rejects any unknown
/// `extensionsRequired` — accepts the rebuilt tile.
///
/// NOTE: use `get_mut`, never `json[key]` — IndexMut on a missing key INSERTS
/// a literal null, which the gltf crate then chokes on.
pub fn strip_handled_extensions(json: &mut serde_json::Value) {
    if let Some(ext) = json.get_mut("extensions").and_then(|e| e.as_object_mut()) {
        ext.remove("CESIUM_RTC");
        if ext.is_empty() {
            json.as_object_mut().unwrap().remove("extensions");
        }
    }
    for list in ["extensionsUsed", "extensionsRequired"] {
        if let Some(arr) = json.get_mut(list).and_then(|v| v.as_array_mut()) {
            arr.retain(|v| {
                !matches!(
                    v.as_str(),
                    Some("KHR_draco_mesh_compression" | "CESIUM_RTC")
                )
            });
            if arr.is_empty() {
                json.as_object_mut().unwrap().remove(list);
            }
        }
    }
}

// ── Draco / CESIUM_RTC preprocessing (T4 — Google P3DT content) ──────────────

/// One `KHR_draco_mesh_compression` primitive found in the document. The
/// Draco *decode* is a platform shim (main-thread JS on wasm) and stays in
/// `bevy_3d_tiles`; only the JSON-side discovery lives here.
pub struct DracoPrim {
    pub mesh: usize,
    pub prim: usize,
    pub buffer_view: usize,
    /// glTF semantic → Draco attribute unique id, straight from the ext JSON.
    pub attributes: Vec<(String, u32)>,
}

pub fn find_draco_prims(json: &serde_json::Value) -> Vec<DracoPrim> {
    let mut out = Vec::new();
    let Some(meshes) = json["meshes"].as_array() else {
        return out;
    };
    for (m, mesh) in meshes.iter().enumerate() {
        let Some(prims) = mesh["primitives"].as_array() else {
            continue;
        };
        for (p, prim) in prims.iter().enumerate() {
            let ext = &prim["extensions"]["KHR_draco_mesh_compression"];
            let Some(view) = ext["bufferView"].as_u64() else {
                continue;
            };
            let Some(attrs) = ext["attributes"].as_object() else {
                continue;
            };
            out.push(DracoPrim {
                mesh: m,
                prim: p,
                buffer_view: view as usize,
                attributes: attrs
                    .iter()
                    .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                    .collect(),
            });
        }
    }
    out
}

pub fn buffer_view_slice<'b>(
    json: &serde_json::Value,
    bin: Option<&'b [u8]>,
    view_ix: usize,
) -> Result<&'b [u8], String> {
    let bv = &json["bufferViews"][view_ix];
    if bv["buffer"].as_u64() != Some(0) {
        return Err("draco bufferView must reference buffer 0 (BIN chunk)".into());
    }
    let offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let len = bv["byteLength"]
        .as_u64()
        .ok_or("bufferView without byteLength")? as usize;
    bin.ok_or("draco bufferView references the BIN chunk but the GLB has none")?
        .get(offset..offset + len)
        .ok_or_else(|| "draco bufferView out of BIN bounds".into())
}

/// One decoded Draco mesh: triangle indices + dequantized float attributes,
/// keyed by glTF attribute unique id. Produced by a platform decoder —
/// `bevy_3d_tiles::draco` (the main-thread JS shim) or a host worker's own —
/// and consumed by [`splice_draco`].
pub struct DracoMesh {
    pub indices: Vec<u32>,
    /// `(unique_id, components_per_element, dequantized values)`.
    pub attributes: Vec<(u32, usize, Vec<f32>)>,
}

/// The compressed payload of one [`DracoPrim`], sliced and OWNED so a host can
/// hand it to a platform decoder in another realm (a Web Worker's JS shim,
/// across a `postMessage` boundary).
pub struct DracoRequest {
    /// glTF attribute unique ids, in [`DracoPrim::attributes`] order.
    pub attr_ids: Vec<u32>,
    pub compressed: Vec<u8>,
}

/// Slice every Draco primitive's compressed payload out of a tile. `[]` = no
/// Draco content, answered from one marker scan (non-Draco tiles pay no JSON
/// parse). Order matches [`find_draco_prims`] on the same document, which is
/// the order [`prepare_tile_extracting_with_draco`] expects its decoded
/// meshes in — decode the requests in order and hand the results back as-is.
pub fn draco_requests(bytes: &[u8]) -> Result<Vec<DracoRequest>, DecodeError> {
    let b3dm = unwrap_b3dm(bytes)?;
    let bytes = match &b3dm {
        Some(b) => b.glb,
        None => bytes,
    };
    let (json_chunk, bin) = split_glb(bytes)?;
    if !Marks::scan(json_chunk).draco {
        return Ok(Vec::new());
    }
    let json: serde_json::Value =
        serde_json::from_slice(json_chunk).map_err(|e| format!("tile json: {e}"))?;
    find_draco_prims(&json)
        .iter()
        .map(|prim| {
            Ok(DracoRequest {
                attr_ids: prim.attributes.iter().map(|(_, id)| *id).collect(),
                compressed: buffer_view_slice(&json, bin, prim.buffer_view)?.to_vec(),
            })
        })
        .collect()
}

/// Splice already-decoded Draco primitives into the document: decoded data
/// appended to the BIN chunk behind fresh accessors, the per-primitive Draco
/// extension removed. Returns the NEW BIN chunk — the caller rebuilds the GLB
/// container once, after every other rewrite pass. The document-level
/// extension strip is [`strip_handled_extensions`] (it must also run for
/// content that declares Draco/RTC without a usable primitive).
pub fn splice_draco(
    json: &mut serde_json::Value,
    bin: Option<&[u8]>,
    prims: &[DracoPrim],
    decoded: Vec<DracoMesh>,
) -> Result<Vec<u8>, String> {
    let mut new_bin: Vec<u8> = bin.unwrap_or_default().to_vec();

    for (prim, dm) in prims.iter().zip(decoded) {
        // Indices.
        while !new_bin.len().is_multiple_of(4) {
            new_bin.push(0);
        }
        let idx_offset = new_bin.len();
        for i in &dm.indices {
            new_bin.extend_from_slice(&i.to_le_bytes());
        }
        let idx_view = push_json(
            json,
            "bufferViews",
            serde_json::json!({
                "buffer": 0, "byteOffset": idx_offset, "byteLength": dm.indices.len() * 4,
            }),
        );
        let idx_accessor = serde_json::json!({
            "bufferView": idx_view, "componentType": 5125,
            "count": dm.indices.len(), "type": "SCALAR",
        });
        // Draco primitives reference accessors WITHOUT bufferViews (count/
        // type only). Overwrite those in place — leaving them orphaned fails
        // the gltf crate's "Missing data" validation.
        set_or_push_accessor(json, prim, None, idx_accessor);

        // Attributes (already dequantized to f32 by the decoder).
        for (semantic, uid) in &prim.attributes {
            let (_, components, data) = dm
                .attributes
                .iter()
                .find(|(id, _, _)| id == uid)
                .ok_or_else(|| format!("draco decoder returned no attribute {uid}"))?;
            let type_str = match components {
                1 => "SCALAR",
                2 => "VEC2",
                3 => "VEC3",
                4 => "VEC4",
                n => return Err(format!("draco attribute with {n} components")),
            };
            let count = data.len() / components;
            let offset = new_bin.len();
            for v in data {
                new_bin.extend_from_slice(&v.to_le_bytes());
            }
            let view = push_json(
                json,
                "bufferViews",
                serde_json::json!({
                    "buffer": 0, "byteOffset": offset, "byteLength": data.len() * 4,
                }),
            );
            let mut accessor = serde_json::json!({
                "bufferView": view, "componentType": 5126,
                "count": count, "type": type_str,
            });
            if semantic == "POSITION" {
                // Spec mandates min/max on POSITION accessors.
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for chunk in data.chunks_exact(3) {
                    for c in 0..3 {
                        lo[c] = lo[c].min(chunk[c]);
                        hi[c] = hi[c].max(chunk[c]);
                    }
                }
                accessor["min"] = serde_json::json!(lo);
                accessor["max"] = serde_json::json!(hi);
            }
            set_or_push_accessor(json, prim, Some(semantic), accessor);
        }

        let p = &mut json["meshes"][prim.mesh]["primitives"][prim.prim];
        if let Some(ext) = p.get_mut("extensions").and_then(|e| e.as_object_mut()) {
            ext.remove("KHR_draco_mesh_compression");
            if ext.is_empty() {
                p.as_object_mut().unwrap().remove("extensions");
            }
        }
    }

    if json["buffers"][0].is_object() {
        json["buffers"][0]["byteLength"] = serde_json::json!(new_bin.len());
    } else if !new_bin.is_empty() {
        json["buffers"] = serde_json::json!([{ "byteLength": new_bin.len() }]);
    }
    Ok(new_bin)
}

/// Point a primitive slot (`indices` when `semantic` is `None`, else
/// `attributes[semantic]`) at `accessor`: overwrite the accessor the slot
/// already references — Draco primitives carry bufferView-less accessors
/// that fail validation if left orphaned — or append it and link the slot.
fn set_or_push_accessor(
    json: &mut serde_json::Value,
    prim: &DracoPrim,
    semantic: Option<&str>,
    accessor: serde_json::Value,
) {
    let slot = {
        let p = &json["meshes"][prim.mesh]["primitives"][prim.prim];
        match semantic {
            Some(s) => p["attributes"][s].as_u64(),
            None => p["indices"].as_u64(),
        }
    };
    match slot {
        Some(existing) => json["accessors"][existing as usize] = accessor,
        None => {
            let ix = push_json(json, "accessors", accessor);
            let p = &mut json["meshes"][prim.mesh]["primitives"][prim.prim];
            match semantic {
                Some(s) => p["attributes"][s] = serde_json::json!(ix),
                None => p["indices"] = serde_json::json!(ix),
            }
        }
    }
}

/// Append `value` to the top-level array `key` (created when absent),
/// returning its index.
fn push_json(json: &mut serde_json::Value, key: &str, value: serde_json::Value) -> usize {
    if !json[key].is_array() {
        json[key] = serde_json::json!([]);
    }
    let arr = json[key].as_array_mut().unwrap();
    arr.push(value);
    arr.len() - 1
}

// ── EXT_meshopt_compression preprocessing (T6 — our emitted geometry) ────────

/// Rewrite an `EXT_meshopt_compression` document into vanilla glTF: decode
/// every meshopt buffer view on the CPU ([`meshopt::decode_buffer_view`]),
/// copy through non-meshopt views (embedded image bytes), collapse to a single
/// buffer (the fallback buffer is virtual — no GLB bytes), and strip the
/// extension. Returns the NEW BIN chunk; the caller rebuilds the container
/// once, after every other rewrite pass.
///
/// Buffer-view *indices* are preserved (accessors and images keep referencing
/// the same slots); only each view's `byteOffset`/`byteLength`/`buffer` are
/// rebuilt against the freshly decoded BIN. The encoder stores compressed data
/// in the GLB BIN (`ext.buffer == 0`) while the view's own `buffer` points at
/// the discarded fallback — so we always read compressed bytes via `ext`.
pub fn decode_meshopt_views(
    json: &mut serde_json::Value,
    bin: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let bin = bin.ok_or("meshopt GLB has no BIN chunk")?;
    let view_count = json["bufferViews"].as_array().map(|a| a.len()).unwrap_or(0);
    let mut new_bin: Vec<u8> = Vec::new();
    let mut new_views: Vec<serde_json::Value> = Vec::with_capacity(view_count);

    for i in 0..view_count {
        let bv = &json["bufferViews"][i];
        let ext = &bv["extensions"]["EXT_meshopt_compression"];
        let def = if ext.is_object() {
            if ext["buffer"].as_u64().unwrap_or(0) != 0 {
                return Err("meshopt ext references a non-BIN buffer".into());
            }
            let off = ext["byteOffset"].as_u64().unwrap_or(0) as usize;
            let len = ext["byteLength"]
                .as_u64()
                .ok_or("meshopt ext without byteLength")? as usize;
            let stride = ext["byteStride"]
                .as_u64()
                .ok_or("meshopt ext without byteStride")? as usize;
            let count = ext["count"].as_u64().ok_or("meshopt ext without count")? as usize;
            let mode = ext["mode"]
                .as_str()
                .ok_or("meshopt ext without mode")?
                .to_string();
            let filter = ext["filter"].as_str().unwrap_or("NONE").to_string();
            let src = bin
                .get(off..off + len)
                .ok_or("meshopt compressed data out of BIN bounds")?;
            let decoded = meshopt::decode_buffer_view(&mode, &filter, count, stride, src)?;
            while !new_bin.len().is_multiple_of(4) {
                new_bin.push(0);
            }
            let new_off = new_bin.len();
            new_bin.extend_from_slice(&decoded);
            let mut def = serde_json::json!({
                "buffer": 0, "byteOffset": new_off, "byteLength": decoded.len(),
            });
            // Vertex views keep their stride (honors interleaving for foreign
            // gltfpack output; == element size for our non-interleaved tiles).
            if mode == "ATTRIBUTES" {
                def["byteStride"] = serde_json::json!(stride);
            }
            def
        } else {
            // Pass-through view (e.g. an embedded image): copy its BIN bytes.
            if bv["buffer"].as_u64().unwrap_or(0) != 0 {
                return Err("non-meshopt bufferView references a non-BIN buffer".into());
            }
            let off = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
            let len = bv["byteLength"]
                .as_u64()
                .ok_or("bufferView without byteLength")? as usize;
            let bytes = bin
                .get(off..off + len)
                .ok_or("bufferView out of BIN bounds")?
                .to_vec();
            while !new_bin.len().is_multiple_of(4) {
                new_bin.push(0);
            }
            let new_off = new_bin.len();
            new_bin.extend_from_slice(&bytes);
            let mut def = serde_json::json!({
                "buffer": 0, "byteOffset": new_off, "byteLength": len,
            });
            if let Some(s) = bv["byteStride"].as_u64() {
                def["byteStride"] = serde_json::json!(s);
            }
            if let Some(t) = bv["target"].as_u64() {
                def["target"] = serde_json::json!(t);
            }
            def
        };
        new_views.push(def);
    }

    json["bufferViews"] = serde_json::Value::Array(new_views);
    json["buffers"] = serde_json::json!([{ "byteLength": new_bin.len() }]);
    for list in ["extensionsUsed", "extensionsRequired"] {
        if let Some(arr) = json.get_mut(list).and_then(|v| v.as_array_mut()) {
            arr.retain(|v| v.as_str() != Some("EXT_meshopt_compression"));
            if arr.is_empty() {
                json.as_object_mut().unwrap().remove(list);
            }
        }
    }
    Ok(new_bin)
}

// ── KHR_texture_basisu preprocessing (T7 — KTX2 tile textures) ───────────────

/// Rewrite `KHR_texture_basisu` textures so the `gltf` crate (which doesn't
/// resolve the extension) finds the KTX2 image: move each texture's
/// `extensions.KHR_texture_basisu.source` to the standard `source`, then strip
/// the extension everywhere. JSON-only — the KTX2 image bytes (mimeType
/// `image/ktx2`, in a buffer view) are untouched; the material decode passes
/// them to the platform KTX2/Basis transcoder later, main-side.
pub fn preprocess_basisu(json: &mut serde_json::Value) {
    if let Some(textures) = json["textures"].as_array_mut() {
        for tex in textures.iter_mut() {
            let Some(src) = tex["extensions"]["KHR_texture_basisu"]["source"].as_u64() else {
                continue;
            };
            tex["source"] = serde_json::json!(src);
            if let Some(ext) = tex.get_mut("extensions").and_then(|e| e.as_object_mut()) {
                ext.remove("KHR_texture_basisu");
                if ext.is_empty() {
                    tex.as_object_mut().unwrap().remove("extensions");
                }
            }
        }
    }
    for list in ["extensionsUsed", "extensionsRequired"] {
        if let Some(arr) = json.get_mut(list).and_then(|v| v.as_array_mut()) {
            arr.retain(|v| v.as_str() != Some("KHR_texture_basisu"));
            if arr.is_empty() {
                json.as_object_mut().unwrap().remove(list);
            }
        }
    }
}

// ── Feature metadata (T8 — EXT_mesh_features + EXT_structural_metadata) ─────

/// Decoded `EXT_mesh_features` + `EXT_structural_metadata` context for a tile
/// (T8). Owns the parsed JSON so `_FEATURE_ID_0` accessors can be read lazily
/// per primitive against the BIN chunk. `bevy_3d_tiles` builds `TileFeatures`
/// from it on the inline path; [`prepare_tile`] materializes it into
/// [`PreparedFeatures`] so the main thread never re-parses the JSON.
pub struct FeatureCtx {
    json: serde_json::Value,
    /// featureId → source-node path (the `/`-joined node names the host's
    /// sections resolver matches against).
    pub node_of_feature: Vec<String>,
    /// (mesh index, primitive index) → `_FEATURE_ID_N` accessor index.
    accessor: HashMap<(u64, u64), usize>,
}

impl FeatureCtx {
    /// Takes the tile's ALREADY-PARSED document (post-rewrite) — the JSON chunk
    /// is parsed once per tile and handed down, never re-parsed here.
    pub fn build(value: serde_json::Value, bin: Option<&[u8]>) -> Result<Self, String> {
        let node_of_feature = read_node_of_feature(&value, bin)?;
        let mut accessor = HashMap::new();
        if let Some(meshes) = value["meshes"].as_array() {
            for (m, mesh) in meshes.iter().enumerate() {
                let Some(prims) = mesh["primitives"].as_array() else {
                    continue;
                };
                for (p, prim) in prims.iter().enumerate() {
                    let ext = &prim["extensions"]["EXT_mesh_features"];
                    // featureIds[0].attribute = N → the `_FEATURE_ID_N` attribute.
                    let Some(n) = ext["featureIds"][0]["attribute"].as_u64() else {
                        continue;
                    };
                    let key = format!("_FEATURE_ID_{n}");
                    if let Some(acc) = prim["attributes"][&key].as_u64() {
                        accessor.insert((m as u64, p as u64), acc as usize);
                    }
                }
            }
        }
        Ok(Self {
            json: value,
            node_of_feature,
            accessor,
        })
    }

    /// Raw per-VERTEX `_FEATURE_ID_0` values of primitive `(mesh_ix, prim_ix)`
    /// (accessor length, NOT padded to the mesh's vertex count — the caller
    /// pads), or `None` when this primitive carries no feature ids.
    pub fn per_vertex_ids(
        &self,
        bin: Option<&[u8]>,
        mesh_ix: u64,
        prim_ix: u64,
    ) -> Result<Option<Vec<f32>>, String> {
        let Some(&acc) = self.accessor.get(&(mesh_ix, prim_ix)) else {
            return Ok(None);
        };
        let vals = read_accessor::<1>(&self.json, bin, acc)?;
        Ok(Some(vals.into_iter().map(|v| v[0]).collect()))
    }

    /// [`FeatureCtx::materialize`] for EXTRACTED geometry: each primitive
    /// gets its tables here ([`ExtractedPrimitive::set_feature_ids`]), on the
    /// preparing thread, so the consumer does no per-vertex pass; the raw ids
    /// are then not carried a second time (`vertex_ids` comes back empty).
    fn attach(
        mut self,
        bin: Option<&[u8]>,
        primitives: &mut [ExtractedPrimitive],
    ) -> Result<PreparedFeatures, String> {
        for p in primitives {
            if let Some(ids) = self.per_vertex_ids(bin, p.mesh_ix, p.prim_ix)? {
                p.set_feature_ids(&ids);
            }
        }
        Ok(PreparedFeatures {
            node_of_feature: std::mem::take(&mut self.node_of_feature),
            vertex_ids: Vec::new(),
        })
    }

    /// Read every feature-carrying primitive's per-vertex ids up front — the
    /// [`PreparedFeatures`] the worker reply carries so the main thread never
    /// re-splits/re-parses the JSON to rebuild feature picking.
    pub fn materialize(mut self, bin: Option<&[u8]>) -> Result<PreparedFeatures, String> {
        let mut keys: Vec<(u64, u64)> = self.accessor.keys().copied().collect();
        keys.sort_unstable(); // deterministic reply layout
        let mut vertex_ids = Vec::with_capacity(keys.len());
        for (m, p) in keys {
            if let Some(ids) = self.per_vertex_ids(bin, m, p)? {
                vertex_ids.push(((m, p), ids));
            }
        }
        Ok(PreparedFeatures {
            node_of_feature: std::mem::take(&mut self.node_of_feature),
            vertex_ids,
        })
    }
}

/// The `EXT_mesh_features` tables of one primitive, from its raw per-vertex
/// `_FEATURE_ID_0` values:
/// * the `ATTRIBUTE_UV_1` layout `[fid, 0]`, padded with feature 0 to
///   `vertex_count` (a mesh attribute must match the position count);
/// * the feature id of each triangle, in `indices` order, so a pick hit's
///   triangle ordinal indexes it directly.
///
/// The one implementation behind every decode route (inline, prepared GLB,
/// extracted), so their picking cannot drift.
pub fn feature_tables(
    per_vertex: &[f32],
    indices: &[u32],
    vertex_count: usize,
) -> (Vec<[f32; 2]>, Vec<u32>) {
    let uv1 = (0..vertex_count)
        .map(|v| [per_vertex.get(v).copied().unwrap_or(0.0), 0.0])
        .collect();
    let by_triangle = indices
        .chunks_exact(3)
        .map(|t| {
            per_vertex
                .get(t[0] as usize)
                .map_or(0, |f| f.round() as u32)
        })
        .collect();
    (uv1, by_triangle)
}

/// Axis-aligned bounds `[min, max]` of `positions`; `None` when there are none.
pub fn bounds_of(positions: &[[f32; 3]]) -> Option<[[f32; 3]; 2]> {
    let (first, rest) = positions.split_first()?;
    Some(rest.iter().fold([*first, *first], |[lo, hi], p| {
        [
            std::array::from_fn(|i| lo[i].min(p[i])),
            std::array::from_fn(|i| hi[i].max(p[i])),
        ]
    }))
}

/// Read the `nodePath` STRING property of `EXT_structural_metadata`'s first
/// property table → `featureId → node path`. UINT32 string offsets (what our
/// writer emits); other offset widths are unsupported (we control the writer).
fn read_node_of_feature(
    json: &serde_json::Value,
    bin: Option<&[u8]>,
) -> Result<Vec<String>, String> {
    let pt = &json["extensions"]["EXT_structural_metadata"]["propertyTables"][0];
    let count = pt["count"].as_u64().ok_or("property table without count")? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    let prop = &pt["properties"]["nodePath"];
    let values_bv = prop["values"]
        .as_u64()
        .ok_or("nodePath property without values")? as usize;
    let offsets_bv = prop["stringOffsets"]
        .as_u64()
        .ok_or("nodePath property without stringOffsets")? as usize;
    let values =
        buffer_view_slice(json, bin, values_bv).map_err(|e| format!("nodePath values: {e}"))?;
    let offsets =
        buffer_view_slice(json, bin, offsets_bv).map_err(|e| format!("nodePath offsets: {e}"))?;
    if offsets.len() < (count + 1) * 4 {
        return Err("nodePath stringOffsets too short".into());
    }
    let read_u32 =
        |i: usize| u32::from_le_bytes(offsets[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let (lo, hi) = (read_u32(i), read_u32(i + 1));
        let s = values
            .get(lo..hi)
            .ok_or("nodePath string range out of bounds")?;
        out.push(String::from_utf8_lossy(s).into_owned());
    }
    Ok(out)
}

/// Read accessor `index` as `Vec<[f32; N]>`. Supports float and the spec's
/// normalized integer encodings; tightly-packed or strided buffer views; no
/// sparse accessors (our tilers never emit them).
pub fn read_accessor<const N: usize>(
    json: &serde_json::Value,
    bin: Option<&[u8]>,
    index: usize,
) -> Result<Vec<[f32; N]>, String> {
    let acc = &json["accessors"][index];
    if acc.is_null() {
        return Err(format!("accessor {index} out of bounds"));
    }
    let count = acc["count"].as_u64().ok_or("accessor without count")? as usize;
    let comp_type = acc["componentType"]
        .as_u64()
        .ok_or("accessor without componentType")?;
    let normalized = acc["normalized"].as_bool().unwrap_or(false);
    let type_str = acc["type"].as_str().ok_or("accessor without type")?;
    let comps = match type_str {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        other => return Err(format!("unsupported accessor type {other}")),
    };
    if comps != N {
        return Err(format!(
            "accessor {index} is {type_str}, expected {N} components"
        ));
    }
    let comp_size = match comp_type {
        5120 | 5121 => 1, // i8 / u8
        5122 | 5123 => 2, // i16 / u16
        5125 | 5126 => 4, // u32 / f32
        other => return Err(format!("unsupported componentType {other}")),
    };
    let bv_ix = acc["bufferView"]
        .as_u64()
        .ok_or("accessor without bufferView")? as usize;
    let bv = &json["bufferViews"][bv_ix];
    if bv["buffer"].as_u64() != Some(0) {
        return Err("accessor bufferView must reference buffer 0 (BIN chunk)".into());
    }
    let bin = bin.ok_or("accessor references the BIN chunk but the GLB has none")?;
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let bv_len = bv["byteLength"]
        .as_u64()
        .ok_or("bufferView without byteLength")? as usize;
    let stride = bv["byteStride"]
        .as_u64()
        .map(|s| s as usize)
        .unwrap_or(comp_size * N);
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let view = bin
        .get(bv_offset..bv_offset + bv_len)
        .ok_or("bufferView out of BIN bounds")?;

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = acc_offset + i * stride;
        let mut vals = [0f32; N];
        for (c, val) in vals.iter_mut().enumerate() {
            let at = base + c * comp_size;
            let bytes = view
                .get(at..at + comp_size)
                .ok_or_else(|| format!("accessor {index} element {i} out of bounds"))?;
            *val = match comp_type {
                5126 => f32::from_le_bytes(bytes.try_into().unwrap()),
                5121 => {
                    let v = bytes[0] as f32;
                    if normalized { v / 255.0 } else { v }
                }
                5120 => {
                    let v = bytes[0] as i8 as f32;
                    if normalized { (v / 127.0).max(-1.0) } else { v }
                }
                5123 => {
                    let v = u16::from_le_bytes(bytes.try_into().unwrap()) as f32;
                    if normalized { v / 65535.0 } else { v }
                }
                5122 => {
                    let v = i16::from_le_bytes(bytes.try_into().unwrap()) as f32;
                    if normalized {
                        (v / 32767.0).max(-1.0)
                    } else {
                        v
                    }
                }
                5125 => u32::from_le_bytes(bytes.try_into().unwrap()) as f32,
                _ => unreachable!(),
            };
        }
        out.push(vals);
    }
    Ok(out)
}

// ── prepare_tile — the S4 hook payload ───────────────────────────────────────

/// Feature-picking side-band of a [`PreparedTile`]: everything the main
/// thread would otherwise re-split + re-parse the tile JSON to rebuild
/// (`EXT_mesh_features` + `EXT_structural_metadata`). Rides the worker reply
/// header, per the offthread-decode plan's "decide before S4 starts" risk item.
pub struct PreparedFeatures {
    /// featureId → source-node path, shared by all of the tile's primitives.
    pub node_of_feature: Vec<String>,
    /// `(mesh index, primitive index)` → raw per-VERTEX `_FEATURE_ID_0`
    /// values (accessor order/length — the consumer pads to vertex count and
    /// derives the per-triangle table from its own index buffer), sorted by
    /// key. **Empty when [`PreparedTile::meshes`] is `Some`**: the extracted
    /// primitives already carry their tables
    /// ([`ExtractedPrimitive::feature_uv1`]), so the ids are not sent twice.
    pub vertex_ids: Vec<((u64, u64), Vec<f32>)>,
}

/// The output of [`prepare_tile`]: a plain (extension-free) glTF binary the
/// strict `gltf` crate accepts, plus the side-band data extracted on the way.
///
/// Geometry travels one of two ways, and `meshes` says which:
/// * `meshes: None` (S4) — `glb` holds the prepared container and the consumer
///   parses it with the `gltf` crate;
/// * `meshes: Some(_)` (S5, [`prepare_tile_extracting`]) — the geometry is
///   already typed buffers (base-colour textures still encoded, missing
///   normals filled), the consumer never parses glTF at all, and `glb`
///   is **empty** (rebuilding a container nobody reads is pure cost, on both
///   the producing thread and the wire).
pub struct PreparedTile {
    /// Vanilla GLB — meshopt views decoded, basisu sources rewritten,
    /// CESIUM_RTC / planetary offsets stripped. For a tile that needed no
    /// rewrite these are the input bytes unchanged; **empty** when `meshes`
    /// carries the geometry instead.
    pub glb: Vec<u8>,
    /// Extracted geometry (S5). `None` = the consumer decodes `glb` itself,
    /// either because extraction was not asked for ([`prepare_tile`]) or
    /// because [`extract_tile_meshes`] declined this tile's content.
    pub meshes: Option<ExtractedMeshes>,
    /// Rtc offset (ECEF metres) the consumer composes innermost, in the glTF
    /// content frame: `CESIUM_RTC` center, extracted planetary root offset,
    /// or a b3dm feature-table `RTC_CENTER` (rotated from the tile frame).
    pub rtc_center: Option<[f64; 3]>,
    /// glTF `asset.copyright` (attribution overlay side-band).
    pub copyright: Option<String>,
    /// Feature-picking data, when the tile carries `EXT_mesh_features`.
    pub features: Option<PreparedFeatures>,
    /// Why extraction declined this tile (`meshes` is `None` although it was
    /// asked for): a short phrase naming the first rule the content failed,
    /// e.g. `"extensionsRequired: KHR_texture_transform"` or
    /// `"TEXCOORD_0: integer (normalized) components"`. `None` when the tile
    /// extracted or extraction was not asked for. Diagnostic text for a log,
    /// not a value to match on: the wording may change. The tile renders the
    /// same either way.
    pub extract_declined: Option<String>,
}

/// Would [`prepare_tile`] hand these bytes straight back — either declined
/// (`Ok(None)`: content it has no decoder for) or echoed verbatim (a vanilla
/// non-georeferenced tile with nothing to rewrite)?
///
/// For callers that pay to ship the tile somewhere else (a Web Worker
/// transfer, an IPC hop): the answer costs one header read plus a marker scan
/// of the kilobyte JSON chunk, and saves a round trip that learns nothing. For
/// a layer that is Draco on every tile, that is a per-tile saving for a whole
/// session. Both answers route the tile to the caller's own inline
/// decode, which is what `Ok(None)` does too.
///
/// Bytes that are not a container at all answer `false` — let [`prepare_tile`]
/// produce the error rather than mirroring its parsing here.
pub fn prepare_would_decline(bytes: &[u8], georeferenced: bool) -> bool {
    let bytes = match unwrap_b3dm(bytes) {
        Ok(Some(b)) => b.glb,
        Ok(None) => bytes,
        Err(_) => return false,
    };
    let Ok((json_chunk, _)) = split_glb(bytes) else {
        return false;
    };
    declines(&Marks::scan(json_chunk), georeferenced)
}

/// [`prepare_would_decline`] for [`prepare_tile_extracting`], relaxed by
/// exactly one case: a vanilla tile — nothing to rewrite, so nothing for
/// `prepare_tile` to do — IS worth the trip under S5, because the parse +
/// attribute collect its geometry extraction saves is the cost S5 exists to
/// move. Textured vanilla tiles included (since prepare 0.3 they extract, and
/// the image decode can move with them).
///
/// The relaxation is conditional on the tile being extractable at all: a
/// vanilla tile [`extract_tile_meshes`] will decline anyway would pay a full
/// round trip (two worker-side copies of a multi-MB GLB) to get its own bytes
/// back and decode inline — strictly worse than S4. The one decline reason a
/// marker scan can see, a surviving `extensionsRequired` (`KHR_mesh_quantization`
/// and anything else no pass here handles), keeps declining here. The scan
/// cannot see WHICH extension, so a tile requiring only `KHR_materials_unlit`
/// (which extracts) declines here too: one skipped extraction, never a wasted
/// trip.
///
/// Ceiling: the decline reasons that live in *values* rather than keys — a
/// non-TRIANGLES `mode`, a sparse or non-`FLOAT` attribute, a VEC3 `COLOR_0`, a
/// texture the consumer cannot decode, or any texture at all under
/// [`ExtractOptions::textures`] off — have no marker to scan for, so those
/// tiles still pay one wasted round trip each. Catching them means parsing the
/// JSON twice, which costs more than the trip saves.
pub fn extract_would_decline(bytes: &[u8], georeferenced: bool) -> bool {
    extract_triage(bytes, georeferenced, false)
}

/// [`extract_would_decline`] for a host whose worker owns a Draco decoder and
/// prepares via [`prepare_tile_extracting_with_draco`]: Draco tiles ARE worth
/// dispatching there (the decode itself moves off-thread), so only splats and
/// the unextractable-vanilla case still decline. A Draco, textured layer then
/// round-trips with the Draco decode + splice done worker-side and (prepare
/// 0.3) comes back extracted, textures included. A photorealistic layer can
/// also arrive as plain glTF (pre-decoded for a client without Draco); that is
/// the vanilla case and extracts the same way.
pub fn extract_would_decline_with_draco(bytes: &[u8], georeferenced: bool) -> bool {
    extract_triage(bytes, georeferenced, true)
}

fn extract_triage(bytes: &[u8], georeferenced: bool, draco_ok: bool) -> bool {
    let bytes = match unwrap_b3dm(bytes) {
        Ok(Some(b)) => b.glb,
        Ok(None) => bytes,
        Err(_) => return false,
    };
    let Ok((json_chunk, _)) = split_glb(bytes) else {
        return false;
    };
    let marks = Marks::scan(json_chunk);
    // Marker scan, not a parse: `"extensionsRequired":[]` reads as required
    // and keeps the S4 answer, which is the safe direction (one skipped
    // extraction, never a wasted trip). Images and textures extract since
    // prepare 0.3, so they no longer count.
    let unextractable = memmem(json_chunk, b"\"extensionsRequired\"");
    let undecodable = if draco_ok {
        marks.splat
    } else {
        marks.draco || marks.splat
    };
    // Minimal form of `(undecodable || vanilla-echo) && (unextractable ||
    // undecodable)`: undecodable content always declines; the vanilla
    // non-georeferenced echo declines only when it is also unextractable
    // (the S5 relaxation — see `extract_would_decline`'s doc).
    undecodable || (!georeferenced && marks.vanilla() && unextractable)
}

/// The predicate [`prepare_tile`] opens with, in ONE place — an off-thread
/// caller needs the same answer before it dispatches ([`prepare_would_decline`])
/// and the two drifting apart is a silent round trip per tile.
fn declines(marks: &Marks, georeferenced: bool) -> bool {
    marks.draco || marks.splat || (!georeferenced && marks.vanilla())
}

/// Run every synchronous, bevy-free decode pass of a tile: marker scan, ONE
/// JSON parse, meshopt BIN decode, basisu/RTC/planetary rewrites, feature
/// extraction, ONE container rebuild — the exact movable set of the
/// offthread-decode plan's S4 seam.
///
/// * `Ok(Some(_))` — prepared; the caller decodes the vanilla GLB.
/// * `Ok(None)` — declined: the tile needs a platform decoder (Draco per
///   `Marks::draco`) or a special renderer path (splats per `Marks::splat`).
///   Not an error — the caller falls back to its inline path.
/// * `Err(_)` — malformed content; the caller warns once and falls back
///   inline (which will surface the same error with full diagnostics).
pub fn prepare_tile(
    bytes: &[u8],
    georeferenced: bool,
) -> Result<Option<PreparedTile>, DecodeError> {
    prepare_tile_inner(bytes, georeferenced, None, None)
}

/// [`prepare_tile`] plus geometry extraction (offthread-decode plan S5): the
/// same single JSON parse also yields [`ExtractedMeshes`], so the consumer
/// builds meshes straight from typed buffers and skips the `gltf` parse and
/// the per-primitive attribute collect entirely (the dominant remaining
/// main-thread streaming cost — 6-9 ms/tile on bevy 0.19).
///
/// Same three outcomes as [`prepare_tile`], plus one shade: extraction is
/// best-effort. Content it cannot reproduce byte-identically (non-triangle
/// primitives, quantized attributes, textures the consumer cannot decode — see
/// [`extract_tile_meshes`]) comes back as an ordinary S4 [`PreparedTile`] with
/// `meshes: None`, which the consumer decodes exactly as before.
///
/// [`ExtractOptions::default`]: textures extract, missing normals are filled.
pub fn prepare_tile_extracting(
    bytes: &[u8],
    georeferenced: bool,
) -> Result<Option<PreparedTile>, DecodeError> {
    prepare_tile_extracting_with(bytes, georeferenced, None, ExtractOptions::default())
}

/// [`prepare_tile_extracting`] for a caller that already decoded this tile's
/// Draco primitives (via [`draco_requests`] + its own platform decoder):
/// the decoded meshes are spliced in ([`splice_draco`]) after the meshopt
/// pass, the extension is stripped, and the rest of the pipeline runs
/// unchanged. `decoded` must be in [`draco_requests`] order — one mesh per
/// request, a mismatch is an error, never a silent truncation.
pub fn prepare_tile_extracting_with_draco(
    bytes: &[u8],
    georeferenced: bool,
    decoded: Vec<DracoMesh>,
) -> Result<Option<PreparedTile>, DecodeError> {
    prepare_tile_extracting_with(
        bytes,
        georeferenced,
        Some(decoded),
        ExtractOptions::default(),
    )
}

/// The general extracting entry point: [`prepare_tile_extracting`] (`draco:
/// None`) or [`prepare_tile_extracting_with_draco`] (`draco: Some`), with the
/// [`ExtractOptions`] a host chose. Pipeline order: extraction, then the
/// feature tables (which may give a non-indexed primitive indices), then the
/// missing normals ([`compute_normals`], so a synthesized index list gets the
/// smooth normals the inline route would compute).
pub fn prepare_tile_extracting_with(
    bytes: &[u8],
    georeferenced: bool,
    draco: Option<Vec<DracoMesh>>,
    opts: ExtractOptions,
) -> Result<Option<PreparedTile>, DecodeError> {
    prepare_tile_inner(bytes, georeferenced, Some(opts), draco)
}

/// `extract: None` = [`prepare_tile`] (no extraction).
fn prepare_tile_inner(
    bytes: &[u8],
    georeferenced: bool,
    extract: Option<ExtractOptions>,
    draco: Option<Vec<DracoMesh>>,
) -> Result<Option<PreparedTile>, DecodeError> {
    // Legacy b3dm containers unwrap to their embedded glb FIRST — a b3dm is
    // not glTF magic, so split_glb would misread it as bare JSON.
    let b3dm = unwrap_b3dm(bytes)?;
    let (bytes, b3dm_rtc) = match &b3dm {
        Some(b) => (b.glb, b.rtc_center),
        None => (bytes, None),
    };
    let (json_chunk, bin) = split_glb(bytes)?;
    let marks = Marks::scan(json_chunk);
    if marks.splat || (marks.draco && draco.is_none()) {
        return Ok(None); // needs a platform decoder/renderer the caller has not got
    }
    // Vanilla and not georeferenced: no rewrite, nothing to extract from the
    // JSON side-band. Without S5 there is no reason to parse it at all; WITH
    // S5 the geometry is the whole point of the trip, so it falls through.
    // `!marks.draco` guards the echo: `declines` counts the Draco mark, but a
    // tile arriving WITH decoded meshes must splice below, never echo.
    if extract.is_none() && !marks.draco && declines(&marks, georeferenced) {
        return Ok(Some(PreparedTile {
            glb: bytes.to_vec(),
            meshes: None,
            rtc_center: b3dm_rtc.map(tile_rtc_to_content_frame),
            copyright: None,
            features: None,
            extract_declined: None,
        }));
    }

    let mut json: serde_json::Value =
        serde_json::from_slice(json_chunk).map_err(|e| format!("tile json: {e}"))?;
    let copyright = json["asset"]["copyright"].as_str().map(str::to_string);
    // b3dm rtc offsets (feature-table RTC_CENTER first, else CESIUM_RTC in
    // the embedded glb) are tile-frame and rotate into the content frame; a
    // bare-glb CESIUM_RTC is already content-frame (see [`B3dm`]).
    let mut rtc_center = b3dm_rtc
        .or_else(|| {
            json["extensions"]["CESIUM_RTC"]["center"]
                .as_array()
                .and_then(|c| {
                    let v: Vec<f64> = c.iter().filter_map(|x| x.as_f64()).collect();
                    <[f64; 3]>::try_from(v).ok()
                })
        })
        .map(|c| {
            if b3dm.is_some() {
                tile_rtc_to_content_frame(c)
            } else {
                c
            }
        });

    // Google P3DT bakes ECEF positions into node MATRICES instead of
    // CESIUM_RTC. Gated on `georeferenced` exactly like the inline path: the
    // rebase MUTATES node matrices and only a georeferenced host re-applies
    // the offset.
    let mut nodes_rebased = false;
    if georeferenced
        && rtc_center.is_none()
        && let Some(center) = extract_planetary_root_offset(&mut json)
    {
        rtc_center = Some(center);
        nodes_rebased = true;
    }

    // Same pass order as the inline path: meshopt first (it REBUILDS the BIN,
    // so every later pass reads decoded bytes; buffer-view indices preserved).
    let mut new_bin: Option<Vec<u8>> = if marks.meshopt {
        Some(decode_meshopt_views(&mut json, bin).map_err(DecodeError::meshopt)?)
    } else {
        None
    };
    // Caller-decoded Draco splices exactly where the inline path splices its
    // shim-decoded meshes: after meshopt, before basisu.
    if marks.draco
        && let Some(decoded) = draco
    {
        let prims = find_draco_prims(&json);
        if prims.len() != decoded.len() {
            return Err(DecodeError::draco(format!(
                "{} draco primitives but {} decoded meshes — decode draco_requests in order",
                prims.len(),
                decoded.len()
            )));
        }
        new_bin = Some(splice_draco(
            &mut json,
            new_bin.as_deref().or(bin),
            &prims,
            decoded,
        )?);
    }
    if marks.basisu {
        preprocess_basisu(&mut json);
    }
    // Runs on the MARKER, like inline — a document can declare Draco/RTC
    // without a usable primitive and the gltf crate still hard-rejects the
    // unknown `extensionsRequired` entry.
    let stripped = marks.rtc || marks.draco;
    if stripped {
        strip_handled_extensions(&mut json);
    }
    let bin = new_bin.as_deref().or(bin);

    // S5: geometry off the document we already hold. Runs BEFORE the feature
    // pass (which consumes `json`) and before the container rebuild — when it
    // succeeds there is no container to rebuild, because nobody will parse one.
    let (mut meshes, extract_declined) = match extract {
        Some(opts) => match extract::extract_tile_meshes_why(&json, bin, opts)? {
            Ok(m) => (Some(m), None),
            Err(why) => (None, Some(why)),
        },
        None => (None, None),
    };

    let glb = if meshes.is_some() {
        Vec::new()
    } else if nodes_rebased || stripped || marks.meshopt || marks.basisu {
        let json_bytes = serde_json::to_vec(&json).map_err(|e| format!("tile splice json: {e}"))?;
        assemble_glb(&json_bytes, bin.unwrap_or(&[]))
    } else {
        bytes.to_vec()
    };

    // Feature metadata reads the post-rewrite document, so the property table
    // + `_FEATURE_ID_0` accessors line up with the rebuilt BIN.
    //
    // Two outcomes, both matching the inline path exactly:
    // * a malformed property TABLE (`build`) loses picking, never geometry —
    //   same policy as the inline path, minus its (bevy) log line;
    // * a bad `_FEATURE_ID_0` ACCESSOR (`materialize`) is an Err, because
    //   inline reads those with `?` and fails the whole tile. Erroring here
    //   routes to warn-once → inline, which reproduces that failure with full
    //   diagnostics. Swallowing it instead would make the same bytes render
    //   (picking silently gone) or fail depending on whether a Worker booted.
    //
    // Extracted geometry gets its feature tables (and synthesized indices)
    // here, off the consumer's thread; only the primitives it extracted are
    // read, exactly the set the inline route would read.
    let features = if marks.features {
        match FeatureCtx::build(json, bin) {
            Ok(ctx) => Some(match meshes.as_mut() {
                Some(m) => ctx.attach(bin, &mut m.primitives)?,
                None => ctx.materialize(bin)?,
            }),
            Err(_) => None,
        }
    } else {
        None
    };

    // Missing normals, LAST: the feature pass above may have given a
    // non-indexed primitive indices, and the inline route computes smooth
    // normals over those, so this must see them too.
    if let (Some(m), Some(opts)) = (meshes.as_mut(), extract)
        && opts.normals
    {
        for p in &mut m.primitives {
            if p.normals.is_none() {
                p.normals = Some(compute_normals(&p.positions, p.indices.as_deref()));
            }
        }
    }

    Ok(Some(PreparedTile {
        glb,
        meshes,
        rtc_center,
        copyright,
        features,
        extract_declined,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a b3dm container: header + feature-table JSON + payload.
    fn b3dm(ft_json: &str, payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"b3dm");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&((28 + ft_json.len() + payload.len()) as u32).to_le_bytes());
        b.extend_from_slice(&(ft_json.len() as u32).to_le_bytes());
        for _ in 0..3 {
            b.extend_from_slice(&0u32.to_le_bytes()); // ft bin, bt json, bt bin
        }
        b.extend_from_slice(ft_json.as_bytes());
        b.extend_from_slice(payload);
        b
    }

    /// The two real-world b3dm rtc shapes, minus their Draco payloads:
    /// swisstopo carries a feature-table `RTC_CENTER`, PLATEAU a `CESIUM_RTC`
    /// inside the embedded glTF. Both are tile-frame and get the z-up→y-up
    /// rotation; a bare-glb `CESIUM_RTC` (offline P3DT) stays untouched.
    #[test]
    fn b3dm_unwraps_and_rotates_tile_frame_rtc() {
        let inner = br#"{"asset":{"version":"2.0"}}"#;
        let b = b3dm(r#"{"BATCH_LENGTH":1,"RTC_CENTER":[1.0,2.0,3.0]}"#, inner);
        let p = prepare_tile(&b, true).expect("prepare").expect("prepared");
        assert_eq!(p.glb, inner);
        assert_eq!(p.rtc_center, Some([1.0, 3.0, -2.0]));

        let inner = br#"{"extensions":{"CESIUM_RTC":{"center":[1.0,2.0,3.0]}}}"#;
        let b = b3dm("{}", inner);
        let p = prepare_tile(&b, true).expect("prepare").expect("prepared");
        assert_eq!(p.rtc_center, Some([1.0, 3.0, -2.0]));

        let p = prepare_tile(inner, true)
            .expect("prepare")
            .expect("prepared");
        assert_eq!(p.rtc_center, Some([1.0, 2.0, 3.0]));

        // Draco b3dm (the real swisstopo/PLATEAU tiles) still declines to the
        // platform decoder, and the off-thread triage agrees.
        let draco = b3dm(
            "{}",
            br#"{"extensionsUsed":["KHR_draco_mesh_compression"]}"#,
        );
        assert!(prepare_tile(&draco, true).unwrap().is_none());
        assert!(prepare_would_decline(&draco, true));
    }

    #[test]
    fn b3dm_malformed_and_sibling_containers_error() {
        assert!(unwrap_b3dm(b"cmpt....").unwrap_err().contains("cmpt"));
        assert!(unwrap_b3dm(b"b3dm").is_err()); // truncated header
        let mut overrun = b3dm("{}", b"x");
        overrun[12] = 255; // feature-table length far past the buffer end
        assert!(unwrap_b3dm(&overrun).is_err());
        // Not a legacy container: pass-through for the glb/bare-JSON path.
        assert!(unwrap_b3dm(b"glTF....").unwrap().is_none());
        assert!(unwrap_b3dm(br#"{"a":1}"#).unwrap().is_none());
    }

    /// The worker-side Draco flow end to end: slice requests out, decode them
    /// "elsewhere" (a fabricated mesh stands in for the platform decoder),
    /// hand them back — the pipeline splices, strips, and the triage
    /// predicates route a Draco tile to the worker only when it can decode.
    #[test]
    fn draco_requests_then_prepare_with_decoded_meshes() {
        let fake_compressed = vec![0xAAu8; 16];
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_draco_mesh_compression"],
            "extensionsRequired": ["KHR_draco_mesh_compression"],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0 },
                "mode": 4,
                "extensions": { "KHR_draco_mesh_compression": {
                    "bufferView": 0,
                    "attributes": { "POSITION": 0 }
                }}
            }]}],
            "accessors": [
                { "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0,0,0], "max": [1,1,0] }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": fake_compressed.len() }
            ],
            "buffers": [{ "byteLength": fake_compressed.len() }]
        });
        let glb = assemble_glb(&serde_json::to_vec(&json).unwrap(), &fake_compressed);

        // Triage: undecodable without a decoder, dispatchable with one.
        assert!(extract_would_decline(&glb, false));
        assert!(!extract_would_decline_with_draco(&glb, false));
        // Without decoded meshes the pipeline still declines to the caller.
        assert!(prepare_tile_extracting(&glb, false).unwrap().is_none());

        let reqs = draco_requests(&glb).expect("requests");
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].attr_ids, vec![0]);
        assert_eq!(reqs[0].compressed, fake_compressed);
        // Non-Draco bytes answer [] from the marker scan alone.
        assert!(
            draco_requests(br#"{"asset":{"version":"2.0"}}"#)
                .unwrap()
                .is_empty()
        );

        let decoded = vec![DracoMesh {
            indices: vec![0, 1, 2],
            attributes: vec![(0, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
        }];
        let p = prepare_tile_extracting_with_draco(&glb, false, decoded)
            .expect("prepare")
            .expect("prepared, not declined");
        // Whichever route extraction picked, no Draco survives the output.
        match &p.meshes {
            Some(m) => {
                assert_eq!(m.primitives.len(), 1);
                assert_eq!(m.primitives[0].positions.len(), 3);
            }
            None => {
                assert!(!p.glb.is_empty());
                let (j, _) = split_glb(&p.glb).unwrap();
                assert!(!memmem(j, b"KHR_draco_mesh_compression"));
            }
        }

        // A count mismatch is an error, never a silent truncation.
        assert!(prepare_tile_extracting_with_draco(&glb, false, Vec::new()).is_err());
    }

    #[test]
    fn declines_draco_and_splat_markers() {
        let draco = br#"{"extensionsRequired":["KHR_draco_mesh_compression"]}"#;
        assert!(prepare_tile(draco, false).unwrap().is_none());
        let splat = br#"{"extensionsUsed":["KHR_gaussian_splatting"]}"#;
        assert!(prepare_tile(splat, false).unwrap().is_none());
    }

    /// `prepare_would_decline` is what an off-thread caller triages on before
    /// it pays for a transfer, so pin it against `prepare_tile` itself: when
    /// it says yes, a round trip really would have returned nothing (`None`)
    /// or the input bytes verbatim with no side-band; when it says no, real
    /// prep is waiting. (`split_glb` treats bytes without the `glTF` magic as
    /// a bare JSON glTF, so these fixtures are just JSON.)
    #[test]
    fn would_decline_agrees_with_prepare_tile() {
        for (json, geo) in [
            // Content with no decoder here — a whole P3DT layer is Draco.
            (r#"{"extensionsUsed":["KHR_draco_mesh_compression"]}"#, true),
            (r#"{"extensionsUsed":["KHR_gaussian_splatting"]}"#, false),
            // Nothing to rewrite and nothing to extract: it echoes the bytes.
            (r#"{"asset":{"version":"2.0"}}"#, false),
        ] {
            let b = json.as_bytes();
            assert!(
                prepare_would_decline(b, geo),
                "should decline: {json} ({geo})"
            );
            if let Some(p) = prepare_tile(b, geo).expect("prepare") {
                assert_eq!(p.glb, b, "declined but prepare_tile rewrote it");
                assert!(p.features.is_none() && p.rtc_center.is_none());
            }
        }
        for (json, geo) in [
            // The same vanilla tile georeferenced — the root-offset pass runs.
            (r#"{"asset":{"version":"2.0"}}"#, true),
            (r#"{"extensionsUsed":["EXT_meshopt_compression"]}"#, false),
            (r#"{"extensionsUsed":["EXT_mesh_features"]}"#, false),
            (r#"{"extensions":{"CESIUM_RTC":{"center":[1,2,3]}}}"#, false),
        ] {
            assert!(
                !prepare_would_decline(json.as_bytes(), geo),
                "should prepare: {json} ({geo})"
            );
        }
    }

    /// The S5 triage relaxes `prepare_would_decline` for vanilla tiles, whose
    /// geometry extraction is the whole point — textured ones included since
    /// prepare 0.3, when textures started extracting too. What a marker scan can
    /// see `extract_tile_meshes` declining (a surviving `extensionsRequired`)
    /// must still be rejected, or the host pays a full round trip (two
    /// multi-MB copies) to get its own bytes back.
    #[test]
    fn triage_dispatches_vanilla_textured_tiles() {
        for json in [
            // Nothing to rewrite, nothing textured.
            r#"{"asset":{"version":"2.0"},"meshes":[]}"#,
            // A plain PNG/JPEG-textured tile: no basisu, no meshopt, no RTC.
            // `prepare_tile` alone would echo it; extraction now takes it.
            r#"{"asset":{"version":"2.0"},"images":[{"mimeType":"image/png"}]}"#,
            r#"{"asset":{"version":"2.0"},"textures":[{"source":0}]}"#,
        ] {
            assert!(prepare_would_decline(json.as_bytes(), false), "{json}");
            assert!(
                !extract_would_decline(json.as_bytes(), false),
                "S5 dispatches: {json}"
            );
        }

        for json in [
            // The pre-existing no-decoder cases stay declined.
            r#"{"extensionsUsed":["KHR_draco_mesh_compression"]}"#,
            r#"{"extensionsUsed":["KHR_gaussian_splatting"]}"#,
            // Untextured, nothing to rewrite, but `extract_tile_meshes` declines
            // any surviving `extensionsRequired` — so does the triage, or every
            // quantized tile pays the round trip to learn that.
            r#"{"extensionsRequired":["KHR_mesh_quantization"],"meshes":[]}"#,
        ] {
            assert!(
                extract_would_decline(json.as_bytes(), false),
                "should decline: {json}"
            );
        }
        // Textured with real prep waiting (basisu) is dispatched as before.
        let basisu =
            br#"{"extensionsUsed":["KHR_texture_basisu"],"images":[{"mimeType":"image/ktx2"}]}"#;
        assert!(!extract_would_decline(basisu, false));

        // A meshopt tile declares EXT_meshopt_compression REQUIRED (our tiler
        // does: `setRequired(true)` in tile_mesh.mjs), so the
        // `extensionsRequired` marker above DOES match it — it is only the
        // `declines()` short-circuit that saves the dispatch, because a meshopt
        // tile has real prep waiting. It must keep dispatching: the decode
        // strips the extension once the views are decoded, and extraction then
        // applies, which is where the whole geometry saving on a meshopt scene
        // comes from. The marker half of this predicate must never be allowed
        // to reach it.
        let meshopt = br#"{"extensionsUsed":["EXT_meshopt_compression"],"extensionsRequired":["EXT_meshopt_compression"]}"#;
        assert!(!extract_would_decline(meshopt, false), "meshopt dispatches");
        assert!(!extract_would_decline(meshopt, true), "meshopt dispatches");
    }

    /// A one-triangle GLB whose two materials use a PNG and a JPEG base-colour
    /// texture, and a third material that reuses the PNG. The image bytes are
    /// opaque to this crate (it never decodes them), so they are short
    /// literals with the right magic, not whole images.
    fn textured_tile() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let png = b"\x89PNG\r\n\x1a\nfake-png".to_vec();
        let jpeg = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9".to_vec();
        let mut bin = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            .iter()
            .flatten()
        {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let png_at = bin.len();
        bin.extend_from_slice(&png);
        let jpeg_at = bin.len();
        bin.extend_from_slice(&jpeg);
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [
                { "attributes": { "POSITION": 0 }, "material": 0 },
                { "attributes": { "POSITION": 0 }, "material": 1 },
                { "attributes": { "POSITION": 0 }, "material": 2 }
            ]}],
            "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": png_at, "byteLength": png.len() },
                { "buffer": 0, "byteOffset": jpeg_at, "byteLength": jpeg.len() }
            ],
            "buffers": [{ "byteLength": bin.len() }],
            "materials": [
                { "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } } },
                { "pbrMetallicRoughness": { "baseColorTexture": { "index": 1 }, "metallicFactor": 0.0 } },
                { "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 }, "roughnessFactor": 0.5 } }
            ],
            "textures": [{ "source": 0, "sampler": 0 }, { "source": 1 }],
            "samplers": [{ "wrapS": 33071, "wrapT": 33648 }],
            "images": [
                { "bufferView": 1, "mimeType": "image/png" },
                { "bufferView": 2, "mimeType": "image/jpeg" }
            ]
        });
        (
            assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin),
            png,
            jpeg,
        )
    }

    /// Textured content extracts (prepare 0.3): each base-colour texture rides
    /// `textures` ENCODED, the bytes exactly the bufferView slice, its wrap
    /// modes from the sampler (absent = REPEAT), and a texture two materials
    /// share is carried once.
    #[test]
    fn extracts_textured_tile_with_encoded_base_color() {
        let (glb, png, jpeg) = textured_tile();
        let m = prepare_tile_extracting(&glb, false)
            .unwrap()
            .expect("prepared")
            .meshes
            .expect("textured content extracts");
        assert_eq!(m.primitives.len(), 3);
        assert_eq!(m.textures.len(), 2, "one entry per glTF texture used");
        assert_eq!(
            m.textures[0],
            ExtractedTexture {
                image: TileImage::Encoded {
                    mime: "image/png".into(),
                    bytes: png
                },
                wrap_s: TextureWrap::ClampToEdge,
                wrap_t: TextureWrap::MirroredRepeat,
            }
        );
        assert_eq!(
            m.textures[1],
            ExtractedTexture {
                image: TileImage::Encoded {
                    mime: "image/jpeg".into(),
                    bytes: jpeg
                },
                wrap_s: TextureWrap::Repeat,
                wrap_t: TextureWrap::Repeat,
            }
        );
        let tex: Vec<_> = m.materials.iter().map(|m| m.base_color_texture).collect();
        assert_eq!(tex, [Some(0), Some(1), Some(0)]);
        assert_eq!(
            (m.materials[1].metallic, m.materials[2].roughness),
            (0.0, 0.5)
        );
    }

    /// The shape of a photorealistic-mesh tile (Draco geometry with UVs and no
    /// normals, one JPEG base-colour texture behind a CLAMP sampler, an unlit
    /// material the document REQUIRES, a planetary node matrix), written from
    /// the format's public description: no captured tile bytes. With Draco
    /// decoded by the host, it extracts: textured, with normals filled in.
    #[test]
    fn photorealistic_shaped_textured_draco_tile_extracts() {
        let draco_payload = [0xAAu8; 16];
        let jpeg = [0xFFu8, 0xD8, 0xFF, 0xD9];
        let mut bin = draco_payload.to_vec();
        bin.extend_from_slice(&jpeg);
        let mut json = serde_json::json!({
            "asset": { "version": "2.0", "copyright": "Data A;Data B" },
            "extensionsUsed": ["KHR_draco_mesh_compression", "KHR_materials_unlit"],
            "extensionsRequired": ["KHR_draco_mesh_compression", "KHR_materials_unlit"],
            "scene": 0, "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0, "matrix": [1,0,0,0, 0,0,-1,0, 0,1,0,0, -1.9e6,-5.0e6,3.3e6,1] }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "TEXCOORD_0": 1 }, "indices": 2, "material": 0,
                "extensions": { "KHR_draco_mesh_compression": {
                    "bufferView": 0, "attributes": { "POSITION": 0, "TEXCOORD_0": 1 }
                }}
            }]}],
            "accessors": [
                { "componentType": 5126, "count": 4, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
                { "componentType": 5126, "count": 4, "type": "VEC2" },
                { "componentType": 5125, "count": 6, "type": "SCALAR" }
            ],
            "materials": [{
                "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 }, "metallicFactor": 0.0 },
                "extensions": { "KHR_materials_unlit": {} }
            }],
            "textures": [{ "source": 0, "sampler": 0 }],
            "samplers": [{ "magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071 }],
            "images": [{ "bufferView": 1, "mimeType": "image/jpeg" }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 16 },
                { "buffer": 0, "byteOffset": 16, "byteLength": 4 }
            ],
            "buffers": [{ "byteLength": bin.len() }]
        });
        let glb = assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin);
        assert!(!extract_would_decline_with_draco(&glb, true), "dispatched");
        let quad = || DracoMesh {
            indices: vec![0, 1, 2, 0, 2, 3],
            attributes: vec![
                (0, 3, vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.]),
                (1, 2, vec![0., 0., 1., 0., 1., 1., 0., 1.]),
            ],
        };
        for georeferenced in [true, false] {
            let p = prepare_tile_extracting_with_draco(&glb, georeferenced, vec![quad()])
                .unwrap()
                .expect("prepared");
            let Some(m) = p.meshes else {
                panic!(
                    "declined to S4 (georeferenced {georeferenced}): {:?}",
                    p.extract_declined
                );
            };
            assert_eq!(p.extract_declined, None);
            assert_eq!(m.primitives.len(), 1);
            let prim = &m.primitives[0];
            assert_eq!(prim.uvs.as_ref().map(Vec::len), Some(4));
            assert_eq!(
                prim.normals.as_ref().map(Vec::len),
                Some(4),
                "normals filled"
            );
            assert_eq!(m.materials[0].base_color_texture, Some(0));
            assert!(m.materials[0].unlit);
            assert_eq!(
                m.textures[0].image,
                TileImage::Encoded {
                    mime: "image/jpeg".into(),
                    bytes: jpeg.to_vec()
                }
            );
            assert_eq!(m.textures[0].wrap_s, TextureWrap::ClampToEdge);
            assert_eq!(p.rtc_center.is_some(), georeferenced, "planetary offset");
        }
        // The same tile with one more required extension goes S4, and says why.
        json["extensionsRequired"] = serde_json::json!([
            "KHR_draco_mesh_compression",
            "KHR_materials_unlit",
            "KHR_texture_transform"
        ]);
        let glb = assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin);
        let p = prepare_tile_extracting_with_draco(&glb, true, vec![quad()])
            .unwrap()
            .expect("prepared");
        assert!(p.meshes.is_none() && !p.glb.is_empty(), "S4");
        assert_eq!(
            p.extract_declined.as_deref(),
            Some("extensionsRequired: KHR_texture_transform")
        );
    }

    /// The worker fills every missing normal (prepare 0.3), so the consumer
    /// never computes them: indexed, and non-indexed made indexed by the
    /// feature pass (smooth normals over the synthesized indices, as inline).
    /// `ExtractOptions::normals` off leaves them to the consumer.
    #[test]
    fn worker_fills_missing_normals() {
        for indexed in [true, false] {
            let glb = feature_tile(indexed);
            let m = prepare_tile_extracting(&glb, false)
                .unwrap()
                .expect("prepared")
                .meshes
                .expect("extracted");
            for p in &m.primitives {
                assert_eq!(
                    p.normals.as_deref(),
                    Some(compute_normals(&p.positions, p.indices.as_deref()).as_slice()),
                    "indexed {indexed}"
                );
            }
            let off = ExtractOptions {
                normals: false,
                ..Default::default()
            };
            let m = prepare_tile_extracting_with(&glb, false, None, off)
                .unwrap()
                .expect("prepared")
                .meshes
                .expect("extracted");
            assert!(m.primitives.iter().all(|p| p.normals.is_none()));
        }
    }

    #[test]
    fn vanilla_tile_passes_through_byte_identical() {
        // A bare-JSON glTF with no markers: the fast path returns the input
        // bytes unchanged and no side-band data.
        let glb = assemble_glb(br#"{"asset":{"version":"2.0"}}"#, &[]);
        let p = prepare_tile(&glb, false).unwrap().expect("accepted");
        assert_eq!(p.glb, glb);
        assert!(p.rtc_center.is_none() && p.copyright.is_none() && p.features.is_none());
    }

    /// The inline path reads `_FEATURE_ID_0` accessors with `?` and fails the
    /// whole tile, so a bad one must be an `Err` here too — swallowing it
    /// would render the same bytes with picking silently missing whenever a
    /// worker happened to be alive.
    #[test]
    fn bad_feature_id_accessor_is_an_error_not_silent_loss() {
        // `count: 0` keeps the property table itself valid, so the failure is
        // squarely the accessor read (index 7 does not exist).
        let json = serde_json::json!({
            "extensions": { "EXT_structural_metadata": { "propertyTables": [{ "count": 0 }] } },
            "meshes": [{ "primitives": [{
                "extensions": { "EXT_mesh_features": { "featureIds": [{ "attribute": 0 }] } },
                "attributes": { "_FEATURE_ID_0": 7 },
            }] }],
        });
        let glb = assemble_glb(&serde_json::to_vec(&json).unwrap(), &[]);
        let Err(err) = prepare_tile(&glb, false) else {
            panic!("a bad feature accessor must fail the tile, not lose picking");
        };
        assert!(err.to_string().contains("accessor 7"), "{err}");
    }

    /// Two triangles carrying features 0 and 1, plus the `nodePath` table our
    /// tiler writes. The index buffer lists the SECOND triangle first, so a
    /// per-triangle table built in vertex order instead of index order fails.
    /// `indexed = false` drops the indices (vertices `3t..3t+3` are triangle t).
    fn feature_tile(indexed: bool) -> Vec<u8> {
        let positions: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, -1.0],
        ];
        let ids: [f32; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices: [u32; 6] = [3, 4, 5, 0, 1, 2];
        let mut bin = Vec::new();
        for v in positions.iter().flatten().chain(&ids) {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin.extend_from_slice(b"AB/c");
        for o in [0u32, 1, 4] {
            bin.extend_from_slice(&o.to_le_bytes());
        }
        let mut json = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["EXT_mesh_features", "EXT_structural_metadata"],
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "_FEATURE_ID_0": 1 },
                "indices": 2,
                "extensions": { "EXT_mesh_features": {
                    "featureIds": [{ "featureCount": 2, "attribute": 0, "propertyTable": 0 }]
                }}
            }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 6, "type": "SCALAR" },
                { "bufferView": 2, "componentType": 5125, "count": 6, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 72 },
                { "buffer": 0, "byteOffset": 72, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 96, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 120, "byteLength": 4 },
                { "buffer": 0, "byteOffset": 124, "byteLength": 12 }
            ],
            "buffers": [{ "byteLength": bin.len() }],
            "extensions": { "EXT_structural_metadata": { "propertyTables": [{
                "count": 2,
                "properties": { "nodePath": { "values": 3, "stringOffsets": 4 } }
            }]}}
        });
        if !indexed {
            json["meshes"][0]["primitives"][0]
                .as_object_mut()
                .unwrap()
                .remove("indices");
        }
        assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin)
    }

    /// The worker builds the feature tables (UV1 `[fid, 0]` + the per-triangle
    /// ids, in INDEX order) so the main thread does no per-vertex pass. They
    /// must equal what the consumer derives from the S4 route's raw per-vertex
    /// ids, and the raw ids are then not sent a second time.
    #[test]
    fn extracted_feature_tables_match_inline() {
        let glb = feature_tile(true);
        let s4 = prepare_tile(&glb, false).unwrap().expect("prepared");
        let mut s4_ids = s4.features.expect("S4 features").vertex_ids;
        assert_eq!(s4_ids.len(), 1);
        let (key, raw) = s4_ids.remove(0);
        assert_eq!(key, (0, 0));

        let s5 = prepare_tile_extracting(&glb, false)
            .unwrap()
            .expect("prepared");
        let feats = s5.features.expect("S5 features");
        assert_eq!(feats.node_of_feature, ["A", "B/c"]);
        assert!(
            feats.vertex_ids.is_empty(),
            "ids ride the primitive, not twice"
        );
        let p = &s5.meshes.expect("extracted").primitives[0];
        let (uv1, by_tri) = feature_tables(&raw, p.indices.as_deref().unwrap(), 6);
        assert_eq!(p.feature_uv1.as_deref(), Some(uv1.as_slice()));
        assert_eq!(p.feature_of_triangle.as_deref(), Some(by_tri.as_slice()));
        // And the values themselves: second triangle (feature 1) listed first.
        assert_eq!(by_tri, [1, 0]);
        assert_eq!(
            uv1,
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ]
        );
    }

    /// Short id accessors pad with feature 0 (a mesh attribute must match the
    /// vertex count or bevy rejects the mesh); the per-triangle id rounds.
    #[test]
    fn feature_tables_pad_and_round() {
        let (uv1, by_tri) = feature_tables(&[2.0, 2.0, 1.6], &[0, 1, 2, 3, 4, 2], 5);
        assert_eq!(
            uv1,
            [[2.0, 0.0], [2.0, 0.0], [1.6, 0.0], [0.0, 0.0], [0.0, 0.0]]
        );
        assert_eq!(by_tri, [2, 0]);
    }

    /// Hiding a feature rewrites index ranges, so a non-indexed FEATURE
    /// primitive gets U32 indices `0..n` where its tables are built: the same
    /// triangles, now addressable.
    #[test]
    fn non_indexed_feature_primitive_gets_sequential_u32_indices() {
        let s5 = prepare_tile_extracting(&feature_tile(false), false)
            .unwrap()
            .expect("prepared");
        let p = &s5.meshes.expect("extracted").primitives[0];
        assert_eq!(p.indices.as_deref(), Some(&[0, 1, 2, 3, 4, 5][..]));
        assert_eq!(p.feature_of_triangle.as_deref(), Some(&[0, 1][..]));
    }

    /// The worker ships each primitive's AABB so the consumer never re-walks
    /// the positions for it.
    #[test]
    fn bounds_match_mesh_min_max() {
        let s5 = prepare_tile_extracting(&feature_tile(true), false)
            .unwrap()
            .expect("prepared");
        let p = &s5.meshes.expect("extracted").primitives[0];
        assert_eq!(p.bounds, Some([[0.0, 0.0, -1.0], [3.0, 1.0, 0.0]]));
        assert_eq!(bounds_of(&[]), None);
        assert_eq!(
            bounds_of(&[[1.0, -2.0, 3.0]]),
            Some([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]])
        );
    }

    #[test]
    fn extracts_rtc_and_copyright_and_strips() {
        let json = serde_json::json!({
            "asset": { "version": "2.0", "copyright": "A;B" },
            "extensions": { "CESIUM_RTC": { "center": [1.0, 2.5, -3.0] } },
            "extensionsRequired": ["CESIUM_RTC"],
        });
        let glb = assemble_glb(&serde_json::to_vec(&json).unwrap(), &[]);
        let p = prepare_tile(&glb, false).unwrap().expect("accepted");
        assert_eq!(p.rtc_center, Some([1.0, 2.5, -3.0]));
        assert_eq!(p.copyright.as_deref(), Some("A;B"));
        let (j, _) = split_glb(&p.glb).unwrap();
        assert!(!memmem(j, b"CESIUM_RTC"), "handled extension stripped");
    }
}
