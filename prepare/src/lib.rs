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
//! What deliberately does NOT live here: anything needing a platform decoder
//! (Draco's JS shim, splat renderers) — [`prepare_tile`] returns `Ok(None)`
//! for those and the caller decodes inline — and anything producing bevy
//! types (`Mesh`/`Image` assembly, KTX2 transcode).

use std::collections::HashMap;

mod extract;
pub mod meshopt;

pub use extract::{ExtractedMaterial, ExtractedMeshes, ExtractedPrimitive, extract_tile_meshes};

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
    /// derives the per-triangle table from its own index buffer).
    pub vertex_ids: Vec<((u64, u64), Vec<f32>)>,
}

/// The output of [`prepare_tile`]: a plain (extension-free) glTF binary the
/// strict `gltf` crate accepts, plus the side-band data extracted on the way.
///
/// Geometry travels one of two ways, and `meshes` says which:
/// * `meshes: None` (S4) — `glb` holds the prepared container and the consumer
///   parses it with the `gltf` crate;
/// * `meshes: Some(_)` (S5, [`prepare_tile_extracting`]) — the geometry is
///   already typed buffers, the consumer never parses glTF at all, and `glb`
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
    /// `CESIUM_RTC` center or extracted planetary root offset (ECEF metres).
    pub rtc_center: Option<[f64; 3]>,
    /// glTF `asset.copyright` (attribution overlay side-band).
    pub copyright: Option<String>,
    /// Feature-picking data, when the tile carries `EXT_mesh_features`.
    pub features: Option<PreparedFeatures>,
}

/// Would [`prepare_tile`] hand these bytes straight back — either declined
/// (`Ok(None)`: content it has no decoder for) or echoed verbatim (a vanilla
/// non-georeferenced tile with nothing to rewrite)?
///
/// For callers that pay to ship the tile somewhere else (a Web Worker
/// transfer, an IPC hop): the answer costs one header read plus a marker scan
/// of the kilobyte JSON chunk, and saves a round trip that learns nothing. A
/// Google P3DT layer is Draco on every tile, so that is a per-tile saving for
/// a whole session. Both answers route the tile to the caller's own inline
/// decode, which is what `Ok(None)` does too.
///
/// Bytes that are not a container at all answer `false` — let [`prepare_tile`]
/// produce the error rather than mirroring its parsing here.
pub fn prepare_would_decline(bytes: &[u8], georeferenced: bool) -> bool {
    let Ok((json_chunk, _)) = split_glb(bytes) else {
        return false;
    };
    declines(&Marks::scan(json_chunk), georeferenced)
}

/// [`prepare_would_decline`] for [`prepare_tile_extracting`], relaxed by
/// exactly one case: a vanilla tile — nothing to rewrite, so nothing for
/// `prepare_tile` to do — IS worth the trip under S5, because the parse +
/// attribute collect its geometry extraction saves is the cost S5 exists to
/// move.
///
/// The relaxation is conditional on the tile being extractable at all: a
/// vanilla tile [`extract_tile_meshes`] will decline anyway would pay a full
/// round trip (two worker-side copies of a multi-MB GLB) to get its own bytes
/// back and decode inline — strictly worse than S4. The two decline reasons a
/// marker scan can see, an image/texture and a surviving `extensionsRequired`
/// (`KHR_mesh_quantization` and anything else no pass here handles), keep
/// declining here.
///
/// Ceiling: the decline reasons that live in *values* rather than keys — a
/// non-TRIANGLES `mode`, a sparse or non-`FLOAT` attribute, a VEC3 `COLOR_0` —
/// have no marker to scan for (`mode` is a number, the rest are accessor
/// properties), so those tiles still pay one wasted round trip each. Catching
/// them means parsing the JSON twice, which costs more than the trip saves.
pub fn extract_would_decline(bytes: &[u8], georeferenced: bool) -> bool {
    let Ok((json_chunk, _)) = split_glb(bytes) else {
        return false;
    };
    let marks = Marks::scan(json_chunk);
    // Marker scan, not a parse: `"images":[]` reads as textured and keeps the
    // S4 answer, which is the safe direction (one skipped extraction, never a
    // wasted trip).
    let unextractable = memmem(json_chunk, b"\"images\"")
        || memmem(json_chunk, b"\"textures\"")
        || memmem(json_chunk, b"\"extensionsRequired\"");
    declines(&marks, georeferenced) && (unextractable || marks.draco || marks.splat)
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
    prepare_tile_inner(bytes, georeferenced, false)
}

/// [`prepare_tile`] plus geometry extraction (offthread-decode plan S5): the
/// same single JSON parse also yields [`ExtractedMeshes`], so the consumer
/// builds meshes straight from typed buffers and skips the `gltf` parse and
/// the per-primitive attribute collect entirely (the dominant remaining
/// main-thread streaming cost — 6-9 ms/tile on bevy 0.19).
///
/// Same three outcomes as [`prepare_tile`], plus one shade: extraction is
/// best-effort. Content it cannot reproduce byte-identically (textures,
/// non-triangle primitives, quantized attributes — see [`extract_tile_meshes`])
/// comes back as an ordinary S4 [`PreparedTile`] with `meshes: None`, which
/// the consumer decodes exactly as before.
pub fn prepare_tile_extracting(
    bytes: &[u8],
    georeferenced: bool,
) -> Result<Option<PreparedTile>, DecodeError> {
    prepare_tile_inner(bytes, georeferenced, true)
}

fn prepare_tile_inner(
    bytes: &[u8],
    georeferenced: bool,
    extract: bool,
) -> Result<Option<PreparedTile>, DecodeError> {
    let (json_chunk, bin) = split_glb(bytes)?;
    let marks = Marks::scan(json_chunk);
    if marks.draco || marks.splat {
        return Ok(None); // needs a platform decoder this crate has not got
    }
    // Vanilla and not georeferenced: no rewrite, nothing to extract from the
    // JSON side-band. Without S5 there is no reason to parse it at all; WITH
    // S5 the geometry is the whole point of the trip, so it falls through.
    if !extract && declines(&marks, georeferenced) {
        return Ok(Some(PreparedTile {
            glb: bytes.to_vec(),
            meshes: None,
            rtc_center: None,
            copyright: None,
            features: None,
        }));
    }

    let mut json: serde_json::Value =
        serde_json::from_slice(json_chunk).map_err(|e| format!("tile json: {e}"))?;
    let copyright = json["asset"]["copyright"].as_str().map(str::to_string);
    let mut rtc_center = json["extensions"]["CESIUM_RTC"]["center"]
        .as_array()
        .and_then(|c| {
            let v: Vec<f64> = c.iter().filter_map(|x| x.as_f64()).collect();
            <[f64; 3]>::try_from(v).ok()
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
    let new_bin: Option<Vec<u8>> = if marks.meshopt {
        Some(decode_meshopt_views(&mut json, bin).map_err(DecodeError::meshopt)?)
    } else {
        None
    };
    if marks.basisu {
        preprocess_basisu(&mut json);
    }
    // Runs on the MARKER, like inline (`marks.draco` is impossible here —
    // declined above).
    let stripped = marks.rtc;
    if stripped {
        strip_handled_extensions(&mut json);
    }
    let bin = new_bin.as_deref().or(bin);

    // S5: geometry off the document we already hold. Runs BEFORE the feature
    // pass (which consumes `json`) and before the container rebuild — when it
    // succeeds there is no container to rebuild, because nobody will parse one.
    let meshes = if extract {
        extract_tile_meshes(&json, bin)?
    } else {
        None
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
    let features = if marks.features {
        match FeatureCtx::build(json, bin) {
            Ok(ctx) => Some(ctx.materialize(bin)?),
            Err(_) => None,
        }
    } else {
        None
    };

    Ok(Some(PreparedTile {
        glb,
        meshes,
        rtc_center,
        copyright,
        features,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The S5 triage relaxes `prepare_would_decline` for exactly one shape —
    /// an untextured vanilla tile, whose geometry extraction is the whole
    /// point. Everything `prepare_would_decline` rejects for having no work to
    /// do AND no extractable geometry must still be rejected, or the host pays
    /// a full round trip (two multi-MB copies) to get its own bytes back.
    #[test]
    fn extract_would_decline_still_rejects_textured_vanilla_tiles() {
        // The relaxation: nothing to rewrite, nothing textured — dispatch it.
        let plain = br#"{"asset":{"version":"2.0"},"meshes":[]}"#;
        assert!(prepare_would_decline(plain, false));
        assert!(!extract_would_decline(plain, false), "S5 dispatches this");

        for json in [
            // A plain PNG/JPEG-textured tile: no basisu, no meshopt, no RTC —
            // `prepare_tile` echoes the bytes and extraction declines on the
            // images, so the trip learns nothing.
            r#"{"asset":{"version":"2.0"},"images":[{"mimeType":"image/png"}]}"#,
            r#"{"asset":{"version":"2.0"},"textures":[{"source":0}]}"#,
            // And the pre-existing no-decoder cases stay declined.
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
        // Textured but with real prep waiting (basisu) is still dispatched —
        // that is S4 work, unchanged.
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
