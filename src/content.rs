//! Tile content decode: GLB bytes → renderable data for the three content
//! types (plan D5) — triangle meshes (T0/T1), point clouds (T2), Gaussian
//! splats (T3). One decoder, three outputs, feeding the existing renderers
//! (`Mesh3d`, vendored `PointCloud`, `PlanarGaussian3d`).
//!
//! Mesh + point tiles decode via the `gltf` crate (no `import` feature — that
//! would pull the `image` crate; embedded textures decode through Bevy's
//! `Image::from_buffer` with the png/jpeg features the GLB twin pipeline
//! already enables). Splat tiles can NOT go through the `gltf` crate: the
//! `KHR_gaussian_splatting` extension (RC) names its vertex attributes
//! `KHR_gaussian_splatting:ROTATION` etc. — not `_`-prefixed — which
//! `gltf-json` rejects as invalid semantics at validation. Splat tiles get a
//! minimal raw JSON+BIN decoder instead ([`decode_splat_gltf`]); our tiler
//! (D3/D4) emits float accessors and single-node scenes, and the decoder
//! checks enough structure to fail cleanly on anything else.
//!
//! Everything runs inside the loader task, off the frame loop: outputs are
//! plain `Send` data the ECS drain turns into entities.
//!
//! Tile GLBs are self-contained by construction (D1/D3: our tilers emit
//! GLB-with-BIN-chunk). External buffer/image URIs are rejected with a clear
//! error rather than fetched — a tile that needs side files defeats the
//! one-blob range-read design.

// With neither `points` nor `splats`, `DecodedItem` collapses to its single
// `Mesh` variant, making the `let DecodedItem::Mesh(_) = … else …` filters
// (the texture-resolve pass + the mesh-extraction tests) irrefutable. Allow it
// only in that degenerate config; the lint stays active in the normal
// multi-variant build so a genuinely irrefutable `let…else` is still caught.
#![cfg_attr(
    not(any(feature = "points", feature = "splats")),
    allow(irrefutable_let_patterns)
)]

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bevy::asset::RenderAssetUsages;
use bevy::image::{
    CompressedImageFormats, Image, ImageAddressMode, ImageFilterMode, ImageSampler,
    ImageSamplerDescriptor, ImageType,
};
use bevy::math::{DVec3, Mat4};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
#[cfg(feature = "splats")]
use bevy_gaussian_splatting::gaussian::formats::planar_3d::Gaussian3d;
#[cfg(feature = "points")]
use bevy_pointcloud::point_cloud::PointCloudData;

use super::draco;
use super::meshopt;

/// Adapter-supported GPU-compressed texture formats (BC on desktop WebGPU,
/// ASTC/ETC on mobile, NONE headless/native-before-init). Latched ONCE at
/// startup ([`set_supported_compressed_formats`]) — the adapter never changes,
/// and the MSAA lesson is latch-don't-toggle. Read by KTX2 transcode in
/// [`decode_material`]: UASTC transcodes to a member format (BC7…) or, when the
/// set is empty, to uncompressed RGBA8 — so KTX2 tiles render everywhere (T7).
static SUPPORTED_FORMATS: OnceLock<CompressedImageFormats> = OnceLock::new();

/// Latch the adapter's supported compressed formats — call once at startup from
/// the `CompressedImageFormatSupport` resource. Idempotent.
pub fn set_supported_compressed_formats(formats: CompressedImageFormats) {
    let _ = SUPPORTED_FORMATS.set(formats);
}

fn supported_formats() -> CompressedImageFormats {
    SUPPORTED_FORMATS
        .get()
        .copied()
        .unwrap_or(CompressedImageFormats::NONE)
}

/// Typed failure surface of tile-content decoding — the error of
/// [`decode_tile`] / [`decode_glb`] and the draco/ktx2 shim modules.
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

    pub(crate) fn draco(message: impl Into<String>) -> Self {
        Self::new(DecodeStage::Draco, message)
    }

    pub(crate) fn ktx2(message: impl Into<String>) -> Self {
        Self::new(DecodeStage::Ktx2, message)
    }

    pub(crate) fn meshopt(message: impl Into<String>) -> Self {
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

/// Resolve deferred KTX2 base-color textures (T7): transcode each pending
/// `image/ktx2` payload to a GPU `Image`. Async because the transcoder is a JS
/// shim on wasm; on native it's bevy's basis transcoder. A failed transcode
/// degrades cleanly to the base-color factor (untextured) — never fatal.
async fn resolve_pending_textures(items: &mut [DecodedItem]) {
    for item in items.iter_mut() {
        let DecodedItem::Mesh(p) = item else { continue };
        let Some(bytes) = p.material.base_color_ktx2.take() else {
            continue;
        };
        match transcode_ktx2(&bytes).await {
            Ok(mut img) => {
                // Stamp the glTF wrap/filter sampler onto the transcoded image
                // (the transcoders return `ImageSampler::Default` = ClampToEdge).
                img.sampler = ImageSampler::Descriptor(p.material.base_color_sampler.clone());
                p.material.base_color_image = Some(img);
            }
            Err(e) => warn_ktx2_once(&e.to_string()),
        }
    }
}

/// wasm: transcode via the `__tt_ktx2_transcode` shim (KTX-Software libktx),
/// targeting BC7 when the adapter supports it, else RGBA8.
#[cfg(target_arch = "wasm32")]
async fn transcode_ktx2(bytes: &[u8]) -> Result<Image, DecodeError> {
    let want_bc = supported_formats().contains(CompressedImageFormats::BC);
    super::ktx2::transcode(bytes, want_bc).await
}

/// native: bevy's `basis-universal` feature (C++; builds off-wasm). Every real
/// native adapter has a block format (llvmpipe included); bevy 0.18's UASTC →
/// uncompressed-RGBA path is broken, so require one rather than hit it.
#[cfg(not(target_arch = "wasm32"))]
async fn transcode_ktx2(bytes: &[u8]) -> Result<Image, DecodeError> {
    if supported_formats() == CompressedImageFormats::NONE {
        return Err(DecodeError::ktx2("no GPU block format for KTX2 transcode"));
    }
    Image::from_buffer(
        bytes,
        ImageType::MimeType("image/ktx2"),
        supported_formats(),
        true, // base color is sRGB
        ImageSampler::Default,
        RenderAssetUsages::RENDER_WORLD,
    )
    .map_err(|e| DecodeError::ktx2(format!("ktx2 native decode: {e}")))
}

/// One-time warning when a KTX2 tile texture can't be transcoded; per-tile spam
/// would bury it, and the geometry still renders (untextured).
fn warn_ktx2_once(detail: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let detail = detail.to_string();
    ONCE.call_once(move || {
        bevy::log::warn!(
            "tiles3d: KTX2 tile texture transcode failed ({detail}); rendering untextured"
        );
    });
}

/// Per-feature picking data for one mesh primitive (T8): `EXT_mesh_features`
/// (`_FEATURE_ID_0`) + the tile's `EXT_structural_metadata` property table.
pub struct TileFeatures {
    /// featureId per triangle, in the spawned mesh's index-buffer order — so
    /// the pick raycast's triangle ordinal indexes straight into it.
    pub feature_of_triangle: Vec<u32>,
    /// featureId per VERTEX (raw `_FEATURE_ID_0` values, length == the
    /// primitive's vertex count). The decode also writes these onto the mesh
    /// as `ATTRIBUTE_UV_1` (`[fid, 0]`), so a host material can style
    /// per-feature in the fragment shader (the Cesium
    /// `Cesium3DTileFeature.color` model) through the standard pipeline's
    /// `VERTEX_UVS_B` path — no custom vertex shader. Feature tiles never
    /// carry a real `TEXCOORD_1` (it was already dropped before 0.1.7), so
    /// nothing is displaced.
    pub feature_of_vertex: Vec<f32>,
    /// Shared per-tile table: featureId → source-node path (the `/`-joined node
    /// names the sections resolver matches `mesh_section` against). `Arc` so
    /// every primitive of one tile shares one decode.
    pub node_of_feature: Arc<Vec<String>>,
}

/// One decoded glTF primitive, positioned by its node's global transform
/// (glTF Y-up frame — the spawned tile entity applies
/// [`super::traversal::TileNode::world_from_content`] above it).
pub struct DecodedPrimitive {
    pub transform: Mat4,
    pub mesh: Mesh,
    pub material: DecodedMaterial,
    /// Feature metadata when the tile carries `EXT_mesh_features` (T8); `None`
    /// for plain/scenery tiles. Drives feature → node → twin picking.
    pub features: Option<TileFeatures>,
}

/// One decoded piece of tile content. A tile may carry several (multiple
/// primitives / nodes); the spawn step turns each into a child entity.
pub enum DecodedItem {
    Mesh(Box<DecodedPrimitive>),
    /// `POINTS`-mode primitive (positions + COLOR_0) → vendored point renderer.
    #[cfg(feature = "points")]
    Points {
        transform: Mat4,
        points: Vec<PointCloudData>,
    },
    /// `KHR_gaussian_splatting` primitive → `PlanarGaussian3d` renderer.
    /// Gaussians are in the primitive's local (glTF Y-up) frame; padded to a
    /// multiple of 32 like the crate's own ply path.
    #[cfg(feature = "splats")]
    Splat {
        transform: Mat4,
        gaussians: Vec<Gaussian3d>,
    },
}

/// Material inputs resolved at decode time; turned into a `StandardMaterial`
/// at spawn (asset insertion needs ECS access).
pub struct DecodedMaterial {
    /// Linear RGBA base color factor.
    pub base_color: [f32; 4],
    pub base_color_image: Option<Image>,
    /// Raw `image/ktx2` (KHR_texture_basisu) base-color bytes awaiting transcode
    /// in the async resolve pass (T7) — the transcoder is a JS shim on wasm /
    /// bevy basis on native, neither callable from the sync decode. Mutually
    /// exclusive with `base_color_image`.
    pub base_color_ktx2: Option<Vec<u8>>,
    /// Wrap/filter sampler for the base-color texture, read from the glTF
    /// sampler (defaulting to REPEAT per the glTF spec). Carried separately so
    /// the deferred KTX2 transcode can stamp it onto its `Image` too.
    pub base_color_sampler: ImageSamplerDescriptor,
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
    /// `KHR_materials_unlit` — photogrammetry/satellite content ships baked
    /// lighting (Google P3DT requires this extension); re-lighting it dims.
    pub unlit: bool,
}

impl Default for DecodedMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            base_color_image: None,
            base_color_ktx2: None,
            base_color_sampler: ImageSamplerDescriptor::default(),
            metallic: 0.0,
            roughness: 1.0,
            double_sided: false,
            unlit: false,
        }
    }
}

/// A fully decoded tile: renderable items plus the side-band data T4 needs.
pub struct DecodedTile {
    pub items: Vec<DecodedItem>,
    /// Raw content (GLB) byte length — the memory-pressure proxy the traversal
    /// sums over resident tiles (decoded CPU+GPU cost is ~2-4x this).
    pub content_bytes: u64,
    /// `CESIUM_RTC` center (ECEF metres, Google P3DT). Composed into the
    /// tile's placement **in f64 at spawn** — never baked into f32 vertex
    /// data or a f32 transform (planetary magnitudes only cancel in f64).
    pub rtc_center: Option<DVec3>,
    /// glTF `asset.copyright` — aggregated into the attribution overlay
    /// (required by the Google ToS, plan D7/L-D5).
    pub copyright: Option<String>,
}

/// Decode a tile, routing by content markers: splats bypass the `gltf` crate
/// (see module docs), Draco/`CESIUM_RTC` content is rewritten to vanilla glTF
/// first ([`preprocess_glb`] — the `gltf` crate rejects unknown
/// `extensionsRequired`), everything else takes the plain [`decode_glb`]
/// path. Async only for the Draco decoder round-trip; plain tiles never
/// yield. `georeferenced` forces the JSON scan — ECEF-tree content can carry
/// planetary node transforms with no marker string to cheaply detect.
pub async fn decode_tile(bytes: &[u8], georeferenced: bool) -> Result<DecodedTile, DecodeError> {
    let (json, bin) = split_glb(bytes)?;
    let has_splat = memmem(json, b"KHR_gaussian_splatting");
    let has_draco = memmem(json, b"KHR_draco_mesh_compression");
    let has_rtc = memmem(json, b"CESIUM_RTC");
    if !(georeferenced || has_splat || has_draco || has_rtc || memmem(json, b"copyright")) {
        let mut items = decode_glb(bytes)?;
        resolve_pending_textures(&mut items).await;
        return Ok(DecodedTile {
            items,
            content_bytes: bytes.len() as u64,
            rtc_center: None,
            copyright: None,
        });
    }

    let mut value: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| format!("tile json: {e}"))?;
    let copyright = value["asset"]["copyright"].as_str().map(str::to_string);
    let mut rtc_center = value["extensions"]["CESIUM_RTC"]["center"]
        .as_array()
        .and_then(|c| {
            let v: Vec<f64> = c.iter().filter_map(|x| x.as_f64()).collect();
            <[f64; 3]>::try_from(v).ok().map(DVec3::from_array)
        });

    #[cfg(feature = "splats")]
    if has_splat {
        let items = decode_splat_gltf(&value, bin)?;
        return Ok(DecodedTile {
            items,
            content_bytes: bytes.len() as u64,
            rtc_center,
            copyright,
        });
    }

    // Google P3DT bakes ECEF positions into node MATRICES instead of
    // CESIUM_RTC — planetary magnitudes that the gltf crate would truncate
    // to f32. Extract the offset in f64 from the raw JSON and route it
    // through the same side-band channel.
    let mut nodes_rebased = false;
    if rtc_center.is_none()
        && let Some(center) = extract_planetary_root_offset(&mut value)
    {
        rtc_center = Some(center);
        nodes_rebased = true;
    }

    let mut items = if has_draco || has_rtc || nodes_rebased {
        let vanilla = preprocess_glb(&mut value, bin).await?;
        decode_glb(&vanilla)?
    } else {
        decode_glb(bytes)?
    };
    resolve_pending_textures(&mut items).await;
    Ok(DecodedTile {
        items,
        content_bytes: bytes.len() as u64,
        rtc_center,
        copyright,
    })
}

/// When any scene-root node sits at planetary magnitude (Google P3DT bakes
/// ECEF into node matrices), pick the first such translation as the tile's
/// offset and subtract it from EVERY root node **in f64**, so the f32 glTF
/// decode only ever sees tile-local values. Returns the extracted offset.
/// The spawn transform re-applies it: `world_from_content × T(offset) ×
/// node'` ≡ `world_from_content × node` exactly.
fn extract_planetary_root_offset(json: &mut serde_json::Value) -> Option<DVec3> {
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
        (DVec3::from_array(t).length() > PLANETARY_M).then_some(DVec3::from_array(t))
    })?;

    // Subtract from EVERY root (small-translation roots become -center —
    // mixing untouched and rebased roots under the re-applied offset would
    // shift the untouched ones).
    for &ix in &roots {
        let t = translation_of(&json["nodes"][ix]);
        let new = [t[0] - center.x, t[1] - center.y, t[2] - center.z];
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

/// Decode a GLB (or self-contained glTF JSON) tile into renderable items.
pub fn decode_glb(bytes: &[u8]) -> Result<Vec<DecodedItem>, DecodeError> {
    let (json, bin) = split_glb(bytes)?;

    // EXT_meshopt_compression (T6/D12 — what our mesh tiler emits, and POINTS
    // tiles too): decode the compressed buffer views on the CPU into a vanilla
    // GLB, then re-enter. Synchronous (no JS shim, unlike Draco), so it lives
    // here rather than in the async `decode_tile` preprocess — both the fast
    // path and direct callers get it. The rebuilt GLB carries no marker, so
    // the recursion routes to the splat/plain path below without looping.
    if memmem(json, b"EXT_meshopt_compression") {
        let mut value: serde_json::Value =
            serde_json::from_slice(json).map_err(|e| format!("meshopt tile json: {e}"))?;
        let vanilla = preprocess_meshopt(&mut value, bin).map_err(DecodeError::meshopt)?;
        return decode_glb(&vanilla);
    }

    // KTX2/Basis textures (T7): the gltf crate (1.4) doesn't resolve
    // KHR_texture_basisu — the KTX2 image hangs off the texture *extension*,
    // not the standard `source`. Rewrite the source onto the standard slot and
    // strip the ext (JSON-only; the KTX2 bytes stay put), then re-enter so the
    // gltf path finds the image and `decode_material` hands the `image/ktx2`
    // bytes to bevy's transcoder.
    if memmem(json, b"KHR_texture_basisu") {
        let mut value: serde_json::Value =
            serde_json::from_slice(json).map_err(|e| format!("basisu tile json: {e}"))?;
        preprocess_basisu(&mut value);
        let json_bytes =
            serde_json::to_vec(&value).map_err(|e| format!("basisu splice json: {e}"))?;
        let glb = assemble_glb(&json_bytes, bin.unwrap_or(&[]));
        return decode_glb(&glb);
    }

    // Splat tiles bypass the gltf crate entirely (see module docs). The
    // marker check is a cheap substring scan of the JSON chunk. Without the
    // `splats` feature a splat tile falls through to the gltf path below, which
    // rejects the unknown required extension — that content simply doesn't show.
    #[cfg(feature = "splats")]
    if memmem(json, b"KHR_gaussian_splatting") {
        let value: serde_json::Value =
            serde_json::from_slice(json).map_err(|e| format!("splat tile json: {e}"))?;
        return decode_splat_gltf(&value, bin).map_err(DecodeError::from);
    }

    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| {
        // Diagnostic: a parse failure here means the bytes reaching the gltf
        // crate aren't the clean vanilla glTF we expect (bad archive range-read,
        // meshopt rebuild, or a stray required extension). Surface the JSON
        // head/tail + length so the cause is visible in the log without a
        // round-trip — the raw tile is usually structurally valid.
        let (j, _) = split_glb(bytes).unwrap_or((bytes, None));
        let head = String::from_utf8_lossy(&j[..j.len().min(180)]);
        let tail = String::from_utf8_lossy(&j[j.len().saturating_sub(180)..]);
        format!(
            "gltf parse: {e} | json_len={} head={head:?} tail={tail:?}",
            j.len()
        )
    })?;
    let doc = gltf.document;
    let blob = gltf.blob;

    // Feature metadata (T8): EXT_mesh_features + EXT_structural_metadata. The
    // gltf crate models neither, so decode them from the raw JSON side-channel
    // (like the splat/draco paths). Built once per tile; attached per primitive.
    // By the time we reach here the GLB is vanilla (meshopt/basisu already
    // preprocessed above), so the property-table + `_FEATURE_ID_0` accessors
    // read plain bytes.
    let feat = if memmem(json, b"EXT_mesh_features") {
        match FeatureCtx::build(json, bin) {
            Ok(ctx) => Some(ctx),
            // A malformed table loses picking, never the geometry.
            Err(e) => {
                bevy::log::warn!("tiles3d: feature metadata ignored ({e})");
                None
            }
        }
    } else {
        None
    };

    let mut out = Vec::new();
    let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) else {
        return Ok(out); // empty content tile — legal, renders nothing
    };
    for node in scene.nodes() {
        decode_node(
            &node,
            Mat4::IDENTITY,
            blob.as_deref(),
            feat.as_ref(),
            &mut out,
        )?;
    }
    Ok(out)
}

/// Decoded `EXT_mesh_features` + `EXT_structural_metadata` context for a tile
/// (T8). Owns the parsed JSON so `_FEATURE_ID_0` accessors can be read lazily
/// per primitive against the BIN chunk.
struct FeatureCtx {
    json: serde_json::Value,
    /// featureId → source-node path, shared across the tile's primitives.
    node_of_feature: Arc<Vec<String>>,
    /// (mesh index, primitive index) → `_FEATURE_ID_N` accessor index.
    accessor: HashMap<(u64, u64), usize>,
}

impl FeatureCtx {
    fn build(json: &[u8], bin: Option<&[u8]>) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_slice(json).map_err(|e| format!("feature json: {e}"))?;
        let node_of_feature = Arc::new(read_node_of_feature(&value, bin)?);
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

    /// `feature_of_triangle` for primitive `(mesh_ix, prim_ix)` in `indices`
    /// order (matching the spawned mesh + pick raycast), or `None` when this
    /// primitive carries no feature ids.
    fn for_primitive(
        &self,
        bin: Option<&[u8]>,
        mesh_ix: u64,
        prim_ix: u64,
        indices: Option<&[u32]>,
        vertex_count: usize,
    ) -> Result<Option<TileFeatures>, String> {
        let Some(&acc) = self.accessor.get(&(mesh_ix, prim_ix)) else {
            return Ok(None);
        };
        let per_vertex = read_accessor::<1>(&self.json, bin, acc)?;
        let feature_of = |v: usize| per_vertex.get(v).map(|f| f[0].round() as u32).unwrap_or(0);
        let feature_of_triangle = match indices {
            Some(idx) => idx
                .chunks_exact(3)
                .map(|t| feature_of(t[0] as usize))
                .collect(),
            // Non-indexed: triangle t spans vertices 3t..3t+3.
            None => (0..vertex_count / 3).map(|t| feature_of(t * 3)).collect(),
        };
        // Exactly vertex_count entries (pad with feature 0) — a mesh attribute
        // must match the position count or bevy rejects the mesh.
        let feature_of_vertex = (0..vertex_count)
            .map(|v| per_vertex.get(v).map(|f| f[0]).unwrap_or(0.0))
            .collect();
        Ok(Some(TileFeatures {
            feature_of_triangle,
            feature_of_vertex,
            node_of_feature: self.node_of_feature.clone(),
        }))
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

/// Split a GLB container into its JSON chunk and optional BIN chunk. Bytes
/// without the `glTF` magic are treated as a bare JSON glTF (no buffer).
fn split_glb(bytes: &[u8]) -> Result<(&[u8], Option<&[u8]>), String> {
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
pub(crate) fn memmem(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

// ── Draco / CESIUM_RTC preprocessing (T4 — Google P3DT content) ──────────────

/// One `KHR_draco_mesh_compression` primitive found in the document.
struct DracoPrim {
    mesh: usize,
    prim: usize,
    buffer_view: usize,
    /// glTF semantic → Draco attribute unique id, straight from the ext JSON.
    attributes: Vec<(String, u32)>,
}

fn find_draco_prims(json: &serde_json::Value) -> Vec<DracoPrim> {
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

fn buffer_view_slice<'b>(
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

/// Decode every Draco primitive (through the platform decoder) and rewrite
/// the document into a vanilla self-contained GLB: decoded data appended to
/// the BIN chunk behind fresh accessors, the Draco extension and `CESIUM_RTC`
/// stripped (the `gltf` crate hard-rejects unknown `extensionsRequired`; the
/// RTC center was already extracted by the caller).
async fn preprocess_glb(
    json: &mut serde_json::Value,
    bin: Option<&[u8]>,
) -> Result<Vec<u8>, DecodeError> {
    let prims = find_draco_prims(json);
    let mut decoded = Vec::with_capacity(prims.len());
    for prim in &prims {
        let compressed = buffer_view_slice(json, bin, prim.buffer_view)?;
        let ids: Vec<u32> = prim.attributes.iter().map(|(_, id)| *id).collect();
        decoded.push(draco::decode(compressed, &ids).await?);
    }
    Ok(splice_glb(json, bin, &prims, decoded)?)
}

/// The synchronous splice half of [`preprocess_glb`] (testable without a
/// real Draco decoder).
fn splice_glb(
    json: &mut serde_json::Value,
    bin: Option<&[u8]>,
    prims: &[DracoPrim],
    decoded: Vec<draco::DracoMesh>,
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

    // Strip the handled extensions; the RTC center is side-band data now.
    // NOTE: use get_mut, never `json[key]` — IndexMut on a missing key
    // INSERTS a literal null, which the gltf crate then chokes on.
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
    if json["buffers"][0].is_object() {
        json["buffers"][0]["byteLength"] = serde_json::json!(new_bin.len());
    } else if !new_bin.is_empty() {
        json["buffers"] = serde_json::json!([{ "byteLength": new_bin.len() }]);
    }

    let json_bytes = serde_json::to_vec(&json).map_err(|e| format!("splice json: {e}"))?;
    Ok(assemble_glb(&json_bytes, &new_bin))
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

/// Assemble a GLB container from JSON + BIN chunks (4-byte padded).
fn assemble_glb(json_bytes: &[u8], bin: &[u8]) -> Vec<u8> {
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

// ── EXT_meshopt_compression preprocessing (T6 — our emitted geometry) ────────

/// Rewrite an `EXT_meshopt_compression` GLB into a vanilla self-contained GLB:
/// decode every meshopt buffer view on the CPU ([`meshopt::decode_buffer_view`]),
/// copy through non-meshopt views (embedded image bytes), collapse to a single
/// buffer (the fallback buffer is virtual — no GLB bytes), and strip the
/// extension. The strict `gltf` crate path then decodes it unchanged.
///
/// Buffer-view *indices* are preserved (accessors and images keep referencing
/// the same slots); only each view's `byteOffset`/`byteLength`/`buffer` are
/// rebuilt against the freshly decoded BIN. The encoder stores compressed data
/// in the GLB BIN (`ext.buffer == 0`) while the view's own `buffer` points at
/// the discarded fallback — so we always read compressed bytes via `ext`.
fn preprocess_meshopt(json: &mut serde_json::Value, bin: Option<&[u8]>) -> Result<Vec<u8>, String> {
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

    let json_bytes = serde_json::to_vec(&json).map_err(|e| format!("meshopt splice json: {e}"))?;
    Ok(assemble_glb(&json_bytes, &new_bin))
}

// ── KHR_texture_basisu preprocessing (T7 — KTX2 tile textures) ───────────────

/// Rewrite `KHR_texture_basisu` textures so the `gltf` crate (which doesn't
/// resolve the extension) finds the KTX2 image: move each texture's
/// `extensions.KHR_texture_basisu.source` to the standard `source`, then strip
/// the extension everywhere. JSON-only — the KTX2 image bytes (mimeType
/// `image/ktx2`, in a buffer view) are untouched; [`decode_material`] passes
/// them to bevy's KTX2/Basis transcoder with the adapter's supported formats.
fn preprocess_basisu(json: &mut serde_json::Value) {
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

/// Resolve a glTF buffer: GLB BIN chunk only (tiles are self-contained).
fn resolve_buffer<'b>(buffer: &gltf::Buffer<'_>, blob: Option<&'b [u8]>) -> Option<&'b [u8]> {
    match buffer.source() {
        gltf::buffer::Source::Bin => blob,
        gltf::buffer::Source::Uri(_) => None,
    }
}

fn decode_node(
    node: &gltf::Node<'_>,
    parent: Mat4,
    blob: Option<&[u8]>,
    feat: Option<&FeatureCtx>,
    out: &mut Vec<DecodedItem>,
) -> Result<(), String> {
    let global = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        let mesh_ix = mesh.index() as u64;
        for primitive in mesh.primitives() {
            match primitive.mode() {
                gltf::mesh::Mode::Triangles => {
                    out.push(DecodedItem::Mesh(Box::new(decode_primitive(
                        &primitive, global, blob, feat, mesh_ix,
                    )?)));
                }
                #[cfg(feature = "points")]
                gltf::mesh::Mode::Points => {
                    out.push(decode_points(&primitive, global, blob)?);
                }
                // Lines/strips/fans (and POINTS without the `points` feature):
                // nothing renders them — skip quietly so a mixed-content tile
                // still shows what it can.
                _ => continue,
            }
        }
    }
    for child in node.children() {
        decode_node(&child, global, blob, feat, out)?;
    }
    Ok(())
}

fn decode_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: Mat4,
    blob: Option<&[u8]>,
    feat: Option<&FeatureCtx>,
    mesh_ix: u64,
) -> Result<DecodedPrimitive, String> {
    let reader = primitive.reader(|buffer| resolve_buffer(&buffer, blob));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("primitive has no POSITION (or buffer is an external URI)")?
        .collect();
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
    let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|tc| tc.into_f32().collect());
    let colors: Option<Vec<[f32; 4]>> = reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
    let indices: Option<Vec<u32>> = reader.read_indices().map(|ix| ix.into_u32().collect());

    // T8: per-feature picking — derive feature_of_triangle from `_FEATURE_ID_0`
    // (raw JSON) in the SAME index order as the mesh below.
    let features = match feat {
        Some(ctx) => ctx.for_primitive(
            blob,
            mesh_ix,
            primitive.index() as u64,
            indices.as_deref(),
            positions.len(),
        )?,
        None => None,
    };

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        // MAIN_WORLD + RENDER_WORLD: the camera-focus/selection raycasts read
        // mesh vertices on the main world (the basemap panic lesson). The CPU
        // copy is a T2 memory-budget follow-up, not a T0 risk.
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    if let Some(uvs) = uvs {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    // Feature ids ride UV1 so a host material can tint per feature in the
    // fragment stage (see `TileFeatures::feature_of_vertex`).
    if let Some(f) = &features {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_1,
            f.feature_of_vertex
                .iter()
                .map(|&id| [id, 0.0])
                .collect::<Vec<[f32; 2]>>(),
        );
    }
    if let Some(colors) = colors {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    }
    if let Some(indices) = indices {
        mesh.insert_indices(Indices::U32(indices));
    }
    match normals {
        Some(n) => mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n),
        // Tiler output may omit normals to save bytes; smooth-compute them
        // (decode-task CPU, not frame time).
        None => mesh.compute_normals(),
    }

    let material = decode_material(&primitive.material(), blob)?;
    Ok(DecodedPrimitive {
        transform,
        mesh,
        material,
        features,
    })
}

/// `POINTS`-mode primitive → point-renderer data. Positions stay in the glTF
/// Y-up content frame (the tile entity transform places them); COLOR_0 when
/// present, white otherwise. `point_size: -1.0` = the shared material's
/// screen-space size, matching the whole-file LAZ loader.
#[cfg(feature = "points")]
fn decode_points(
    primitive: &gltf::Primitive<'_>,
    transform: Mat4,
    blob: Option<&[u8]>,
) -> Result<DecodedItem, String> {
    let reader = primitive.reader(|buffer| resolve_buffer(&buffer, blob));
    let positions = reader
        .read_positions()
        .ok_or("points primitive has no POSITION (or buffer is an external URI)")?;
    let mut colors = reader.read_colors(0).map(|c| c.into_rgba_f32());
    let points: Vec<PointCloudData> = positions
        .map(|p| PointCloudData {
            position: bevy::math::Vec3::from(p),
            point_size: -1.0,
            color: colors
                .as_mut()
                .and_then(|c| c.next())
                .unwrap_or([1.0, 1.0, 1.0, 1.0]),
        })
        .collect();
    Ok(DecodedItem::Points { transform, points })
}

/// glTF `WrappingMode` → bevy `ImageAddressMode`.
fn gltf_address_mode(w: gltf::texture::WrappingMode) -> ImageAddressMode {
    use gltf::texture::WrappingMode;
    match w {
        WrappingMode::ClampToEdge => ImageAddressMode::ClampToEdge,
        WrappingMode::MirroredRepeat => ImageAddressMode::MirrorRepeat,
        WrappingMode::Repeat => ImageAddressMode::Repeat,
    }
}

/// Build a bevy sampler descriptor from a glTF texture's sampler. The `gltf`
/// crate returns the spec default (REPEAT, linear) for an unauthored sampler,
/// so this both honours authored wrap modes and revives tiling textures the old
/// `ImageSampler::Default` (ClampToEdge) silently flattened. Linear filtering;
/// mips are deferred crate-wide.
fn sampler_from_gltf(texture: &gltf::Texture<'_>) -> ImageSamplerDescriptor {
    let s = texture.sampler();
    ImageSamplerDescriptor {
        address_mode_u: gltf_address_mode(s.wrap_s()),
        address_mode_v: gltf_address_mode(s.wrap_t()),
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        // Trilinear + anisotropic over the baked mip pyramid: terrain is viewed
        // at grazing angles where isotropic mips alone still shimmer. Clamped to
        // the device max by wgpu; a no-op on single-mip fallback (png/jpeg).
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::default()
    }
}

fn decode_material(
    material: &gltf::Material<'_>,
    blob: Option<&[u8]>,
) -> Result<DecodedMaterial, String> {
    let pbr = material.pbr_metallic_roughness();
    let mut out = DecodedMaterial {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        double_sided: material.double_sided(),
        base_color_image: None,
        base_color_ktx2: None,
        base_color_sampler: ImageSamplerDescriptor::default(),
        unlit: material.unlit(),
    };
    if let Some(info) = pbr.base_color_texture() {
        let texture = info.texture();
        // Honour the glTF wrap mode. The loader used to hardcode
        // `ImageSampler::Default` (bevy's engine default is ClampToEdge), so a
        // material whose UVs run past [0,1] — tiling/atlas-wrapping textures —
        // sampled the EDGE texel everywhere and the surface read as a flat smear
        // ("the tiled texture disappears in the 3dtiles version"). The glTF spec
        // default wrap is REPEAT, which the `gltf` crate returns when no sampler
        // is authored, so this revives tiling textures and respects authored
        // CLAMP (e.g. the tiler's per-footprint crops).
        out.base_color_sampler = sampler_from_gltf(&texture);
        let image = texture.source();
        match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                let buf = resolve_buffer(&view.buffer(), blob)
                    .ok_or("texture bufferView points at an external buffer")?;
                let bytes = buf
                    .get(view.offset()..view.offset() + view.length())
                    .ok_or("texture bufferView out of bounds")?;
                if mime_type == "image/ktx2" {
                    // T7: defer the UASTC transcode to the async resolve pass
                    // (JS shim on wasm / bevy basis on native) — neither is
                    // callable from this sync decode. The sampler rides
                    // `base_color_sampler` and is stamped on after transcode.
                    out.base_color_ktx2 = Some(bytes.to_vec());
                } else {
                    let decoded = Image::from_buffer(
                        bytes,
                        ImageType::MimeType(mime_type),
                        CompressedImageFormats::NONE, // png/jpeg: format irrelevant
                        true,                         // base color is sRGB
                        ImageSampler::Descriptor(out.base_color_sampler.clone()),
                        // GPU-only: tile textures are never read back on the CPU.
                        RenderAssetUsages::RENDER_WORLD,
                    )
                    .map_err(|e| format!("texture decode ({mime_type}): {e}"))?;
                    out.base_color_image = Some(decoded);
                }
            }
            gltf::image::Source::Uri { .. } => {
                return Err("external texture URIs unsupported in tile content".into());
            }
        }
    }
    Ok(out)
}

// ── Raw KHR_gaussian_splatting decode ────────────────────────────────────────

/// Spec attribute names (KHR_gaussian_splatting RC).
#[cfg(feature = "splats")]
const ATTR_ROTATION: &str = "KHR_gaussian_splatting:ROTATION";
#[cfg(feature = "splats")]
const ATTR_SCALE: &str = "KHR_gaussian_splatting:SCALE";
#[cfg(feature = "splats")]
const ATTR_OPACITY: &str = "KHR_gaussian_splatting:OPACITY";
#[cfg(feature = "splats")]
const ATTR_SH0: &str = "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0";

/// `SH_0` basis constant: `color = 0.5 + C0 × f_dc` (and its inverse for the
/// COLOR_0 fallback).
#[cfg(feature = "splats")]
const SH_C0: f32 = 0.282_095;

/// Decode every splat primitive in a raw glTF document. Node transforms are
/// honored (matrix or TRS); non-splat primitives in the same file are skipped.
#[cfg(feature = "splats")]
fn decode_splat_gltf(
    json: &serde_json::Value,
    bin: Option<&[u8]>,
) -> Result<Vec<DecodedItem>, String> {
    let mut out = Vec::new();
    let scene_ix = json["scene"].as_u64().unwrap_or(0) as usize;
    let roots = json["scenes"][scene_ix]["nodes"]
        .as_array()
        .ok_or("splat tile has no scene nodes")?;
    for root in roots {
        let ix = root.as_u64().ok_or("bad node index")? as usize;
        decode_splat_node(json, bin, ix, Mat4::IDENTITY, &mut out)?;
    }
    Ok(out)
}

#[cfg(feature = "splats")]
fn decode_splat_node(
    json: &serde_json::Value,
    bin: Option<&[u8]>,
    node_ix: usize,
    parent: Mat4,
    out: &mut Vec<DecodedItem>,
) -> Result<(), String> {
    let node = &json["nodes"][node_ix];
    if node.is_null() {
        return Err(format!("node {node_ix} out of bounds"));
    }
    let global = parent * node_transform(node);
    if let Some(mesh_ix) = node["mesh"].as_u64() {
        let prims = json["meshes"][mesh_ix as usize]["primitives"]
            .as_array()
            .ok_or("mesh without primitives")?;
        for prim in prims {
            let attrs = &prim["attributes"];
            if attrs[ATTR_ROTATION].is_null() {
                continue; // not a splat primitive
            }
            out.push(DecodedItem::Splat {
                transform: global,
                gaussians: decode_splat_primitive(json, bin, attrs)?,
            });
        }
    }
    if let Some(children) = node["children"].as_array() {
        for child in children {
            let ix = child.as_u64().ok_or("bad child index")? as usize;
            decode_splat_node(json, bin, ix, global, out)?;
        }
    }
    Ok(())
}

/// A raw glTF node's local transform: `matrix` (column-major) or TRS.
#[cfg(feature = "splats")]
fn node_transform(node: &serde_json::Value) -> Mat4 {
    if let Some(m) = node["matrix"].as_array() {
        let vals: Vec<f32> = m
            .iter()
            .filter_map(|v| v.as_f64())
            .map(|v| v as f32)
            .collect();
        if vals.len() == 16 {
            return Mat4::from_cols_array(&vals.try_into().unwrap());
        }
    }
    let vec3 = |key: &str, default: [f32; 3]| -> bevy::math::Vec3 {
        node[key]
            .as_array()
            .and_then(|a| {
                let v: Vec<f32> = a
                    .iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect();
                <[f32; 3]>::try_from(v).ok()
            })
            .map(bevy::math::Vec3::from)
            .unwrap_or(bevy::math::Vec3::from(default))
    };
    let rotation = node["rotation"]
        .as_array()
        .and_then(|a| {
            let v: Vec<f32> = a
                .iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect();
            <[f32; 4]>::try_from(v).ok()
        })
        .map(bevy::math::Quat::from_array)
        .unwrap_or(bevy::math::Quat::IDENTITY);
    Mat4::from_scale_rotation_translation(
        vec3("scale", [1.0; 3]),
        rotation,
        vec3("translation", [0.0; 3]),
    )
}

#[cfg(feature = "splats")]
fn decode_splat_primitive(
    json: &serde_json::Value,
    bin: Option<&[u8]>,
    attrs: &serde_json::Value,
) -> Result<Vec<Gaussian3d>, String> {
    let accessor_of = |name: &str| -> Result<usize, String> {
        attrs[name]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("splat primitive missing {name}"))
    };
    let positions = read_accessor::<3>(json, bin, accessor_of("POSITION")?)?;
    let rotations = read_accessor::<4>(json, bin, accessor_of(ATTR_ROTATION)?)?;
    let scales = read_accessor::<3>(json, bin, accessor_of(ATTR_SCALE)?)?;
    let opacities = read_accessor::<1>(json, bin, accessor_of(ATTR_OPACITY)?)?;
    // Color source: SH degree 0 (required by the spec); COLOR_0 as a
    // defensive fallback for foreign files.
    let sh0 = match attrs[ATTR_SH0].as_u64() {
        Some(ix) => Some(read_accessor::<3>(json, bin, ix as usize)?),
        None => None,
    };
    let color0 = match (&sh0, attrs["COLOR_0"].as_u64()) {
        (None, Some(ix)) => Some(read_accessor::<4>(json, bin, ix as usize)?),
        _ => None,
    };
    if sh0.is_none() && color0.is_none() {
        return Err("splat primitive has neither SH_DEGREE_0_COEF_0 nor COLOR_0".into());
    }

    let n = positions.len();
    if [rotations.len(), scales.len(), opacities.len()]
        .iter()
        .any(|&l| l != n)
    {
        return Err(format!(
            "splat attribute counts disagree: pos={n} rot={} scale={} opacity={}",
            rotations.len(),
            scales.len(),
            opacities.len()
        ));
    }

    let mut gaussians = Vec::with_capacity(n.div_ceil(32) * 32);
    for i in 0..n {
        let mut g = Gaussian3d::default();
        g.position_visibility.position = [positions[i][0], positions[i][1], positions[i][2]];
        g.position_visibility.visibility = 1.0;
        // glTF quaternion order is xyzw; the crate stores wxyz (the INRIA ply
        // rot_0..3 layout). Spec guarantees unit quaternions; normalize anyway
        // (quantized foreign data).
        let [x, y, z, w] = rotations[i];
        let norm = (x * x + y * y + z * z + w * w).sqrt().max(1e-12);
        g.rotation.rotation = [w / norm, x / norm, y / norm, z / norm];
        // Spec: linear, non-negative scale; linear opacity (sigmoid already
        // applied at training) — both match the crate's post-ply-parse state.
        g.scale_opacity.scale = [scales[i][0], scales[i][1], scales[i][2]];
        g.scale_opacity.opacity = opacities[i][0].clamp(0.0, 1.0);
        let f_dc = match (&sh0, &color0) {
            (Some(sh), _) => [sh[i][0], sh[i][1], sh[i][2]],
            (None, Some(c)) => [
                (c[i][0] - 0.5) / SH_C0,
                (c[i][1] - 0.5) / SH_C0,
                (c[i][2] - 0.5) / SH_C0,
            ],
            (None, None) => unreachable!(),
        };
        g.spherical_harmonic.set(0, f_dc[0]);
        g.spherical_harmonic.set(1, f_dc[1]);
        g.spherical_harmonic.set(2, f_dc[2]);
        gaussians.push(g);
    }
    // Pad to a multiple of 32 (the crate's own ply path does the same — the
    // GPU sort works in 32-wide groups). Default gaussians are invisible.
    let pad = (32 - gaussians.len() % 32) % 32;
    gaussians.extend(std::iter::repeat_n(Gaussian3d::default(), pad));
    Ok(gaussians)
}

/// Read accessor `index` as `Vec<[f32; N]>`. Supports float and the spec's
/// normalized integer encodings; tightly-packed or strided buffer views; no
/// sparse accessors (our tilers never emit them).
fn read_accessor<const N: usize>(
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic GLB: one triangle with COLOR_0, no normals.
    /// (The fixture generator writes real tiles with the same layout.)
    fn tiny_glb() -> Vec<u8> {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let colors: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 1.0]; 3];
        let indices: [u16; 4] = [0, 1, 2, 0]; // padded to 4-byte alignment

        let mut bin: Vec<u8> = Vec::new();
        for p in positions.iter().flatten() {
            bin.extend_from_slice(&p.to_le_bytes());
        }
        for c in colors.iter().flatten() {
            bin.extend_from_slice(&c.to_le_bytes());
        }
        let idx_offset = bin.len();
        for i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }

        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0, "translation": [0.0, 2.0, 0.0] }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "COLOR_0": 1 },
                "indices": 2,
                "mode": 4
            }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" },
                { "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 48 },
                { "buffer": 0, "byteOffset": idx_offset, "byteLength": 6 }
            ],
            "buffers": [{ "byteLength": bin.len() }]
        });
        glb_from_parts(&serde_json::to_vec(&json).unwrap(), &bin)
    }

    fn glb_from_parts(json_bytes: &[u8], bin: &[u8]) -> Vec<u8> {
        assemble_glb(json_bytes, bin)
    }

    #[test]
    fn decodes_positions_colors_and_node_transform() {
        let items = decode_glb(&tiny_glb()).expect("decode");
        assert_eq!(items.len(), 1);
        let DecodedItem::Mesh(p) = &items[0] else {
            panic!("expected mesh")
        };
        // Node translation carried into the primitive transform.
        assert_eq!(p.transform.w_axis.y, 2.0);
        assert_eq!(p.mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(), 3);
        // Normals were computed (absent in the GLB).
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        assert_eq!(p.material.base_color, [1.0; 4]);
        assert!(p.material.base_color_image.is_none());
    }

    #[test]
    fn garbage_bytes_error_cleanly() {
        assert!(decode_glb(b"not a glb").is_err());
    }

    /// `decode_tile` on a plain tile takes the fast path and carries no
    /// side-band data; with `asset.copyright` + `CESIUM_RTC` it extracts
    /// both, strips the extensions (the gltf crate rejects unknown
    /// `extensionsRequired`), and still decodes the geometry.
    #[test]
    fn decode_tile_extracts_copyright_and_rtc() {
        use bevy::tasks::block_on;

        let plain = block_on(decode_tile(&tiny_glb(), false)).expect("plain decode");
        assert_eq!(plain.items.len(), 1);
        assert!(plain.rtc_center.is_none() && plain.copyright.is_none());

        // tiny_glb's JSON + copyright + a required CESIUM_RTC extension.
        let glb_bytes = tiny_glb();
        let (json, bin) = split_glb(&glb_bytes).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(json).unwrap();
        value["asset"]["copyright"] = serde_json::json!("Data A;Data B");
        value["extensions"] =
            serde_json::json!({ "CESIUM_RTC": { "center": [6378137.0, 1000.5, -2000.25] } });
        value["extensionsRequired"] = serde_json::json!(["CESIUM_RTC"]);
        let glb = assemble_glb(&serde_json::to_vec(&value).unwrap(), bin.unwrap());

        let tile = block_on(decode_tile(&glb, false)).expect("rtc decode");
        assert_eq!(tile.items.len(), 1, "geometry survives the strip");
        assert_eq!(tile.copyright.as_deref(), Some("Data A;Data B"));
        let rtc = tile.rtc_center.expect("rtc center");
        assert!((rtc - DVec3::new(6_378_137.0, 1000.5, -2000.25)).length() < 1e-9);
    }

    /// Google P3DT shape: ECEF baked into the node MATRIX (no CESIUM_RTC),
    /// `KHR_materials_unlit` required. The planetary translation must come
    /// out in f64 as the rtc side-band; decoded node transforms stay
    /// tile-local; the material decodes unlit.
    #[test]
    fn decode_tile_extracts_planetary_node_matrix() {
        use bevy::tasks::block_on;

        let (cx, cy, cz) = (
            -2_398_029.177_060_164,
            3_361_082.915_181_850_5,
            2_398_029.177_060_164_5,
        );
        let glb_bytes = tiny_glb();
        let (json, bin) = split_glb(&glb_bytes).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(json).unwrap();
        // The real Google node shape: rotation + planetary translation in
        // one matrix (column-major).
        value["nodes"] = serde_json::json!([{
            "matrix": [1,0,0,0, 0,0,-1,0, 0,1,0,0, cx, cy, cz, 1],
            "mesh": 0
        }]);
        value["asset"]["copyright"] = serde_json::json!("Google");
        value["extensionsUsed"] = serde_json::json!(["KHR_materials_unlit"]);
        value["extensionsRequired"] = serde_json::json!(["KHR_materials_unlit"]);
        value["materials"] = serde_json::json!([{
            "pbrMetallicRoughness": { "baseColorFactor": [1.0,1.0,1.0,1.0] },
            "extensions": { "KHR_materials_unlit": {} }
        }]);
        value["meshes"][0]["primitives"][0]["material"] = serde_json::json!(0);
        let glb = assemble_glb(&serde_json::to_vec(&value).unwrap(), bin.unwrap());

        let tile = block_on(decode_tile(&glb, true)).expect("decode");
        let rtc = tile.rtc_center.expect("planetary offset extracted");
        assert!(
            (rtc - DVec3::new(cx, cy, cz)).length() < 1e-6,
            "rtc = {rtc:?}"
        );
        let DecodedItem::Mesh(p) = &tile.items[0] else {
            panic!("expected mesh")
        };
        // The decoded transform keeps the rotation but the translation is
        // tile-local now (zero here — single node).
        assert!(
            p.transform.w_axis.truncate().length() < 1e-3,
            "{:?}",
            p.transform.w_axis
        );
        assert!(
            (p.transform.y_axis.z - (-1.0)).abs() < 1e-6,
            "rotation preserved"
        );
        assert!(p.material.unlit, "KHR_materials_unlit decoded");
        assert_eq!(tile.copyright.as_deref(), Some("Google"));
    }

    /// The Draco splice: a primitive whose data only exists Draco-compressed
    /// rewrites into a vanilla GLB that the standard path decodes (mock
    /// decoder output stands in for the real decoder, which is wasm-only).
    #[test]
    fn splice_glb_rewrites_draco_primitive() {
        let fake_compressed = vec![0xAAu8; 16];
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_draco_mesh_compression", "CESIUM_RTC"],
            "extensionsRequired": ["KHR_draco_mesh_compression", "CESIUM_RTC"],
            "extensions": { "CESIUM_RTC": { "center": [1.0, 2.0, 3.0] } },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                // Spec shape: attributes reference accessors WITHOUT
                // bufferViews; the extension maps semantics → draco ids.
                "attributes": { "POSITION": 0, "COLOR_0": 1 },
                "mode": 4,
                "extensions": { "KHR_draco_mesh_compression": {
                    "bufferView": 0,
                    "attributes": { "POSITION": 0, "COLOR_0": 1 }
                }}
            }]}],
            "accessors": [
                { "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0,0,0], "max": [1,1,0] },
                { "componentType": 5126, "count": 3, "type": "VEC4" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": fake_compressed.len() }
            ],
            "buffers": [{ "byteLength": fake_compressed.len() }]
        });

        let positions = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let colors = vec![1.0f32; 12];
        let decoded = vec![draco::DracoMesh {
            indices: vec![0, 1, 2],
            attributes: vec![(0, 3, positions), (1, 4, colors)],
        }];
        let prims = find_draco_prims(&json);
        assert_eq!(prims.len(), 1);
        assert_eq!(prims[0].buffer_view, 0);

        let mut value = json;
        let vanilla =
            splice_glb(&mut value, Some(&fake_compressed), &prims, decoded).expect("splice");
        // The spliced GLB decodes through the strict gltf-crate path.
        let items = decode_glb(&vanilla).expect("decode spliced");
        assert_eq!(items.len(), 1);
        let DecodedItem::Mesh(p) = &items[0] else {
            panic!("expected mesh")
        };
        assert_eq!(p.mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(), 3);
        assert!(p.mesh.indices().is_some());
        // All handled extensions stripped.
        let (j, _) = split_glb(&vanilla).unwrap();
        assert!(!memmem(j, b"KHR_draco_mesh_compression"));
        assert!(!memmem(j, b"CESIUM_RTC"));
    }

    /// POINTS-mode GLB: positions + u8-normalized COLOR_0 → point items in
    /// the glTF frame with material-driven sizes.
    #[cfg(feature = "points")]
    #[test]
    fn decodes_points_primitive() {
        let positions: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let colors: [[u8; 4]; 2] = [[255, 0, 0, 255], [0, 255, 0, 255]];
        let mut bin: Vec<u8> = Vec::new();
        for p in positions.iter().flatten() {
            bin.extend_from_slice(&p.to_le_bytes());
        }
        let color_offset = bin.len();
        for c in colors.iter().flatten() {
            bin.push(*c);
        }
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "COLOR_0": 1 },
                "mode": 0
            }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3",
                  "min": [0.0, 1.0, 2.0], "max": [3.0, 4.0, 5.0] },
                { "bufferView": 1, "componentType": 5121, "normalized": true,
                  "count": 2, "type": "VEC4" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 24 },
                { "buffer": 0, "byteOffset": color_offset, "byteLength": 8 }
            ],
            "buffers": [{ "byteLength": bin.len() }]
        });
        let glb = glb_from_parts(&serde_json::to_vec(&json).unwrap(), &bin);
        let items = decode_glb(&glb).expect("decode");
        assert_eq!(items.len(), 1);
        let DecodedItem::Points { points, .. } = &items[0] else {
            panic!("expected points")
        };
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].position, bevy::math::Vec3::new(0.0, 1.0, 2.0));
        assert_eq!(points[0].point_size, -1.0);
        assert!((points[0].color[0] - 1.0).abs() < 1e-6);
        assert!((points[1].color[1] - 1.0).abs() < 1e-6);
    }

    /// KHR_gaussian_splatting GLB (float accessors, like our tiler emits):
    /// bypasses the gltf crate, maps quaternions xyzw→wxyz, keeps linear
    /// scale/opacity, reads SH degree 0, pads to 32.
    #[cfg(feature = "splats")]
    #[test]
    fn decodes_splat_primitive_via_raw_path() {
        let positions: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rotations: [[f32; 4]; 2] = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]; // xyzw
        let scales: [[f32; 3]; 2] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let opacities: [f32; 2] = [0.25, 1.0];
        let sh0: [[f32; 3]; 2] = [[1.0, -0.5, 0.0], [0.0, 0.0, 2.0]];

        let mut bin: Vec<u8> = Vec::new();
        let mut offsets = Vec::new();
        let mut push = |vals: &[f32]| {
            offsets.push(bin.len());
            for v in vals {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        };
        push(&positions.iter().flatten().copied().collect::<Vec<_>>());
        push(&rotations.iter().flatten().copied().collect::<Vec<_>>());
        push(&scales.iter().flatten().copied().collect::<Vec<_>>());
        push(&opacities);
        push(&sh0.iter().flatten().copied().collect::<Vec<_>>());

        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_gaussian_splatting"],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0, "translation": [10.0, 0.0, 0.0] }],
            "meshes": [{ "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "KHR_gaussian_splatting:ROTATION": 1,
                    "KHR_gaussian_splatting:SCALE": 2,
                    "KHR_gaussian_splatting:OPACITY": 3,
                    "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4
                },
                "mode": 0,
                "extensions": { "KHR_gaussian_splatting": {} }
            }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC4" },
                { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3" },
                { "bufferView": 3, "componentType": 5126, "count": 2, "type": "SCALAR" },
                { "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC3" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": offsets[0], "byteLength": 24 },
                { "buffer": 0, "byteOffset": offsets[1], "byteLength": 32 },
                { "buffer": 0, "byteOffset": offsets[2], "byteLength": 24 },
                { "buffer": 0, "byteOffset": offsets[3], "byteLength": 8 },
                { "buffer": 0, "byteOffset": offsets[4], "byteLength": 24 }
            ],
            "buffers": [{ "byteLength": bin.len() }]
        });
        let glb = glb_from_parts(&serde_json::to_vec(&json).unwrap(), &bin);
        let items = decode_glb(&glb).expect("decode");
        assert_eq!(items.len(), 1);
        let DecodedItem::Splat {
            transform,
            gaussians,
        } = &items[0]
        else {
            panic!("expected splat")
        };
        assert_eq!(transform.w_axis.x, 10.0);
        assert_eq!(gaussians.len(), 32, "2 real + 30 pad");
        let g = &gaussians[0];
        assert_eq!(g.position_visibility.position, [1.0, 2.0, 3.0]);
        assert_eq!(g.position_visibility.visibility, 1.0);
        // xyzw [0,0,0,1] → wxyz [1,0,0,0].
        assert_eq!(g.rotation.rotation, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(g.scale_opacity.scale, [0.1, 0.2, 0.3]);
        assert_eq!(g.scale_opacity.opacity, 0.25);
        let g1 = &gaussians[1];
        // xyzw [1,0,0,0] → wxyz [0,1,0,0].
        assert_eq!(g1.rotation.rotation, [0.0, 1.0, 0.0, 0.0]);
    }

    /// End-to-end T6: an `EXT_meshopt_compression` GLB produced by the exact
    /// writer config (`tile_mesh.mjs`: QUANTIZE method, no quantization → filter
    /// NONE → lossless) decodes through `preprocess_meshopt` + the strict gltf
    /// path to **byte-identical** positions/colors and the same triangle set.
    /// The GLB bytes are captured from `@gltf-transform` + `meshoptimizer` (see
    /// the BEVY-3D-TILES T6 commit notes).
    #[test]
    fn decodes_meshopt_tile_byte_identical() {
        use base64::Engine;

        const GLB_B64: &str = "Z2xURgIAAABMBgAAaAUAAEpTT057ImFzc2V0Ijp7ImdlbmVyYXRvciI6ImdsVEYtVHJhbnNmb3JtIHY0LjMuMCIsInZlcnNpb24iOiIyLjAifSwiYWNjZXNzb3JzIjpbeyJ0eXBlIjoiVkVDMyIsImNvbXBvbmVudFR5cGUiOjUxMjYsImNvdW50Ijo2LCJtYXgiOlsyLDMsM10sIm1pbiI6Wy0xLDAsMF0sIm5vcm1hbGl6ZWQiOmZhbHNlLCJieXRlT2Zmc2V0IjowLCJidWZmZXJWaWV3IjowfSx7InR5cGUiOiJWRUM0IiwiY29tcG9uZW50VHlwZSI6NTEyNiwiY291bnQiOjYsIm5vcm1hbGl6ZWQiOmZhbHNlLCJieXRlT2Zmc2V0IjowLCJidWZmZXJWaWV3IjoxfSx7InR5cGUiOiJTQ0FMQVIiLCJjb21wb25lbnRUeXBlIjo1MTI1LCJjb3VudCI6MTIsIm5vcm1hbGl6ZWQiOmZhbHNlLCJieXRlT2Zmc2V0IjowLCJidWZmZXJWaWV3IjoyfV0sImJ1ZmZlclZpZXdzIjpbeyJidWZmZXIiOjEsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjcyLCJ0YXJnZXQiOjM0OTYyLCJieXRlU3RyaWRlIjoxMiwiZXh0ZW5zaW9ucyI6eyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiI6eyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjgwLCJtb2RlIjoiQVRUUklCVVRFUyIsImJ5dGVTdHJpZGUiOjEyLCJjb3VudCI6Nn19fSx7ImJ1ZmZlciI6MSwiYnl0ZU9mZnNldCI6NzIsImJ5dGVMZW5ndGgiOjk2LCJ0YXJnZXQiOjM0OTYyLCJieXRlU3RyaWRlIjoxNiwiZXh0ZW5zaW9ucyI6eyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiI6eyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjgwLCJieXRlTGVuZ3RoIjo5NSwibW9kZSI6IkFUVFJJQlVURVMiLCJieXRlU3RyaWRlIjoxNiwiY291bnQiOjZ9fX0seyJidWZmZXIiOjEsImJ5dGVPZmZzZXQiOjE2OCwiYnl0ZUxlbmd0aCI6NDgsInRhcmdldCI6MzQ5NjMsImV4dGVuc2lvbnMiOnsiRVhUX21lc2hvcHRfY29tcHJlc3Npb24iOnsiYnVmZmVyIjowLCJieXRlT2Zmc2V0IjoxNzYsImJ5dGVMZW5ndGgiOjIyLCJtb2RlIjoiVFJJQU5HTEVTIiwiYnl0ZVN0cmlkZSI6NCwiY291bnQiOjEyfX19XSwiYnVmZmVycyI6W3siYnl0ZUxlbmd0aCI6MjAwfSx7ImJ5dGVMZW5ndGgiOjIxNiwiZXh0ZW5zaW9ucyI6eyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiI6eyJmYWxsYmFjayI6dHJ1ZX19fV0sIm1lc2hlcyI6W3sicHJpbWl0aXZlcyI6W3siYXR0cmlidXRlcyI6eyJQT1NJVElPTiI6MCwiQ09MT1JfMCI6MX0sIm1vZGUiOjQsImluZGljZXMiOjJ9XX1dLCJub2RlcyI6W3sidHJhbnNsYXRpb24iOlsxMCwwLDBdLCJtZXNoIjowfV0sInNjZW5lcyI6W3sibm9kZXMiOlswXX1dLCJleHRlbnNpb25zVXNlZCI6WyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiJdLCJleHRlbnNpb25zUmVxdWlyZWQiOlsiRVhUX21lc2hvcHRfY29tcHJlc3Npb24iXX0gyAAAAEJJTgCgAAABAMAAAP8BM/AAAIB/fv8AAAEA8AAA/38BDGAAAIAAAAEA8AAAgIABANAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAAAEz8AAA/////wEz8AAAfX59fgAAAT8wAAD/////AT8wAAB+fX59AAABD8AAAP///wEPwAAAfn1+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AOHwAP4FAgB2h1ZneKmGZYlomAFpAAAAAA==";
        let glb = base64::engine::general_purpose::STANDARD
            .decode(GLB_B64)
            .unwrap();

        let items = decode_glb(&glb).expect("meshopt decode");
        assert_eq!(items.len(), 1);
        let DecodedItem::Mesh(p) = &items[0] else {
            panic!("expected mesh")
        };

        // Node translation [10,0,0] carried onto the primitive transform.
        assert_eq!(
            p.transform.w_axis.truncate(),
            bevy::math::Vec3::new(10.0, 0.0, 0.0)
        );

        // Positions are byte-identical (lossless meshopt vertex codec).
        let pos = p
            .mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let expect_pos: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 3.0],
            [-1.0, 3.0, 1.0],
        ];
        assert_eq!(pos, &expect_pos, "positions must round-trip byte-identical");
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());

        // The triangle SET is preserved (meshopt may cyclically rotate each
        // triangle — winding kept, a rendering no-op).
        let Some(Indices::U32(idx)) = p.mesh.indices() else {
            panic!("expected u32 indices")
        };
        assert_eq!(idx.len(), 12);
        let as_sorted_tris = |flat: &[u32]| {
            let mut tris: Vec<[u32; 3]> = flat
                .chunks_exact(3)
                .map(|t| {
                    let mut v = [t[0], t[1], t[2]];
                    v.sort_unstable(); // set comparison ignores winding/rotation
                    v
                })
                .collect();
            tris.sort_unstable();
            tris
        };
        let got = as_sorted_tris(idx);
        let want = as_sorted_tris(&[0, 1, 2, 0, 2, 3, 2, 4, 5, 0, 4, 2]);
        assert_eq!(got, want, "same triangle set");
    }

    /// T8: a GLB with `EXT_mesh_features` (`_FEATURE_ID_0`, FLOAT) + a minimal
    /// `EXT_structural_metadata` STRING property table (the exact shape our
    /// tiler injects). Two triangles, two features; decode must produce a
    /// per-triangle featureId array (index-buffer order) and the node-path
    /// table — the inputs the pick → node → twin resolver consumes.
    #[test]
    fn decodes_feature_metadata() {
        // 6 verts (2 tris). Tri 0 → feature 0, tri 1 → feature 1.
        let positions: [[f32; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let feature_ids: [f32; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let strings: [&str; 2] = ["AlphaModule", "BetaModule/sub"];

        let mut bin: Vec<u8> = Vec::new();
        for p in positions.iter().flatten() {
            bin.extend_from_slice(&p.to_le_bytes());
        }
        let feat_off = bin.len();
        for f in feature_ids {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        let idx_off = bin.len();
        for i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        // EXT_structural_metadata STRING column: values + UINT32 stringOffsets.
        let values_off = bin.len();
        let mut offsets = vec![0u32];
        for s in strings {
            bin.extend_from_slice(s.as_bytes());
            offsets.push((bin.len() - values_off) as u32);
        }
        let values_len = bin.len() - values_off;
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let offsets_off = bin.len();
        for o in &offsets {
            bin.extend_from_slice(&o.to_le_bytes());
        }

        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["EXT_mesh_features", "EXT_structural_metadata"],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "_FEATURE_ID_0": 1 },
                "indices": 2,
                "mode": 4,
                "extensions": { "EXT_mesh_features": {
                    "featureIds": [{ "featureCount": 2, "attribute": 0, "propertyTable": 0 }]
                }}
            }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3",
                  "min": [0.0, 0.0, 0.0], "max": [3.0, 1.0, 0.0] },
                { "bufferView": 1, "componentType": 5126, "count": 6, "type": "SCALAR" },
                { "bufferView": 2, "componentType": 5125, "count": 6, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": feat_off },
                { "buffer": 0, "byteOffset": feat_off, "byteLength": idx_off - feat_off },
                { "buffer": 0, "byteOffset": idx_off, "byteLength": values_off - idx_off },
                { "buffer": 0, "byteOffset": values_off, "byteLength": values_len },
                { "buffer": 0, "byteOffset": offsets_off, "byteLength": offsets.len() * 4 }
            ],
            "buffers": [{ "byteLength": bin.len() }],
            "extensions": { "EXT_structural_metadata": {
                "schema": { "id": "tt_features", "classes": { "feature": {
                    "properties": { "nodePath": { "type": "STRING" } }
                }}},
                "propertyTables": [{
                    "class": "feature", "count": 2,
                    "properties": { "nodePath": { "values": 3, "stringOffsets": 4 } }
                }]
            }}
        });
        let glb = assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin);

        let items = decode_glb(&glb).expect("decode features");
        assert_eq!(items.len(), 1);
        let DecodedItem::Mesh(p) = &items[0] else {
            panic!("expected mesh")
        };
        let feats = p.features.as_ref().expect("feature metadata decoded");
        // featureId per triangle, in index-buffer order.
        assert_eq!(feats.feature_of_triangle, vec![0, 1]);
        // Property table → node paths (one carries a `/` path the resolver splits).
        assert_eq!(
            &**feats.node_of_feature,
            &["AlphaModule".to_string(), "BetaModule/sub".to_string()]
        );
        // Per-vertex ids are kept AND written onto the mesh as UV1, so a host
        // feature-tint material can read them in the fragment stage (0.1.7).
        assert_eq!(feats.feature_of_vertex, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let uv1 = p
            .mesh
            .attribute(Mesh::ATTRIBUTE_UV_1)
            .expect("feature ids as UV1");
        assert_eq!(uv1.len(), 6);
    }

    /// T7: a GLB whose base-color texture is a `KHR_texture_basisu` KTX2 (UASTC,
    /// the writer's exact output) decodes through `preprocess_basisu` + the gltf
    /// path + the async texture-resolve pass. The `gltf` crate can't resolve the
    /// extension and the transcoder isn't callable from the sync decode, so this
    /// proves the source rewrite + deferred transcode work end-to-end. On native
    /// (this test) the resolve uses bevy's `basis-universal`; we latch `BC` (as a
    /// desktop adapter would) so UASTC → BC7 on the CPU. GLB captured from
    /// `@gltf-transform` + `ktx create --encode uastc` (BEVY-3D-TILES T7).
    #[test]
    fn decodes_basisu_ktx2_base_color() {
        use base64::Engine;
        use bevy::tasks::block_on;

        // Pretend the adapter supports BC (desktop WebGPU). OnceLock first-wins;
        // no other test latches it, and it only affects KTX2 decode.
        set_supported_compressed_formats(CompressedImageFormats::BC);

        const GLB_B64: &str = "Z2xURgIAAACUBQAAMAQAAEpTT057ImFzc2V0Ijp7ImdlbmVyYXRvciI6ImdsVEYtVHJhbnNmb3JtIHY0LjMuMCIsInZlcnNpb24iOiIyLjAifSwiYWNjZXNzb3JzIjpbeyJ0eXBlIjoiVkVDMyIsImNvbXBvbmVudFR5cGUiOjUxMjYsImNvdW50IjozLCJtYXgiOlsxLDEsMF0sIm1pbiI6WzAsMCwwXSwiYnVmZmVyVmlldyI6MSwiYnl0ZU9mZnNldCI6MH0seyJ0eXBlIjoiVkVDMiIsImNvbXBvbmVudFR5cGUiOjUxMjYsImNvdW50IjozLCJidWZmZXJWaWV3IjoxLCJieXRlT2Zmc2V0IjoxMn0seyJ0eXBlIjoiU0NBTEFSIiwiY29tcG9uZW50VHlwZSI6NTEyNSwiY291bnQiOjMsImJ1ZmZlclZpZXciOjIsImJ5dGVPZmZzZXQiOjB9XSwiYnVmZmVyVmlld3MiOlt7ImJ1ZmZlciI6MCwiYnl0ZU9mZnNldCI6NzIsImJ5dGVMZW5ndGgiOjI1NH0seyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjYwLCJieXRlU3RyaWRlIjoyMCwidGFyZ2V0IjozNDk2Mn0seyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjYwLCJieXRlTGVuZ3RoIjoxMiwidGFyZ2V0IjozNDk2M31dLCJzYW1wbGVycyI6W3sid3JhcFMiOjEwNDk3LCJ3cmFwVCI6MTA0OTd9XSwidGV4dHVyZXMiOlt7InNhbXBsZXIiOjAsImV4dGVuc2lvbnMiOnsiS0hSX3RleHR1cmVfYmFzaXN1Ijp7InNvdXJjZSI6MH19fV0sImltYWdlcyI6W3sibmFtZSI6ImJhc2UiLCJtaW1lVHlwZSI6ImltYWdlL2t0eDIiLCJidWZmZXJWaWV3IjowfV0sImJ1ZmZlcnMiOlt7ImJ5dGVMZW5ndGgiOjMyOH1dLCJtYXRlcmlhbHMiOlt7Im5hbWUiOiJtIiwicGJyTWV0YWxsaWNSb3VnaG5lc3MiOnsiYmFzZUNvbG9yVGV4dHVyZSI6eyJpbmRleCI6MH19fV0sIm1lc2hlcyI6W3sicHJpbWl0aXZlcyI6W3siYXR0cmlidXRlcyI6eyJQT1NJVElPTiI6MCwiVEVYQ09PUkRfMCI6MX0sIm1vZGUiOjQsIm1hdGVyaWFsIjowLCJpbmRpY2VzIjoyfV19XSwibm9kZXMiOlt7Im1lc2giOjB9XSwic2NlbmVzIjpbeyJub2RlcyI6WzBdfV0sImV4dGVuc2lvbnNVc2VkIjpbIktIUl90ZXh0dXJlX2Jhc2lzdSJdLCJleHRlbnNpb25zUmVxdWlyZWQiOlsiS0hSX3RleHR1cmVfYmFzaXN1Il19SAEAAEJJTgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAgD8AAAAAAQAAAAIAAACrS1RYIDIwuw0KGgoAAAAAAQAAAAgAAAAIAAAAAAAAAAAAAAABAAAAAQAAAAIAAABoAAAALAAAAJQAAABQAAAAAAAAAAAAAAAAAAAAAAAAAOQAAAAAAAAAGgAAAAAAAABAAAAAAAAAACwAAAAAAAAAAgAoAKYBAgADAwAAEAAAAAAAAAAAAH8AAAAAAAAAAAD/////LAAAAEtUWHdyaXRlcgBrdHggY3JlYXRlIHY0LjQuMiAvIGxpYmt0eCB2NC40LjIAHAAAAEtUWHdyaXRlclNjUGFyYW1zAC0tenN0ZCAxOAAotS/9IECNAABIVwGZ5/87vgEAAgDNjCADRwAA";
        let glb = base64::engine::general_purpose::STANDARD
            .decode(GLB_B64)
            .unwrap();

        let tile = block_on(decode_tile(&glb, false)).expect("ktx2 decode");
        assert_eq!(tile.items.len(), 1);
        let DecodedItem::Mesh(p) = &tile.items[0] else {
            panic!("expected mesh")
        };
        // Resolved to a real image; the pending KTX2 bytes were consumed.
        assert!(
            p.material.base_color_ktx2.is_none(),
            "pending KTX2 must be taken"
        );
        let img = p
            .material
            .base_color_image
            .as_ref()
            .expect("KTX2 base color transcoded");
        assert_eq!(
            (img.width(), img.height()),
            (8, 8),
            "8x8 source dimensions kept"
        );
    }
}
