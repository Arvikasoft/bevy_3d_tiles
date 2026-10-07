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
//! Everything runs inside the loader task: outputs are plain `Send` data the
//! ECS drain turns into entities. That task is an OS thread on native, but on
//! wasm it is `spawn_local` on the MAIN thread — every synchronous span
//! between `.await` points is frame time, which is why this module works hard
//! to parse the JSON chunk once and rebuild the GLB container at most once.
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

use std::sync::{Arc, OnceLock};

use bevy::asset::RenderAssetUsages;
use bevy::image::{
    CompressedImageFormats, Image, ImageAddressMode, ImageFilterMode, ImageSampler,
    ImageSamplerDescriptor, ImageType,
};
use bevy::math::{DVec3, Mat4};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
// ponytail: `performance.now()` on wasm via bevy_platform's `web` feature —
// already on transitively through bevy_render/bevy_winit; nothing here asks
// for it explicitly.
use bevy::platform::time::Instant;
#[cfg(feature = "splats")]
use bevy_gaussian_splatting::gaussian::formats::planar_3d::Gaussian3d;
#[cfg(feature = "points")]
use bevy_pointcloud::point_cloud::PointCloudData;

use super::draco;

// The bevy-free CPU half of tile decode lives in the sibling
// `bevy_3d_tiles_prepare` crate (offthread-decode plan S4) — moved, never
// copied, and re-exported here so `content::DecodeError` etc. keep working.
pub use bevy_3d_tiles_prepare::{
    DecodeError, DecodeStage, ExtractedMaterial, ExtractedMeshes, ExtractedPrimitive, PreparedTile,
    extract_tile_meshes, prepare_tile, prepare_tile_extracting,
};

#[cfg(feature = "splats")]
use bevy_3d_tiles_prepare::read_accessor;
use bevy_3d_tiles_prepare::{
    DracoPrim, FeatureCtx, Marks, PreparedFeatures, assemble_glb, bounds_of, buffer_view_slice,
    decode_meshopt_views, extract_planetary_root_offset, find_draco_prims, preprocess_basisu,
    splice_draco, split_glb, strip_handled_extensions, tile_rtc_to_content_frame, unwrap_b3dm,
};
// `lib.rs` sniffs external tilesets with it (`looks_like_external_tileset`).
pub(crate) use bevy_3d_tiles_prepare::memmem;

use crate::api::TilePrepareFn;

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
///
/// The per-VERTEX ids live on the mesh itself, as `ATTRIBUTE_UV_1` (`[fid, 0]`),
/// so a host material can style per feature in the fragment shader (the Cesium
/// `Cesium3DTileFeature.color` model) through the standard pipeline's
/// `VERTEX_UVS_B` path, with no custom vertex shader. Feature tiles never carry
/// a real `TEXCOORD_1` (it was already dropped before 0.1.7), so nothing is
/// displaced. Read them from [`DecodedPrimitive::mesh`] before it is spawned.
pub struct TileFeatures {
    /// featureId per triangle, in the spawned mesh's index-buffer order — so
    /// the pick raycast's triangle ordinal indexes straight into it. A feature
    /// primitive is always indexed (a non-indexed one gets `0..n`).
    pub feature_of_triangle: Vec<u32>,
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
    /// Axis-aligned bounds `[min, max]` of the mesh positions, in the
    /// primitive's own frame (before `transform`). Computed off-thread on the
    /// extracted route. `None` only for a primitive with no positions.
    pub bounds: Option<[[f32; 3]; 2]>,
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

/// Decoded geometry cost of a tile's items, in bytes: mesh vertex attributes +
/// indices, plus point/splat buffers. Textures are NOT counted — images upload
/// `RENDER_WORLD`-only, so their CPU copy is transient decode traffic. This is
/// what `TileSlot::Ready.bytes` stores and the memory-pressure valve sums.
/// (0.2.0; ≤0.1.x summed the RAW compressed content bytes, undercounting by the
/// meshopt/draco expansion factor of roughly 3–10×.) Unchanged in 0.5 although
/// tile meshes no longer stay in the main world, so the same scene budgets the
/// same cut; `Tiles3dSets::resident_cpu_bytes` reports the heap figure.
pub fn resident_cost_bytes(items: &[DecodedItem]) -> u64 {
    items.iter().map(item_cost_bytes).sum()
}

/// [`resident_cost_bytes`] of one item.
pub(crate) fn item_cost_bytes(item: &DecodedItem) -> u64 {
    match item {
        DecodedItem::Mesh(prim) => {
            let attributes: u64 = prim
                .mesh
                .attributes()
                .map(|(_, values)| values.get_bytes().len() as u64)
                .sum();
            attributes
                + match prim.mesh.indices() {
                    Some(Indices::U16(v)) => (v.len() * 2) as u64,
                    Some(Indices::U32(v)) => (v.len() * 4) as u64,
                    None => 0,
                }
        }
        #[cfg(feature = "points")]
        DecodedItem::Points { points, .. } => {
            (points.len() * std::mem::size_of::<PointCloudData>()) as u64
        }
        #[cfg(feature = "splats")]
        DecodedItem::Splat { gaussians, .. } => {
            (gaussians.len() * std::mem::size_of::<Gaussian3d>()) as u64
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
    /// Per-span decode cost in ms, cut ALONG THE S4 SEAM (offthread-decode
    /// plan S1(b)) — the boundaries are load-bearing for the S4 go/no-go gate:
    /// * `[0]` prep — `split_glb` + `Marks::scan` + parse 1
    ///   (`serde_json::from_slice`) + `decode_meshopt_views` +
    ///   `serde_json::to_vec` + `assemble_glb`. Exactly the movable set.
    /// * `[1]` parse 2 — `gltf::Gltf::from_slice`. Its OWN span, never merged
    ///   into `[0]`: it does not move under S4, and merging would make
    ///   "span 0 dominant ⇒ build S4" unfalsifiable.
    /// * `[2]` geometry — `decode_node`/`decode_primitive` attribute collect +
    ///   `compute_normals` + inline PNG/JPEG decode.
    /// * `[3]` textures — `resolve_pending_textures` (KTX2 transcode). **Wall
    ///   time, not CPU time**: it brackets an `.await`, and on wasm every
    ///   decode task shares one thread (`spawn_local`), so any other tile's
    ///   synchronous work that interleaves while this future is suspended is
    ///   counted here too. It therefore over-reads, without bound, whenever
    ///   more than one tile is in flight. The CPU truth for transcode is the
    ///   host's `window.__tt_ktx2_stats` counter; span 3 alone must never
    ///   decide the S2/S4 gate.
    ///
    /// No inflate span: `.3tz` entries are STORED.
    ///
    /// On the **extracted** route (S5 — the hook returned
    /// [`PreparedTile::meshes`]) spans `[0]` and `[1]` read 0 because both ran
    /// on the hook's thread, and `[2]` measures ONLY the `Mesh` build from the
    /// hook's buffers (`insert_attribute` + `compute_normals`) — no attribute
    /// collect, no image decode. Comparing `[2]` across the two routes is
    /// therefore comparing two different quantities; compare route totals.
    pub stage_ms: [f32; 4],
}

/// Elapsed milliseconds since `t` (span accumulator for [`DecodedTile::stage_ms`]).
fn span_ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

/// Decode a tile, routing by content markers ([`Marks`]): splats bypass the
/// `gltf` crate (see module docs), Draco / `CESIUM_RTC` / meshopt / basisu
/// content is rewritten to vanilla glTF first (the `gltf` crate rejects
/// unknown `extensionsRequired`), everything else goes straight to the `gltf`
/// crate. Async only for the Draco decoder round-trip; plain tiles never
/// yield. `georeferenced` forces the JSON parse — ECEF-tree content can carry
/// planetary node transforms with no marker string to cheaply detect.
///
/// The JSON chunk is parsed AT MOST ONCE and the GLB container rebuilt at most
/// once, however many rewrites the tile needs. (The previous shape re-entered
/// [`decode_glb`] once per extension: a georeferenced meshopt+basisu+features
/// leaf cost five serde parses, two re-serializes, three container rebuilds
/// and ~6 copies of a multi-MB BIN before a vertex was read.)
pub async fn decode_tile(bytes: &[u8], georeferenced: bool) -> Result<DecodedTile, DecodeError> {
    decode_tile_with(bytes, georeferenced, None).await
}

/// [`decode_tile`], but the prep half (the S4 movable set — container split,
/// marker scan, JSON parse 1, meshopt BIN decode, rewrites, container
/// rebuild, feature extraction) can be delegated to a host
/// [`crate::api::TilePrepareHook`] — typically a Web Worker running
/// `bevy_3d_tiles_prepare` in its own wasm module.
///
/// Every fallback route is the inline path that already exists, byte-identical:
/// * `hook = None` → inline (this IS [`decode_tile`]).
/// * hook returns `Ok(None)` → the tile needs a platform decoder (Draco,
///   splats) — inline.
/// * hook returns `Err` → warn once, inline.
///
/// A returned [`PreparedTile`] is consumed WITHOUT re-parsing its JSON: the
/// feature side-band ([`PreparedFeatures`]) replaces the `FeatureCtx` rebuild
/// (main-thread parse count for feature tiles drops from 2 to 1).
///
/// The hook BORROWS `bytes` (no copy): they have to outlive the hook anyway,
/// because every fallback above decodes them inline.
pub async fn decode_tile_with(
    bytes: &[u8],
    georeferenced: bool,
    hook: Option<&Arc<TilePrepareFn>>,
) -> Result<DecodedTile, DecodeError> {
    if let Some(hook) = hook {
        match hook(bytes, georeferenced).await {
            Ok(Some(prepared)) => return decode_prepared(prepared, bytes.len() as u64).await,
            Ok(None) => {} // declined (Draco/splat) — the inline path handles those
            Err(e) => warn_prepare_hook_once(&e.to_string()),
        }
    }
    decode_tile_inline(bytes, georeferenced).await
}

/// One-time warning when the prepare hook errors; the tile still decodes
/// inline (which surfaces a per-tile error with full diagnostics if the
/// content itself is bad).
fn warn_prepare_hook_once(detail: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let detail = detail.to_string();
    ONCE.call_once(move || {
        bevy::log::warn!("tiles3d: tile prepare hook failed ({detail}); decoding inline");
    });
}

/// Decode a hook-prepared tile.
///
/// Two routes, and the hook picks by what it put in [`PreparedTile::meshes`]:
/// * **extracted (S5)** — typed vertex buffers; only the `Mesh` build runs
///   here (span 2), spans 0-1 moved into the hook and read 0;
/// * **prepared GLB (S4)** — the glb is already vanilla glTF, so spans 1-3
///   run here and only span 0 moved.
async fn decode_prepared(
    prepared: PreparedTile,
    content_bytes: u64,
) -> Result<DecodedTile, DecodeError> {
    let mut stage_ms = [0f32; 4];
    let feat = prepared.features.map(FeatSource::from_prepared);
    let mut items = match prepared.meshes {
        Some(meshes) => {
            let t = Instant::now();
            let items = items_from_extracted(meshes, feat.as_ref())?;
            stage_ms[2] += span_ms(t);
            items
        }
        None => decode_vanilla(&prepared.glb, feat.as_ref(), &mut stage_ms)?,
    };
    let t = Instant::now();
    resolve_pending_textures(&mut items).await;
    stage_ms[3] += span_ms(t);
    Ok(DecodedTile {
        items,
        content_bytes,
        rtc_center: prepared.rtc_center.map(DVec3::from_array),
        copyright: prepared.copyright,
        stage_ms,
    })
}

/// The inline (main-thread / in-task) decode pipeline — today's path,
/// untouched by the hook seam.
async fn decode_tile_inline(bytes: &[u8], georeferenced: bool) -> Result<DecodedTile, DecodeError> {
    let mut stage_ms = [0f32; 4];
    let t = Instant::now();
    // Legacy b3dm containers (swisstopo, PLATEAU) unwrap to their embedded
    // glb FIRST — b3dm is not glTF magic, split_glb would misread it as bare
    // JSON. Their rtc offsets are tile-frame (see [`B3dm`]) and rotate into
    // the content frame the compose applies rtc in.
    let b3dm = unwrap_b3dm(bytes)?;
    let (bytes, b3dm_rtc) = match &b3dm {
        Some(b) => (b.glb, b.rtc_center),
        None => (bytes, None),
    };
    let (json_chunk, bin) = split_glb(bytes)?;
    let marks = Marks::scan(json_chunk);
    stage_ms[0] += span_ms(t);

    if !georeferenced && marks.vanilla() {
        let mut items = decode_vanilla(bytes, None, &mut stage_ms)?;
        let t = Instant::now();
        resolve_pending_textures(&mut items).await;
        stage_ms[3] += span_ms(t);
        return Ok(DecodedTile {
            items,
            content_bytes: bytes.len() as u64,
            rtc_center: b3dm_rtc.map(|c| DVec3::from_array(tile_rtc_to_content_frame(c))),
            copyright: None,
            stage_ms,
        });
    }

    let t = Instant::now();
    let mut json: serde_json::Value =
        serde_json::from_slice(json_chunk).map_err(|e| format!("tile json: {e}"))?;
    stage_ms[0] += span_ms(t);
    let copyright = json["asset"]["copyright"].as_str().map(str::to_string);
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
        })
        .map(DVec3::from_array);

    #[cfg(feature = "splats")]
    if marks.splat {
        let items = decode_splat_gltf(&json, bin)?;
        return Ok(DecodedTile {
            items,
            content_bytes: bytes.len() as u64,
            rtc_center,
            copyright,
            stage_ms,
        });
    }

    // Google P3DT bakes ECEF positions into node MATRICES instead of
    // CESIUM_RTC — planetary magnitudes that the gltf crate would truncate
    // to f32. Extract the offset in f64 from the raw JSON and route it
    // through the same side-band channel.
    //
    // Gated on `georeferenced`: this rebase MUTATES the node matrices and hands
    // the removed offset back as `rtc_center`, which only a georeferenced host
    // consumes. An anchored set discards it, so the tile would render rebased
    // with its offset thrown away — i.e. in the wrong place. Before the
    // single-parse rewrite, meshopt/basisu/feature tiles never reached here at
    // all; `!marks.vanilla()` widened the door, so the guard has to be explicit.
    let mut nodes_rebased = false;
    if georeferenced
        && rtc_center.is_none()
        && let Some(center) = extract_planetary_root_offset(&mut json)
    {
        rtc_center = Some(DVec3::from_array(center));
        nodes_rebased = true;
    }

    // Draco is the one ASYNC pass (a platform decoder shim). Run it here so
    // the rewrite below stays synchronous and single-pass.
    let draco = if marks.draco {
        let prims = find_draco_prims(&json);
        let mut decoded = Vec::with_capacity(prims.len());
        for prim in &prims {
            let compressed = buffer_view_slice(&json, bin, prim.buffer_view)?;
            let ids: Vec<u32> = prim.attributes.iter().map(|(_, id)| *id).collect();
            decoded.push(draco::decode(compressed, &ids).await?);
        }
        Some((prims, decoded))
    } else {
        None
    };

    let mut items =
        rewrite_and_decode(json, bytes, bin, marks, draco, nodes_rebased, &mut stage_ms)?;
    let t = Instant::now();
    resolve_pending_textures(&mut items).await;
    stage_ms[3] += span_ms(t);
    Ok(DecodedTile {
        items,
        content_bytes: bytes.len() as u64,
        rtc_center,
        copyright,
        stage_ms,
    })
}

/// Decode a GLB (or self-contained glTF JSON) tile into renderable items.
/// The SYNCHRONOUS entry: Draco content can't be decoded here (its decoder is
/// async) and falls through to the `gltf` crate, which rejects it — use
/// [`decode_tile`] for that. Same single-parse/single-rebuild pipeline.
pub fn decode_glb(bytes: &[u8]) -> Result<Vec<DecodedItem>, DecodeError> {
    let mut discard_ms = [0f32; 4]; // spans reported only via `decode_tile`
    let (json_chunk, bin) = split_glb(bytes)?;
    let marks = Marks::scan(json_chunk);
    if marks.vanilla() {
        return decode_vanilla(bytes, None, &mut discard_ms);
    }
    let json: serde_json::Value =
        serde_json::from_slice(json_chunk).map_err(|e| format!("tile json: {e}"))?;
    rewrite_and_decode(json, bytes, bin, marks, None, false, &mut discard_ms)
}

/// Every synchronous rewrite a tile can need, in ONE pass over the
/// already-parsed document: decode `EXT_meshopt_compression` buffer views,
/// splice pre-decoded Draco primitives, point `KHR_texture_basisu` textures at
/// their standard `source`, strip the handled extensions. Then re-serialize
/// and rebuild the GLB container **exactly once** — and not at all when no
/// pass touched the document — before handing it to the `gltf` crate.
///
/// `json_dirty`: the caller already mutated the document (the planetary
/// root-offset rebase), so the container needs rebuilding even if no pass here
/// fires. `draco`: primitives already decoded by the async shim.
fn rewrite_and_decode(
    mut json: serde_json::Value,
    original: &[u8],
    bin: Option<&[u8]>,
    marks: Marks,
    draco: Option<(Vec<DracoPrim>, Vec<draco::DracoMesh>)>,
    json_dirty: bool,
    stage_ms: &mut [f32; 4],
) -> Result<Vec<DecodedItem>, DecodeError> {
    // meshopt first (T6/D12 — what our mesh tiler emits, and POINTS tiles
    // too): it REBUILDS the BIN, so every later pass reads decoded bytes.
    // Buffer-view indices are preserved, so nothing else has to move.
    let mut new_bin: Option<Vec<u8>> = if marks.meshopt {
        let t = Instant::now();
        let b = decode_meshopt_views(&mut json, bin).map_err(DecodeError::meshopt)?;
        stage_ms[0] += span_ms(t);
        Some(b)
    } else {
        None
    };
    if let Some((prims, decoded)) = draco {
        let current = new_bin.as_deref().or(bin);
        new_bin = Some(splice_draco(&mut json, current, &prims, decoded)?);
    }
    // KTX2/Basis textures (T7): the gltf crate (1.4) doesn't resolve
    // KHR_texture_basisu — the KTX2 image hangs off the texture *extension*,
    // not the standard `source`. JSON-only; the KTX2 bytes stay put and
    // `decode_material` hands them to the transcoder.
    if marks.basisu {
        preprocess_basisu(&mut json);
    }
    // The gltf crate hard-rejects unknown `extensionsRequired`; the RTC center
    // is side-band data by now. Runs on the MARKER, not on "we spliced
    // something" — a document can declare draco/RTC without a usable primitive.
    let stripped = marks.draco || marks.rtc;
    if stripped {
        strip_handled_extensions(&mut json);
    }
    let bin = new_bin.as_deref().or(bin);

    // Splat tiles bypass the gltf crate entirely (see module docs), but only
    // AFTER the meshopt pass so a compressed splat tile still decodes. Without
    // the `splats` feature they fall through to the gltf path, which rejects
    // the unknown required extension — that content simply doesn't show.
    #[cfg(feature = "splats")]
    if marks.splat {
        return decode_splat_gltf(&json, bin).map_err(DecodeError::from);
    }

    let rebuilt;
    let glb: &[u8] = if json_dirty || stripped || marks.meshopt || marks.basisu {
        let t = Instant::now();
        let json_bytes = serde_json::to_vec(&json).map_err(|e| format!("tile splice json: {e}"))?;
        rebuilt = assemble_glb(&json_bytes, bin.unwrap_or(&[]));
        stage_ms[0] += span_ms(t);
        &rebuilt
    } else {
        original
    };

    // Feature metadata (T8): EXT_mesh_features + EXT_structural_metadata. The
    // gltf crate models neither, so they read from the JSON we already parsed —
    // post-rewrite, so the property-table + `_FEATURE_ID_0` accessors line up
    // with the rebuilt BIN. Built once per tile; attached per primitive.
    let feat = if marks.features {
        match FeatureCtx::build(json, bin) {
            Ok(ctx) => Some(FeatSource::from_ctx(ctx)),
            // A malformed table loses picking, never the geometry.
            Err(e) => {
                bevy::log::warn!("tiles3d: feature metadata ignored ({e})");
                None
            }
        }
    } else {
        None
    };

    decode_vanilla(glb, feat.as_ref(), stage_ms)
}

/// Decode a vanilla (no unhandled extension) GLB through the `gltf` crate.
fn decode_vanilla(
    bytes: &[u8],
    feat: Option<&FeatSource>,
    stage_ms: &mut [f32; 4],
) -> Result<Vec<DecodedItem>, DecodeError> {
    // Span 1 (parse 2): its own cut — this parse does NOT move under S4.
    let t = Instant::now();
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
    stage_ms[1] += span_ms(t);
    let doc = gltf.document;
    let blob = gltf.blob;

    let t = Instant::now();
    let mut out = Vec::new();
    let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) else {
        return Ok(out); // empty content tile — legal, renders nothing
    };
    for node in scene.nodes() {
        decode_node(&node, Mat4::IDENTITY, blob.as_deref(), feat, 0, &mut out)?;
    }
    stage_ms[2] += span_ms(t);
    Ok(out)
}

/// Feature-picking source for one tile's decode: the parsed-JSON context on
/// the inline path (accessors read lazily against the BIN), or the
/// worker-materialized arrays of a [`PreparedTile`] — which is exactly what
/// lets the hook path skip re-parsing the JSON. Both funnel into
/// [`ExtractedPrimitive::set_feature_ids`] (the prepare crate's one table
/// builder, which the worker's extraction runs too) so no route can drift.
struct FeatSource {
    /// featureId → source-node path, shared across the tile's primitives.
    node_of_feature: Arc<Vec<String>>,
    kind: FeatKind,
}

enum FeatKind {
    Json(FeatureCtx),
    /// Sorted by key, so a lookup is a binary search.
    Prepared(Vec<((u64, u64), Vec<f32>)>),
}

impl FeatSource {
    fn from_ctx(mut ctx: FeatureCtx) -> Self {
        Self {
            node_of_feature: Arc::new(std::mem::take(&mut ctx.node_of_feature)),
            kind: FeatKind::Json(ctx),
        }
    }

    fn from_prepared(mut f: PreparedFeatures) -> Self {
        // `materialize` sorts; a hand-built hook reply might not.
        f.vertex_ids.sort_unstable_by_key(|(key, _)| *key);
        Self {
            node_of_feature: Arc::new(f.node_of_feature),
            kind: FeatKind::Prepared(f.vertex_ids),
        }
    }

    /// Give primitive `p` its feature tables (and, if it is non-indexed,
    /// indices) from this tile's raw per-vertex ids. No-op when the primitive
    /// carries no feature ids.
    fn attach(&self, bin: Option<&[u8]>, p: &mut ExtractedPrimitive) -> Result<(), String> {
        let key = (p.mesh_ix, p.prim_ix);
        match &self.kind {
            FeatKind::Json(ctx) => {
                if let Some(ids) = ctx.per_vertex_ids(bin, key.0, key.1)? {
                    p.set_feature_ids(&ids);
                }
            }
            FeatKind::Prepared(prims) => {
                if let Ok(i) = prims.binary_search_by_key(&key, |(k, _)| *k) {
                    p.set_feature_ids(&prims[i].1);
                }
            }
        }
        Ok(())
    }

    /// The pick table of a primitive whose tables are built, taken out of it.
    fn take_features(&self, p: &mut ExtractedPrimitive) -> Option<TileFeatures> {
        Some(TileFeatures {
            feature_of_triangle: p.feature_of_triangle.take()?,
            node_of_feature: self.node_of_feature.clone(),
        })
    }
}

// ── Draco splice: moved to `bevy_3d_tiles_prepare` (0.2.1) ──────────────────
// `splice_draco` + its accessor helpers live in the prepare crate now, so a
// host worker that owns its own Draco decoder can splice off-thread
// (`draco_requests` → decode → `prepare_tile_extracting_with_draco`). The
// inline path imports them above; the decode shim itself stays in `draco.rs`.

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
    feat: Option<&FeatSource>,
    depth: usize,
    out: &mut Vec<DecodedItem>,
) -> Result<(), String> {
    // Tile bytes are untrusted network input and the `gltf` crate does not
    // reject a cyclic `children` chain, so this recursion is unbounded without
    // a cap — on wasm it runs on the FRAME thread, where a stack overflow
    // takes the whole page. Same limit as the off-thread extraction's
    // `MAX_NODE_DEPTH`, so a tile that declines there for depth errors here
    // rather than crashing (a >256-deep node chain is not real tile content).
    if depth > 256 {
        return Err("node graph deeper than 256 — cyclic or malformed".into());
    }
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
        decode_node(&child, global, blob, feat, depth + 1, out)?;
    }
    Ok(())
}

fn decode_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: Mat4,
    blob: Option<&[u8]>,
    feat: Option<&FeatSource>,
    mesh_ix: u64,
) -> Result<DecodedPrimitive, String> {
    let reader = primitive.reader(|buffer| resolve_buffer(&buffer, blob));

    // Collected into the SAME buffer struct the off-thread extraction
    // produces, so both routes then run one shared `Mesh` build and cannot
    // drift (offthread-decode plan S5).
    let mut buffers = ExtractedPrimitive {
        transform: transform.to_cols_array(),
        mesh_ix,
        prim_ix: primitive.index() as u64,
        material: primitive.material().index(),
        positions: reader
            .read_positions()
            .ok_or("primitive has no POSITION (or buffer is an external URI)")?
            .collect(),
        normals: reader.read_normals().map(|it| it.collect()),
        uvs: reader.read_tex_coords(0).map(|tc| tc.into_f32().collect()),
        colors: reader.read_colors(0).map(|c| c.into_rgba_f32().collect()),
        indices: reader.read_indices().map(|ix| ix.into_u32().collect()),
        ..Default::default()
    };
    buffers.bounds = bounds_of(&buffers.positions);

    // T8: per-feature picking — the tables from `_FEATURE_ID_0` (raw JSON) in
    // the SAME index order as the mesh below.
    let features = match feat {
        Some(ctx) => {
            ctx.attach(blob, &mut buffers)?;
            ctx.take_features(&mut buffers)
        }
        None => None,
    };

    let mesh = mesh_from_buffers(&mut buffers);
    let material = decode_material(&primitive.material(), blob)?;
    Ok(DecodedPrimitive {
        transform,
        mesh,
        material,
        features,
        bounds: buffers.bounds,
    })
}

/// Build one `Mesh` from plain typed vertex buffers — the ONE place a tile
/// mesh is assembled, shared by the inline `gltf` route and the off-thread
/// extracted route (S5). Consumes the buffers (`take`), so nothing is copied:
/// the feature UV1 the worker built is moved in as-is.
fn mesh_from_buffers(p: &mut ExtractedPrimitive) -> Mesh {
    let vertex_count = p.positions.len();
    let has_uv0 = p.uvs.is_some();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        // MAIN_WORLD + RENDER_WORLD while it is a decoded item: the spawn
        // step copies the pick geometry out of it, then (since 0.5) flips it
        // to RENDER_WORLD unless the set keeps main-world meshes.
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, std::mem::take(&mut p.positions));
    if let Some(uvs) = p.uvs.take() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    // Feature ids ride UV1 so a host material can tint per feature in the
    // fragment stage (see [`TileFeatures`]).
    if let Some(uv1) = p.feature_uv1.take() {
        // UV1 WITHOUT UV0 is a combination bevy 0.18's pbr shader never
        // handles: `pbr_fragment.wgsl` declares `var uv` only under
        // VERTEX_UVS_A but references it in VERTEX_UVS-gated code (defined by
        // EITHER uv set), so pipeline creation fails and the geometry
        // silently vanishes — for ANY StandardMaterial-derived material, not
        // just a host tint. Untextured feature tiles get zero UV0s.
        if !has_uv0 {
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32, 0.0]; vertex_count]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    }
    if let Some(colors) = p.colors.take() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    }
    if let Some(indices) = p.indices.take() {
        mesh.insert_indices(Indices::U32(indices));
    }
    match p.normals.take() {
        Some(n) => mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n),
        // Tiler output may omit normals to save bytes; smooth-compute them.
        // On wasm the decode task IS the frame thread (`spawn_local`), so this
        // is frame time — counted in `DecodedTile::stage_ms[2]`, and on a
        // normal-less tileset it DOMINATES that span even on the extracted
        // route. ponytail: it stays here anyway. Moving it needs a bit-exact
        // copy of bevy's angle-weighted `compute_smooth_normals` (glam
        // `angle_between`/`acos_approx`/`try_normalize`) inside the
        // math-dependency-free prepare crate, and "almost the same normals"
        // is the one failure mode the parity lattice exists to forbid. Do it
        // only if a real flight shows the extracted route is actually taken
        // (read the `extracted / off_thread` counters) AND span 2 is still
        // over budget — then take a `glam` dep rather than hand-rolling it.
        None => mesh.compute_normals(),
    }
    mesh
}

/// The extracted route (S5): meshes straight from the hook's typed buffers —
/// no `gltf` parse, no attribute collect, just `Mesh` assembly. Item order is
/// the extraction's node-traversal order, which is the inline route's.
fn items_from_extracted(
    meshes: ExtractedMeshes,
    feat: Option<&FeatSource>,
) -> Result<Vec<DecodedItem>, String> {
    let ExtractedMeshes {
        mut primitives,
        materials,
    } = meshes;
    let mut out = Vec::with_capacity(primitives.len());
    for p in &mut primitives {
        let features = match feat {
            Some(ctx) => {
                // The worker built the tables (prepare 0.3). A hook that sends
                // only the raw ids (the 0.2 shape) gets them built here. `bin`
                // is unused for a prepared source: there is no BIN on this route.
                if p.feature_uv1.is_none() {
                    ctx.attach(None, p)?;
                }
                ctx.take_features(p)
            }
            None => None,
        };
        // No material index = the glTF default material, which is what the
        // `gltf` crate hands the inline route for the same primitive.
        let material = p
            .material
            .and_then(|ix| materials.get(ix))
            .copied()
            .unwrap_or_default();
        let transform = Mat4::from_cols_array(&p.transform);
        let bounds = p.bounds.or_else(|| bounds_of(&p.positions));
        let mesh = mesh_from_buffers(p);
        out.push(DecodedItem::Mesh(Box::new(DecodedPrimitive {
            transform,
            mesh,
            material: material_from_extracted(&material),
            features,
            bounds,
        })));
    }
    Ok(out)
}

/// Extracted material factors → the decode-time material. Textured tiles never
/// reach this route (the extraction declines any document with an image), so
/// there is nothing to resolve later.
fn material_from_extracted(m: &ExtractedMaterial) -> DecodedMaterial {
    DecodedMaterial {
        base_color: m.base_color,
        metallic: m.metallic,
        roughness: m.roughness,
        double_sided: m.double_sided,
        unlit: m.unlit,
        base_color_image: None,
        base_color_ktx2: None,
        base_color_sampler: ImageSamplerDescriptor::default(),
    }
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

#[cfg(test)]
pub(crate) mod tests {
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

    /// S1(b): `Tiles3dDecodeStats::record` accumulates spans across two
    /// decoded tiles — count, per-span sums, worst-tile total, averages.
    #[test]
    fn decode_stats_accumulate_across_two_tiles() {
        use bevy::tasks::block_on;

        let a = block_on(decode_tile(&tiny_glb(), false)).expect("decode a");
        let b = block_on(decode_tile(&tiny_glb(), false)).expect("decode b");

        let mut stats = crate::Tiles3dDecodeStats::default();
        stats.record(&a);
        stats.record(&b);

        assert_eq!(stats.tiles, 2);
        for i in 0..4 {
            let want = f64::from(a.stage_ms[i]) + f64::from(b.stage_ms[i]);
            assert!((stats.stage_ms[i] - want).abs() < 1e-9, "span {i} sums");
        }
        let worst = a
            .stage_ms
            .iter()
            .sum::<f32>()
            .max(b.stage_ms.iter().sum::<f32>());
        assert_eq!(stats.worst_ms, worst, "worst = max single-tile total");
        for (i, avg) in stats.avg_ms().iter().enumerate() {
            assert!((avg * 2.0 - stats.stage_ms[i]).abs() < 1e-12, "avg {i}");
        }
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
    fn splice_draco_rewrites_draco_primitive() {
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
        // The same sequence `rewrite_and_decode` runs for a draco tile:
        // splice → strip → rebuild the container ONCE.
        let new_bin =
            splice_draco(&mut value, Some(&fake_compressed), &prims, decoded).expect("splice");
        strip_handled_extensions(&mut value);
        let vanilla = assemble_glb(&serde_json::to_vec(&value).unwrap(), &new_bin);
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
    /// NONE → lossless) decodes through `decode_meshopt_views` + the strict gltf
    /// path to **byte-identical** positions/colors and the same triangle set.
    /// The GLB bytes are captured from `@gltf-transform` + `meshoptimizer` (see
    /// the BEVY-3D-TILES T6 commit notes).
    /// The T6 meshopt fixture, shared by the byte-identity test and the
    /// combined single-pass rewrite test below.
    fn meshopt_fixture() -> Vec<u8> {
        use base64::Engine;

        const GLB_B64: &str = "Z2xURgIAAABMBgAAaAUAAEpTT057ImFzc2V0Ijp7ImdlbmVyYXRvciI6ImdsVEYtVHJhbnNmb3JtIHY0LjMuMCIsInZlcnNpb24iOiIyLjAifSwiYWNjZXNzb3JzIjpbeyJ0eXBlIjoiVkVDMyIsImNvbXBvbmVudFR5cGUiOjUxMjYsImNvdW50Ijo2LCJtYXgiOlsyLDMsM10sIm1pbiI6Wy0xLDAsMF0sIm5vcm1hbGl6ZWQiOmZhbHNlLCJieXRlT2Zmc2V0IjowLCJidWZmZXJWaWV3IjowfSx7InR5cGUiOiJWRUM0IiwiY29tcG9uZW50VHlwZSI6NTEyNiwiY291bnQiOjYsIm5vcm1hbGl6ZWQiOmZhbHNlLCJieXRlT2Zmc2V0IjowLCJidWZmZXJWaWV3IjoxfSx7InR5cGUiOiJTQ0FMQVIiLCJjb21wb25lbnRUeXBlIjo1MTI1LCJjb3VudCI6MTIsIm5vcm1hbGl6ZWQiOmZhbHNlLCJieXRlT2Zmc2V0IjowLCJidWZmZXJWaWV3IjoyfV0sImJ1ZmZlclZpZXdzIjpbeyJidWZmZXIiOjEsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjcyLCJ0YXJnZXQiOjM0OTYyLCJieXRlU3RyaWRlIjoxMiwiZXh0ZW5zaW9ucyI6eyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiI6eyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjgwLCJtb2RlIjoiQVRUUklCVVRFUyIsImJ5dGVTdHJpZGUiOjEyLCJjb3VudCI6Nn19fSx7ImJ1ZmZlciI6MSwiYnl0ZU9mZnNldCI6NzIsImJ5dGVMZW5ndGgiOjk2LCJ0YXJnZXQiOjM0OTYyLCJieXRlU3RyaWRlIjoxNiwiZXh0ZW5zaW9ucyI6eyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiI6eyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjgwLCJieXRlTGVuZ3RoIjo5NSwibW9kZSI6IkFUVFJJQlVURVMiLCJieXRlU3RyaWRlIjoxNiwiY291bnQiOjZ9fX0seyJidWZmZXIiOjEsImJ5dGVPZmZzZXQiOjE2OCwiYnl0ZUxlbmd0aCI6NDgsInRhcmdldCI6MzQ5NjMsImV4dGVuc2lvbnMiOnsiRVhUX21lc2hvcHRfY29tcHJlc3Npb24iOnsiYnVmZmVyIjowLCJieXRlT2Zmc2V0IjoxNzYsImJ5dGVMZW5ndGgiOjIyLCJtb2RlIjoiVFJJQU5HTEVTIiwiYnl0ZVN0cmlkZSI6NCwiY291bnQiOjEyfX19XSwiYnVmZmVycyI6W3siYnl0ZUxlbmd0aCI6MjAwfSx7ImJ5dGVMZW5ndGgiOjIxNiwiZXh0ZW5zaW9ucyI6eyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiI6eyJmYWxsYmFjayI6dHJ1ZX19fV0sIm1lc2hlcyI6W3sicHJpbWl0aXZlcyI6W3siYXR0cmlidXRlcyI6eyJQT1NJVElPTiI6MCwiQ09MT1JfMCI6MX0sIm1vZGUiOjQsImluZGljZXMiOjJ9XX1dLCJub2RlcyI6W3sidHJhbnNsYXRpb24iOlsxMCwwLDBdLCJtZXNoIjowfV0sInNjZW5lcyI6W3sibm9kZXMiOlswXX1dLCJleHRlbnNpb25zVXNlZCI6WyJFWFRfbWVzaG9wdF9jb21wcmVzc2lvbiJdLCJleHRlbnNpb25zUmVxdWlyZWQiOlsiRVhUX21lc2hvcHRfY29tcHJlc3Npb24iXX0gyAAAAEJJTgCgAAABAMAAAP8BM/AAAIB/fv8AAAEA8AAA/38BDGAAAIAAAAEA8AAAgIABANAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKAAAAEz8AAA/////wEz8AAAfX59fgAAAT8wAAD/////AT8wAAB+fX59AAABD8AAAP///wEPwAAAfn1+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAIA/AOHwAP4FAgB2h1ZneKmGZYlomAFpAAAAAA==";
        base64::engine::general_purpose::STANDARD
            .decode(GLB_B64)
            .unwrap()
    }

    /// The 6 vertices the fixture round-trips to (lossless meshopt codec).
    const MESHOPT_POSITIONS: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 3.0],
        [-1.0, 3.0, 1.0],
    ];

    #[test]
    fn decodes_meshopt_tile_byte_identical() {
        let glb = meshopt_fixture();

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
        assert_eq!(
            pos, &MESHOPT_POSITIONS,
            "positions must round-trip byte-identical"
        );
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

    /// A tile that needs SEVERAL rewrites at once — meshopt geometry, an
    /// extracted+stripped `CESIUM_RTC`, `asset.copyright`, `georeferenced` — is
    /// decoded in ONE pass: one JSON parse, one container rebuild, in an order
    /// where every later pass sees the meshopt-decoded BIN. Guards the shape
    /// the old per-extension `decode_glb` re-entry hid: a strip that never runs
    /// (the gltf crate then rejects the required extension), a `buffers`
    /// byteLength left describing the compressed BIN, or geometry decoded
    /// against the wrong chunk.
    /// The meshopt fixture wrapped in copyright + a required `CESIUM_RTC` —
    /// exercises every synchronous rewrite at once. Shared by the one-pass
    /// test and the S4 hook-parity test.
    fn combined_fixture() -> Vec<u8> {
        let fixture = meshopt_fixture();
        let (json, bin) = split_glb(&fixture).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(json).unwrap();
        value["asset"]["copyright"] = serde_json::json!("Fixture Co");
        value["extensions"] =
            serde_json::json!({ "CESIUM_RTC": { "center": [6378137.0, -1000.5, 2000.25] } });
        // Both extensions REQUIRED: the strict gltf crate rejects either one
        // surviving into the rebuilt container.
        value["extensionsRequired"] = serde_json::json!(["EXT_meshopt_compression", "CESIUM_RTC"]);
        value["extensionsUsed"] = serde_json::json!(["EXT_meshopt_compression", "CESIUM_RTC"]);
        assemble_glb(&serde_json::to_vec(&value).unwrap(), bin.unwrap())
    }

    #[test]
    fn combined_rewrites_decode_in_one_pass() {
        use bevy::tasks::block_on;

        let glb = combined_fixture();
        let tile = block_on(decode_tile(&glb, true)).expect("combined decode");
        assert_eq!(tile.copyright.as_deref(), Some("Fixture Co"));
        let rtc = tile.rtc_center.expect("rtc center");
        assert!((rtc - DVec3::new(6_378_137.0, -1000.5, 2000.25)).length() < 1e-9);

        assert_eq!(tile.items.len(), 1);
        let DecodedItem::Mesh(p) = &tile.items[0] else {
            panic!("expected mesh")
        };
        // The RTC center is side-band only — it must never reach the f32 transform.
        assert_eq!(
            p.transform.w_axis.truncate(),
            bevy::math::Vec3::new(10.0, 0.0, 0.0)
        );
        // Geometry still decodes from the meshopt BIN, byte-identical.
        let pos = p
            .mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert_eq!(pos, &MESHOPT_POSITIONS);
    }

    /// T8: a GLB with `EXT_mesh_features` (`_FEATURE_ID_0`, FLOAT) + a minimal
    /// `EXT_structural_metadata` STRING property table (the exact shape our
    /// tiler injects). Two triangles, two features; decode must produce a
    /// per-triangle featureId array (index-buffer order) and the node-path
    /// table — the inputs the pick → node → twin resolver consumes.
    /// The `EXT_mesh_features` GLB the feature tests share: 6 verts (2 tris),
    /// tri 0 → feature 0, tri 1 → feature 1, node paths
    /// `["AlphaModule", "BetaModule/sub"]`.
    fn feature_fixture() -> Vec<u8> {
        feature_glb(true, true)
    }

    /// The per-vertex feature ids a decoded mesh carries (`ATTRIBUTE_UV_1`'s
    /// first component), the only place they live since 0.5.
    fn feature_ids_of(mesh: &Mesh) -> Vec<f32> {
        match mesh.attribute(Mesh::ATTRIBUTE_UV_1) {
            Some(bevy::mesh::VertexAttributeValues::Float32x2(v)) => {
                assert!(v.iter().all(|uv| uv[1] == 0.0), "UV1 is [fid, 0]");
                v.iter().map(|uv| uv[0]).collect()
            }
            other => panic!("feature UV1 missing or mistyped: {other:?}"),
        }
    }

    /// [`feature_fixture`] with no `indices`: a NON-indexed TRIANGLES primitive
    /// (vertices `3t..3t+3` are triangle `t`). `with_ids = false` also drops the
    /// primitive's `EXT_mesh_features`, leaving the tile-level table: the
    /// control for "only FEATURE primitives get synthesized indices".
    pub(crate) fn non_indexed_feature_fixture(with_ids: bool) -> Vec<u8> {
        feature_glb(false, with_ids)
    }

    fn feature_glb(indexed: bool, with_ids: bool) -> Vec<u8> {
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
        let mut json = json;
        let prim = json["meshes"][0]["primitives"][0].as_object_mut().unwrap();
        if !indexed {
            prim.remove("indices");
        }
        if !with_ids {
            prim.remove("extensions");
        }
        assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin)
    }

    #[test]
    fn decodes_feature_metadata() {
        let glb = feature_fixture();
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
        // Per-vertex ids are written onto the mesh as UV1, so a host
        // feature-tint material can read them in the fragment stage (0.1.7).
        assert_eq!(feature_ids_of(&p.mesh), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        // This fixture has no TEXCOORD_0 — UV1 without UV0 kills pipeline
        // creation in bevy's pbr shader (`uv` only declared under
        // VERTEX_UVS_A), so the decode must backfill zero UV0s (0.1.8).
        let uv0 = p
            .mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .expect("zero UV0 backfilled alongside feature UV1");
        assert_eq!(uv0.len(), 6);
    }

    /// Hiding a feature rewrites its triangles' INDICES, so every feature
    /// primitive has to be indexed. A non-indexed one gets U32 indices `0..n` on
    /// every route (inline, prepared GLB, extracted): the same triangles in the
    /// same order, now addressable. A primitive without feature ids keeps
    /// whatever the glTF has.
    #[test]
    fn non_indexed_feature_primitive_gets_sequential_u32_indices() {
        use bevy::tasks::block_on;

        for with_ids in [true, false] {
            let glb = non_indexed_feature_fixture(with_ids);
            // The extracted arm below really is the extracted route.
            assert!(
                prepare_tile_extracting(&glb, false)
                    .unwrap()
                    .is_some_and(|p| p.meshes.is_some()),
                "fixture must extract"
            );
            for (route, tile) in [
                ("inline", block_on(decode_tile(&glb, false))),
                (
                    "prepared glb",
                    block_on(decode_tile_with(&glb, false, Some(&canned_hook()))),
                ),
                (
                    "extracted",
                    block_on(decode_tile_with(&glb, false, Some(&canned_extract_hook()))),
                ),
            ] {
                let tile = tile.expect(route);
                let DecodedItem::Mesh(p) = &tile.items[0] else {
                    panic!("{route}: expected mesh")
                };
                if !with_ids {
                    assert!(p.mesh.indices().is_none(), "{route}: no ids, no indices");
                    assert!(p.features.is_none(), "{route}");
                    continue;
                }
                assert!(
                    matches!(p.mesh.indices(), Some(Indices::U32(_))),
                    "{route}: U32 indices"
                );
                assert_eq!(indices_u32(&p.mesh), Some((0..6).collect()), "{route}");
                let f = p.features.as_ref().expect(route);
                assert_eq!(f.feature_of_triangle, vec![0, 1], "{route}");
            }
        }
    }

    /// The hidden-feature GPU write (`index_writes`) writes 4-byte elements over
    /// the mesh's index slot, so every tile mesh with indices must carry U32 —
    /// whatever the glTF stored — and every FEATURE mesh must have indices. On
    /// every decode route.
    #[test]
    fn tile_meshes_are_always_u32_indexed_feature_meshes() {
        use bevy::tasks::block_on;
        // (fixture, is a feature mesh): u16 source indices, u32, non-indexed.
        let fixtures = [
            (tiny_glb(), false),
            (nested_transform_fixture(), false),
            (feature_fixture(), true),
            (non_indexed_feature_fixture(true), true),
        ];
        for (glb, feature) in fixtures {
            for (route, tile) in [
                ("inline", block_on(decode_tile(&glb, false))),
                (
                    "prepared glb",
                    block_on(decode_tile_with(&glb, false, Some(&canned_hook()))),
                ),
                (
                    "extracted",
                    block_on(decode_tile_with(&glb, false, Some(&canned_extract_hook()))),
                ),
            ] {
                for item in tile.expect(route).items {
                    let DecodedItem::Mesh(p) = item else {
                        continue;
                    };
                    assert_eq!(p.features.is_some(), feature, "{route}");
                    assert!(
                        matches!(p.mesh.indices(), Some(Indices::U32(_))),
                        "{route}: {:?}",
                        p.mesh.indices().map(|i| i.len())
                    );
                }
            }
        }
    }

    /// A GLB whose base-color texture is a `KHR_texture_basisu` KTX2 (UASTC,
    /// the writer's exact output). GLB captured from `@gltf-transform` +
    /// `ktx create --encode uastc` (BEVY-3D-TILES T7). Shared by the transcode
    /// test and the S5 decline lattice (a textured tile takes the S4 route).
    fn basisu_fixture() -> Vec<u8> {
        use base64::Engine;

        const GLB_B64: &str = "Z2xURgIAAACUBQAAMAQAAEpTT057ImFzc2V0Ijp7ImdlbmVyYXRvciI6ImdsVEYtVHJhbnNmb3JtIHY0LjMuMCIsInZlcnNpb24iOiIyLjAifSwiYWNjZXNzb3JzIjpbeyJ0eXBlIjoiVkVDMyIsImNvbXBvbmVudFR5cGUiOjUxMjYsImNvdW50IjozLCJtYXgiOlsxLDEsMF0sIm1pbiI6WzAsMCwwXSwiYnVmZmVyVmlldyI6MSwiYnl0ZU9mZnNldCI6MH0seyJ0eXBlIjoiVkVDMiIsImNvbXBvbmVudFR5cGUiOjUxMjYsImNvdW50IjozLCJidWZmZXJWaWV3IjoxLCJieXRlT2Zmc2V0IjoxMn0seyJ0eXBlIjoiU0NBTEFSIiwiY29tcG9uZW50VHlwZSI6NTEyNSwiY291bnQiOjMsImJ1ZmZlclZpZXciOjIsImJ5dGVPZmZzZXQiOjB9XSwiYnVmZmVyVmlld3MiOlt7ImJ1ZmZlciI6MCwiYnl0ZU9mZnNldCI6NzIsImJ5dGVMZW5ndGgiOjI1NH0seyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjAsImJ5dGVMZW5ndGgiOjYwLCJieXRlU3RyaWRlIjoyMCwidGFyZ2V0IjozNDk2Mn0seyJidWZmZXIiOjAsImJ5dGVPZmZzZXQiOjYwLCJieXRlTGVuZ3RoIjoxMiwidGFyZ2V0IjozNDk2M31dLCJzYW1wbGVycyI6W3sid3JhcFMiOjEwNDk3LCJ3cmFwVCI6MTA0OTd9XSwidGV4dHVyZXMiOlt7InNhbXBsZXIiOjAsImV4dGVuc2lvbnMiOnsiS0hSX3RleHR1cmVfYmFzaXN1Ijp7InNvdXJjZSI6MH19fV0sImltYWdlcyI6W3sibmFtZSI6ImJhc2UiLCJtaW1lVHlwZSI6ImltYWdlL2t0eDIiLCJidWZmZXJWaWV3IjowfV0sImJ1ZmZlcnMiOlt7ImJ5dGVMZW5ndGgiOjMyOH1dLCJtYXRlcmlhbHMiOlt7Im5hbWUiOiJtIiwicGJyTWV0YWxsaWNSb3VnaG5lc3MiOnsiYmFzZUNvbG9yVGV4dHVyZSI6eyJpbmRleCI6MH19fV0sIm1lc2hlcyI6W3sicHJpbWl0aXZlcyI6W3siYXR0cmlidXRlcyI6eyJQT1NJVElPTiI6MCwiVEVYQ09PUkRfMCI6MX0sIm1vZGUiOjQsIm1hdGVyaWFsIjowLCJpbmRpY2VzIjoyfV19XSwibm9kZXMiOlt7Im1lc2giOjB9XSwic2NlbmVzIjpbeyJub2RlcyI6WzBdfV0sImV4dGVuc2lvbnNVc2VkIjpbIktIUl90ZXh0dXJlX2Jhc2lzdSJdLCJleHRlbnNpb25zUmVxdWlyZWQiOlsiS0hSX3RleHR1cmVfYmFzaXN1Il19SAEAAEJJTgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAgD8AAAAAAQAAAAIAAACrS1RYIDIwuw0KGgoAAAAAAQAAAAgAAAAIAAAAAAAAAAAAAAABAAAAAQAAAAIAAABoAAAALAAAAJQAAABQAAAAAAAAAAAAAAAAAAAAAAAAAOQAAAAAAAAAGgAAAAAAAABAAAAAAAAAACwAAAAAAAAAAgAoAKYBAgADAwAAEAAAAAAAAAAAAH8AAAAAAAAAAAD/////LAAAAEtUWHdyaXRlcgBrdHggY3JlYXRlIHY0LjQuMiAvIGxpYmt0eCB2NC40LjIAHAAAAEtUWHdyaXRlclNjUGFyYW1zAC0tenN0ZCAxOAAotS/9IECNAABIVwGZ5/87vgEAAgDNjCADRwAA";
        base64::engine::general_purpose::STANDARD
            .decode(GLB_B64)
            .unwrap()
    }

    /// T7: the basisu tile decodes through `preprocess_basisu`, the gltf path
    /// and the async texture-resolve pass. The `gltf` crate can't resolve the
    /// extension and the transcoder isn't callable from the sync decode, so
    /// this proves the source rewrite plus deferred transcode work end-to-end.
    /// On native (this test) the resolve uses bevy's `basis-universal`; we
    /// latch `BC` (as a desktop adapter would) so UASTC becomes BC7 on the CPU.
    #[test]
    fn decodes_basisu_ktx2_base_color() {
        use bevy::tasks::block_on;

        // Pretend the adapter supports BC (desktop WebGPU). OnceLock first-wins;
        // no other test latches it, and it only affects KTX2 decode.
        set_supported_compressed_formats(CompressedImageFormats::BC);

        let glb = basisu_fixture();
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

    #[test]
    fn resident_cost_counts_decoded_buffers_not_raw_len() {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        // 3 vertices: position (3×f32) + uv (2×f32) = 36 + 24 bytes.
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0f32; 3], [1.0; 3], [2.0; 3]],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2], [0.5; 2], [1.0; 2]]);
        mesh.insert_indices(Indices::U32(vec![0, 1, 2])); // 12 bytes
        let items = vec![DecodedItem::Mesh(Box::new(DecodedPrimitive {
            transform: Mat4::IDENTITY,
            mesh,
            material: DecodedMaterial::default(),
            features: None,
            bounds: None,
        }))];
        assert_eq!(resident_cost_bytes(&items), 36 + 24 + 12);
    }

    // ── S4 hook seam (offthread-decode plan): hook path ≡ inline path ───────

    /// Canned in-process prepare hook — exactly what the real worker does
    /// minus the postMessage: `bevy_3d_tiles_prepare::prepare_tile`.
    fn canned_hook() -> Arc<crate::api::TilePrepareFn> {
        Arc::new(|bytes, geo| Box::pin(async move { prepare_tile(bytes, geo) }))
    }

    fn indices_u32(mesh: &Mesh) -> Option<Vec<u32>> {
        match mesh.indices() {
            Some(Indices::U32(v)) => Some(v.clone()),
            Some(Indices::U16(v)) => Some(v.iter().map(|&i| u32::from(i)).collect()),
            None => None,
        }
    }

    /// Byte-level equality of two decode outputs: geometry (positions +
    /// normals byte-for-byte, same indices/transforms), feature picking, and
    /// the side-band (rtc/copyright/content_bytes). `stage_ms` is timing and
    /// deliberately not compared.
    fn assert_tiles_equal(a: &DecodedTile, b: &DecodedTile) {
        assert_eq!(a.items.len(), b.items.len(), "item count");
        assert_eq!(a.content_bytes, b.content_bytes, "content_bytes");
        assert_eq!(a.rtc_center, b.rtc_center, "rtc_center");
        assert_eq!(a.copyright, b.copyright, "copyright");
        for (x, y) in a.items.iter().zip(&b.items) {
            let (DecodedItem::Mesh(x), DecodedItem::Mesh(y)) = (x, y) else {
                panic!("expected mesh items");
            };
            assert_eq!(x.transform, y.transform, "primitive transform");
            for attr in [
                Mesh::ATTRIBUTE_POSITION,
                Mesh::ATTRIBUTE_NORMAL,
                Mesh::ATTRIBUTE_UV_0,
                Mesh::ATTRIBUTE_UV_1,
                Mesh::ATTRIBUTE_COLOR,
            ] {
                assert_eq!(
                    x.mesh.attribute(attr).map(|v| v.get_bytes()),
                    y.mesh.attribute(attr).map(|v| v.get_bytes()),
                    "attribute bytes"
                );
            }
            assert_eq!(x.mesh.attributes().count(), y.mesh.attributes().count());
            assert_eq!(indices_u32(&x.mesh), indices_u32(&y.mesh), "indices");
            // Material factors/flags, and whether a texture resolved at all
            // (the image CONTENTS are the transcoder's business, tested
            // separately — presence is what differs if a route drops one).
            let (mx, my) = (&x.material, &y.material);
            assert_eq!(mx.base_color, my.base_color, "base_color");
            assert_eq!(mx.metallic, my.metallic, "metallic");
            assert_eq!(mx.roughness, my.roughness, "roughness");
            assert_eq!(mx.double_sided, my.double_sided, "double_sided");
            assert_eq!(mx.unlit, my.unlit, "unlit");
            assert_eq!(
                mx.base_color_image.is_some(),
                my.base_color_image.is_some(),
                "base color texture presence"
            );
            assert_eq!(
                mx.base_color_ktx2.is_some(),
                my.base_color_ktx2.is_some(),
                "pending ktx2 presence"
            );
            // Per-vertex feature ids are UV1, compared byte-for-byte above.
            match (&x.features, &y.features) {
                (None, None) => {}
                (Some(fx), Some(fy)) => {
                    assert_eq!(fx.feature_of_triangle, fy.feature_of_triangle);
                    assert_eq!(fx.node_of_feature, fy.node_of_feature);
                }
                _ => panic!("feature presence differs between paths"),
            }
            assert_eq!(x.bounds, y.bounds, "bounds");
        }
    }

    /// S4 gate test (a): the hook path — a canned in-process hook running
    /// `prepare_tile`, the same function the real worker wasm links — decodes
    /// the meshopt+RTC+copyright fixture byte-identically to the inline path.
    #[test]
    fn hook_path_matches_inline_on_meshopt_fixture() {
        use bevy::tasks::block_on;

        let glb = combined_fixture();
        let hook = canned_hook();
        let inline = block_on(decode_tile(&glb, true)).expect("inline decode");
        let hooked = block_on(decode_tile_with(&glb, true, Some(&hook))).expect("hook-path decode");
        assert_tiles_equal(&inline, &hooked);
        // Prep really moved: span 0 is the hook's, so the main-side report is 0.
        assert_eq!(hooked.stage_ms[0], 0.0, "span 0 lives in the hook now");
        // And the plain meshopt fixture (anchored set, no side-band) too.
        let glb = meshopt_fixture();
        let inline = block_on(decode_tile(&glb, false)).expect("inline decode");
        let hooked =
            block_on(decode_tile_with(&glb, false, Some(&hook))).expect("hook-path decode");
        assert_tiles_equal(&inline, &hooked);
    }

    /// S4 feature side-band: the hook reply's `PreparedFeatures` — not a JSON
    /// re-parse — must rebuild picking data identical to the inline path's
    /// `FeatureCtx` route.
    #[test]
    fn hook_path_consumes_prepared_features() {
        use bevy::tasks::block_on;

        let glb = feature_fixture();
        // The prepared reply really carries the feature fields.
        let prepared = prepare_tile(&glb, false).unwrap().expect("accepted");
        let feats = prepared.features.as_ref().expect("features in the reply");
        assert_eq!(
            feats.node_of_feature,
            vec!["AlphaModule".to_string(), "BetaModule/sub".to_string()]
        );
        assert_eq!(feats.vertex_ids.len(), 1, "one feature-carrying primitive");

        let inline = block_on(decode_tile(&glb, false)).expect("inline decode");
        let hooked = block_on(decode_tile_with(&glb, false, Some(&canned_hook())))
            .expect("hook-path decode");
        assert_tiles_equal(&inline, &hooked);
        let DecodedItem::Mesh(p) = &hooked.items[0] else {
            panic!("expected mesh")
        };
        let f = p.features.as_ref().expect("features decoded via hook");
        assert_eq!(f.feature_of_triangle, vec![0, 1]);
        assert_eq!(feature_ids_of(&p.mesh), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    /// S4 gate test (b): a declining hook (`Ok(None)` — the Draco/splat
    /// platform-decoder answer) falls back to the inline path and decodes the
    /// same tile.
    #[test]
    fn declining_hook_falls_back_inline() {
        use bevy::tasks::block_on;

        let decline: Arc<crate::api::TilePrepareFn> =
            Arc::new(|_, _| Box::pin(async { Ok::<_, DecodeError>(None) }));
        let glb = combined_fixture();
        let inline = block_on(decode_tile(&glb, true)).expect("inline decode");
        let fallen =
            block_on(decode_tile_with(&glb, true, Some(&decline))).expect("fallback decode");
        assert_tiles_equal(&inline, &fallen);
        // The fallback IS the inline path — span 0 is back on this side.
        assert!(fallen.stage_ms[0] > 0.0, "inline prep span recorded");
    }

    // ── S5 extracted route (offthread-decode plan): buffers ≡ inline ────────

    /// Canned in-process S5 hook: `prepare_tile_extracting` — prep AND
    /// geometry extraction, the pair the real worker runs in its own wasm.
    fn canned_extract_hook() -> Arc<crate::api::TilePrepareFn> {
        Arc::new(|bytes, geo| Box::pin(async move { prepare_tile_extracting(bytes, geo) }))
    }

    /// A THREE-deep node chain — TRS with a real rotation + non-uniform scale,
    /// then a `matrix` node, then a translated mesh node — plus NORMAL,
    /// TEXCOORD_0, u16 indices and an authored material. Everything the
    /// extracted route composes by hand instead of through `gltf` + `glam`.
    fn nested_transform_fixture() -> Vec<u8> {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals: [[f32; 3]; 3] = [[0.0, 0.0, 1.0]; 3];
        let uvs: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let indices: [u16; 3] = [0, 1, 2];

        let mut bin: Vec<u8> = Vec::new();
        for v in positions.iter().flatten().chain(normals.iter().flatten()) {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let uv_off = bin.len();
        for v in uvs.iter().flatten() {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let idx_off = bin.len();
        for i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [
                // 22.5° about Y, non-uniform scale, offset — T×R×S order matters.
                { "children": [1], "translation": [5.0, -2.0, 0.5],
                  "rotation": [0.0, 0.19509032, 0.0, 0.98078528], "scale": [2.0, 3.0, 4.0] },
                // A `matrix` node in the middle of the chain.
                { "children": [2],
                  "matrix": [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0, 0.25, 0.5, -0.75, 1.0] },
                { "mesh": 0, "translation": [1.0, 2.0, 3.0] }
            ],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                "indices": 3, "mode": 4, "material": 0
            }]}],
            "materials": [{
                "doubleSided": true,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.4, 0.6, 1.0],
                    "metallicFactor": 0.25, "roughnessFactor": 0.75
                }
            }],
            "accessors": [
                { "bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
                { "bufferView": 0, "byteOffset": 36, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" },
                { "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": uv_off },
                { "buffer": 0, "byteOffset": uv_off, "byteLength": idx_off - uv_off },
                { "buffer": 0, "byteOffset": idx_off, "byteLength": 6 }
            ],
            "buffers": [{ "byteLength": bin.len() }],
        });
        assemble_glb(&serde_json::to_vec(&json).unwrap(), &bin)
    }

    /// The transform math is the one part of the extraction written by hand
    /// (the prepare crate has no `glam`), so pin it against the real thing:
    /// a rotated + non-uniformly scaled chain through a `matrix` node must
    /// come out of the extracted route BIT-identical to `gltf` + `glam`.
    /// Materials, NORMAL/UV attributes and u16→u32 index widening ride along.
    #[test]
    fn extracted_route_matches_inline_on_a_nested_transform_chain() {
        use bevy::tasks::block_on;

        let glb = nested_transform_fixture();
        // The route under test really is the extracted one — without this a
        // future decline rule makes the whole test compare inline to inline.
        assert!(
            prepare_tile_extracting(&glb, false)
                .unwrap()
                .is_some_and(|p| p.meshes.is_some()),
            "fixture must extract"
        );
        let inline = block_on(decode_tile(&glb, false)).expect("inline decode");
        let hooked = block_on(decode_tile_with(&glb, false, Some(&canned_extract_hook())))
            .expect("extracted decode");
        assert_tiles_equal(&inline, &hooked);

        let DecodedItem::Mesh(p) = &hooked.items[0] else {
            panic!("expected mesh")
        };
        // Not identity — a fixture that composed to identity would prove nothing.
        assert_ne!(p.transform, Mat4::IDENTITY);
        assert_eq!(p.material.base_color, [0.2, 0.4, 0.6, 1.0]);
        assert_eq!(indices_u32(&p.mesh), Some(vec![0, 1, 2]), "u16 → u32");
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
    }

    /// S5 gate test (a): the extracted route — typed vertex buffers, no glTF
    /// parse on this side at all — decodes byte-identically to the inline
    /// path, on the meshopt fixture (COLOR_0, u32 indices, a TRS node) and on
    /// the meshopt+RTC+copyright+georeferenced one.
    ///
    /// Both fixtures declare `EXT_meshopt_compression` **required**, like our
    /// own tiler's output, so the `meshes.is_some()` guards below are also what
    /// pins `decode_meshopt_views`' extension strip: leave the extension in
    /// `extensionsRequired` and `extract_tile_meshes` declines the document,
    /// silently costing a meshopt scene the whole S5 geometry saving.
    #[test]
    fn extracted_route_matches_inline_on_meshopt_fixtures() {
        use bevy::tasks::block_on;

        let hook = canned_extract_hook();
        for (glb, geo) in [
            (meshopt_fixture(), false),
            (meshopt_fixture(), true),
            (combined_fixture(), true),
        ] {
            // The route under test really is the extracted one.
            assert!(
                prepare_tile_extracting(&glb, geo)
                    .unwrap()
                    .is_some_and(|p| p.meshes.is_some()),
                "fixture must extract (geo={geo})"
            );
            let inline = block_on(decode_tile(&glb, geo)).expect("inline decode");
            let hooked =
                block_on(decode_tile_with(&glb, geo, Some(&hook))).expect("extracted decode");
            assert_tiles_equal(&inline, &hooked);
            // Both movable spans really moved: prep AND the glTF parse are the
            // hook's now, and span 2 is Mesh-from-buffers only.
            assert_eq!(hooked.stage_ms[0], 0.0, "span 0 lives in the hook");
            assert_eq!(hooked.stage_ms[1], 0.0, "no gltf parse on this route");
        }
    }

    /// The extracted reply carries buffers INSTEAD of a container — a rebuilt
    /// GLB nobody parses is wasted CPU on the worker and wasted bytes on the
    /// wire, so `glb` must come back empty.
    #[test]
    fn extracted_reply_carries_buffers_not_a_container() {
        let glb = meshopt_fixture();
        let prepared = prepare_tile_extracting(&glb, false)
            .unwrap()
            .expect("accepted");
        assert!(
            prepared.glb.is_empty(),
            "no container on the extracted route"
        );
        let meshes = prepared.meshes.expect("geometry extracted");
        assert_eq!(meshes.primitives.len(), 1);
        let p = &meshes.primitives[0];
        assert_eq!(p.positions.len(), 6);
        assert_eq!(p.colors.as_ref().map(Vec::len), Some(6));
        assert_eq!(p.indices.as_ref().map(Vec::len), Some(12));
        assert!(p.normals.is_none(), "fixture has no NORMAL");
        // Node translation [10,0,0] flattened into the primitive transform.
        assert_eq!(&p.transform[12..15], &[10.0, 0.0, 0.0]);
        // `prepare_tile` (S4) still returns a container and no buffers.
        let s4 = prepare_tile(&glb, false).unwrap().expect("accepted");
        assert!(!s4.glb.is_empty() && s4.meshes.is_none());
    }

    /// S5 gate test (c): feature picking survives the extracted route — the
    /// per-vertex ids ride the same side-band, and the triangle table is built
    /// from the extracted index buffer.
    #[test]
    fn extracted_route_keeps_feature_picking() {
        use bevy::tasks::block_on;

        let glb = feature_fixture();
        // As above: guard against the test degenerating into inline-vs-inline.
        assert!(
            prepare_tile_extracting(&glb, false)
                .unwrap()
                .is_some_and(|p| p.meshes.is_some()),
            "fixture must extract"
        );
        let inline = block_on(decode_tile(&glb, false)).expect("inline decode");
        let hooked = block_on(decode_tile_with(&glb, false, Some(&canned_extract_hook())))
            .expect("extracted decode");
        assert_tiles_equal(&inline, &hooked);
        let DecodedItem::Mesh(p) = &hooked.items[0] else {
            panic!("expected mesh")
        };
        let f = p
            .features
            .as_ref()
            .expect("features on the extracted route");
        assert_eq!(f.feature_of_triangle, vec![0, 1]);
        // UV1 (feature ids, built by the worker) + the backfilled zero UV0
        // (the shared mesh builder) must be on this route too.
        assert_eq!(feature_ids_of(&p.mesh), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
    }

    /// The worker builds the feature tables; the main thread only MOVES them
    /// onto the mesh. Hand the extracted route tables no recompute could
    /// produce (features 7 and 9), next to the raw per-vertex ids a 0.4
    /// consumer recomputed them from: the mesh UV1 and the pick table must be
    /// the provided ones.
    #[test]
    fn extracted_route_uses_worker_feature_tables() {
        use bevy::mesh::VertexAttributeValues;
        use bevy::tasks::block_on;

        let glb = feature_fixture();
        let mut prepared = prepare_tile_extracting(&glb, false)
            .unwrap()
            .expect("accepted");
        let uv1 = vec![[7.0f32, 0.0]; 6];
        let by_triangle = vec![9u32, 7];
        let p = &mut prepared.meshes.as_mut().expect("extracted").primitives[0];
        p.feature_uv1 = Some(uv1.clone());
        p.feature_of_triangle = Some(by_triangle.clone());
        prepared.features.as_mut().expect("features").vertex_ids =
            vec![((0, 0), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])];

        let tile = block_on(decode_prepared(prepared, glb.len() as u64)).expect("decode");
        let DecodedItem::Mesh(p) = &tile.items[0] else {
            panic!("expected mesh")
        };
        match p.mesh.attribute(Mesh::ATTRIBUTE_UV_1) {
            Some(VertexAttributeValues::Float32x2(v)) => assert_eq!(v, &uv1),
            other => panic!("UV1 missing or mistyped: {other:?}"),
        }
        let f = p.features.as_ref().expect("features");
        assert_eq!(f.feature_of_triangle, by_triangle);
    }

    /// Every route hands spawn a decode-time AABB equal to the mesh's own
    /// position min/max (the extracted route computes it off-thread). The
    /// parity lattice compares it across routes; this pins its value.
    #[test]
    fn bounds_match_mesh_min_max() {
        use bevy::math::Vec3;
        use bevy::tasks::block_on;

        for glb in [
            feature_fixture(),
            nested_transform_fixture(),
            meshopt_fixture(),
        ] {
            for (route, tile) in [
                ("inline", block_on(decode_tile(&glb, false))),
                (
                    "prepared glb",
                    block_on(decode_tile_with(&glb, false, Some(&canned_hook()))),
                ),
                (
                    "extracted",
                    block_on(decode_tile_with(&glb, false, Some(&canned_extract_hook()))),
                ),
            ] {
                for item in tile.expect(route).items {
                    let DecodedItem::Mesh(p) = item else {
                        panic!("{route}: expected mesh")
                    };
                    let pos = p.mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
                    let pos = pos.as_float3().unwrap();
                    let (lo, hi) = pos
                        .iter()
                        .map(|&v| Vec3::from(v))
                        .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), v| {
                            (lo.min(v), hi.max(v))
                        });
                    assert_eq!(p.bounds, Some([lo.to_array(), hi.to_array()]), "{route}");
                }
            }
        }
    }

    /// S5 gate test (b), the decline lattice: content the extraction cannot
    /// reproduce exactly falls back to the S4 container route (or all the way
    /// inline) and renders identically. A textured tile is the live one — it
    /// decodes for real through both routes.
    #[test]
    fn extraction_declines_textured_tile_to_the_s4_route() {
        use bevy::tasks::block_on;

        set_supported_compressed_formats(CompressedImageFormats::BC);
        let glb = basisu_fixture();
        // Declined: buffers absent, container present — the S4 payload.
        let prepared = prepare_tile_extracting(&glb, false)
            .unwrap()
            .expect("accepted");
        assert!(prepared.meshes.is_none(), "a textured tile must decline");
        assert!(!prepared.glb.is_empty(), "declining still prepares the glb");

        let inline = block_on(decode_tile(&glb, false)).expect("inline decode");
        let hooked = block_on(decode_tile_with(&glb, false, Some(&canned_extract_hook())))
            .expect("declined-to-S4 decode");
        assert_tiles_equal(&inline, &hooked);
        let DecodedItem::Mesh(p) = &hooked.items[0] else {
            panic!("expected mesh")
        };
        assert!(
            p.material.base_color_image.is_some(),
            "texture still resolves through the declined route"
        );
    }

    /// The rest of the lattice: Draco and splat content declines the whole
    /// trip (`Ok(None)` — there is no decoder for it here at all), and the
    /// cheap pre-dispatch triage agrees, so a host never pays a round trip to
    /// learn it. Their rendering fallback is covered by
    /// `declining_hook_falls_back_inline`.
    #[test]
    fn extraction_declines_draco_and_splat_content() {
        for json in [
            r#"{"extensionsRequired":["KHR_draco_mesh_compression"]}"#,
            r#"{"extensionsUsed":["KHR_gaussian_splatting"]}"#,
        ] {
            let b = json.as_bytes();
            assert!(
                prepare_tile_extracting(b, true).unwrap().is_none(),
                "{json}"
            );
            assert!(
                bevy_3d_tiles_prepare::extract_would_decline(b, true),
                "{json}"
            );
        }
        // ...and a vanilla tile is NOT declined by the S5 triage, unlike S4's:
        // its geometry is exactly what the trip is for.
        let vanilla = br#"{"asset":{"version":"2.0"}}"#;
        assert!(bevy_3d_tiles_prepare::prepare_would_decline(vanilla, false));
        assert!(!bevy_3d_tiles_prepare::extract_would_decline(
            vanilla, false
        ));
    }

    /// The prepare hook BORROWS the fetched tile: the caller keeps the bytes
    /// alive for the inline fallback anyway, so handing the hook its own copy
    /// was a full-tile allocation + memcpy on the frame thread per request.
    #[test]
    fn hook_borrows_the_fetched_bytes() {
        use bevy::tasks::block_on;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let seen = Arc::new(AtomicUsize::new(0));
        let record = seen.clone();
        let hook: Arc<crate::api::TilePrepareFn> = Arc::new(move |bytes, _| {
            record.store(bytes.as_ptr() as usize, Ordering::SeqCst);
            Box::pin(async { Ok::<_, DecodeError>(None) })
        });
        let glb = tiny_glb();
        block_on(decode_tile_with(&glb, false, Some(&hook))).expect("declined → inline decode");
        assert_eq!(
            seen.load(Ordering::SeqCst),
            glb.as_ptr() as usize,
            "the hook saw the caller's buffer, not a copy of it"
        );
    }

    /// An erroring hook warns once and falls back inline — never fatal.
    #[test]
    fn erroring_hook_falls_back_inline() {
        use bevy::tasks::block_on;

        let broken: Arc<crate::api::TilePrepareFn> =
            Arc::new(|_, _| Box::pin(async { Err(DecodeError::from("canned hook failure")) }));
        let glb = meshopt_fixture();
        let inline = block_on(decode_tile(&glb, false)).expect("inline decode");
        let fallen =
            block_on(decode_tile_with(&glb, false, Some(&broken))).expect("fallback decode");
        assert_tiles_equal(&inline, &fallen);
    }
}
