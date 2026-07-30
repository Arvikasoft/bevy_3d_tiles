//! Public, backend-agnostic API surface for `bevy_3d_tiles`.
//!
//! These types are the seam that keeps the crate free of any application (or
//! SpacetimeDB) coupling. Most are supplied by the host and read by the crate
//! ([`EcefOrigin`], [`TileFeatureResolver`], [`Tiles3dCamera`],
//! [`PointTileMaterial`]); [`TileOwner`] and [`TileGeometry`] run the other way —
//! the crate stamps them onto spawned geometry for the host to react to.
//!
//! In the TurboTwin reference app (`bevy-client`) thin adapter systems map these
//! to the twin-specific machinery (`ProjectOrigin`, `TwinMeshGroup`, the section
//! map, the orbit camera) — but a standalone viewer can ignore all of them and
//! still stream local/relative tilesets.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bevy::math::DMat4;
use bevy::prelude::*;

use bevy_3d_tiles_prepare::{DecodeError, PreparedTile};

/// The crate's `Update` system chain, as public [`SystemSet`]s so a host can
/// order its own systems against it (e.g. run a memory ledger after
/// [`Tiles3dSet::Receive`] and before [`Tiles3dSet::Drive`], instead of
/// racing the chain by a frame). `Receive` runs before `Drive`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tiles3dSet {
    /// Attach/detach intake + draining the async channel: tileset opens, tile
    /// content landing/spawning, decode stats.
    Receive,
    /// Per-frame traversal/selection, request issue (where the
    /// [`TilePrepareHook`] is consulted), eviction, attribution.
    Drive,
}

/// The prepare-hook signature: `(tile GLB bytes, georeferenced)` → future of
/// [`bevy_3d_tiles_prepare::prepare_tile`]'s result. `Ok(None)` = the hook
/// declines (Draco/splat content needing a platform decoder) and the crate
/// decodes inline; `Err` = warn-once, then inline. On wasm the future may hold
/// JS values (a Web Worker round-trip), so it is deliberately non-`Send`
/// there; native hooks run on IO threads and must be `Send`.
#[cfg(target_arch = "wasm32")]
pub type TilePrepareFn =
    dyn Fn(
        Vec<u8>,
        bool,
    ) -> Pin<Box<dyn Future<Output = Result<Option<PreparedTile>, DecodeError>>>>;
/// The prepare-hook signature (native: `Send` — hooks run on IO threads).
#[cfg(not(target_arch = "wasm32"))]
pub type TilePrepareFn = dyn Fn(
        Vec<u8>,
        bool,
    ) -> Pin<Box<dyn Future<Output = Result<Option<PreparedTile>, DecodeError>> + Send>>
    + Send
    + Sync;

/// Host-supplied off-thread tile-prepare hook (offthread-decode plan S4/S5).
/// `None` (the default) = today's inline decode, byte-identical. Insert it
/// *before* `add_plugins(Tiles3dPlugin)` — the same `init_resource` override
/// contract as `Tiles3dConfig`. The crate clones the `Arc` out per request
/// and calls it from the fetch task; it never learns what a Worker is.
///
/// A hook returns one of two payload shapes in the same [`PreparedTile`], and
/// picks by how much work it did off-thread:
/// * `meshes: None` (S4, [`bevy_3d_tiles_prepare::prepare_tile`]) — prepared
///   glTF bytes in `glb`; the crate parses them and collects attributes.
/// * `meshes: Some(_)` (S5,
///   [`bevy_3d_tiles_prepare::prepare_tile_extracting`]) — typed vertex
///   buffers; the crate only builds `Mesh` objects and uploads them, and `glb`
///   is empty. Extraction declines content it cannot reproduce exactly
///   (textures, non-triangle, quantized attributes), which lands back on the
///   first shape — so a hook can always return the richer call and let the
///   fallbacks sort it out.
#[derive(Resource, Default, Clone)]
pub struct TilePrepareHook(pub Option<Arc<TilePrepareFn>>);

// SAFETY: on wasm32-unknown-unknown this build is single-threaded (no wasm
// threads — both wasm modules declare unshared memory; bevy runs its
// single-threaded executor), so the non-`Send` hook is only ever touched from
// the one thread. `Resource` demands `Send + Sync` unconditionally; these
// impls exist solely to satisfy that bound on the target where it is vacuous.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for TilePrepareHook {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for TilePrepareHook {}

/// The ECEF→world transform the host supplies so the crate can place
/// planet-georeferenced tilesets (a `region` root, a planetary-scale bounding
/// volume, or Google Photorealistic 3D Tiles) into the app's local world frame.
///
/// `None` until the host resolves an origin: ECEF-framed tilesets wait, and
/// re-place all resident tiles when it changes (the "rebase" path). A purely
/// local/relative tileset never reads this.
///
/// The host computes `world_from_ecef` however it likes (TurboTwin derives it
/// from its ENU project origin via `world_from_ecef(lat0, lon0, elev0)`); the
/// crate only consumes the resulting matrix.
#[derive(Resource, Default, Clone, Copy, PartialEq, Debug)]
pub struct EcefOrigin {
    pub world_from_ecef: Option<DMat4>,
}

/// Generic "which entity owns this tile geometry" tag the crate inserts on every
/// spawned tile mesh / point / splat entity (carrying the `owner_id` from
/// [`crate::Tiles3dAttach`]). The crate never reads it — it exists so the host
/// can wire selection/highlight/picking back to its own domain. The TurboTwin
/// app runs an `Added<TileOwner>` adapter that mirrors it to `TwinMeshGroup`.
#[derive(Component, Clone, Debug)]
pub struct TileOwner {
    pub id: String,
}

/// Marker the crate inserts on **every** entity it spawns for tile content —
/// mesh primitives (and each per-feature submesh), point clouds, splats —
/// carrying the id of the tileset the geometry streamed from.
///
/// The crate never reads it. It exists so a host can post-process tile geometry
/// **per tileset**: swap in a custom material, apply clipping planes or a
/// cross-section, x-ray/ghost a set, tint by classification, move a set to
/// another render layer. Without it a host cannot even tell which entities are
/// tile geometry — the spawned meshes are otherwise indistinguishable from any
/// other child entity, and walking up to [`crate::Tiles3dTile`] to find out is
/// both awkward and O(depth) per primitive.
///
/// Distinct from [`TileOwner`], which answers *"whose is this?"* and exists only
/// for owner-anchored sets ([`crate::Tiles3dAttach::owner_id`]). `TileGeometry`
/// answers *"which tileset is this?"* and is present on every set, including
/// world-layer / basemap tilesets that have no owner.
///
/// Pair it with [`crate::Tiles3dSets::set_id_for_anchor`] to key host-side state
/// off the tileset the host attached. Tile content spawns **hidden** (the render
/// cut reveals it), so a host system reacting to `Added<TileGeometry>` lands its
/// changes before the geometry is ever drawn — no first-frame flicker.
///
/// ```ignore
/// // Replace the crate's StandardMaterial with an extended one, per tileset.
/// fn extend_tile_materials(
///     mut commands: Commands,
///     added: Query<(Entity, &MeshMaterial3d<StandardMaterial>, &TileGeometry), Added<TileGeometry>>,
///     standard: Res<Assets<StandardMaterial>>,
///     mut extended: ResMut<Assets<ExtendedMaterial<StandardMaterial, MyExt>>>,
///     my_sets: Res<MyPerTilesetState>,
/// ) {
///     for (entity, mat, content) in &added {
///         let Some(state) = my_sets.get(content.set_id) else { continue };
///         let Some(base) = standard.get(&mat.0).cloned() else { continue };
///         let handle = extended.add(ExtendedMaterial { base, extension: state.ext() });
///         commands
///             .entity(entity)
///             .remove::<MeshMaterial3d<StandardMaterial>>()
///             .insert(MeshMaterial3d(handle));
///     }
/// }
/// ```
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGeometry {
    /// Id of the streaming tileset this geometry belongs to. Stable for the
    /// life of the set; a detached-and-reattached tileset gets a fresh id.
    pub set_id: u64,
}

/// Pick-time feature resolution for a tile mesh entity (`EXT_mesh_features`,
/// T8) — the Cesium model: ONE mesh per primitive, feature identity resolved
/// from the HIT, never by splitting geometry per feature. (Splitting was
/// measured at seconds of main-thread hang per refine wave: up to
/// `max_feature_submeshes` mesh builds + GPU uploads per tile. Cesium3DTileFeature
/// works the same way — batch ids + per-feature render state, one draw.)
///
/// The crate never reads it. A host raycaster that knows the hit triangle's
/// index-buffer ordinal resolves `owner_of_feature[feature_of_triangle[tri]]`
/// — the same owner string the per-feature submeshes used to carry in their
/// [`TileOwner`] tags.
#[derive(Component, Clone, Debug)]
pub struct TileFeaturePick {
    /// Index-buffer triangle ordinal → LOCAL feature id.
    pub feature_of_triangle: Vec<u32>,
    /// LOCAL feature id → resolved owner id (host domain — twin id, node
    /// path under an identity resolver, …).
    pub owner_of_feature: Vec<String>,
}

/// Optional per-feature resolver for tiles that carry `EXT_mesh_features`.
///
/// Given the owning tile's id (`anchor`) and **all** of a tile's feature node
/// paths, returns one sub-owner id per path to tag that feature's triangles
/// with. `None` (the default) → no feature splitting: every feature gets the
/// anchor id. Lets the host resolve features to sub-entities (TurboTwin:
/// sub-twins via its section map) without the crate knowing anything about the
/// host's domain.
///
/// Resolving a whole tile's paths in **one** call is deliberate: it lets the
/// host build any per-anchor lookup (e.g. the section map) ONCE per tile and
/// reuse it across the tile's many features, instead of rebuilding it per
/// feature.
#[derive(Resource, Default, Clone)]
pub struct TileFeatureResolver(pub Option<Arc<FeatureResolverFn>>);

/// The resolver signature: `(anchor id, all node paths of one tile)` → one
/// sub-owner id per path.
pub type FeatureResolverFn = dyn Fn(&str, &[&str]) -> Vec<String> + Send + Sync;

impl TileFeatureResolver {
    /// Resolve every feature path of one tile, or fall back to the anchor id
    /// for each path when no resolver is set.
    pub fn resolve(&self, anchor: &str, node_paths: &[&str]) -> Vec<String> {
        match &self.0 {
            Some(f) => f(anchor, node_paths),
            None => node_paths.iter().map(|_| anchor.to_string()).collect(),
        }
    }
}

/// Per-tileset streaming priority class, read live from the tileset's **anchor
/// entity** ([`crate::Tiles3dAttach::anchor`]). `0` = highest; default `1`.
///
/// When the global load pool ([`crate::Tiles3dConfig::max_concurrent_loads`])
/// is contended, class orders requests across ALL sets — but it is **not** the
/// primary key. Any set with nothing Ready yet gets its first (root) request
/// before any set's refinement, class notwithstanding; a class-0 world layer
/// refines without bound, and letting class win would let it starve a class-1
/// twin's root forever. Class then orders *among* starving roots, and
/// separately among refinements, with sets round-robining inside each.
///
/// So the class is for *semantic* precedence (TurboTwin marks ground-context
/// sets — terrain surfaces, background layers — class 0 so the ground appears
/// before twin tilesets fight over the pool), never a starvation lever.
///
/// Host-supplied like [`EcefOrigin`] / [`TileFeatureResolver`]: insert it on
/// the anchor entity, before or after attach. Absent — or on a world-anchored
/// dev set with no anchor — the set streams at the default class 1.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TilePriorityClass(pub u8);

impl Default for TilePriorityClass {
    fn default() -> Self {
        Self(1)
    }
}

/// Per-tileset screen-space-error **multiplier**, read live from the tileset's
/// **anchor entity** — the same entity that carries [`TilePriorityClass`].
///
/// The effective refine threshold of the set becomes `threshold × multiplier`,
/// so `>1.0` coarsens the cut (fewer, coarser tiles selected) and `<1.0`
/// sharpens it. Absent, or ≤0 / non-finite, means `1.0`. Read once per set per
/// frame — this is a per-SET knob, never per-tile.
///
/// This is the **"ground tilesets don't need twin-grade density"** knob. A cut
/// tuned for a twin (texel≈pixel on a machine you are inspecting) is far too
/// fine for ground context: measured 2026-07-30 on a settled close orbit, the
/// ground layers alone put ~4.7 k meshes / 9 Mtri into the main pass, and draw
/// submission — not triangles — was the frame cost. Stamp `TileSseMultiplier`
/// (with `TilePriorityClass(0)`) on ground/background anchors to buy that back;
/// the twins keep the sharp default.
///
/// Distinct from [`crate::Tiles3dAttach::sse_threshold_px`], which sets a set's
/// absolute threshold once at attach. This is a live, relative dial a host can
/// change per frame (a quality slider, a per-class policy) without re-attaching,
/// and it composes with both that override and the memory-pressure valve.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct TileSseMultiplier(pub f32);

impl Default for TileSseMultiplier {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Marker the host inserts on the camera the streamer uses for screen-space-error
/// tile selection. Add it alongside your `Camera3d` (TurboTwin adds it next to
/// the orbit camera). Replaces the former hard-coded `PanOrbitCamera` filter.
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct Tiles3dCamera;

/// Shared material handle for `POINTS`-mode tile content (`points` feature).
///
/// The crate spawns every point-cloud tile entity with this material so the
/// host owns point shading/sizing. The TurboTwin app sets it from its
/// `SharedPointMaterial` (the same material its whole-file `.laz` renditions
/// use); a standalone host inserts its own. Defaults to an empty handle — set
/// it before any `POINTS` tile spawns or those points render with the default
/// material.
#[cfg(feature = "points")]
#[derive(Resource, Default, Clone)]
pub struct PointTileMaterial(
    pub bevy::asset::Handle<bevy_pointcloud::point_cloud_material::PointCloudMaterial>,
);
