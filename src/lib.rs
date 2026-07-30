//! 3D Tiles 1.1 streaming plugin (BEVY-3D-TILES-PLAN, phases T0/T1).
//!
//! One traversal engine for tiled meshes (T0/T1), point clouds (T2), and
//! splats (T3) — the generalization of the basemap streamer's proven
//! selection/fetch machinery from quadtree arithmetic to tileset-defined
//! trees (plan D5):
//!
//! * [`schema`] — `tileset.json` serde model.
//! * [`archive`] — `.3tz` ranged reader (tail-scan → `@3dtilesIndex1@` →
//!   two range-GETs per entry; D2's one-blob-per-asset artifact).
//! * [`traversal`] — flattened tile tree + the per-frame selection algorithm
//!   (per-tile geometricError SSE, zoom-out protection, frame-history
//!   kicking, Urgent/Normal/Preload priorities — plan §7).
//! * [`fetch`] — byte sources (HTTP range / file / memory), Cache-Storage CAS
//!   for tile entries, abort plumbing, and the never-block-the-executor task
//!   spawning discipline.
//! * [`content`] — tile GLB → mesh / point / splat data (plan D5: one
//!   decoder, three renderers).
//! * this module — ECS wiring: per-frame selection, the request scheduler
//!   (priorities recomputed each frame, out-of-cut requests aborted),
//!   time-boxed content spawning, visibility cut, eviction, and the
//!   attach/detach surface the asset loader drives (D6).
//!
//! **Anchoring (T1)**: a tileset attaches to an *anchor entity* — the twin
//! entity (ENU placement + per-frame twin transform) or a preview root. The
//! tileset's root entity is parented under the anchor with the rendition
//! correction as its local transform, so tiles inherit world placement the
//! exact way whole-file scenes do. Selection math runs in the tileset's local
//! frame: the camera is pulled into that frame per set, which keeps SSE exact
//! under rigid/uniform anchor transforms without rebuilding tree volumes.
//! Dev-trigger tilesets (`TT_TILES3D=…` / `?tiles3d=…`) stay world-anchored.
//!
//! ## Error convention (public SDK surface)
//!
//! Public fallible APIs return `thiserror` types, never `String`/`anyhow`:
//! [`fetch::FetchError`] (with cancellation as the distinct
//! [`fetch::FetchError::Aborted`] variant — callers treat an abort differently
//! from a failure), [`archive::ArchiveError`], and
//! [`content::DecodeError`] (staged: permanent content failures vs
//! environmental transcoder-shim failures). Internal helpers may pass plain
//! `String` messages; the typed boundary is the public function signature.

use std::collections::BTreeSet;
use std::sync::Arc;

use bevy::camera::Projection;
use bevy::camera::primitives::{Frustum, Sphere};
use bevy::math::{DMat4, DVec3, Vec3A};
use bevy::prelude::*;
use bevy::window::RequestRedraw;

pub mod api;
pub mod archive;
pub mod content;
pub mod draco;
pub mod fetch;
pub mod geo;
pub mod geodesy;
// wasm-only KTX2 transcode shim binding (T7); native uses bevy basis-universal.
#[cfg(target_arch = "wasm32")]
pub mod ktx2;
pub mod schema;
pub mod traversal;

// The bevy-free CPU half of tile decode (offthread-decode plan S4) — split
// into the sibling `bevy_3d_tiles_prepare` crate so a host worker can link it
// without bevy. Re-exported wholesale (and `meshopt` at its old path) so
// nothing downstream breaks.
pub use bevy_3d_tiles_prepare as prepare;
pub use bevy_3d_tiles_prepare::meshopt;

#[cfg(feature = "points")]
pub use api::PointTileMaterial;
pub use api::{
    EcefOrigin, TileFeaturePick, TileFeatureResolver, TileGeometry, TileOwner, TilePrepareFn,
    TilePrepareHook, TilePriorityClass, TileSseMultiplier, Tiles3dCamera, Tiles3dSet,
};

use archive::Archive3tz;
use content::{DecodedItem, DecodedPrimitive, DecodedTile};
use fetch::{BudgetCounter, ByteSource, ExplodedBase, LiveSession, TilesetSource};
use traversal::{History, SelectParams, TileContent, TileTree, TreeFrame, ZUP_TO_BEVY};

use geodesy::WGS84_EQUATORIAL_RADIUS_M;

// Heavy tile-content renderer types — point clouds (`points`) and Gaussian
// splats (`splats`). The host supplies the `Assets` stores (its own render
// plugins register them) and, for points, the shared [`PointTileMaterial`].
#[cfg(feature = "splats")]
use bevy_gaussian_splatting::{CloudSettings, PlanarGaussian3d, PlanarGaussian3dHandle};
#[cfg(feature = "points")]
use bevy_pointcloud::point_cloud::{PointCloud, PointCloud3d};
#[cfg(feature = "points")]
use bevy_pointcloud::point_cloud_material::PointCloudMaterial3d;

/// The Google Photorealistic 3D Tiles root tileset (D7). The org's API key
/// is appended per request, never stored in the row.
pub const GOOGLE_P3DT_ROOT_URL: &str = "https://tile.googleapis.com/v1/3dtiles/root.json";

/// Committed demo tileset (see `examples/gen_tiles3d_fixture.rs`). The path
/// doubles as a relative URL under Trunk (assets are `copy-dir`'d) and a
/// relative file path for native runs from the crate root.
const FIXTURE_SPEC: &str = "assets/fixtures/tiles3d-demo/tileset.json";

/// App-global streamer tuning, read fresh every traversal.
///
/// **Host override contract:** insert a `Tiles3dConfig` resource *before*
/// `add_plugins(Tiles3dPlugin)` to override any field — [`Tiles3dPlugin::build`]
/// uses `init_resource`, which never overwrites an already-present resource, so
/// the host's values win. This is a supported, tested seam
/// (`host_inserted_config_wins`), not an accident; the TurboTwin host relies on
/// it to lower `max_concurrent_loads`/`max_spawns_per_frame` on wasm.
///
/// These knobs are app-global. For a threshold that differs *per tileset* in one
/// app — a dense single-asset preview framed close vs. the globe basemap — use
/// [`Tiles3dAttach::sse_threshold_px`] instead of mutating this at runtime.
#[derive(Resource, Debug, Clone)]
pub struct Tiles3dConfig {
    /// Refine while a tile's screen-space error exceeds this (px). App-global
    /// default; a set attached with [`Tiles3dAttach::sse_threshold_px`] overrides
    /// it for that set only.
    pub sse_threshold_px: f64,
    /// Distance-relaxed detail falloff (metres) — see
    /// [`SelectParams::detail_falloff_m`]. Caps how far the cut refines toward
    /// the horizon so a grazing view doesn't graft+stream the whole visible
    /// hemisphere (the P3DT "tilt → 98 k-tile tree" finding). `0` disables.
    pub detail_falloff_m: f64,
    /// Max tile fetch+decode tasks in flight, **GLOBAL across every tileset**
    /// (0.2.0; ≤0.1.x applied this per set, so a 20-tileset scene streamed
    /// with 20× the intended concurrency and the decode spikes ratcheted a
    /// wasm host's grows-only heap). Sets are served in iteration order each
    /// frame; earlier sets' tiles going in-flight frees slots for later sets
    /// on the following frames, so everyone converges.
    pub max_concurrent_loads: usize,
    /// Main-thread time box: max decoded tiles turned into entities per frame.
    pub max_spawns_per_frame: usize,
    /// Main-thread time box for RE-spawns: max hidden-tile re-entries turned
    /// back into entities per frame, **GLOBAL across every tileset** (like
    /// `max_concurrent_loads`, unlike the per-set counter this replaced).
    ///
    /// Deliberately NOT `max_spawns_per_frame`: that one boxes DECODE (a wasm
    /// host lowers it to 2 because one tile's GLB+meshopt decode is tens of ms
    /// on the single main thread), while a respawn re-uses the already-decoded
    /// assets and costs a `spawn` plus whatever the host's `Added<…>` adapters
    /// do. Sharing the field made a 23-tileset scene refill a re-entered area
    /// at 2 tiles/set/frame — seconds of visible catch-up under an orbit.
    pub max_respawns_per_frame: usize,
    /// Frames an out-of-cut tile stays resident before eviction (zoom/orbit
    /// in-and-back reuse, mirrors basemap). Generous: P3DT content is never
    /// CAS-cached (ToS), so every eviction is a real re-download and the
    /// view rebuilds coarse-to-fine.
    pub grace_frames: u64,
    /// Hard cap on resident (spawned) tiles per tileset.
    pub max_resident_tiles: usize,
    /// Tree-compaction floor: don't reclaim grafted subtrees until the tree has
    /// at least this many nodes (and has grown ≥50% since the last pass). The
    /// P3DT tree grows monotonically as external tilesets graft in while you
    /// fly — without reclamation it crept 16k→43k nodes in ~30 s, slowing every
    /// per-frame O(tree) pass. The compactor drops whole grafted subtrees that
    /// have been out of view past the grace window; revisiting re-grafts them.
    pub tree_compact_min: usize,
    /// VESTIGIAL since 0.1.6 (kept for struct-literal compatibility): tiles no
    /// longer split into per-feature submeshes at all — every primitive spawns
    /// as one mesh and features resolve at pick time via [`TileFeaturePick`]
    /// (the Cesium model). The split, even capped, cost seconds of
    /// main-thread hang per refine wave.
    pub max_feature_submeshes: usize,
    /// Memory-pressure valve: when the **decoded main-world bytes** of all
    /// resident tiles (summed across tilesets — mesh attributes + indices,
    /// see `content::resident_cost_bytes`) exceed this budget, the effective
    /// SSE threshold inflates by the overshoot ratio (clamped ×8) — the cut
    /// coarsens, `want_visible` shrinks, and eviction can actually reclaim.
    /// Past 1.5× the budget, NEW loads stop starting entirely (the hard
    /// stop): with many tilesets even the coarsest cut has a byte floor no
    /// SSE inflation can go below, and bounded degradation must win over a
    /// wasm host dying with "memory access out of bounds". `0` disables both.
    ///
    /// 0.2.0 semantics change: ≤0.1.x compared this against RAW compressed
    /// content bytes (~3-10× smaller than what actually stays in the heap) —
    /// re-tune stored budgets upward accordingly.
    pub memory_budget_bytes: u64,
    /// Extra pressure multiplier supplied by the HOST's global memory ledger
    /// (its own GLB budgets, wasm heap high-water, …), folded into the SSE
    /// inflation product each traversal (result still clamped ×8). 1.0 = no
    /// external pressure. Lets tiles coarsen when memory is eaten by things
    /// this crate can't see.
    pub external_pressure: f32,
    /// Host-controlled emergency brake: while `true` no NEW tile loads start.
    /// Resident tiles keep rendering, eviction keeps running. For a wasm host
    /// to latch near the 4 GiB linear-memory wall, where another decode spike
    /// would trap the module.
    pub halt_new_loads: bool,
}

/// SSE-threshold inflation for the memory-pressure valve: 1.0 under budget,
/// the overshoot ratio above it, clamped so a pathological set degrades to
/// visibly-coarse rather than unbounded thrash.
fn memory_pressure_factor(resident_bytes: u64, budget_bytes: u64) -> f64 {
    if budget_bytes == 0 || resident_bytes <= budget_bytes {
        return 1.0;
    }
    (resident_bytes as f64 / budget_bytes as f64).min(8.0)
}

impl Default for Tiles3dConfig {
    fn default() -> Self {
        Self {
            // 10 px, below the 3D-Tiles standard 16 (`DEFAULT_SSE_THRESHOLD_PX`):
            // a tile selected at 16 can cover ~16 screen px per geometric-error
            // unit, so its baked texture upscales and reads blurry — especially
            // zoomed out, where each tile fills more screen. 10 pulls one finer
            // level so texel≈pixel (crisper), affordable now compaction bounds
            // the tree. Costs more P3DT requests; raise toward 16 to cut quota.
            sse_threshold_px: 10.0,
            // ~2 km: near terrain stays sharp, the horizon coarsens. Tuned
            // against the live autzen P3DT view (cam ~10–20 m up); raise for
            // high-altitude orbits, lower if the tree still grows too far out.
            detail_falloff_m: 2000.0,
            // 16 (the basemap baseline): on wasm every tile decode runs on the
            // single main thread, so a full-LOD wave of large leaves at 32-wide
            // floods it and stutters. 16 halves that burst while still resolving
            // P3DT's ~20-deep tileset.json graft chain with ample breadth. The
            // real fix for the decode hitch is a worker pool (off-main-thread).
            max_concurrent_loads: 16,
            max_spawns_per_frame: 4,
            // Generous next to the decode box: no fetch, no decode, no asset
            // insert and no GPU upload — a re-entering tile is a `spawn` against
            // handles that are already live. Sized so a whole re-entered ground
            // area (hundreds of tiles) refills in a handful of frames instead of
            // trickling in visibly behind the camera.
            max_respawns_per_frame: 64,
            grace_frames: 600,
            max_resident_tiles: 1024,
            // Comfortable working set before reclaiming: the grace window keeps
            // recently-seen grafts, so this is mostly the floor that stops us
            // compacting tiny trees. Re-grafting a reclaimed area is a real
            // re-fetch (P3DT is never CAS-cached), so don't set it too tight.
            tree_compact_min: 16_384,
            // Generous for real part hierarchies (a valve skid is dozens of
            // features); far below the pathological hundreds-of-export-parts
            // case the cap exists for.
            max_feature_submeshes: 64,
            // Off by default — native address space is effectively unbounded.
            // wasm hosts should set a budget (see the field docs).
            memory_budget_bytes: 0,
            external_pressure: 1.0,
            halt_new_loads: false,
        }
    }
}

/// Google P3DT per-layer config, denormalized from the project row (L3).
#[derive(Debug, Clone)]
pub struct P3dtParams {
    /// Org's Map Tiles API key (client-visible by design, L-D4).
    pub api_key: String,
    /// Hard per-day request stop; 0 = no client-side cap (D7 guardrail).
    pub daily_request_cap: u32,
}

/// Attach a streaming tileset under an anchor entity (D6 resolver routing —
/// sent by the asset loader for `"3dtiles"` renditions, and by the layers
/// resolver for world layers).
#[derive(Message, Debug, Clone)]
pub struct Tiles3dAttach {
    /// Entity the tileset root parents under (twin entity / preview root).
    /// Tile placement = anchor's world transform × `local` × tile transforms.
    /// Georeferenced (ECEF) tilesets ignore the anchor's transform — they
    /// place themselves via the project origin's ENU frame; the anchor only
    /// scopes their lifecycle (detach-by-anchor, GC).
    pub anchor: Entity,
    /// `.3tz` blob URL (SAS-signed) or an exploded `tileset.json` URL.
    pub url: String,
    /// Per-rendition correction transform (pivot/facing/unit fix-up).
    pub local: Transform,
    /// Owning entity id, when anchored to one — spawned tile content gets a
    /// [`TileOwner`] tag so the host's selection / highlight / focus keep
    /// working. `None` for world-anchored / standalone tilesets.
    pub owner_id: Option<String>,
    /// Display label for logs/debug (asset id, twin id…).
    pub label: String,
    /// Google P3DT session config: routes the open through a live, keyed,
    /// budget-capped, never-cached source (D7). `None` = a normal tileset.
    pub p3dt: Option<P3dtParams>,
    /// Per-tileset screen-space-error refine threshold (physical px), overriding
    /// [`Tiles3dConfig::sse_threshold_px`] for this set only. `None` = use the
    /// app-global config value.
    ///
    /// Set this when one app streams tilesets with very different framings. The
    /// config default (10 px) is tuned for a globe basemap seen from far off,
    /// where a low threshold buys crisp texels cheaply. A dense *single-asset*
    /// preview framed close to its bounding radius is the opposite case: that
    /// same 10 px over-refines a 500 k-tri root a level or two into millions of
    /// resident triangles for no visible gain. Raise it (e.g. 24) for such sets
    /// while the basemap keeps the sharp global default.
    pub sse_threshold_px: Option<f64>,
}

/// Tear down any tileset anchored to this entity (rebind / mode switch).
/// Despawning the anchor outright works too — sets garbage-collect when
/// their root entity dies with the hierarchy.
#[derive(Message, Debug, Clone, Copy)]
pub struct Tiles3dDetach {
    pub anchor: Entity,
}

/// Marker on each spawned tile root entity.
#[derive(Component, Debug)]
pub struct Tiles3dTile {
    pub set_id: u64,
    /// Spawn-time tile index. **Debug/`Name` only** — never read for logic, and
    /// it goes stale after `compact_grafted_subtrees` renumbers the tree (the
    /// authoritative tile→entity map is the set's `slots`, which IS remapped).
    pub tile: usize,
}

/// Per-tile load slot.
#[derive(Debug, Clone, Copy)]
enum TileSlot {
    NotLoaded,
    /// Fetch+decode task running; results carry the generation so a
    /// cancelled-then-reissued tile drops the stale payload. The generation
    /// also keys the abort registry (`fetch::trigger_abort`).
    InFlight {
        generation: u64,
    },
    /// Content decoded and RESIDENT (its assets are alive in `Assets<*>`,
    /// held by `ActiveTileset::caches[tile]`).
    Ready {
        /// The spawned tile-root entity, or `None` while the tile is out of the
        /// render cut: a hidden tile gives up its ENTITIES but keeps its assets
        /// (0.2.4 hidden-tile despawn — see [`CachedItem`]). Re-entry respawns
        /// from the cache; only eviction drops the assets.
        entity: Option<Entity>,
        /// Decoded main-world CPU bytes (`content::resident_cost_bytes`) —
        /// what the memory-pressure valve sums; see
        /// [`Tiles3dConfig::memory_budget_bytes`].
        bytes: u64,
    },
    /// Terminal fetch/decode failure — never re-queued this session.
    Failed,
}

/// One resident tile's spawn recipe, built once at decode time and kept for the
/// life of the [`TileSlot::Ready`] slot (0.2.4 hidden-tile despawn).
///
/// Holding these handles is what keeps a hidden tile's meshes/textures in
/// `Assets<*>` — [`Tiles3dSets::resident_content_bytes`] is deliberately
/// unchanged by a despawn, because the memory really is still resident. A
/// re-entering tile therefore costs a `spawn` (the GPU upload reuses the live
/// `RenderAssets`), never a re-download or re-decode; only eviction drops the
/// cache and reclaims the bytes.
///
/// The tile-root transform is NOT cached: it is recomposed at respawn from the
/// CURRENT [`EcefOrigin`], so a tile that was hidden across an origin rebase
/// comes back in the right place.
#[derive(Clone)]
enum CachedItem {
    Mesh {
        mesh: Handle<Mesh>,
        material: Handle<StandardMaterial>,
        transform: Transform,
        pick: Option<TileFeaturePick>,
    },
    #[cfg(feature = "points")]
    Points {
        cloud: Handle<PointCloud>,
        transform: Transform,
    },
    #[cfg(feature = "splats")]
    Splat {
        cloud: Handle<PlanarGaussian3d>,
        transform: Transform,
    },
}

/// How a set's tree coordinates reach Bevy world space.
enum SetFrame {
    /// Set-local frame: the root entity's `GlobalTransform` (anchor chain ×
    /// rendition correction) places the set; selection pulls the camera into
    /// set-local coordinates (T1).
    Anchored,
    /// Tree coordinates are ECEF (T4): placement = the ENU frame at the
    /// project origin, recomputed from absolutes in f64 on origin change
    /// (basemap's rebase model — no accumulated drift; one view, true world
    /// positions — a spaceborne anchor puts ground tiles at their real
    /// height far below, exactly like basemap terrain). `built` = the
    /// origin resident tile transforms were composed at.
    Ecef { built: Option<DMat4> },
}

/// One external-tileset graft, recorded so the tree compactor can drop a stale
/// grafted subtree and restore its graft-point's content for re-fetching.
#[derive(Debug, Clone)]
struct GraftRecord {
    /// The host tile the external tileset was grafted under (its `content.uri`
    /// was cleared at graft time).
    at: usize,
    /// Root of the grafted subtree (the external tileset's root node).
    child_root: usize,
    /// The graft-point's original `content.uri` — restored verbatim if the
    /// subtree is reclaimed, so a later visit re-fetches and re-grafts it.
    uri: String,
}

/// One streaming tileset.
pub struct ActiveTileset {
    id: u64,
    label: String,
    tree: TileTree,
    source: TilesetSource,
    slots: Vec<TileSlot>,
    /// Per-tile spawn recipe of every [`TileSlot::Ready`] tile (empty for every
    /// other state) — see [`CachedItem`]. Index-aligned with `slots`, so it
    /// rides `resize`/`gather` with the rest of the per-tile arrays.
    caches: Vec<Vec<CachedItem>>,
    history: History,
    /// Frame each tile was last in the wanted set (eviction clock).
    last_touched: Vec<u64>,
    /// External tilesets grafted into `tree`, for compaction (reclaim + restore).
    grafts: Vec<GraftRecord>,
    /// `tree.len()` at the last compaction pass — the compactor only re-scans
    /// once the tree has grown ≥50% past this (amortizes its O(tree) cost).
    compact_high_water: usize,
    root_entity: Entity,
    /// Anchor entity when attached via D6 (None = world-anchored dev set).
    anchor: Option<Entity>,
    /// Owning entity id ([`TileOwner`] tagging + placeholder clearing).
    owner_id: Option<String>,
    /// Per-set SSE threshold override (physical px) from
    /// [`Tiles3dAttach::sse_threshold_px`]; `None` falls back to the app-global
    /// [`Tiles3dConfig::sse_threshold_px`] in the traversal.
    sse_threshold_px: Option<f64>,
    /// Whether the anchor's placeholder cube has been stripped yet.
    placeholder_cleared: bool,
    /// Last logged render-cut shape `(tiles, min_depth, max_depth)` —
    /// transitions are the observable trace of LOD swaps.
    last_cut: Option<(usize, u32, u32)>,
    /// Placement frame (T4): anchored set-local vs georeferenced ECEF.
    frame: SetFrame,
    /// Per-tile `CESIUM_RTC` centers (ECEF) — composed into the spawn
    /// transform in f64, kept for origin rebases.
    rtc_centers: Vec<Option<DVec3>>,
    /// Aggregated tile `asset.copyright` fragments (P3DT attribution, D7).
    copyrights: BTreeSet<String>,
    /// Budget-exhausted warning emitted (log once, not per frame).
    budget_warned: bool,
}

impl ActiveTileset {
    fn tree_frame(&self) -> TreeFrame {
        match self.frame {
            SetFrame::Anchored => TreeFrame::Local,
            SetFrame::Ecef { .. } => TreeFrame::Ecef,
        }
    }

    fn is_live(&self) -> bool {
        matches!(self.source, TilesetSource::Live(_))
    }
}

/// Attribution side-band of the streaming tilesets, read by the basemap's
/// overlay system: aggregated tile copyrights (P3DT ToS requires showing
/// them) and whether Google content is on screen (logo requirement).
#[derive(Resource, Default, PartialEq, Eq)]
pub struct TilesetCredits {
    pub lines: Vec<String>,
    pub google_visible: bool,
    /// A georeferenced (ECEF) tileset is rendering a cut — it IS the ground,
    /// so the metric ground grid should hide exactly like it does for the
    /// basemap (read by `basemap::toggle_ground_grid`).
    pub ground_covering: bool,
}

/// Cumulative tile-decode span stats (offthread-decode plan S1(b)) — the
/// host's F3 instrument. Accumulated per landed tile in `receive_tiles3d`
/// from [`content::DecodedTile::stage_ms`]; see that field's doc for the
/// exact span boundaries (they are load-bearing for the S4 go/no-go gate).
///
/// **Read spans 0–2, never span 3 alone.** Span 3 (tex) is wall time around an
/// `.await`, so on wasm it absorbs whatever other tiles decode while it is
/// suspended and over-reads by an unbounded amount. The S2/S4 gate inputs are
/// spans 0–2 plus `window.__tt_ktx2_stats` (the host's synchronous-transcode
/// counter), which is the CPU truth for what span 3 is trying to measure.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct Tiles3dDecodeStats {
    /// Content tiles decoded (subtree grafts don't decode geometry).
    pub tiles: u64,
    /// Cumulative ms per span: `[prep, parse, geom, tex]`.
    pub stage_ms: [f64; 4],
    /// Worst single-tile total decode ms.
    pub worst_ms: f32,
}

impl Tiles3dDecodeStats {
    /// Per-span averages in ms (zeros before the first tile).
    pub fn avg_ms(&self) -> [f64; 4] {
        let n = self.tiles.max(1) as f64;
        self.stage_ms.map(|s| s / n)
    }

    pub(crate) fn record(&mut self, stage_ms: [f32; 4]) {
        self.tiles += 1;
        for (acc, s) in self.stage_ms.iter_mut().zip(stage_ms) {
            *acc += f64::from(s);
        }
        self.worst_ms = self.worst_ms.max(stage_ms.iter().sum());
    }
}

/// Live tilesets + scheduler counters.
#[derive(Resource, Default)]
pub struct Tiles3dSets {
    sets: Vec<ActiveTileset>,
    /// Anchors whose tileset open is still in flight — counted by
    /// [`Tiles3dSets::has_anchor`] so the asset loader doesn't double-attach
    /// while `tileset.json` streams, and cleared by a detach so the landing
    /// open is dropped instead of resurrecting a torn-down anchor.
    pending_anchors: std::collections::HashSet<Entity>,
    /// Anchors whose open failed terminally — never retried for the same
    /// entity (no per-frame retry storms; a respawned twin is a new entity
    /// and gets a fresh attempt).
    failed_anchors: std::collections::HashSet<Entity>,
    frame: u64,
    next_set_id: u64,
    next_generation: u64,
}

impl Tiles3dSets {
    /// Whether `anchor` is taken: streaming, opening, or terminally failed.
    /// The asset loader treats `true` as "nothing to do this frame".
    pub fn has_anchor(&self, anchor: Entity) -> bool {
        self.pending_anchors.contains(&anchor)
            || self.failed_anchors.contains(&anchor)
            || self.sets.iter().any(|s| s.anchor == Some(anchor))
    }

    /// Decoded main-world bytes of every resident tile across all sets — the
    /// same sum the memory-pressure valve uses. For the host's global memory
    /// ledger (feed the overshoot back as
    /// [`Tiles3dConfig::external_pressure`]).
    pub fn resident_content_bytes(&self) -> u64 {
        self.sets
            .iter()
            .flat_map(|s| s.slots.iter())
            .map(|slot| match slot {
                TileSlot::Ready { bytes, .. } => *bytes,
                _ => 0,
            })
            .sum()
    }

    /// Root-volume bounding sphere of the tileset anchored to `anchor`, in
    /// the tileset's local (root-entity) frame — for camera framing. The
    /// caller composes the root entity's `GlobalTransform`, so ECEF sets
    /// (whose volumes are planetary ECEF, not root-entity-local) return
    /// `None` — world layers aren't camera-framing targets.
    pub fn root_volume_for_anchor(&self, anchor: Entity) -> Option<(Entity, Vec3, f32)> {
        let set = self.sets.iter().find(|s| s.anchor == Some(anchor))?;
        if !matches!(set.frame, SetFrame::Anchored) {
            return None;
        }
        let (center, radius) = set.tree.nodes.first()?.volume.bounding_sphere();
        Some((set.root_entity, center.as_vec3(), radius as f32))
    }

    /// Root entity of the **anchored** tileset on `anchor`, whose local
    /// `Transform` is the rendition correction (the asset loader re-applies a
    /// changed correction to it for live alignment — Phase 1 hot-reload). ECEF
    /// (world-layer / P3DT) sets place themselves via the project origin, so
    /// their root carries no editable correction and is excluded.
    pub fn root_entity_for_anchor(&self, anchor: Entity) -> Option<Entity> {
        let set = self.sets.iter().find(|s| s.anchor == Some(anchor))?;
        matches!(set.frame, SetFrame::Anchored).then_some(set.root_entity)
    }

    /// Id of the tileset attached to `anchor`, in **any** frame (anchored or
    /// ECEF/world-layer). It is the same id the crate stamps onto every piece of
    /// that set's geometry as [`TileGeometry::set_id`], so a host can key its own
    /// per-tileset state — materials, clip layers, styling, render layers — off
    /// the anchor it attached and then apply it to the spawned content.
    ///
    /// `None` while the tileset's open is still in flight, or after a detach.
    ///
    /// Deliberately does **not** filter on [`SetFrame::Anchored`] the way
    /// `root_entity_for_anchor` above does. That one is about the editable
    /// rendition correction, which only anchored sets carry. This one is about
    /// identifying a tileset at all — and ECEF world-layer sets (basemaps,
    /// terrain, P3DT) are precisely the ones a host most wants to key state off.
    /// Adding a frame filter here would silently blind hosts to them.
    pub fn set_id_for_anchor(&self, anchor: Entity) -> Option<u64> {
        self.sets
            .iter()
            .find(|s| s.anchor == Some(anchor))
            .map(|s| s.id)
    }
}

/// Anchor info carried through the async tileset open.
#[derive(Debug, Clone)]
struct AttachTarget {
    anchor: Entity,
    local: Transform,
    owner_id: Option<String>,
    sse_threshold_px: Option<f64>,
}

/// What one tile's content fetch produced.
enum TileOutput {
    Content(Box<DecodedTile>),
    /// `content.uri` named another tileset.json (external tileset — the
    /// P3DT tree is built of these): graft it under the tile.
    Subtree(Box<schema::Tileset>),
}

/// Async-task → ECS messages.
enum Tiles3dMsg {
    TilesetOpened {
        label: String,
        attach: Option<AttachTarget>,
        /// Boxed: a parsed tileset tree dwarfs the per-tile variant.
        result: Result<(TilesetSource, Box<schema::Tileset>), String>,
    },
    TileContent {
        set_id: u64,
        /// Resolves to the slot at receive time by matching this generation —
        /// NOT by a captured tile index, which `compact_grafted_subtrees` may
        /// renumber while the fetch is in flight.
        generation: u64,
        result: Result<TileOutput, String>,
    },
}

#[derive(Resource)]
struct Tiles3dChannel {
    tx: crossbeam_channel::Sender<Tiles3dMsg>,
    rx: crossbeam_channel::Receiver<Tiles3dMsg>,
}

impl Default for Tiles3dChannel {
    fn default() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        Self { tx, rx }
    }
}

/// The 3D Tiles streaming plugin.
///
/// Every resource it owns is registered with `init_resource`, which does **not**
/// overwrite a resource the host already inserted. That makes pre-insertion the
/// supported override seam: insert a tuned [`Tiles3dConfig`] (or any host-supplied
/// seam resource) *before* `add_plugins(Tiles3dPlugin)` and the host's value wins.
/// Guarded by the `host_inserted_config_wins` test so a future refactor to
/// `insert_resource` can't silently break it.
pub struct Tiles3dPlugin;

impl Plugin for Tiles3dPlugin {
    fn build(&self, app: &mut App) {
        // `init_resource`, NOT `insert_resource`: a host-inserted config survives
        // (the documented override contract — see the struct + `Tiles3dConfig` docs).
        app.init_resource::<Tiles3dConfig>()
            .init_resource::<Tiles3dSets>()
            .init_resource::<Tiles3dChannel>()
            .init_resource::<TilesetCredits>()
            .init_resource::<Tiles3dDecodeStats>()
            // Host-supplied seams (defaults are inert: no origin, no resolver,
            // no prepare hook). The host overwrites these via its own adapter
            // systems (or pre-inserts them); a standalone viewer leaves them
            // and streams local/relative sets.
            .init_resource::<EcefOrigin>()
            .init_resource::<TileFeatureResolver>()
            .init_resource::<TilePrepareHook>()
            .add_message::<Tiles3dAttach>()
            .add_message::<Tiles3dDetach>()
            .add_systems(Startup, (latch_compressed_formats, init_dev_tileset))
            // The public ordering seam: hosts order against `Tiles3dSet`
            // (Receive → Drive), e.g. a memory ledger between the two.
            .configure_sets(Update, (Tiles3dSet::Receive, Tiles3dSet::Drive).chain())
            .add_systems(
                Update,
                (
                    (apply_attach_detach, receive_tiles3d)
                        .chain()
                        .in_set(Tiles3dSet::Receive),
                    (drive_tiles3d, update_google_logo)
                        .chain()
                        .in_set(Tiles3dSet::Drive),
                ),
            );
        // The shared point material the host sets before any POINTS tile spawns.
        #[cfg(feature = "points")]
        app.init_resource::<PointTileMaterial>();
    }
}

/// Latch the adapter's supported GPU-compressed texture formats for KTX2 tile
/// decode (T7). `CompressedImageFormatSupport` is inserted into the main world
/// by `RenderPlugin::finish` from the render device; absent on a headless build,
/// where KTX2/UASTC transcodes to RGBA8 instead. One-shot — the OnceLock ignores
/// later sets (the latch-don't-toggle discipline from the MSAA work).
fn latch_compressed_formats(support: Option<Res<bevy::image::CompressedImageFormatSupport>>) {
    if let Some(support) = support {
        content::set_supported_compressed_formats(support.0);
    }
}

// ── Tileset opening ──────────────────────────────────────────────────────────

/// The dev trigger: `TT_TILES3D` env var (native) or `?tiles3d=` query param
/// (wasm). Value `fixture`, a `.3tz` path/URL, or a `tileset.json` path/URL.
fn dev_source_spec() -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("TT_TILES3D").ok().filter(|s| !s.is_empty())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let search = web_sys::window()?.location().search().ok()?;
        let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
        params.get("tiles3d").filter(|s| !s.is_empty())
    }
}

fn init_dev_tileset(channel: Res<Tiles3dChannel>) {
    let Some(spec) = dev_source_spec() else {
        return;
    };
    let spec = if spec == "fixture" {
        FIXTURE_SPEC.to_string()
    } else {
        spec
    };
    info!("tiles3d: dev trigger — opening {spec}");
    spawn_tileset_open(spec, None, None, channel.tx.clone());
}

/// Drain attach/detach messages from the asset loader (D6 routing).
fn apply_attach_detach(
    mut attaches: MessageReader<Tiles3dAttach>,
    mut detaches: MessageReader<Tiles3dDetach>,
    channel: Res<Tiles3dChannel>,
    mut sets: ResMut<Tiles3dSets>,
    mut commands: Commands,
) {
    for msg in detaches.read() {
        // Cancel a still-opening attach: when its TilesetOpened lands, the
        // missing pending entry drops it. A rebind also forgives an earlier
        // terminal failure — the new asset deserves its own attempt.
        sets.pending_anchors.remove(&msg.anchor);
        sets.failed_anchors.remove(&msg.anchor);
        let Tiles3dSets { sets, .. } = &mut *sets;
        sets.retain(|set| {
            if set.anchor != Some(msg.anchor) {
                return true;
            }
            info!("tiles3d: detaching {} from {:?}", set.label, msg.anchor);
            abort_in_flight(set);
            if let Ok(mut e) = commands.get_entity(set.root_entity) {
                e.despawn();
            }
            false
        });
    }
    for msg in attaches.read() {
        // One set per anchor: duplicate sends (resolver retries while the
        // open is in flight) are absorbed here.
        if sets.has_anchor(msg.anchor) {
            continue;
        }
        info!(
            "tiles3d: attaching {} ({}) to {:?}",
            msg.label, msg.url, msg.anchor
        );
        sets.pending_anchors.insert(msg.anchor);
        spawn_tileset_open(
            msg.url.clone(),
            msg.p3dt.clone(),
            Some(AttachTarget {
                anchor: msg.anchor,
                local: msg.local,
                owner_id: msg.owner_id.clone(),
                sse_threshold_px: msg.sse_threshold_px,
            }),
            channel.tx.clone(),
        );
    }
}

/// Abort every in-flight request of a set (detach/GC path).
fn abort_in_flight(set: &ActiveTileset) {
    for slot in &set.slots {
        if let TileSlot::InFlight { generation } = slot {
            fetch::trigger_abort(*generation);
        }
    }
}

/// Open a tileset (async): resolve the source kind, fetch + parse the root
/// `tileset.json`, report back on the channel.
fn spawn_tileset_open(
    spec: String,
    p3dt: Option<P3dtParams>,
    attach: Option<AttachTarget>,
    tx: crossbeam_channel::Sender<Tiles3dMsg>,
) {
    fetch::spawn_io(async move {
        let result = open_tileset(&spec, p3dt).await;
        let _ = tx.send(Tiles3dMsg::TilesetOpened {
            label: spec,
            attach,
            result,
        });
    });
}

fn byte_source_for(spec: &str) -> ByteSource {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        ByteSource::Http(spec.to_string())
    } else {
        #[cfg(target_arch = "wasm32")]
        {
            // Relative paths are same-origin URLs in the browser.
            ByteSource::Http(spec.to_string())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            ByteSource::File(spec.into())
        }
    }
}

/// Whether a spec names a packed archive. Checked on the URL *path* — a SAS
/// query string (`…/demo.3tz?se=…&sig=…`) must not hide the extension.
fn is_archive_spec(spec: &str) -> bool {
    spec.split(['?', '#'])
        .next()
        .unwrap_or(spec)
        .ends_with(".3tz")
}

/// Directory prefix (with trailing `/`) of a tileset-relative URI, for
/// rebasing an external tileset's content URIs onto its own location.
/// `None` when the URI has no directory part (root-level subtree).
fn uri_dir_prefix(uri: &str) -> Option<String> {
    let path = uri.split(['?', '#']).next().unwrap_or(uri);
    path.rsplit_once('/').map(|(dir, _)| format!("{dir}/"))
}

/// Whether fetched content bytes are an external `tileset.json` rather than
/// tile geometry. By CONTENT, never by URI: P3DT serves both from
/// extensionless paths. A tileset is JSON with `root` + `geometricError`
/// (no glTF document has the latter); GLBs carry the `glTF` magic.
fn looks_like_external_tileset(bytes: &[u8]) -> bool {
    if bytes.starts_with(b"glTF") {
        return false;
    }
    let first = bytes.iter().find(|b| !b.is_ascii_whitespace());
    first == Some(&b'{')
        && content::memmem(bytes, b"\"geometricError\"")
        && content::memmem(bytes, b"\"root\"")
}

/// Walk a parsed tileset for the first content URI carrying a `session`
/// query param and adopt it into the live session (the P3DT protocol: the
/// root response embeds the token in its child URIs; every subsequent
/// request must echo it).
fn adopt_session(live: &LiveSession, tileset: &schema::Tileset) {
    fn find(tile: &schema::Tile) -> Option<String> {
        if let Some(content) = &tile.content
            && let Some(session) = fetch::extract_session_param(&content.uri)
        {
            return Some(session);
        }
        tile.children.iter().find_map(find)
    }
    if let Some(session) = find(&tileset.root) {
        let fresh = !live.has_session();
        live.set_session(session);
        if fresh {
            info!("tiles3d: P3DT session established");
        }
    }
}

async fn open_tileset(
    spec: &str,
    p3dt: Option<P3dtParams>,
) -> Result<(TilesetSource, Box<schema::Tileset>), String> {
    if let Some(p3dt) = p3dt {
        // Live sessioned endpoint (Google P3DT, D7): keyed, budget-capped,
        // never CAS-cached. The root fetch is the billed "root request".
        let budget = BudgetCounter::new(p3dt.daily_request_cap, Some("p3dt"));
        let live = Arc::new(LiveSession::new(spec, p3dt.api_key, budget));
        let source = TilesetSource::Live(live.clone());
        let bytes = source
            .read_entry_cached(spec, None)
            .await
            .map_err(|e| format!("fetch P3DT root: {e}"))?;
        let tileset = schema::parse_tileset(&bytes).map_err(|e| format!("parse P3DT root: {e}"))?;
        adopt_session(&live, &tileset);
        return Ok((source, Box::new(tileset)));
    }
    let source = if is_archive_spec(spec) {
        let archive = Archive3tz::open(byte_source_for(spec))
            .await
            .map_err(|e| format!("open 3tz: {e}"))?;
        TilesetSource::Archive(Arc::new(archive))
    } else {
        // `…/tileset.json` (or a bare base): entries resolve against the base.
        let base = spec.strip_suffix("tileset.json").unwrap_or(spec);
        let base = base.trim_end_matches('/');
        let exploded = if base.starts_with("http://") || base.starts_with("https://") {
            ExplodedBase::Url(base.to_string())
        } else {
            #[cfg(target_arch = "wasm32")]
            {
                ExplodedBase::Url(base.to_string())
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                ExplodedBase::Dir(base.into())
            }
        };
        TilesetSource::Exploded(exploded)
    };
    let bytes = source
        .read_entry_cached("tileset.json", None)
        .await
        .map_err(|e| format!("fetch tileset.json: {e}"))?;
    let tileset = schema::parse_tileset(&bytes).map_err(|e| format!("parse tileset.json: {e}"))?;
    Ok((source, Box::new(tileset)))
}

// ── ECS drain: tilesets + decoded tile content ───────────────────────────────

/// Drain async results into the ECS, time-boxed: at most
/// `max_spawns_per_frame` content spawns per frame (§7's main-thread budget);
/// the rest stay queued in the channel for the next frame.
#[allow(clippy::too_many_arguments)]
fn receive_tiles3d(
    channel: Res<Tiles3dChannel>,
    config: Res<Tiles3dConfig>,
    origin: Res<EcefOrigin>,
    // Per-feature owner resolver: at spawn, a feature tile's triangles are
    // tagged with their resolved owner id ([`TileOwner`]) so the host's
    // click/hover/outline machinery treats each feature as its own entity.
    // Inert by default — every feature falls back to the tile's anchor owner.
    resolver: Res<TileFeatureResolver>,
    mut sets: ResMut<Tiles3dSets>,
    mut decode_stats: ResMut<Tiles3dDecodeStats>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    #[cfg(feature = "points")] mut clouds: ResMut<Assets<PointCloud>>,
    #[cfg(feature = "points")] point_material: Res<PointTileMaterial>,
    #[cfg(feature = "splats")] mut splats: ResMut<Assets<PlanarGaussian3d>>,
    mut commands: Commands,
) {
    let mut spawned = 0usize;
    while spawned < config.max_spawns_per_frame {
        let Ok(msg) = channel.rx.try_recv() else {
            break;
        };
        match msg {
            Tiles3dMsg::TilesetOpened {
                label,
                attach,
                result,
            } => match result {
                Ok((source, tileset)) => {
                    // An anchored open must still be wanted (not detached
                    // while in flight) and its anchor must still exist (the
                    // twin/preview may have despawned during the fetch).
                    if let Some(a) = &attach {
                        let still_pending = sets.pending_anchors.remove(&a.anchor);
                        if !still_pending || commands.get_entity(a.anchor).is_err() {
                            info!("tiles3d: {label}: anchor gone before open finished — dropping");
                            continue;
                        }
                    }
                    let anchor = attach.as_ref().map(|a| a.anchor);
                    // Frame decision (T4): live P3DT and detected
                    // georeferenced tilesets are ECEF trees placed via the
                    // project origin's ENU frame; everything else is a
                    // local-metres Z-up set placed by its anchor entity.
                    let georef = matches!(source, TilesetSource::Live(_))
                        || geo::tileset_is_georeferenced(&tileset);
                    let (frame, world_from_tileset, tree_frame) = if georef {
                        (
                            SetFrame::Ecef { built: None },
                            DMat4::IDENTITY,
                            TreeFrame::Ecef,
                        )
                    } else {
                        (SetFrame::Anchored, ZUP_TO_BEVY, TreeFrame::Local)
                    };
                    match TileTree::build(&tileset, world_from_tileset, tree_frame) {
                        Ok(tree) => {
                            let n = tree.len();
                            let id = sets.next_set_id;
                            sets.next_set_id += 1;
                            // ECEF sets are NOT parented under the anchor:
                            // their placement comes from the ENU frame, and
                            // an anchor transform (twin placement) must not
                            // shift them. The anchor still scopes lifecycle.
                            let mut root = commands.spawn((
                                Name::new(format!("Tiles3d({label})")),
                                Visibility::default(),
                            ));
                            if georef {
                                root.insert(Transform::IDENTITY);
                            } else {
                                root.insert(attach.as_ref().map(|a| a.local).unwrap_or_default());
                                if let Some(a) = &attach {
                                    root.insert(ChildOf(a.anchor));
                                }
                            }
                            let root_entity = root.id();
                            let mut history = History::default();
                            history.resize(n);
                            info!(
                                "tiles3d: {label}: {n} tiles{}",
                                if georef {
                                    " (georeferenced — ECEF frame)"
                                } else {
                                    ""
                                }
                            );
                            sets.sets.push(ActiveTileset {
                                id,
                                label,
                                tree,
                                source,
                                slots: vec![TileSlot::NotLoaded; n],
                                caches: vec![Vec::new(); n],
                                history,
                                last_touched: vec![0; n],
                                grafts: Vec::new(),
                                compact_high_water: n,
                                root_entity,
                                anchor: attach.as_ref().map(|a| a.anchor),
                                sse_threshold_px: attach.as_ref().and_then(|a| a.sse_threshold_px),
                                owner_id: attach.and_then(|a| a.owner_id),
                                placeholder_cleared: false,
                                last_cut: None,
                                frame,
                                rtc_centers: vec![None; n],
                                copyrights: BTreeSet::new(),
                                budget_warned: false,
                            });
                        }
                        Err(e) => {
                            if let Some(anchor) = anchor {
                                sets.failed_anchors.insert(anchor);
                            }
                            error!("tiles3d: {label}: unusable tileset: {e}");
                        }
                    }
                }
                Err(e) => {
                    if let Some(a) = &attach {
                        sets.pending_anchors.remove(&a.anchor);
                        sets.failed_anchors.insert(a.anchor);
                    }
                    error!("tiles3d: {label}: {e}");
                }
            },
            Tiles3dMsg::TileContent {
                set_id,
                generation,
                result,
            } => {
                let Some(set) = sets.sets.iter_mut().find(|s| s.id == set_id) else {
                    continue;
                };
                // Resolve the slot by GENERATION, never the message's captured
                // tile index: `compact_grafted_subtrees` renumbers the tree, so
                // a result landing after a compaction carries a stale (possibly
                // out-of-range) index — indexing `slots[tile]` with it panicked.
                // The generation is globally unique, so this finds the in-flight
                // slot at its CURRENT index and naturally drops a cancelled,
                // reissued, or compacted-away payload (no matching InFlight gen).
                let Some(tile) = set.slots.iter().position(
                    |s| matches!(s, TileSlot::InFlight { generation: g } if *g == generation),
                ) else {
                    continue;
                };
                match result {
                    Ok(TileOutput::Subtree(external)) => {
                        spawned += 1;
                        if let TilesetSource::Live(live) = &set.source {
                            adopt_session(live, &external);
                        }
                        let consumed_uri = set.tree.nodes[tile].content_uri.clone();
                        match set.tree.graft(tile, &external, set.tree_frame()) {
                            Ok(new_root) => {
                                // Per spec, an external tileset's relative
                                // content URIs resolve against ITS location,
                                // not the host root — rebase them onto the
                                // subtree's directory. (P3DT URIs are
                                // absolute paths; untouched.)
                                if let Some(prefix) =
                                    consumed_uri.as_deref().and_then(uri_dir_prefix)
                                {
                                    for node in &mut set.tree.nodes[new_root..] {
                                        if let Some(u) = &node.content_uri
                                            && !u.starts_with('/')
                                            && !u.contains("://")
                                        {
                                            node.content_uri = Some(format!("{prefix}{u}"));
                                        }
                                    }
                                }
                                // The graft consumed the content: the tile is
                                // a plain interior node now; its subtree's
                                // slots ride the same per-tile arrays.
                                set.tree.nodes[tile].content_uri = None;
                                let n = set.tree.len();
                                set.slots.resize(n, TileSlot::NotLoaded);
                                set.caches.resize(n, Vec::new());
                                set.last_touched.resize(n, 0);
                                set.rtc_centers.resize(n, None);
                                set.history.resize(n);
                                set.slots[tile] = TileSlot::NotLoaded;
                                // Record the graft so the compactor can later
                                // reclaim this subtree and restore the host's
                                // content for re-fetching (consumed_uri is Some —
                                // the tile had external-tileset content).
                                if let Some(uri) = consumed_uri {
                                    set.grafts.push(GraftRecord {
                                        at: tile,
                                        child_root: new_root,
                                        uri,
                                    });
                                }
                                info!(
                                    "tiles3d: {}: external tileset grafted at tile {tile} \
                                     (tree now {n} tiles)",
                                    set.label
                                );
                            }
                            Err(e) => {
                                error!(
                                    "tiles3d: {}: unusable external tileset at tile {tile}: {e}",
                                    set.label
                                );
                                set.slots[tile] = TileSlot::Failed;
                            }
                        }
                    }
                    Ok(TileOutput::Content(decoded)) => {
                        spawned += 1;
                        let DecodedTile {
                            items,
                            content_bytes: _,
                            rtc_center,
                            copyright,
                            stage_ms,
                        } = *decoded;
                        decode_stats.record(stage_ms);
                        set.rtc_centers[tile] = rtc_center;
                        if let Some(c) = copyright {
                            for frag in c.split(';') {
                                let frag = frag.trim();
                                if !frag.is_empty() {
                                    set.copyrights.insert(frag.to_string());
                                }
                            }
                        }
                        // ECEF sets compose placement against the CURRENT
                        // origin in f64; tiles landing before the origin
                        // resolves wait (re-requested once it exists).
                        let Some(transform) =
                            tile_spawn_transform(set, tile, origin.world_from_ecef)
                        else {
                            set.slots[tile] = TileSlot::NotLoaded;
                            continue;
                        };
                        let renderers = ContentRenderers {
                            #[cfg(feature = "points")]
                            clouds: &mut clouds,
                            #[cfg(feature = "splats")]
                            splats: &mut splats,
                            _marker: std::marker::PhantomData,
                        };
                        // Resident cost = decoded main-world bytes, measured
                        // from the actual buffers (not the raw content len —
                        // see `content::resident_cost_bytes`).
                        let resident_cost = content::resident_cost_bytes(&items);
                        let cache = build_tile_cache(
                            &mut meshes,
                            &mut materials,
                            &mut images,
                            renderers,
                            &resolver,
                            set,
                            items,
                        );
                        // Spawned HIDDEN; `drive_tiles3d` flips it visible in
                        // this same frame if the cut selects it (the documented
                        // `Added<TileGeometry>` window for host adapters).
                        let entity = spawn_tile_entities(
                            &mut commands,
                            #[cfg(feature = "points")]
                            &point_material,
                            set,
                            tile,
                            transform,
                            &cache,
                            Visibility::Hidden,
                        );
                        set.caches[tile] = cache;
                        set.slots[tile] = TileSlot::Ready {
                            entity: Some(entity),
                            bytes: resident_cost,
                        };
                    }
                    Err(e) => {
                        warn!(
                            "tiles3d: {}: tile {tile} ({:?}) failed terminally: {e}",
                            set.label, set.tree.nodes[tile].content_uri
                        );
                        set.slots[tile] = TileSlot::Failed;
                    }
                }
            }
        }
    }
}

/// A tile's spawn transform in its set's frame. Anchored sets: the
/// precomposed `world_from_content` (the root entity's transform chain
/// finishes placement). ECEF sets: `world_from_ecef × ecef_from_content ×
/// rtc` composed **in f64** against the current origin — the planetary
/// magnitudes cancel before the f32 cast (the altitude-anchor jitter
/// lesson). `None` = the origin isn't resolved yet — the caller re-queues
/// the tile.
fn tile_spawn_transform(
    set: &ActiveTileset,
    tile: usize,
    origin: Option<DMat4>,
) -> Option<Transform> {
    let node = &set.tree.nodes[tile];
    match set.frame {
        SetFrame::Anchored => Some(Transform::from_matrix(node.world_from_content.as_mat4())),
        SetFrame::Ecef { .. } => {
            let o = origin?;
            let m = compose_ecef_tile_matrix(o, node.world_from_content, set.rtc_centers[tile]);
            Some(Transform::from_matrix(m.as_mat4()))
        }
    }
}

/// Which out-of-cut tiles must NOT give up their entities this frame, and
/// whether any wanted tile is still waiting for one (0.2.4 hidden-tile despawn).
///
/// A wanted tile with `entity: None` has nothing of itself on screen — it is
/// starved by the respawn budget, or waiting for the origin — so whatever is
/// currently covering its screen area has to stay. Under REPLACE refinement
/// that is exactly its
/// **ancestors** (zooming in: the coarse parent holds until the children land)
/// and its **descendants** (zooming out: the fine children hold until the parent
/// lands). Everything else in the set is unrelated screen area and leaves on the
/// frame it leaves the cut — which is the entire point of despawning.
///
/// A set-wide hold instead of this one was the first cut of the feature and it
/// undid the feature: while ANY tile was starved, every out-of-cut tile of that
/// set stayed spawned AND VISIBLE, so `drawn` climbed toward `resident` for as
/// long as the camera moved and a refining parent drew on top of its own
/// children. Under a host budget of 2 respawns per frame that state was
/// effectively permanent during motion.
fn refinement_hold(
    tree: &TileTree,
    want_visible: &[bool],
    slots: &[TileSlot],
) -> (Vec<bool>, bool) {
    let mut hold = vec![false; tree.len()];
    let mut starved = false;
    let mut down: Vec<usize> = Vec::new();
    for i in 0..tree.len() {
        if !want_visible[i] || !matches!(slots[i], TileSlot::Ready { entity: None, .. }) {
            continue;
        }
        starved = true;
        let mut up = tree.nodes[i].parent;
        while let Some(a) = up {
            hold[a] = true;
            up = tree.nodes[a].parent;
        }
        down.extend(tree.nodes[i].children.iter().copied());
        while let Some(d) = down.pop() {
            if hold[d] {
                continue; // subtree already marked
            }
            hold[d] = true;
            down.extend(tree.nodes[d].children.iter().copied());
        }
    }
    (hold, starved)
}

/// `world_from_ecef(origin) × ecef_from_content × T(rtc_center)`, in f64.
fn compose_ecef_tile_matrix(
    world_from_ecef: DMat4,
    ecef_from_content: DMat4,
    rtc_center: Option<DVec3>,
) -> DMat4 {
    let mut m = world_from_ecef * ecef_from_content;
    if let Some(rtc) = rtc_center {
        m *= DMat4::from_translation(rtc);
    }
    m
}

/// Heavy-renderer asset stores threaded into [`build_tile_cache`] for
/// point-cloud (`points`) and Gaussian-splat (`splats`) tile content. Each
/// field exists only under its feature; with neither, this is just the
/// lifetime marker. The host's render plugins own the `Assets` stores.
struct ContentRenderers<'a> {
    #[cfg(feature = "points")]
    clouds: &'a mut Assets<PointCloud>,
    #[cfg(feature = "splats")]
    splats: &'a mut Assets<PlanarGaussian3d>,
    _marker: std::marker::PhantomData<&'a ()>,
}

/// Decode-side half: insert one tile's decoded items into the asset stores
/// ONCE and return the [`CachedItem`] spawn recipe. The returned handles are
/// what keep the tile resident across hidden-tile despawns — dropping them
/// (eviction) is what actually reclaims the memory.
#[allow(clippy::too_many_arguments)]
// `renderers` is only read by the cfg-gated point/splat arms; with neither
// feature it's unused. Scope the allow to that config so a genuinely unused
// param in the always-on mesh path is still caught.
#[cfg_attr(
    not(any(feature = "points", feature = "splats")),
    allow(unused_variables)
)]
fn build_tile_cache(
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    renderers: ContentRenderers<'_>,
    resolver: &TileFeatureResolver,
    set: &ActiveTileset,
    items: Vec<DecodedItem>,
) -> Vec<CachedItem> {
    let anchor = set.owner_id.as_deref();
    // T8 highlight: a feature tile under an owner resolves its feature paths to
    // host sub-owners via the host [`TileFeatureResolver`], so the host's
    // click/hover/outline machinery can treat each feature as its own thing.
    // `has_resolver` gates the work: with no resolver every feature resolves to
    // the anchor anyway.
    let has_resolver = anchor.is_some() && resolver.0.is_some();
    let mut cache = Vec::with_capacity(items.len());

    for item in items {
        match item {
            DecodedItem::Mesh(prim) => {
                let DecodedPrimitive {
                    transform: ptf,
                    mesh,
                    material,
                    features,
                } = *prim;
                let prim_transform = Transform::from_matrix(ptf);
                // One OPAQUE StandardMaterial per primitive, SHARED by all of its
                // feature submeshes (no per-submesh texture duplication). Opaque:
                // no discard/alpha so the GPU keeps early-Z (dense P3DT overdraw).
                let standard = StandardMaterial {
                    base_color: Color::LinearRgba(LinearRgba::new(
                        material.base_color[0],
                        material.base_color[1],
                        material.base_color[2],
                        material.base_color[3],
                    )),
                    base_color_texture: material.base_color_image.map(|img| images.add(img)),
                    metallic: material.metallic,
                    perceptual_roughness: material.roughness,
                    unlit: material.unlit,
                    cull_mode: if material.double_sided {
                        None
                    } else {
                        Some(bevy::render::render_resource::Face::Back)
                    },
                    ..default()
                };
                let mat_handle = materials.add(standard);

                // ONE mesh per primitive, always — the Cesium model. Features
                // resolve at PICK time from the hit triangle via
                // [`TileFeaturePick`]; the old per-owner submesh split
                // (build_submesh + a mesh asset + a GPU upload PER FEATURE per
                // tile) was measured at seconds of main-thread hang per refine
                // wave even capped, while pure-decode tilesets only
                // micro-stuttered. Per-feature hover highlight moves to
                // render-state (a feature-id tint — Phase B); selection
                // correctness is carried entirely by the pick table.
                let pick = match (features, has_resolver) {
                    (Some(f), true) => {
                        let fallback = anchor.unwrap_or("");
                        // Resolve ALL of the tile's feature paths in one call
                        // so the host builds its per-anchor lookup once per
                        // tile, not once per feature.
                        let paths: Vec<&str> =
                            f.node_of_feature.iter().map(String::as_str).collect();
                        let mut owner_of_feature = resolver.resolve(fallback, &paths);
                        // Out-of-range/unresolved ids fall back to the anchor
                        // rather than dropping picks (matches the old split's
                        // per-triangle fallback).
                        for owner in &mut owner_of_feature {
                            if owner.is_empty() {
                                *owner = fallback.to_string();
                            }
                        }
                        Some(TileFeaturePick {
                            feature_of_triangle: f.feature_of_triangle,
                            owner_of_feature,
                        })
                    }
                    _ => None,
                };
                cache.push(CachedItem::Mesh {
                    mesh: meshes.add(mesh),
                    material: mat_handle,
                    transform: prim_transform,
                    pick,
                });
            }
            #[cfg(feature = "points")]
            DecodedItem::Points { transform, points } => {
                cache.push(CachedItem::Points {
                    cloud: renderers.clouds.add(PointCloud { points }),
                    transform: Transform::from_matrix(transform),
                });
            }
            #[cfg(feature = "splats")]
            DecodedItem::Splat {
                transform,
                gaussians,
            } => {
                cache.push(CachedItem::Splat {
                    cloud: renderers.splats.add(PlanarGaussian3d::from(gaussians)),
                    transform: Transform::from_matrix(transform),
                });
            }
        }
    }
    cache
}

/// Spawn one tile's cached content under a tile-root entity, at `visibility`
/// (children inherit it). Used both by the fresh-decode path — `Hidden`, so a
/// host reacting to `Added<TileGeometry>` lands its changes before the geometry
/// is drawn — and by the hidden-tile respawn path, which re-runs it from the
/// cache alone: no fetch, no decode, no `Assets` insert.
///
/// Every re-spawn is a brand-new entity, so every `Added<…>` host adapter
/// (`TileOwner` → domain groups, `TileGeometry` → materials/clipping,
/// `TileFeaturePick` → tint/outline) re-fires exactly as on first spawn. Nothing
/// may cache an `Entity` per tile: the set's `slots` is the only authority.
fn spawn_tile_entities(
    commands: &mut Commands,
    #[cfg(feature = "points")] point_material: &PointTileMaterial,
    set: &ActiveTileset,
    tile: usize,
    transform: Transform,
    cache: &[CachedItem],
    visibility: Visibility,
) -> Entity {
    let tile_root = commands
        .spawn((
            Tiles3dTile {
                set_id: set.id,
                tile,
            },
            transform,
            visibility,
            ChildOf(set.root_entity),
            Name::new(format!("Tiles3dTile({} #{tile})", set.label)),
        ))
        .id();
    let anchor_group = set
        .owner_id
        .as_deref()
        .map(|id| TileOwner { id: id.to_string() });
    // Goes on EVERY content entity, owned or not, so a host can post-process
    // tile geometry per tileset (custom materials, clipping, styling). Unlike
    // `TileOwner` this is unconditional — world-layer sets have no owner but
    // are exactly the ones a host most wants to treat specially.
    let content_tag = TileGeometry { set_id: set.id };
    for item in cache {
        let child = match item {
            CachedItem::Mesh {
                mesh,
                material,
                transform,
                pick,
            } => {
                let mut e = commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    *transform,
                    ChildOf(tile_root),
                    content_tag,
                ));
                if let Some(pick) = pick {
                    e.insert(pick.clone());
                }
                e.id()
            }
            #[cfg(feature = "points")]
            CachedItem::Points { cloud, transform } => commands
                .spawn((
                    PointCloud3d(cloud.clone()),
                    PointCloudMaterial3d(point_material.0.clone()),
                    *transform,
                    ChildOf(tile_root),
                    content_tag,
                ))
                .id(),
            #[cfg(feature = "splats")]
            CachedItem::Splat { cloud, transform } => commands
                .spawn((
                    PlanarGaussian3dHandle(cloud.clone()),
                    CloudSettings::default(),
                    *transform,
                    ChildOf(tile_root),
                    content_tag,
                ))
                .id(),
        };
        if let Some(group) = &anchor_group {
            commands.entity(child).insert(group.clone());
        }
    }
    tile_root
}

/// Build a sub-mesh from a subset of `mesh`'s triangles (by triangle ordinal),
/// remapped to a compact vertex range — splits a feature tile into per-section-
/// twin pieces at spawn (T8 highlight). Copies POSITION plus whatever of
/// NORMAL/UV0/COLOR the source carries; `MAIN_WORLD` usage so the pick raycast
/// can read it.
/// Public since 0.1.9: the lazy-extraction seam for hosts. Per-feature render
/// styling is a material concern ([`TileFeaturePick`] + the UV1 feature ids),
/// but effects that need REAL per-feature geometry — a selection outline pass,
/// a physics proxy, an export — extract just the wanted triangles on demand
/// (e.g. once per click), which is why the eager per-feature split of ≤0.1.5
/// is gone: this is its surviving, on-demand half.
pub fn build_submesh(mesh: &Mesh, tris: &[usize]) -> Mesh {
    use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues as Vav};
    let mut out = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    let Some(positions) = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|a| a.as_float3())
    else {
        return out;
    };
    let normals = mesh
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .and_then(|a| a.as_float3());
    let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(Vav::Float32x2(v)) => Some(v.as_slice()),
        _ => None,
    };
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(Vav::Float32x4(v)) => Some(v.as_slice()),
        _ => None,
    };
    let vertex_of = |t: usize, k: usize| -> Option<usize> {
        match mesh.indices() {
            Some(Indices::U32(v)) => v.get(t * 3 + k).map(|&i| i as usize),
            Some(Indices::U16(v)) => v.get(t * 3 + k).map(|&i| i as usize),
            None => Some(t * 3 + k),
        }
    };
    let mut remap: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut out_pos: Vec<[f32; 3]> = Vec::new();
    let mut out_nrm: Vec<[f32; 3]> = Vec::new();
    let mut out_uv: Vec<[f32; 2]> = Vec::new();
    let mut out_col: Vec<[f32; 4]> = Vec::new();
    let mut out_idx: Vec<u32> = Vec::new();
    for &t in tris {
        for k in 0..3 {
            let Some(v) = vertex_of(t, k).filter(|&v| v < positions.len()) else {
                continue;
            };
            let local = match remap.get(&v) {
                Some(&l) => l,
                None => {
                    let l = out_pos.len() as u32;
                    remap.insert(v, l);
                    out_pos.push(positions[v]);
                    if let Some(n) = normals {
                        out_nrm.push(n[v]);
                    }
                    if let Some(u) = uvs {
                        out_uv.push(u[v]);
                    }
                    if let Some(c) = colors {
                        out_col.push(c[v]);
                    }
                    l
                }
            };
            out_idx.push(local);
        }
    }
    out.insert_attribute(Mesh::ATTRIBUTE_POSITION, out_pos);
    if !out_nrm.is_empty() {
        out.insert_attribute(Mesh::ATTRIBUTE_NORMAL, out_nrm);
    }
    if !out_uv.is_empty() {
        out.insert_attribute(Mesh::ATTRIBUTE_UV_0, out_uv);
    }
    if !out_col.is_empty() {
        out.insert_attribute(Mesh::ATTRIBUTE_COLOR, out_col);
    }
    out.insert_indices(Indices::U32(out_idx));
    out
}

/// Keep the `true`-flagged entries of `v`, preserving order — the index-aligned
/// twin of [`TileTree::retain`] for the per-tile side arrays.
fn gather<T: Clone>(v: &[T], keep: &[bool]) -> Vec<T> {
    v.iter()
        .zip(keep)
        .filter(|&(_, &k)| k)
        .map(|(x, _)| x.clone())
        .collect()
}

/// Reclaim grafted subtrees that have fallen out of view past the grace window,
/// bounding the otherwise monotonically-growing P3DT tree (external tilesets
/// graft in as you fly and were never removed — the 16k→43k-node session creep
/// that slowed every per-frame O(tree) pass). Keeps the base tileset and
/// everything recently touched / still resident / in flight; drops whole stale
/// grafted subtrees and restores each graft-point's `content_uri` so revisiting
/// re-grafts. Renumbers tiles, so it runs occasionally (amortized), never per
/// frame. Returns the number of tiles reclaimed.
fn compact_grafted_subtrees(set: &mut ActiveTileset, frame: u64, grace: u64) -> usize {
    let n = set.tree.len();
    let stale = |i: usize| frame.saturating_sub(set.last_touched[i]) > grace;

    // `keep[i]` defaults true; a prunable grafted subtree flips its nodes false.
    // Because we only ever drop COMPLETE subtrees rooted at a grafted child,
    // `keep` stays ancestor-closed (a kept node's parent is kept) — the
    // invariant `TileTree::retain` relies on.
    let mut keep = vec![true; n];
    let mut reclaimed = 0usize;
    for g in &set.grafts {
        let r = g.child_root;
        // Already inside an outer pruned subtree, or still wanted near-view.
        if !keep[r] || !stale(r) {
            continue;
        }
        // Scan the subtree: prune only if NOTHING in it is resident or in flight
        // (else we'd leak its entity or abort a live fetch). `r` stale ⇒ every
        // descendant is stale too (a touched tile always touches its parent),
        // so the root's staleness check covers the subtree; here we just guard
        // against lingering content.
        let mut subtree = Vec::new();
        let mut stack = vec![r];
        let mut prunable = true;
        while let Some(x) = stack.pop() {
            if matches!(
                set.slots[x],
                TileSlot::Ready { .. } | TileSlot::InFlight { .. }
            ) {
                prunable = false;
                break;
            }
            subtree.push(x);
            stack.extend(set.tree.nodes[x].children.iter().copied());
        }
        if prunable {
            for x in subtree {
                keep[x] = false;
            }
            reclaimed += 1;
        }
    }
    if reclaimed == 0 {
        return 0;
    }

    // Restore content on surviving graft-points whose child was pruned, so a
    // later visit re-fetches + re-grafts the external tileset (OLD indices).
    for g in &set.grafts {
        if keep[g.at] && !keep[g.child_root] {
            set.tree.nodes[g.at].content_uri = Some(g.uri.clone());
        }
    }

    // Gather the parallel per-tile arrays with the same mask, then remap the
    // tree (parent/children) and the graft records to the new indices. A
    // surviving record's `child_root` is kept ⇒ its `at` is kept too
    // (ancestor-closed), so both remap cleanly.
    set.slots = gather(&set.slots, &keep);
    set.caches = gather(&set.caches, &keep);
    set.last_touched = gather(&set.last_touched, &keep);
    set.rtc_centers = gather(&set.rtc_centers, &keep);
    set.history.rendered = gather(&set.history.rendered, &keep);
    set.history.refined = gather(&set.history.refined, &keep);
    set.grafts.retain(|g| keep[g.child_root]);
    let map = set.tree.retain(&keep);
    for g in &mut set.grafts {
        g.at = map[g.at];
        g.child_root = map[g.child_root];
    }
    n - set.tree.len()
}

// ── Per-frame manager ────────────────────────────────────────────────────────

/// One issuable tile request, gathered across every set each frame so the
/// global load pool is granted with cross-set fairness (many-tileset scenes
/// used to starve later sets — see [`sort_load_candidates`]).
struct LoadCandidate {
    /// Index into `Tiles3dSets::sets` THIS frame (stable within one system
    /// run: the GC `retain` happens before collection).
    set_idx: usize,
    /// The set's stable id — the round-robin cursor is keyed on it so
    /// rotation survives sets detaching/reshuffling.
    set_id: u64,
    /// Tile index within the set's tree.
    tile: usize,
    /// [`api::TilePriorityClass`] of the set (default 1).
    class: u8,
    /// First request of a set with zero Ready content — its entry into
    /// visibility (typically its root tile).
    starving_root: bool,
    /// Rank within the set's own `sel.loads` order this frame.
    within: usize,
}

/// Order candidates for issue:
///
/// * (a) **Starving sets first, across every class.** A set with nothing Ready
///   gets its first (root) request before ANY set's refinement, whatever the
///   classes. Class must not be able to gate this: a class-0 world layer
///   (Google P3DT) refines without bound, so keying on class first lets it
///   starve a class-1 twin's root forever — the original many-tileset
///   starvation incident, one class up.
/// * (b) Class ascending — ordering starving roots among themselves (terrain
///   root before twin roots) and, separately, refinements among themselves.
/// * (c) Breadth-first across sets — every set's k-th request before any set's
///   (k+1)-th — rotated by `cursors`, which holds the set id last granted a
///   slot **per class** (missing ⇒ start at the lowest id), so a saturated
///   pool still rotates across sets over successive frames. Per class, not
///   global: a frame whose slots all go to class 0 would otherwise leave the
///   cursor pointing at a class-0 id, and every leftover slot thereafter would
///   land on the same lowest-id class-1 set.
/// * (d) Within one set, selection order is preserved.
fn sort_load_candidates(cands: &mut [LoadCandidate], cursors: &std::collections::HashMap<u8, u64>) {
    cands.sort_by_key(|c| {
        let cursor = cursors.get(&c.class).copied().unwrap_or(u64::MAX);
        (
            !c.starving_root,
            c.class,
            c.within,
            // Rotation: ids past this class's cursor first (ascending), then wrap.
            c.set_id <= cursor,
            c.set_id,
        )
    });
}

/// Run the selection pass per tileset, apply the render cut as visibility,
/// schedule loads by priority (recomputed every frame, out-of-cut requests
/// aborted), and evict stale residents.
#[allow(clippy::too_many_arguments)]
fn drive_tiles3d(
    config: Res<Tiles3dConfig>,
    channel: Res<Tiles3dChannel>,
    origin: Res<EcefOrigin>,
    mut sets: ResMut<Tiles3dSets>,
    mut credits: ResMut<TilesetCredits>,
    camera: Query<(&Camera, &GlobalTransform, &Projection, &Frustum), With<Tiles3dCamera>>,
    transforms: Query<&GlobalTransform>,
    mut vis_q: Query<&mut Visibility, With<Tiles3dTile>>,
    mut tile_transforms: Query<&mut Transform, With<Tiles3dTile>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
    // Last logged state of the load halt (hard stop / host brake) — log
    // transitions, not every frame.
    mut halt_logged: Local<bool>,
    // Host-supplied per-set tuning, both looked up on the set's ANCHOR entity:
    // streaming priority class and SSE relaxation. One query, not two — bevy
    // caps a system at 16 params and the `points` feature adds one.
    anchor_tuning: Query<(
        Option<&api::TilePriorityClass>,
        Option<&api::TileSseMultiplier>,
    )>,
    // Shared material for respawned POINTS tiles (the fresh-decode path reads
    // the same resource in `receive_tiles3d`).
    #[cfg(feature = "points")] point_material: Res<PointTileMaterial>,
    // Round-robin cursors, one per priority class: id of the set that class
    // last granted a load slot to. Persists across frames so fairness holds
    // under a saturated pool, and is keyed by set id (not position) so it
    // stays sane when sets detach.
    mut rr_cursors: Local<std::collections::HashMap<u8, u64>>,
    // Host off-thread prepare hook (S4). Cloned out of the Res BEFORE
    // `fetch::spawn_io` — the task must not capture the Res.
    prepare_hook: Res<TilePrepareHook>,
) {
    let prepare_hook = prepare_hook.0.clone();
    let Tiles3dSets {
        sets,
        frame,
        next_generation,
        ..
    } = &mut *sets;
    *frame += 1;

    // GC: a set whose root entity died — or whose anchor died (ECEF roots
    // are NOT parented under their anchor, so the hierarchy can't cascade
    // for them) — is torn down here; its in-flight requests abort and late
    // results drop harmlessly. (`pending_anchors` / `failed_anchors` keep
    // dead Entity ids — harmless: entity generations never repeat, and the
    // per-id cost is 8 bytes.)
    sets.retain(|set| {
        let root_alive = transforms.contains(set.root_entity);
        let anchor_alive = set.anchor.is_none_or(|a| transforms.contains(a));
        if root_alive && anchor_alive {
            return true;
        }
        info!("tiles3d: {}: anchor gone — dropping tileset", set.label);
        abort_in_flight(set);
        if root_alive && let Ok(mut e) = commands.get_entity(set.root_entity) {
            e.despawn();
        }
        false
    });
    if sets.is_empty() {
        if *credits != TilesetCredits::default() {
            *credits = TilesetCredits::default();
        }
        return;
    }
    let Ok((cam, cam_gt, proj, frustum)) = camera.single() else {
        return;
    };

    let fov_y = match proj {
        Projection::Perspective(p) => p.fov as f64,
        _ => std::f64::consts::FRAC_PI_4,
    };
    // PHYSICAL pixels, not logical: the tile geometry rasterises at the
    // framebuffer resolution, so SSE must too. On a high-DPI/retina display
    // logical = physical / devicePixelRatio, so a logical-pixel k underestimated
    // the error by ~2× and selected one LOD too coarse — the "blurry zoomed out"
    // finding. `sse_threshold_px` is now in physical pixels (texel ≈ device pixel).
    let viewport_h = cam
        .physical_viewport_size()
        .map(|v| v.y as f64)
        .unwrap_or(1080.0);
    let k_px = viewport_h / (2.0 * (fov_y * 0.5).tan()).max(1e-6);
    let cam_pos_world = cam_gt.translation().as_dvec3();
    let cam_forward_world = Vec3::from(cam_gt.forward()).as_dvec3();

    // Memory-pressure valve: sum resident decoded bytes across ALL sets once
    // per traversal (re-summed, never incrementally tracked — zero drift), and
    // inflate every set's SSE threshold by the overshoot so the wanted cut
    // coarsens and eviction can actually reclaim. The host's external
    // pressure (its global memory ledger) folds into the same product.
    // Quality degradation beats "memory access out of bounds"
    // (see Tiles3dConfig::memory_budget_bytes).
    let mut resident_bytes: u64 = 0;
    let mut in_flight_total: usize = 0;
    for slot in sets.iter().flat_map(|s| s.slots.iter()) {
        match slot {
            TileSlot::Ready { bytes, .. } => resident_bytes += *bytes,
            TileSlot::InFlight { .. } => in_flight_total += 1,
            _ => {}
        }
    }
    let pressure = (memory_pressure_factor(resident_bytes, config.memory_budget_bytes)
        * f64::from(config.external_pressure.max(1.0)))
    .min(8.0);
    if pressure > 1.0 {
        debug!(
            "tiles3d: memory pressure {:.2} ({} MB resident / {} MB budget, external {:.2}) — \
             SSE threshold inflated",
            pressure,
            resident_bytes / 1_048_576,
            config.memory_budget_bytes / 1_048_576,
            config.external_pressure,
        );
    }
    // The load-slot pool is GLOBAL (0.2.0): per-set caps multiplied by the
    // set count and let a many-tileset scene spike the decode working set.
    let mut load_slots = config.max_concurrent_loads.saturating_sub(in_flight_total);
    // Hard stop: past 1.5× budget the SSE valve has demonstrably not held the
    // line (a many-tileset coarse cut has a byte floor) — stop STARTING loads
    // until eviction brings residency back down. The host brake latches the
    // same lever.
    let hard_stopped = config.memory_budget_bytes > 0
        && resident_bytes > config.memory_budget_bytes.saturating_mul(3) / 2;
    let halt_loads = hard_stopped || config.halt_new_loads;
    if halt_loads != *halt_logged {
        *halt_logged = halt_loads;
        if halt_loads {
            warn!(
                "tiles3d: NEW TILE LOADS HALTED ({} MB resident / {} MB budget, host brake: {}) \
                 — rendering the resident cut until eviction reclaims",
                resident_bytes / 1_048_576,
                config.memory_budget_bytes / 1_048_576,
                config.halt_new_loads,
            );
        } else {
            info!("tiles3d: tile loads resumed");
        }
    }

    let mut any_in_flight = false;
    // A wanted tile is spawned-but-not-yet-shown (or still awaiting its respawn
    // budget) somewhere, so the next frame has work to do even if nothing streams.
    let mut pending_respawns = false;
    let mut google_visible = false;
    let mut ground_covering = false;
    let mut compacted_this_frame = false;
    // Issuable requests across ALL sets — collected per set below, issued
    // after the loop under the global pool (cross-set fairness).
    let mut candidates: Vec<LoadCandidate> = Vec::new();
    // Hidden-tile respawn budget, GLOBAL across sets like `max_concurrent_loads`
    // — per set it multiplied by tileset count (23 on the Hermosa site).
    // ponytail: earlier sets in iteration order drain it first; that only delays
    // a later set's refill (its hold keeps the coverage), so the round-robin
    // machinery below is not worth spending on it until a scene actually starves.
    let mut respawns_left = config.max_respawns_per_frame;
    for (set_idx, set) in sets.iter_mut().enumerate() {
        // Reclaim stale grafted subtrees once the tree has grown well past the
        // last pass — bounds the monotonic P3DT graft creep so the per-frame
        // O(tree) bookkeeping below stops getting slower the longer you fly.
        // Renumbers tiles, so it's amortized (≥50% growth) and capped to one
        // set per frame to avoid a multi-set spike. Runs BEFORE selection so
        // everything downstream sees the compacted indices.
        if !compacted_this_frame
            && set.tree.len() >= config.tree_compact_min
            && set.tree.len() >= set.compact_high_water.saturating_mul(3) / 2
        {
            let reclaimed = compact_grafted_subtrees(set, *frame, config.grace_frames);
            set.compact_high_water = set.tree.len();
            compacted_this_frame = true;
            if reclaimed > 0 {
                info!(
                    "tiles3d: {}: compacted tree — reclaimed {reclaimed} stale tile(s) \
                     ({} remain)",
                    set.label,
                    set.tree.len()
                );
            }
        }

        // The set's frame. Anchored: world_from_set = anchor chain ×
        // correction (the root entity's GlobalTransform — last frame's
        // propagation, fine for streaming decisions); selection runs in
        // set-local coordinates so SSE is exact under rigid/uniform anchor
        // transforms. ECEF (T4): world_from_set = the ENU frame at the
        // project origin, recomputed from absolutes in f64 — one view, true
        // world positions (the one-view atmosphere model).
        // `planet_radius`: Some(R) for ECEF/globe sets enables horizon culling
        // (the set frame is centred on the planet, so the camera and every
        // tile volume are already in globe coordinates); None for set-local
        // tilesets (no globe to occlude behind).
        let (world_from_set, set_scale, planet_radius) = match &mut set.frame {
            SetFrame::Anchored => {
                let m = transforms
                    .get(set.root_entity)
                    .map(|gt| gt.to_matrix().as_dmat4())
                    .unwrap_or(DMat4::IDENTITY);
                let scale = traversal::max_scale(&m).max(1e-12);
                (m, scale, None)
            }
            SetFrame::Ecef { built } => {
                let Some(o) = origin.world_from_ecef else {
                    // No ENU datum yet — hold the set entirely.
                    continue;
                };
                if *built != Some(o) {
                    // ORIGIN REBASE (basemap's model, exact-recompute form):
                    // re-place every resident tile from absolutes in f64.
                    *built = Some(o);
                    // Despawned-but-cached tiles need nothing here: their
                    // transform is recomposed from the current origin when they
                    // respawn (`tile_spawn_transform`).
                    for (i, slot) in set.slots.iter_mut().enumerate() {
                        let TileSlot::Ready {
                            entity: Some(entity),
                            ..
                        } = *slot
                        else {
                            continue;
                        };
                        if let Ok(mut t) = tile_transforms.get_mut(entity) {
                            let m = compose_ecef_tile_matrix(
                                o,
                                set.tree.nodes[i].world_from_content,
                                set.rtc_centers[i],
                            );
                            *t = Transform::from_matrix(m.as_mat4());
                        }
                    }
                }
                (o, 1.0, Some(WGS84_EQUATORIAL_RADIUS_M))
            }
        };
        let set_from_world = world_from_set.inverse();
        let cam_pos = set_from_world.transform_point3(cam_pos_world);
        let cam_forward = set_from_world
            .transform_vector3(cam_forward_world)
            .normalize_or(DVec3::NEG_Z);
        // Camera height above the origin ground plane (globe sets only). World
        // +Y is ENU up, and the world↔set transform is rigid (distance-
        // preserving), so `cam_pos_world.y` is a valid metres floor for the
        // set-frame tile distances in the traversal. NOTE: use the world Y, NOT
        // `cam_pos.length() - equatorial_radius` — that sphere approximation is
        // off by up to ~21 km away from the equator, which would wreck the floor.
        let cam_height_m = if planet_radius.is_some() {
            cam_pos_world.y.max(0.0)
        } else {
            0.0
        };
        // The distance falloff is a globe/horizon guard: it bounds the P3DT
        // graft+stream storm toward the horizon on an ECEF set. A local-frame
        // tileset is a BOUNDED model with no horizon hemisphere to stream, and
        // `cam_height_m == 0` there makes the falloff measure RAW camera
        // distance — so it coarsens the ENTIRE model the moment you pull back
        // (the "local terrain stays blurry until I'm much closer than I expect"
        // finding). Natural 1/dist SSE already coarsens the far edge on its own;
        // disable the extra relaxation for local sets, keep it for ECEF/globe.
        let detail_falloff_m = if planet_radius.is_some() {
            config.detail_falloff_m
        } else {
            0.0
        };
        // Per-class SSE relaxation (0.2.4): read once per set per frame off the
        // anchor, the same entity that carries `TilePriorityClass`. Absent, or
        // nonsense (≤0 / NaN), means 1.0.
        let sse_mult = set
            .anchor
            .and_then(|a| anchor_tuning.get(a).ok())
            .and_then(|(_, mult)| mult)
            .map(|m| f64::from(m.0))
            .filter(|m| m.is_finite() && *m > 0.0)
            .unwrap_or(1.0);
        let params = SelectParams {
            cam_pos,
            cam_forward,
            k_px,
            // Per-set override (dense single-asset preview) wins; else the
            // app-global config default (globe basemap). The memory-pressure
            // factor and the anchor's `TileSseMultiplier` scale EITHER — over
            // budget, or on a ground-context set, everything coarsens.
            sse_threshold_px: set.sse_threshold_px.unwrap_or(config.sse_threshold_px)
                * pressure
                * sse_mult,
            detail_falloff_m,
            cam_height_m,
        };

        // Content readiness as the traversal sees it.
        let tiles_content: Vec<TileContent> = set
            .slots
            .iter()
            .zip(&set.tree.nodes)
            .map(|(slot, node)| {
                if node.content_uri.is_none() {
                    TileContent::None
                } else {
                    match slot {
                        TileSlot::Ready { .. } => TileContent::Ready,
                        TileSlot::Failed => TileContent::Failed,
                        TileSlot::NotLoaded | TileSlot::InFlight { .. } => TileContent::Pending,
                    }
                }
            })
            .collect();

        let tree = &set.tree;
        let culled = |i: usize| {
            let (center, radius) = tree.nodes[i].volume.bounding_sphere();
            // Frustum test in world space: local volume → world. Inflate the test
            // sphere 25% so tiles whose extent sits just past the edge don't pop
            // out (and stop loading) as the view rotates. `intersect_far = false`
            // like the basemap: distant tiles coarsen via SSE; clipping the rest.
            //
            // NO horizon/limb cull: it removed tiles that were genuinely on
            // screen — you could see ground PAST the limb that it had culled,
            // which broke the whole view — and the distance falloff (bounds far
            // refinement) + compaction (reclaims stale grafts) cover its old job.
            // Removed 2026-06-14; see git history for `beyond_horizon`.
            let world_center = world_from_set.transform_point3(center);
            let sphere = Sphere {
                center: Vec3A::from(world_center.as_vec3()),
                radius: (radius * set_scale * 1.25) as f32,
            };
            !frustum.intersects_sphere(&sphere, false)
        };

        let sel = traversal::select(tree, &tiles_content, &set.history, &culled, params);

        // Eviction clock: everything the pass wanted stays fresh.
        for (i, &touched) in sel.touched.iter().enumerate() {
            if touched {
                set.last_touched[i] = *frame;
            }
        }

        // Apply the render cut. A selected tile shows; an unselected one gives
        // up its ENTITIES entirely (0.2.4) while keeping its decoded assets in
        // `caches[i]`, so ~5k hidden mesh entities stop paying per-frame ECS tax
        // and a re-entry costs a spawn instead of a download.
        //
        // Three things make a REPLACE refinement show EXACTLY ONE rung per frame
        // — never a hole, never coarse over fine (the coarsen direction keeps its
        // one-frame coarse-over-fine overlap on purpose; see `covered_by_coarser`):
        //
        // * respawns are issued BEFORE any despawn, in this same system, so both
        //   land at the same command flush — spawn → show → despawn inside one
        //   frame, never across two;
        // * an out-of-cut tile that is currently covering for a wanted tile with
        //   no entity yet is HELD: kept spawned and visible until that tile
        //   confirms. `refinement_hold` is what picks those, and it is
        //   deliberately narrow — ancestors and descendants of the starved tile,
        //   not the whole set;
        // * while such a hold is PAINTING this tile's footprint, the tile waits
        //   (`covered_by_coarser`) instead of drawing on top of it — and when no
        //   hold is painting it, it shows the frame it is wanted instead of
        //   waiting. One predicate decides both.
        let mut want_visible = vec![false; tree.len()];
        for &t in &sel.render {
            want_visible[t] = true;
        }
        // An ECEF set has no placement at all until the host publishes an
        // `EcefOrigin` (`tile_spawn_transform` → `None`). Such a set can't
        // respawn, so it must not despawn either — it keeps what it has, and
        // deliberately does NOT report a pending respawn, which would pin the
        // host's reactive loop at full frame rate on a parked camera for as long
        // as the origin stays missing.
        let respawnable =
            !matches!(set.frame, SetFrame::Ecef { .. }) || origin.world_from_ecef.is_some();
        let (hold, starved) = refinement_hold(tree, &want_visible, &set.slots);
        let starved = starved && respawnable;
        // Which held tiles are actually PAINTING right now. A held tile that was
        // never shown (decoded straight into an out-of-cut slot) covers nothing,
        // and a spawned tile that is not held leaves this frame — neither can be
        // waited behind. A tile that is itself in the cut is excluded: it draws
        // on its own merit, so a `Refine::Add` parent never gates its own
        // children here. Reading `Visibility` per tile is cheap now — after the
        // despawn only ~cut-many tiles have an entity at all.
        let covering: Vec<bool> = (0..tree.len())
            .map(|i| {
                hold[i]
                    && !want_visible[i]
                    && matches!(set.slots[i],
                        TileSlot::Ready { entity: Some(e), .. }
                            if vis_q.get(e).is_ok_and(|v| *v == Visibility::Visible))
            })
            .collect();
        // The atomicity predicate. A painting, held ANCESTOR covers this tile's
        // whole footprint (its bounding volume contains the child's), so a wanted
        // tile that has one waits — hidden — until the last of its starved
        // siblings arrives and the hold lifts. That is what makes a refinement
        // swap flip every sibling at once instead of trickling fine tiles on top
        // of the coarse they replace. With NO painting ancestor, nothing is on
        // screen here at all and waiting a frame IS the hole — that is cut entry
        // from cache, where the footprint's whole ancestor chain is despawned, so
        // the tile draws the frame it is wanted.
        //
        // Ancestors only, deliberately: a held DESCENDANT is the coarsen
        // direction, where the incoming parent overlapping its outgoing children
        // for one frame is the accepted deviation (coarse over fine beats a gap).
        let covered_by_coarser = |i: usize| {
            let mut up = tree.nodes[i].parent;
            while let Some(a) = up {
                if covering[a] {
                    return true;
                }
                up = tree.nodes[a].parent;
            }
            false
        };
        let mut hide: Vec<usize> = Vec::new();
        for (i, &want) in want_visible.iter().enumerate() {
            let TileSlot::Ready { entity, bytes } = set.slots[i] else {
                continue;
            };
            match (want, entity) {
                (true, Some(e)) => {
                    // Its coverage is still painting: wait for the siblings
                    // instead of drawing fine on top of the coarse. `starved` is
                    // necessarily true whenever this fires (a hold implies one),
                    // so the reactive loop keeps ticking until the swap lands.
                    if covered_by_coarser(i) {
                        continue;
                    }
                    // Visible from this frame on: the write is immediate, so a
                    // tile flipped here is on screen for the same frame the
                    // despawns below take effect.
                    if let Ok(mut vis) = vis_q.get_mut(e)
                        && *vis != Visibility::Visible
                    {
                        *vis = Visibility::Visible;
                    }
                }
                (true, None) => {
                    // Re-entry: respawn from the cache against the CURRENT
                    // origin (never the one captured at first spawn).
                    if respawns_left == 0 {
                        continue;
                    }
                    let Some(transform) = tile_spawn_transform(set, i, origin.world_from_ecef)
                    else {
                        continue;
                    };
                    respawns_left -= 1;
                    // Same predicate as the show arm, for the same reason: land
                    // hidden behind live coverage (and join the swap when it
                    // lifts), land VISIBLE when there is none — a hidden frame
                    // with nothing covering it is a hole.
                    let visibility = if covered_by_coarser(i) {
                        Visibility::Hidden
                    } else {
                        Visibility::Visible
                    };
                    let e = spawn_tile_entities(
                        &mut commands,
                        #[cfg(feature = "points")]
                        &point_material,
                        set,
                        i,
                        transform,
                        &set.caches[i],
                        visibility,
                    );
                    set.slots[i] = TileSlot::Ready {
                        entity: Some(e),
                        bytes,
                    };
                }
                // Held tiles keep BOTH their entity and their visibility: they
                // are the coverage a starved tile has not provided yet.
                (false, Some(_)) if respawnable && !hold[i] => hide.push(i),
                (false, _) => {}
            }
        }
        for i in hide {
            let TileSlot::Ready {
                entity: Some(e),
                bytes,
            } = set.slots[i]
            else {
                continue;
            };
            commands.entity(e).despawn();
            // Slot stays Ready — the assets (and the ledger bytes) are
            // deliberately still resident; only eviction reclaims them.
            set.slots[i] = TileSlot::Ready {
                entity: None,
                bytes,
            };
        }
        if starved {
            // Keep the reactive loop ticking until the pending respawns have
            // shown, or a settled camera would leave them hidden.
            pending_respawns = true;
        }

        // First painted cut: strip the anchor's placeholder cube geometry
        // (same contract as `bind_spawned_scenes` for whole-file scenes —
        // remove `Mesh3d`, keep the entity as the transform anchor).
        if !set.placeholder_cleared && !sel.render.is_empty() && set.owner_id.is_some() {
            if let Some(anchor) = set.anchor
                && let Ok(mut e) = commands.get_entity(anchor)
            {
                e.remove::<Mesh3d>();
            }
            set.placeholder_cleared = true;
        }

        // Scheduler. Cancel first: an in-flight tile that fell out of this
        // frame's wanted loads aborts its network transfer (T1) and frees its
        // slot now; a landed stale payload is dropped by the
        // InFlight/generation guard in `receive_tiles3d`.
        let mut wanted = vec![false; tree.len()];
        for req in &sel.loads {
            wanted[req.tile] = true;
        }
        for (i, slot) in set.slots.iter_mut().enumerate() {
            if let TileSlot::InFlight { generation } = slot
                && !wanted[i]
            {
                fetch::trigger_abort(*generation);
                *slot = TileSlot::NotLoaded;
            }
        }
        // Budget guardrail (D7): a live set whose daily request cap is
        // exhausted issues nothing more — hard stop, warn once.
        let budget_exhausted = match &set.source {
            TilesetSource::Live(live) => {
                let exhausted = live.budget().exhausted();
                if exhausted && !set.budget_warned {
                    warn!(
                        "tiles3d: {}: daily request cap reached ({} requests) — \
                         P3DT streaming halted until tomorrow (org admins set the \
                         cap on the layer entry)",
                        set.label,
                        live.budget().cap(),
                    );
                    set.budget_warned = true;
                }
                exhausted
            }
            _ => false,
        };

        // Collect this set's issuable requests for the cross-set scheduler
        // after the loop — nothing is issued per set any more, so the first
        // sets in iteration order can no longer starve the rest of the pool.
        // `load_slots == 0` (pool fully in flight) skips collection outright:
        // nothing above this point depends on it, and the abort pass, budget
        // warn and halt logging all still ran.
        if load_slots > 0 && !budget_exhausted && !halt_loads {
            let class = set
                .anchor
                .and_then(|a| anchor_tuning.get(a).ok())
                .and_then(|(class, _)| class)
                .copied()
                .unwrap_or_default()
                .0;
            let has_ready = set
                .slots
                .iter()
                .any(|s| matches!(s, TileSlot::Ready { .. }));
            let mut within = 0usize;
            for req in &sel.loads {
                if !matches!(set.slots[req.tile], TileSlot::NotLoaded) {
                    continue; // already in flight, ready, or failed
                }
                if tree.nodes[req.tile].content_uri.is_none() {
                    continue;
                }
                candidates.push(LoadCandidate {
                    set_idx,
                    set_id: set.id,
                    tile: req.tile,
                    class,
                    starving_root: !has_ready && within == 0,
                    within,
                });
                within += 1;
            }
        }

        // Eviction: out-of-cut residents past the grace window, then the
        // oldest extras over the hard budget (memory wins over reuse).
        let mut resident: Vec<(usize, u64)> = Vec::new();
        for (i, slot) in set.slots.iter().enumerate() {
            if matches!(slot, TileSlot::Ready { .. }) {
                resident.push((i, set.last_touched[i]));
            }
        }
        // `!hold[i]`: eviction is the one despawn path that would otherwise
        // ignore the refinement hold, and it drops the CACHE too — evicting a
        // held tile opens the hole the hold exists to prevent and pays a
        // re-download to close it. Both arms are gated: the grace arm because a
        // long starved motion can outlast `grace_frames`, the overflow arm
        // because it consults no clock at all.
        let mut evict: Vec<usize> = resident
            .iter()
            .filter(|(i, seen)| {
                !want_visible[*i] && !hold[*i] && frame.saturating_sub(*seen) > config.grace_frames
            })
            .map(|(i, _)| *i)
            .collect();
        if resident.len() - evict.len() > config.max_resident_tiles {
            let mut extras: Vec<(usize, u64)> = resident
                .iter()
                .filter(|(i, _)| !want_visible[*i] && !hold[*i] && !evict.contains(i))
                .copied()
                .collect();
            extras.sort_by_key(|(_, seen)| *seen);
            let over = resident.len() - evict.len() - config.max_resident_tiles;
            evict.extend(extras.iter().take(over).map(|(i, _)| *i));
        }
        for i in evict {
            if let TileSlot::Ready { entity, .. } = set.slots[i] {
                if let Some(e) = entity {
                    commands.entity(e).despawn();
                }
                // Dropping the cache drops the asset handles — THIS is what
                // reclaims the memory a hidden tile deliberately kept.
                set.caches[i] = Vec::new();
                set.slots[i] = TileSlot::NotLoaded;
            }
        }

        if !sel.render.is_empty() {
            let (mut dmin, mut dmax) = (u32::MAX, 0);
            for &t in &sel.render {
                dmin = dmin.min(tree.nodes[t].depth);
                dmax = dmax.max(tree.nodes[t].depth);
            }
            let cut = (sel.render.len(), dmin, dmax);
            if set.last_cut != Some(cut) {
                if set.last_cut.is_none() {
                    info!("tiles3d: {}: first cut visible", set.label);
                }
                set.last_cut = Some(cut);
                info!(
                    "tiles3d: {}: cut {} tile(s) at depth {dmin}..{dmax} (covered={})",
                    set.label, cut.0, sel.covered,
                );
            }
        }
        set.history.absorb(&sel, tree.len());
        google_visible |= set.is_live() && !sel.render.is_empty();
        ground_covering |= matches!(set.frame, SetFrame::Ecef { .. }) && !sel.render.is_empty();
        any_in_flight |= set
            .slots
            .iter()
            .any(|s| matches!(s, TileSlot::InFlight { .. }));
    }

    // Issue across ALL sets under the GLOBAL concurrency pool (`load_slots`),
    // in cross-set fairness order — see `sort_load_candidates`. Halt/budget
    // gating and the empty-pool skip already happened at collection time
    // (`candidates` is empty in either case, so this is a no-op then).
    sort_load_candidates(&mut candidates, &rr_cursors);
    for c in candidates {
        if load_slots == 0 {
            break;
        }
        let set = &mut sets[c.set_idx];
        if !matches!(set.slots[c.tile], TileSlot::NotLoaded) {
            continue; // defensive: collection guarantees this today
        }
        let Some(uri) = set.tree.nodes[c.tile].content_uri.clone() else {
            continue;
        };
        let generation = *next_generation;
        *next_generation += 1;
        set.slots[c.tile] = TileSlot::InFlight { generation };
        load_slots -= 1;
        // Advance ONLY this class's cursor — see `sort_load_candidates` (c).
        rr_cursors.insert(c.class, c.set_id);
        any_in_flight = true;
        let abort = fetch::register_abort(generation);
        let source = set.source.clone();
        let tx = channel.tx.clone();
        let set_id = set.id;
        let georeferenced = matches!(set.frame, SetFrame::Ecef { .. });
        let hook = prepare_hook.clone();
        fetch::spawn_io(async move {
            // Fetch + decode entirely inside the task (wasm: every IO step
            // awaits a JS future and yields; decode is small-tile CPU).
            // External tilesets are detected by CONTENT, not URI — P3DT
            // serves subtree JSON and GLBs from the same extensionless
            // /files/<id> namespace.
            let result = match source.read_entry_cached(&uri, Some(&abort)).await {
                Ok(bytes) if looks_like_external_tileset(&bytes) => schema::parse_tileset(&bytes)
                    .map(|ts| TileOutput::Subtree(Box::new(ts)))
                    .map_err(|e| format!("parse external tileset: {e}")),
                Ok(bytes) => content::decode_tile_with(&bytes, georeferenced, hook.as_ref())
                    .await
                    .map(|tile| TileOutput::Content(Box::new(tile)))
                    .map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            fetch::unregister_abort(generation);
            // Receiver gone (plugin torn down) is fine — drop silently.
            let _ = tx.send(Tiles3dMsg::TileContent {
                set_id,
                generation,
                result,
            });
        });
    }

    // Attribution side-band (D7/L-D5): aggregated tile copyrights + the
    // Google-logo flag, consumed by the basemap overlay system. Change-gated
    // to avoid resource churn.
    let mut lines: BTreeSet<&String> = BTreeSet::new();
    for set in sets.iter() {
        lines.extend(&set.copyrights);
    }
    let want = TilesetCredits {
        lines: lines.into_iter().cloned().collect(),
        google_visible,
        ground_covering,
    };
    if *credits != want {
        *credits = want;
    }

    // Keep the reactive loop awake while content streams (or while a respawn
    // still has to be shown) — without this the idle 200 ms tick would crawl
    // through the decode queue (the same lesson as `keep_awake_while_loading`
    // in the asset loader).
    if any_in_flight || pending_respawns {
        redraw.write(RequestRedraw);
    }
}

/// Show/remove the Google logo overlay while Photorealistic 3D Tiles render
/// (Map Tiles API attribution policy: the logo must be visible whenever
/// Google content is; bottom-left, clear of the data attributions at
/// bottom-right). Only touches the DOM on a state change.
fn update_google_logo(credits: Res<TilesetCredits>, mut last: Local<Option<bool>>) {
    let want = credits.google_visible;
    if *last == Some(want) {
        return;
    }
    *last = Some(want);
    set_google_logo_dom(want);
}

/// Create/remove the `#tt-google-logo` overlay div (the official Google
/// wordmark served from gstatic, on a subtle backing chip for contrast per
/// the brand guidance).
#[cfg(target_arch = "wasm32")]
fn set_google_logo_dom(show: bool) {
    const ID: &str = "tt-google-logo";
    const LOGO_URL: &str =
        "https://www.gstatic.com/images/branding/googlelogo/svg/googlelogo_clr_74x24px.svg";
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    if show {
        if doc.get_element_by_id(ID).is_some() {
            return;
        }
        let Some(body) = doc.body() else { return };
        if let Ok(el) = doc.create_element("div") {
            el.set_id(ID);
            let _ = el.set_attribute(
                "style",
                "position:fixed;left:6px;bottom:4px;z-index:30;\
                 background:rgba(255,255,255,0.85);padding:2px 7px;border-radius:4px;\
                 pointer-events:none;user-select:none;line-height:0;",
            );
            el.set_inner_html(&format!(
                "<img src=\"{LOGO_URL}\" alt=\"Google\" style=\"height:19px\">"
            ));
            let _ = body.append_child(&el);
        }
    } else if let Some(el) = doc.get_element_by_id(ID) {
        el.remove();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn set_google_logo_dom(_show: bool) {}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::tasks::block_on;

    /// The host-override contract (see [`Tiles3dConfig`] + [`Tiles3dPlugin`]
    /// docs): a config inserted before the plugin survives `add_plugins`, because
    /// `build` registers it with `init_resource` (never overwrites). The TurboTwin
    /// wasm host depends on this to lower its load budget; locking it here stops a
    /// refactor to `insert_resource` from silently reverting every host to the
    /// defaults.
    /// The memory-pressure valve's transfer function: identity under budget
    /// (and when disabled), overshoot ratio above it, clamped at ×8.
    #[test]
    fn memory_pressure_factor_curve() {
        assert_eq!(memory_pressure_factor(500, 0), 1.0, "0 budget = disabled");
        assert_eq!(memory_pressure_factor(400, 400), 1.0, "at budget");
        assert_eq!(memory_pressure_factor(100, 400), 1.0, "under budget");
        assert_eq!(memory_pressure_factor(800, 400), 2.0, "2x overshoot");
        assert_eq!(memory_pressure_factor(400_000, 400), 8.0, "clamped at 8x");
    }

    #[test]
    fn host_inserted_config_wins() {
        let mut app = App::new();
        app.insert_resource(Tiles3dConfig {
            sse_threshold_px: 24.0,
            max_concurrent_loads: 4,
            ..Default::default()
        });
        app.add_plugins(Tiles3dPlugin);
        let cfg = app.world().resource::<Tiles3dConfig>();
        assert_eq!(cfg.sse_threshold_px, 24.0, "host sse_threshold_px kept");
        assert_eq!(
            cfg.max_concurrent_loads, 4,
            "host max_concurrent_loads kept"
        );
    }

    /// `build_submesh` (T8 highlight) extracts a triangle subset into a compact
    /// mesh, copying the attributes the source has and remapping indices.
    #[test]
    fn build_submesh_extracts_triangle_subset() {
        use bevy::mesh::{Indices, PrimitiveTopology};
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        // A quad: 4 verts, 2 triangles [0,1,2] and [0,2,3].
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0, 0.0, 0.0, 1.0]; 4]);
        mesh.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));

        // Keep only triangle 1 (verts 0,2,3) → 3 verts, 1 triangle, remapped.
        let sub = build_submesh(&mesh, &[1]);
        let pos = sub
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert_eq!(pos, &[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(
            sub.attribute(Mesh::ATTRIBUTE_COLOR).is_some(),
            "color carried through"
        );
        let Some(Indices::U32(idx)) = sub.indices() else {
            panic!("u32 indices")
        };
        assert_eq!(idx, &[0, 1, 2]);
    }

    #[test]
    fn archive_spec_detection_ignores_query_strings() {
        assert!(is_archive_spec("assets/fixtures/tiles3d-demo.3tz"));
        assert!(is_archive_spec(
            "https://x.blob.core.windows.net/a/demo.3tz?se=2026&sig=abc"
        ));
        assert!(is_archive_spec("https://x/a/demo.3tz#frag"));
        assert!(!is_archive_spec(
            "assets/fixtures/tiles3d-demo/tileset.json"
        ));
        assert!(!is_archive_spec("https://x/a/tileset.json?sas=1"));
    }

    #[test]
    fn external_tileset_detection_is_content_based() {
        let tileset = br#"{"asset":{"version":"1.1"},"geometricError":1e100,
            "root":{"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]},"geometricError":0}}"#;
        assert!(looks_like_external_tileset(tileset));
        assert!(looks_like_external_tileset(
            b"  \n{\"geometricError\": 1, \"root\": {}}"
        ));
        // GLB magic → content, regardless of any JSON-chunk strings.
        assert!(!looks_like_external_tileset(b"glTF\x02\x00\x00\x00..."));
        // Bare glTF JSON (no geometricError) → content.
        assert!(!looks_like_external_tileset(
            br#"{"asset":{"version":"2.0"},"scenes":[{"nodes":[0]}],"meshes":[]}"#
        ));
    }

    #[test]
    fn uri_dir_prefix_for_subtree_rebasing() {
        assert_eq!(uri_dir_prefix("sub/tileset.json"), Some("sub/".to_string()));
        assert_eq!(
            uri_dir_prefix("a/b/c.json?session=x"),
            Some("a/b/".to_string())
        );
        assert_eq!(uri_dir_prefix("tileset.json"), None);
        // Absolute-path subtrees (P3DT) yield their directory; the rebase
        // loop skips absolute CONTENT uris anyway.
        assert_eq!(
            uri_dir_prefix("/v1/3dtiles/datasets/x/files/a.json"),
            Some("/v1/3dtiles/datasets/x/files/".to_string())
        );
    }

    /// The committed fixture round-trips through the full native stack:
    /// exploded read → schema parse → tree build → GLB decode, and the
    /// packed `.3tz` twin through the ranged reader. Regenerate with
    /// `cargo run --example gen_tiles3d_fixture` if this fails after
    /// intentional fixture changes.
    #[test]
    fn committed_fixture_parses_and_decodes() {
        let source =
            TilesetSource::Exploded(ExplodedBase::Dir("assets/fixtures/tiles3d-demo".into()));
        let bytes = block_on(source.read_entry("tileset.json")).expect("fixture tileset.json");
        let tileset = schema::parse_tileset(&bytes).expect("parse");
        assert!(
            !geo::tileset_is_georeferenced(&tileset),
            "fixture is local-metres"
        );
        let tree = TileTree::build(&tileset, ZUP_TO_BEVY, TreeFrame::Local).expect("build");
        assert_eq!(tree.len(), 21, "1 root + 4 children + 16 leaves");
        // Mixed volume kinds present.
        let spheres = tree
            .nodes
            .iter()
            .filter(|n| matches!(n.volume, traversal::WorldVolume::Sphere { .. }))
            .count();
        assert!(
            spheres > 0 && spheres < tree.len(),
            "mixed box/sphere volumes"
        );
        // Every content GLB decodes.
        for node in &tree.nodes {
            let uri = node
                .content_uri
                .as_ref()
                .expect("all fixture tiles carry content");
            let glb = block_on(source.read_entry(uri)).expect("fixture glb");
            let items = content::decode_glb(&glb).expect("decode");
            assert!(!items.is_empty(), "{uri} has geometry");
        }
    }

    #[test]
    fn committed_fixture_3tz_roundtrips() {
        let ar = block_on(Archive3tz::open(ByteSource::File(
            "assets/fixtures/tiles3d-demo.3tz".into(),
        )))
        .expect("open fixture 3tz");
        assert_eq!(ar.index().len(), 22, "tileset.json + 21 GLBs");
        let bytes = block_on(ar.read_entry("tileset.json")).expect("tileset.json");
        let tileset = schema::parse_tileset(&bytes).expect("parse");
        let tree = TileTree::build(&tileset, ZUP_TO_BEVY, TreeFrame::Local).expect("build");
        let uri = tree.nodes[5].content_uri.clone().unwrap();
        let glb = block_on(ar.read_entry(&uri)).expect("tile glb via ranged read");
        assert!(content::decode_glb(&glb).is_ok());
    }

    fn cand(set_id: u64, class: u8, starving_root: bool, within: usize) -> LoadCandidate {
        LoadCandidate {
            set_idx: set_id as usize,
            set_id,
            tile: within,
            class,
            starving_root,
            within,
        }
    }

    /// No rotation history — every class starts at its lowest set id.
    fn no_cursors() -> std::collections::HashMap<u8, u64> {
        std::collections::HashMap::new()
    }

    /// (a) A set with zero Ready content gets its first (root) request before
    /// an older, already-visible set's refinements at equal class.
    #[test]
    fn starving_set_root_beats_refinement() {
        let mut c = vec![
            cand(1, 1, false, 0), // old set's refinements
            cand(1, 1, false, 1),
            cand(2, 1, true, 0), // new set's root
        ];
        sort_load_candidates(&mut c, &no_cursors());
        assert_eq!((c[0].set_id, c[0].within), (2, 0), "root request first");
    }

    /// (a) Root-first outranks class: an unbounded class-0 refiner (a Google
    /// P3DT world layer) must NOT be able to starve a class-1 twin's root.
    #[test]
    fn class1_starving_root_beats_class0_refinement() {
        let mut c = vec![
            cand(2, 0, false, 3), // class-0 world layer, refining forever
            cand(2, 0, false, 4),
            cand(1, 1, true, 0), // class-1 twin with nothing Ready yet
        ];
        sort_load_candidates(&mut c, &no_cursors());
        assert_eq!(c[0].set_id, 1, "starving root wins across classes");
    }

    /// (b) Class orders like-for-like: refinement vs refinement, class 0 first.
    /// (Same among starving roots — terrain root before twin roots.)
    #[test]
    fn class_zero_beats_class_one() {
        let mut c = vec![
            cand(1, 1, false, 0), // class-1 refinement
            cand(2, 0, false, 3), // class-0 refinement
        ];
        sort_load_candidates(&mut c, &no_cursors());
        assert_eq!(c[0].set_id, 2, "class 0 refinement first");

        let mut c = vec![cand(1, 1, true, 0), cand(2, 0, true, 0)];
        sort_load_candidates(&mut c, &no_cursors());
        assert_eq!(c[0].set_id, 2, "class 0 root first among starving roots");
    }

    /// (c) Round-robin: under a saturated pool (one grant per frame), the
    /// cursor rotates the grant across equal-class sets over successive calls.
    #[test]
    fn round_robin_rotates_across_equal_class_sets() {
        let fresh = || {
            vec![
                cand(1, 1, false, 0),
                cand(1, 1, false, 1),
                cand(2, 1, false, 0),
                cand(2, 1, false, 1),
                cand(3, 1, false, 0),
                cand(3, 1, false, 1),
            ]
        };
        let mut cursors = no_cursors();
        let mut served = Vec::new();
        for _ in 0..4 {
            let mut c = fresh();
            sort_load_candidates(&mut c, &cursors);
            cursors.insert(c[0].class, c[0].set_id); // pool of 1: only the first is granted
            served.push(c[0].set_id);
        }
        assert_eq!(served, vec![1, 2, 3, 1], "rotation wraps across frames");
    }

    /// (c) Rotation is PER CLASS, and this is what a shared cursor breaks:
    /// alternate a frame whose whole pool goes to class 0 with a frame that
    /// leaves one slot over. One cursor would end each saturated frame keyed
    /// to a class-0 id ABOVE every class-1 id, so every leftover slot sorts
    /// class 1 from scratch and lands on set 1 forever (`[1, 1, 1, 1]`);
    /// per-class cursors keep class 1 rotating across the leftover slots.
    #[test]
    fn round_robin_rotates_per_class() {
        // Class-0 ids sit above the class-1 ids, as a late-attached world
        // layer's would. Refinements only — a starving root would jump ahead.
        let frame = |class0_reqs: usize| {
            let mut c: Vec<LoadCandidate> = (0..class0_reqs)
                .map(|i| cand(100 + i as u64, 0, false, 0))
                .collect();
            c.extend([
                cand(1, 1, false, 0),
                cand(2, 1, false, 0),
                cand(3, 1, false, 0),
            ]);
            c
        };
        let mut cursors = no_cursors();
        let mut leftover_served = Vec::new();
        for _ in 0..4 {
            // Saturated frame: pool of 2, both slots taken by class 0.
            let mut c = frame(2);
            sort_load_candidates(&mut c, &cursors);
            assert!(c[..2].iter().all(|g| g.class == 0), "class 0 saturates");
            for g in c.iter().take(2) {
                cursors.insert(g.class, g.set_id);
            }
            // Leftover frame: class 0 wants one slot, class 1 gets the other.
            let mut c = frame(1);
            sort_load_candidates(&mut c, &cursors);
            for g in c.iter().take(2) {
                cursors.insert(g.class, g.set_id);
            }
            assert_eq!(c[1].class, 1, "the leftover slot is class 1's");
            leftover_served.push(c[1].set_id);
        }
        assert_eq!(
            leftover_served,
            vec![1, 2, 3, 1],
            "class-1 rotation survives class-0 saturation frames"
        );
    }

    /// (c)+(d) Within one frame the order is breadth-first across sets
    /// (every set's k-th request before any set's (k+1)-th), and within one
    /// set the `sel.loads` order is preserved.
    #[test]
    fn within_set_selection_order_preserved() {
        let mut c = vec![
            cand(2, 1, false, 1),
            cand(1, 1, false, 2),
            cand(2, 1, false, 0),
            cand(1, 1, false, 0),
            cand(1, 1, false, 1),
        ];
        sort_load_candidates(&mut c, &no_cursors());
        let order: Vec<(u64, usize)> = c.iter().map(|c| (c.set_id, c.within)).collect();
        assert_eq!(order, vec![(1, 0), (2, 0), (1, 1), (2, 1), (1, 2)]);
        let set1: Vec<usize> = c
            .iter()
            .filter(|c| c.set_id == 1)
            .map(|c| c.within)
            .collect();
        assert_eq!(set1, vec![0, 1, 2], "within-set order intact");
    }

    /// The cursor is keyed by set id, so it stays sane when the set it points
    /// at detaches: rotation continues from the next id past it.
    #[test]
    fn rotation_cursor_survives_set_detach() {
        // Cursor points at set 2, which has since detached.
        let mut c = vec![cand(1, 1, false, 0), cand(3, 1, false, 0)];
        sort_load_candidates(&mut c, &std::collections::HashMap::from([(1u8, 2u64)]));
        assert_eq!(c[0].set_id, 3, "next id past the dead cursor goes first");
        // Cursor past every live id wraps to the lowest.
        let mut c = vec![cand(1, 1, false, 0), cand(3, 1, false, 0)];
        sort_load_candidates(&mut c, &std::collections::HashMap::from([(1u8, 99u64)]));
        assert_eq!(c[0].set_id, 1, "wraps to the lowest live id");
    }

    // ── Hidden-tile despawn / respawn (0.2.4) ────────────────────────────────
    //
    // These drive the real `receive_tiles3d`/`drive_tiles3d` chain over a
    // synthetic all-Ready set, which is the only way to test the swap ORDERING
    // (the invariant that actually bites) rather than the selection math alone.

    /// Per-tile ledger cost of the synthetic sets below.
    const TEST_TILE_BYTES: u64 = 100;

    /// Streaming config for the synthetic sets: a spawn budget big enough that a
    /// whole cut lands in one frame (tests that care about the budget lower it),
    /// and no distance falloff so the SSE assertions stay exact.
    fn test_config() -> Tiles3dConfig {
        Tiles3dConfig {
            max_spawns_per_frame: 64,
            detail_falloff_m: 0.0,
            ..Default::default()
        }
    }

    /// Just enough app to run the streamer's two systems: real `Assets<Mesh>` (so
    /// cached handles are real handles), the `RequestRedraw` message, and the
    /// plugin. No render/window/transform plugins — nothing here draws, and
    /// `GlobalTransform`s are set by hand.
    fn despawn_test_app(config: Tiles3dConfig) -> App {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .add_message::<RequestRedraw>()
            .insert_resource(config)
            .add_plugins(Tiles3dPlugin);
        app
    }

    /// root(depth 0) → 4 children(depth 1) → 4 leaves each(depth 2), REPLACE,
    /// all content-bearing, spheres around the origin. `ge = [root, child, leaf]`
    /// drives the cut: for a 1080p 45° camera at z = 600, `[100, 6, 0]` refines
    /// the root at any sane threshold and the children only below ~13 px.
    fn synth_tree(ge: [f64; 3]) -> TileTree {
        let node = |parent, depth, geometric_error, center, radius| traversal::TileNode {
            parent,
            children: vec![],
            depth,
            geometric_error,
            refine: schema::Refine::Replace,
            content_uri: Some("t.glb".into()),
            volume: traversal::WorldVolume::Sphere { center, radius },
            world_from_content: DMat4::IDENTITY,
            world_from_tile: DMat4::IDENTITY,
        };
        let mut tree = TileTree::default();
        tree.nodes.push(node(None, 0, ge[0], DVec3::ZERO, 100.0));
        let quad = [(-30.0, -30.0), (30.0, -30.0), (-30.0, 30.0), (30.0, 30.0)];
        for (cx, cz) in quad {
            let c = tree.nodes.len();
            tree.nodes
                .push(node(Some(0), 1, ge[1], DVec3::new(cx, 0.0, cz), 25.0));
            tree.nodes[0].children.push(c);
            for (lx, lz) in quad {
                let l = tree.nodes.len();
                let center = DVec3::new(cx + lx * 0.25, 0.0, cz + lz * 0.25);
                tree.nodes.push(node(Some(c), 2, ge[2], center, 8.0));
                tree.nodes[c].children.push(l);
            }
        }
        tree
    }

    fn tiles_at_depth(tree: &TileTree, depth: u32) -> Vec<usize> {
        (0..tree.len())
            .filter(|&i| tree.nodes[i].depth == depth)
            .collect()
    }

    /// Install one synthetic set with every tile already `Ready` but
    /// despawned-but-cached (one cached mesh each), plus the streamer camera.
    /// Returns `(anchor, camera, mesh handles)`.
    fn install_set(
        app: &mut App,
        tree: TileTree,
        frame: SetFrame,
        cam: Vec3,
    ) -> (Entity, Entity, Vec<Handle<Mesh>>) {
        let n = tree.len();
        let world = app.world_mut();
        let anchor = world
            .spawn((Transform::IDENTITY, GlobalTransform::IDENTITY))
            .id();
        let root_entity = world
            .spawn((
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                Visibility::default(),
            ))
            .id();
        let camera = world
            .spawn((
                Camera::default(),
                Projection::default(),
                Frustum::default(),
                GlobalTransform::from(Transform::from_translation(cam)),
                Tiles3dCamera,
            ))
            .id();
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        let handles: Vec<Handle<Mesh>> = (0..n)
            .map(|_| {
                meshes.add(Mesh::new(
                    bevy::mesh::PrimitiveTopology::TriangleList,
                    bevy::asset::RenderAssetUsages::default(),
                ))
            })
            .collect();
        let caches: Vec<Vec<CachedItem>> = handles
            .iter()
            .map(|h| {
                vec![CachedItem::Mesh {
                    mesh: h.clone(),
                    material: Handle::default(),
                    transform: Transform::IDENTITY,
                    pick: None,
                }]
            })
            .collect();
        let mut history = History::default();
        history.resize(n);
        let mut sets = world.resource_mut::<Tiles3dSets>();
        sets.sets.push(ActiveTileset {
            id: 1,
            label: "synth".into(),
            tree,
            source: TilesetSource::Exploded(ExplodedBase::Url("http://x".into())),
            slots: vec![
                TileSlot::Ready {
                    entity: None,
                    bytes: TEST_TILE_BYTES,
                };
                n
            ],
            caches,
            history,
            last_touched: vec![0; n],
            grafts: Vec::new(),
            compact_high_water: n,
            root_entity,
            anchor: Some(anchor),
            owner_id: None,
            sse_threshold_px: None,
            placeholder_cleared: true,
            last_cut: None,
            frame,
            rtc_centers: vec![None; n],
            copyrights: BTreeSet::new(),
            budget_warned: false,
        });
        (anchor, camera, handles)
    }

    /// Tiles of set 0 that are actually on screen: entity spawned AND visible.
    fn visible_tiles(app: &App) -> Vec<usize> {
        let world = app.world();
        world.resource::<Tiles3dSets>().sets[0]
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                matches!(slot, TileSlot::Ready { entity: Some(e), .. }
                    if world.get::<Visibility>(*e) == Some(&Visibility::Visible))
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn move_camera(app: &mut App, camera: Entity, pos: Vec3) {
        *app.world_mut().get_mut::<GlobalTransform>(camera).unwrap() =
            GlobalTransform::from(Transform::from_translation(pos));
    }

    /// Pan the set out of view (and back). There are no render plugins here, so
    /// the camera's `Frustum` is inert — `default()`'s zero half-spaces intersect
    /// everything — and a pan has to be written by hand: one half-space that
    /// rejects everything near the origin. Slot 0 because `intersects_sphere(_,
    /// false)` only tests the first five.
    fn look_away(app: &mut App, camera: Entity, away: bool) {
        let mut frustum = Frustum::default();
        if away {
            frustum.half_spaces[0] =
                bevy::camera::primitives::HalfSpace::new(Vec4::new(0.0, 0.0, 1.0, -1.0e9));
        }
        *app.world_mut().get_mut::<Frustum>(camera).unwrap() = frustum;
    }

    /// (a) A tile that leaves the cut gives up its ENTITIES but keeps its assets:
    /// the slot stays `Ready`, the cached mesh is still in `Assets<Mesh>`, and the
    /// memory ledger is byte-for-byte unchanged — the memory IS still resident,
    /// deliberately.
    #[test]
    fn hidden_tile_despawns_entities_but_keeps_assets() {
        let mut app = despawn_test_app(test_config());
        let tree = synth_tree([100.0, 6.0, 0.0]);
        let leaves = tiles_at_depth(&tree, 2);
        let (_, cam, handles) = install_set(
            &mut app,
            tree,
            SetFrame::Anchored,
            Vec3::new(0.0, 0.0, 600.0),
        );
        let ledger = app
            .world()
            .resource::<Tiles3dSets>()
            .resident_content_bytes();

        // Close in: the cut is the leaf level (spawn hidden, then show).
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), leaves, "leaf cut on screen");

        // Pull back: the root alone covers the view, so every leaf is hidden.
        move_camera(&mut app, cam, Vec3::new(0.0, 0.0, 60_000.0));
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), vec![0], "root cut on screen");

        let sets = app.world().resource::<Tiles3dSets>();
        for &l in &leaves {
            assert!(
                matches!(
                    sets.sets[0].slots[l],
                    TileSlot::Ready {
                        entity: None,
                        bytes: TEST_TILE_BYTES
                    }
                ),
                "leaf {l} despawned but still Ready"
            );
            assert_eq!(sets.sets[0].caches[l].len(), 1, "leaf {l} kept its cache");
        }
        assert_eq!(
            sets.resident_content_bytes(),
            ledger,
            "ledger unchanged by despawn"
        );
        let meshes = app.world().resource::<Assets<Mesh>>();
        assert!(
            handles.iter().all(|h| meshes.get(h).is_some()),
            "every cached mesh is still in Assets<Mesh>"
        );
    }

    /// (b) A REPLACE **refinement** swap shows EXACTLY ONE rung per frame: either
    /// the coarse parent, or the WHOLE finer cut. Never neither (a hole), and
    /// never both — a parent's geometry sits slightly off the surface its children
    /// draw, so any overlap reads as z-shimmer for as long as it lasts. Driven at
    /// one respawn per frame so the children trickle in over many frames, which is
    /// the fast-orbit case the hold and the deferred show exist for.
    #[test]
    fn replace_swap_shows_exactly_one_rung() {
        let mut app = despawn_test_app(Tiles3dConfig {
            max_respawns_per_frame: 1,
            ..test_config()
        });
        let tree = synth_tree([100.0, 6.0, 0.0]);
        let leaves = tiles_at_depth(&tree, 2);
        // Settled far away first: the root alone is the cut.
        let (_, cam, _) = install_set(
            &mut app,
            tree,
            SetFrame::Anchored,
            Vec3::new(0.0, 0.0, 60_000.0),
        );
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), vec![0], "root cut first");

        // Zoom in: the cut becomes the 16 leaves, one respawn per frame.
        move_camera(&mut app, cam, Vec3::new(0.0, 0.0, 600.0));
        let mut swapped = false;
        for f in 0..40 {
            app.update();
            let vis = visible_tiles(&app);
            let parent_up = vis.contains(&0);
            let children_up = leaves.iter().all(|l| vis.contains(l));
            let any_child_up = leaves.iter().any(|l| vis.contains(l));
            assert!(
                !(parent_up && any_child_up),
                "frame {f}: coarse parent co-rendering with arrived children ({vis:?})"
            );
            assert!(
                parent_up || children_up,
                "frame {f}: neither the parent nor the full child cut is visible ({vis:?})"
            );
            swapped |= children_up;
        }
        assert!(swapped, "the swap completed inside 40 frames");
    }

    /// (b2) The COARSENING direction, which an ancestors-only hold would miss:
    /// zooming out makes the parent the cut and its children the out-of-cut tiles,
    /// so the CHILDREN are what has to hold until the parent's respawn is on
    /// screen. Weaker on purpose — NO HOLE is the invariant here, and the incoming
    /// parent overlapping its outgoing children for one frame is the accepted
    /// deviation: coarse draws over fine, which beats a gap. (b) is the strict
    /// exactly-one-rung direction.
    #[test]
    fn coarsening_swap_never_shows_a_hole() {
        let mut app = despawn_test_app(Tiles3dConfig {
            max_respawns_per_frame: 1,
            ..test_config()
        });
        let tree = synth_tree([100.0, 6.0, 0.0]);
        let leaves = tiles_at_depth(&tree, 2);
        let (_, cam, _) = install_set(
            &mut app,
            tree,
            SetFrame::Anchored,
            Vec3::new(0.0, 0.0, 600.0),
        );
        // `install_set` starts every tile despawned-but-cached, so the FIRST cut
        // is a respawn burst too — settle it at one per frame before measuring.
        for _ in 0..40 {
            app.update();
        }
        assert_eq!(visible_tiles(&app), leaves, "leaf cut first");

        // Pull back: the root alone becomes the cut, and it has no entity.
        move_camera(&mut app, cam, Vec3::new(0.0, 0.0, 60_000.0));
        let mut swapped = false;
        for f in 0..10 {
            app.update();
            let vis = visible_tiles(&app);
            let parent_up = vis.contains(&0);
            let children_up = leaves.iter().all(|l| vis.contains(l));
            assert!(
                parent_up || children_up,
                "frame {f}: neither the parent nor the full child cut is visible ({vis:?})"
            );
            swapped |= parent_up && !children_up;
        }
        assert!(swapped, "the coarsening swap completed");
    }

    /// (b2b) Cut ENTRY from cache with NOTHING to hold: a footprint that panned
    /// out of view has its whole ancestor chain despawned, so no coarse rung is on
    /// screen to cover the re-entry and `refinement_hold` has nothing to pick.
    /// The respawn must therefore PAINT on the frame it is wanted — landing hidden
    /// for the next frame's show pass is a hole (~90 ms of nothing at the frame
    /// rates this feature exists for).
    #[test]
    fn cut_entry_from_cache_paints_the_same_frame() {
        let mut app = despawn_test_app(test_config());
        let tree = synth_tree([100.0, 6.0, 0.0]);
        let leaves = tiles_at_depth(&tree, 2);
        let (_, cam, _) = install_set(
            &mut app,
            tree,
            SetFrame::Anchored,
            Vec3::new(0.0, 0.0, 600.0),
        );
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), leaves, "leaf cut on screen");

        // Pan away: the cut empties, so the whole set — ancestors included —
        // gives up its entities and keeps only its caches.
        look_away(&mut app, cam, true);
        app.update();
        assert!(visible_tiles(&app).is_empty(), "nothing on screen off-view");
        assert!(
            app.world().resource::<Tiles3dSets>().sets[0]
                .slots
                .iter()
                .all(|s| matches!(s, TileSlot::Ready { entity: None, .. })),
            "the whole set is despawned-but-cached — no rung left to hold"
        );

        // Pan back: ONE frame, and the cut is painting.
        look_away(&mut app, cam, false);
        app.update();
        assert_eq!(
            visible_tiles(&app),
            leaves,
            "re-entry paints the frame it is wanted"
        );
    }

    /// (b3) The hold is NARROW: only the starved tile's own ancestors and
    /// descendants, never the rest of the set. A set-wide hold kept every
    /// out-of-cut tile spawned AND visible for as long as the camera moved,
    /// which is the whole cost this feature exists to remove.
    #[test]
    fn the_hold_covers_only_the_starved_tile_s_line() {
        let tree = synth_tree([100.0, 6.0, 0.0]);
        let starved = *tiles_at_depth(&tree, 2).first().unwrap(); // leaf under child 1
        let parent = tree.nodes[starved].parent.unwrap();
        let mut slots = vec![
            TileSlot::Ready {
                entity: Some(Entity::from_raw_u32(1).unwrap()),
                bytes: TEST_TILE_BYTES,
            };
            tree.len()
        ];
        slots[starved] = TileSlot::Ready {
            entity: None,
            bytes: TEST_TILE_BYTES,
        };
        let mut want = vec![false; tree.len()];
        want[starved] = true;

        let (hold, is_starved) = refinement_hold(&tree, &want, &slots);
        assert!(is_starved);
        assert!(hold[0], "the root ancestor holds");
        assert!(hold[parent], "the immediate parent holds");
        assert!(!hold[starved], "the starved tile itself is not 'held'");
        // A sibling subtree is unrelated screen area — it must be free to go.
        let other = tiles_at_depth(&tree, 1)
            .into_iter()
            .find(|&c| c != parent)
            .unwrap();
        assert!(!hold[other], "an unrelated child does NOT hold");
        for &l in &tree.nodes[other].children {
            assert!(!hold[l], "an unrelated leaf does NOT hold");
        }

        // Starve the ROOT instead: now every descendant is the coverage.
        let mut slots = vec![
            TileSlot::Ready {
                entity: Some(Entity::from_raw_u32(1).unwrap()),
                bytes: TEST_TILE_BYTES,
            };
            tree.len()
        ];
        slots[0] = TileSlot::Ready {
            entity: None,
            bytes: TEST_TILE_BYTES,
        };
        let mut want = vec![false; tree.len()];
        want[0] = true;
        let (hold, _) = refinement_hold(&tree, &want, &slots);
        assert!(
            (1..tree.len()).all(|i| hold[i]),
            "a starved root holds its whole subtree"
        );
    }

    /// (c) A tile hidden across an ECEF origin rebase respawns against the
    /// CURRENT origin, not the one it was first spawned at.
    #[test]
    fn respawn_places_against_the_current_ecef_origin() {
        let mut app = despawn_test_app(test_config());
        let o1 = DMat4::from_translation(DVec3::new(1000.0, 0.0, 0.0));
        let o2 = DMat4::from_translation(DVec3::new(0.0, 0.0, -2000.0));
        app.insert_resource(EcefOrigin {
            world_from_ecef: Some(o1),
        });
        // Children carry no geometric error, so the cut stops at depth 1 and the
        // root is the hidden tile under test.
        let tree = synth_tree([100.0, 0.0, 0.0]);
        let children = tiles_at_depth(&tree, 1);
        let (_, cam, _) = install_set(
            &mut app,
            tree,
            SetFrame::Ecef { built: None },
            Vec3::new(1000.0, 0.0, 600.0),
        );
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), children, "child cut at origin 1");
        assert!(
            matches!(
                app.world().resource::<Tiles3dSets>().sets[0].slots[0],
                TileSlot::Ready { entity: None, .. }
            ),
            "root is despawned-but-cached"
        );

        // Rebase the origin AND pull back so the root becomes the cut.
        app.insert_resource(EcefOrigin {
            world_from_ecef: Some(o2),
        });
        move_camera(&mut app, cam, Vec3::new(0.0, 0.0, 58_000.0));
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), vec![0], "root cut after the rebase");
        let TileSlot::Ready {
            entity: Some(root), ..
        } = app.world().resource::<Tiles3dSets>().sets[0].slots[0]
        else {
            panic!("root respawned");
        };
        let t = app.world().get::<Transform>(root).unwrap();
        assert!(
            t.translation
                .abs_diff_eq(Vec3::new(0.0, 0.0, -2000.0), 1e-3),
            "respawned at the CURRENT origin, got {:?}",
            t.translation
        );
    }

    /// (d) `TileSseMultiplier` is the "ground tilesets don't need twin-grade
    /// density" knob: the same tree and camera that refine to the leaf level at
    /// 1.0 stop one level coarser at 2.0.
    #[test]
    fn sse_multiplier_coarsens_the_cut() {
        let cut = |mult: Option<f32>| -> Vec<usize> {
            let mut app = despawn_test_app(test_config());
            let (anchor, _, _) = install_set(
                &mut app,
                synth_tree([100.0, 6.0, 0.0]),
                SetFrame::Anchored,
                Vec3::new(0.0, 0.0, 600.0),
            );
            if let Some(m) = mult {
                app.world_mut()
                    .entity_mut(anchor)
                    .insert(TileSseMultiplier(m));
            }
            app.update();
            app.update();
            visible_tiles(&app)
        };
        let tree = synth_tree([100.0, 6.0, 0.0]);
        assert_eq!(cut(None), tiles_at_depth(&tree, 2), "absent → leaf cut");
        assert_eq!(cut(Some(1.0)), tiles_at_depth(&tree, 2), "1.0 == absent");
        assert_eq!(
            cut(Some(2.0)),
            tiles_at_depth(&tree, 1),
            "2.0 → the coarser child cut"
        );
    }

    /// (e) A tile inside the eviction grace window is despawned-but-CACHED: its
    /// slot stays `Ready`, it is never re-requested, and coming back into view
    /// costs a spawn — the decode counter never moves.
    #[test]
    fn grace_tile_keeps_its_cache_and_never_re_decodes() {
        let mut app = despawn_test_app(test_config());
        let tree = synth_tree([100.0, 6.0, 0.0]);
        let leaves = tiles_at_depth(&tree, 2);
        let (_, cam, _) = install_set(
            &mut app,
            tree,
            SetFrame::Anchored,
            Vec3::new(0.0, 0.0, 600.0),
        );
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), leaves, "leaf cut on screen");

        // Out of the cut, well inside `grace_frames`.
        move_camera(&mut app, cam, Vec3::new(0.0, 0.0, 60_000.0));
        app.update();
        app.update();
        {
            let sets = app.world().resource::<Tiles3dSets>();
            for &l in &leaves {
                assert!(
                    matches!(sets.sets[0].slots[l], TileSlot::Ready { entity: None, .. }),
                    "grace tile {l} kept its Ready slot"
                );
                assert_eq!(sets.sets[0].caches[l].len(), 1, "tile {l} kept its cache");
            }
        }

        // Back in: respawn only — no fetch was issued, nothing decoded.
        move_camera(&mut app, cam, Vec3::new(0.0, 0.0, 600.0));
        app.update();
        app.update();
        assert_eq!(visible_tiles(&app), leaves, "leaf cut back on screen");
        let sets = app.world().resource::<Tiles3dSets>();
        assert!(
            !sets.sets[0]
                .slots
                .iter()
                .any(|s| matches!(s, TileSlot::InFlight { .. } | TileSlot::NotLoaded)),
            "re-entry re-requested nothing"
        );
        assert_eq!(
            app.world().resource::<Tiles3dDecodeStats>().tiles,
            0,
            "no tile was decoded — re-entry is a spawn, not a decode"
        );
    }

    /// Aborting a registered generation flips its handle; a triggered source
    /// read returns `Aborted` instead of bytes.
    #[test]
    fn abort_registry_cancels_reads() {
        let abort = fetch::register_abort(987_654);
        assert!(!abort.is_triggered());
        fetch::trigger_abort(987_654);
        assert!(abort.is_triggered());
        let src = ByteSource::Mem(Arc::new(vec![0u8; 64]));
        let res = block_on(src.read_abortable(0, 8, Some(&abort)));
        assert!(matches!(res, Err(fetch::FetchError::Aborted)));
        fetch::unregister_abort(987_654);
        // Unregistered generations are no-ops.
        fetch::trigger_abort(987_654);
    }
}
