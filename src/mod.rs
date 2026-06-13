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

use std::collections::BTreeSet;
use std::sync::Arc;

use bevy::camera::primitives::{Frustum, Sphere};
use bevy::camera::Projection;
use bevy::math::{DMat4, DVec3, Vec3A};
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use bevy_panorbit_camera::PanOrbitCamera;

pub mod archive;
pub mod content;
pub mod draco;
pub mod fetch;
pub mod geo;
// wasm-only KTX2 transcode shim binding (T7); native uses bevy basis-universal.
#[cfg(target_arch = "wasm32")]
pub mod ktx2;
pub mod meshopt;
pub mod schema;
pub mod traversal;

use archive::Archive3tz;
use content::{DecodedItem, DecodedTile};
use fetch::{BudgetCounter, ByteSource, ExplodedBase, LiveSession, TilesetSource};
use traversal::{History, SelectParams, TileContent, TileTree, TreeFrame, ZUP_TO_BEVY};

use turbotwin_sdk_rs::enu::WGS84_EQUATORIAL_RADIUS_M;

use crate::plugins::scene_layout::TwinMeshGroup;
use crate::plugins::spatial_source::{ProjectOrigin, ProjectOriginInner};

/// The Google Photorealistic 3D Tiles root tileset (D7). The org's API key
/// is appended per request, never stored in the row.
pub const GOOGLE_P3DT_ROOT_URL: &str = "https://tile.googleapis.com/v1/3dtiles/root.json";

/// Committed demo tileset (see `examples/gen_tiles3d_fixture.rs`). The path
/// doubles as a relative URL under Trunk (assets are `copy-dir`'d) and a
/// relative file path for native runs from the crate root.
const FIXTURE_SPEC: &str = "assets/fixtures/tiles3d-demo/tileset.json";

#[derive(Resource, Debug, Clone)]
pub struct Tiles3dConfig {
    /// Refine while a tile's screen-space error exceeds this (px).
    pub sse_threshold_px: f64,
    /// Distance-relaxed detail falloff (metres) — see
    /// [`SelectParams::detail_falloff_m`]. Caps how far the cut refines toward
    /// the horizon so a grazing view doesn't graft+stream the whole visible
    /// hemisphere (the P3DT "tilt → 98 k-tile tree" finding). `0` disables.
    pub detail_falloff_m: f64,
    /// Max tile fetch+decode tasks in flight per tileset. Basemap's proven
    /// starting point; retune against CesiumJS's 50/18 once Front Door
    /// HTTP/2 carries tile traffic (D10).
    pub max_concurrent_loads: usize,
    /// Main-thread time box: max decoded tiles turned into entities per frame.
    pub max_spawns_per_frame: usize,
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
}

impl Default for Tiles3dConfig {
    fn default() -> Self {
        Self {
            sse_threshold_px: traversal::DEFAULT_SSE_THRESHOLD_PX,
            // ~2 km: near terrain stays sharp, the horizon coarsens. Tuned
            // against the live autzen P3DT view (cam ~10–20 m up); raise for
            // high-altitude orbits, lower if the tree still grows too far out.
            detail_falloff_m: 2000.0,
            // 32 (up from 16): the startup descent walks a ~20-deep chain of
            // nested tileset.json grafts to reach street-level detail; more
            // parallel slots resolve the breadth of that descent faster.
            max_concurrent_loads: 32,
            max_spawns_per_frame: 4,
            grace_frames: 600,
            max_resident_tiles: 1024,
            // Comfortable working set before reclaiming: the grace window keeps
            // recently-seen grafts, so this is mostly the floor that stops us
            // compacting tiny trees. Re-grafting a reclaimed area is a real
            // re-fetch (P3DT is never CAS-cached), so don't set it too tight.
            tree_compact_min: 16_384,
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
    /// Owning twin, when anchored to one — spawned tile content gets
    /// `TwinMeshGroup` so click-select / highlight / focus keep working, and
    /// the twin's placeholder cube clears on the first rendered cut.
    pub twin_id: Option<String>,
    /// Display label for logs/debug (asset id, twin id…).
    pub label: String,
    /// Google P3DT session config: routes the open through a live, keyed,
    /// budget-capped, never-cached source (D7). `None` = a normal tileset.
    pub p3dt: Option<P3dtParams>,
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

/// Per-feature picking table on a tile mesh entity (T8). The tile mesh is
/// already `TwinMeshGroup`-tagged with the tileset's anchor twin, so a click
/// resolves to the whole tileset by default; this refines a *specific*
/// triangle to the section twin that embodies it. `selection` reads it on a
/// pick: triangle ordinal → `feature_of_triangle` → `node_of_feature` path →
/// `sections::resolve_feature_twin(.., anchor_twin)` → the twin to select.
#[derive(Component, Clone)]
pub struct TileFeatureTable {
    /// featureId per triangle (index-buffer order; the pick raycast's triangle
    /// ordinal indexes straight in).
    pub feature_of_triangle: Vec<u32>,
    /// featureId → source-node path (shared across the tile's primitives).
    pub node_of_feature: std::sync::Arc<Vec<String>>,
    /// The tileset's owning twin — the anchor the node→twin resolution is
    /// scoped to (and the fallback when no path segment matches a section).
    pub anchor_twin: String,
}

/// Per-tile load slot.
#[derive(Debug, Clone, Copy)]
enum TileSlot {
    NotLoaded,
    /// Fetch+decode task running; results carry the generation so a
    /// cancelled-then-reissued tile drops the stale payload. The generation
    /// also keys the abort registry (`fetch::trigger_abort`).
    InFlight { generation: u64 },
    /// Content spawned (hidden until selected by the render cut).
    Ready { entity: Entity },
    /// Terminal fetch/decode failure — never re-queued this session.
    Failed,
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
    Ecef { built: Option<ProjectOriginInner> },
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
    /// Owning twin id (TwinMeshGroup tagging + placeholder clearing).
    twin_id: Option<String>,
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

/// Snapshot for the debug overlay / tests.
pub struct Tiles3dDebug {
    pub tilesets: usize,
    pub resident: usize,
    pub in_flight: usize,
}

impl Tiles3dSets {
    pub fn debug_summary(&self) -> Tiles3dDebug {
        let (mut resident, mut in_flight) = (0, 0);
        for set in &self.sets {
            for slot in &set.slots {
                match slot {
                    TileSlot::Ready { .. } => resident += 1,
                    TileSlot::InFlight { .. } => in_flight += 1,
                    _ => {}
                }
            }
        }
        Tiles3dDebug { tilesets: self.sets.len(), resident, in_flight }
    }

    /// Whether `anchor` is taken: streaming, opening, or terminally failed.
    /// The asset loader treats `true` as "nothing to do this frame".
    pub fn has_anchor(&self, anchor: Entity) -> bool {
        self.pending_anchors.contains(&anchor)
            || self.failed_anchors.contains(&anchor)
            || self.sets.iter().any(|s| s.anchor == Some(anchor))
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
}

/// Anchor info carried through the async tileset open.
#[derive(Debug, Clone)]
struct AttachTarget {
    anchor: Entity,
    local: Transform,
    twin_id: Option<String>,
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

pub struct Tiles3dPlugin;

impl Plugin for Tiles3dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tiles3dConfig>()
            .init_resource::<Tiles3dSets>()
            .init_resource::<Tiles3dChannel>()
            .init_resource::<TilesetCredits>()
            .add_message::<Tiles3dAttach>()
            .add_message::<Tiles3dDetach>()
            .add_systems(Startup, (latch_compressed_formats, init_dev_tileset))
            .add_systems(
                Update,
                (
                    apply_attach_detach,
                    receive_tiles3d,
                    drive_tiles3d,
                    update_google_logo,
                )
                    .chain(),
            );
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
    let Some(spec) = dev_source_spec() else { return };
    let spec = if spec == "fixture" { FIXTURE_SPEC.to_string() } else { spec };
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
        info!("tiles3d: attaching {} ({}) to {:?}", msg.label, msg.url, msg.anchor);
        sets.pending_anchors.insert(msg.anchor);
        spawn_tileset_open(
            msg.url.clone(),
            msg.p3dt.clone(),
            Some(AttachTarget {
                anchor: msg.anchor,
                local: msg.local,
                twin_id: msg.twin_id.clone(),
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
        let _ = tx.send(Tiles3dMsg::TilesetOpened { label: spec, attach, result });
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
    spec.split(['?', '#']).next().unwrap_or(spec).ends_with(".3tz")
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
        let tileset =
            schema::parse_tileset(&bytes).map_err(|e| format!("parse P3DT root: {e}"))?;
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
    origin: Res<ProjectOrigin>,
    mut sets: ResMut<Tiles3dSets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut clouds: ResMut<Assets<bevy_pointcloud::point_cloud::PointCloud>>,
    mut splats: ResMut<Assets<bevy_gaussian_splatting::PlanarGaussian3d>>,
    shared_point_material: Res<crate::plugins::point_cloud::SharedPointMaterial>,
    mut commands: Commands,
) {
    let mut spawned = 0usize;
    while spawned < config.max_spawns_per_frame {
        let Ok(msg) = channel.rx.try_recv() else { break };
        match msg {
            Tiles3dMsg::TilesetOpened { label, attach, result } => match result {
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
                        (SetFrame::Ecef { built: None }, DMat4::IDENTITY, TreeFrame::Ecef)
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
                                root.insert(
                                    attach.as_ref().map(|a| a.local).unwrap_or_default(),
                                );
                                if let Some(a) = &attach {
                                    root.insert(ChildOf(a.anchor));
                                }
                            }
                            let root_entity = root.id();
                            let mut history = History::default();
                            history.resize(n);
                            info!(
                                "tiles3d: {label}: {n} tiles{}",
                                if georef { " (georeferenced — ECEF frame)" } else { "" }
                            );
                            sets.sets.push(ActiveTileset {
                                id,
                                label,
                                tree,
                                source,
                                slots: vec![TileSlot::NotLoaded; n],
                                history,
                                last_touched: vec![0; n],
                                grafts: Vec::new(),
                                compact_high_water: n,
                                root_entity,
                                anchor: attach.as_ref().map(|a| a.anchor),
                                twin_id: attach.and_then(|a| a.twin_id),
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
            Tiles3dMsg::TileContent { set_id, generation, result } => {
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
                        let DecodedTile { items, rtc_center, copyright } = *decoded;
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
                        let Some(transform) = tile_spawn_transform(set, tile, origin.get())
                        else {
                            set.slots[tile] = TileSlot::NotLoaded;
                            continue;
                        };
                        let entity = spawn_tile_content(
                            &mut commands,
                            &mut meshes,
                            &mut materials,
                            &mut images,
                            &mut clouds,
                            &mut splats,
                            &shared_point_material,
                            set,
                            tile,
                            transform,
                            items,
                        );
                        set.slots[tile] = TileSlot::Ready { entity };
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
    origin: Option<ProjectOriginInner>,
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

/// `world_from_ecef(origin) × ecef_from_content × T(rtc_center)`, in f64.
fn compose_ecef_tile_matrix(
    origin: ProjectOriginInner,
    ecef_from_content: DMat4,
    rtc_center: Option<DVec3>,
) -> DMat4 {
    let world_from_ecef = crate::plugins::spatial_source::enu::world_from_ecef(origin);
    let mut m = world_from_ecef * ecef_from_content;
    if let Some(rtc) = rtc_center {
        m *= DMat4::from_translation(rtc);
    }
    m
}

/// Whether a tile's bounding sphere is entirely beyond the horizon — occluded
/// by the planet (radius `planet_r`, centred at the **set-frame origin**) as
/// seen from `cam`. Both `cam` and `center` are in the set frame (ECEF metres
/// for globe sets). Cesium's scaled-space occlusion test, specialised to a
/// sphere; the test point is lifted to the top of the bounding sphere (the
/// part most likely to peek over the limb) so tiles straddling the horizon are
/// kept — it only culls tiles that are *fully* behind the curve.
fn beyond_horizon(cam: DVec3, center: DVec3, radius: f64, planet_r: f64) -> bool {
    let up = center.length();
    if up < 1.0 {
        return false; // planet-centred (whole-globe) volume — never cull
    }
    // Lift the test point radially out to the top of the bounding sphere.
    let test = center * (1.0 + radius / up);
    // Scale so the planet is the unit sphere.
    let c = cam / planet_r;
    let p = test / planet_r;
    let vh2 = c.length_squared() - 1.0; // squared scaled camera→horizon distance
    if vh2 <= 0.0 {
        return false; // camera at/under the surface — disable the cull
    }
    let vt = p - c;
    let vt_dot_vc = -vt.dot(c);
    // Occluded ⇔ the point is past the horizon plane AND inside the limb cone.
    vt_dot_vc > vh2 && (vt_dot_vc * vt_dot_vc) / vt.length_squared() > vh2
}

/// Spawn one tile's decoded items under a hidden tile-root entity. The
/// render cut flips the root's visibility; children inherit. Twin-anchored
/// sets tag content with `TwinMeshGroup` so selection/highlight/focus treat
/// tiles like any other twin geometry.
#[allow(clippy::too_many_arguments)]
fn spawn_tile_content(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    clouds: &mut Assets<bevy_pointcloud::point_cloud::PointCloud>,
    splats: &mut Assets<bevy_gaussian_splatting::PlanarGaussian3d>,
    shared_point_material: &crate::plugins::point_cloud::SharedPointMaterial,
    set: &ActiveTileset,
    tile: usize,
    transform: Transform,
    items: Vec<DecodedItem>,
) -> Entity {
    let tile_root = commands
        .spawn((
            Tiles3dTile { set_id: set.id, tile },
            transform,
            // Spawned hidden; `drive_tiles3d` flips it visible once the render
            // cut selects this tile (children inherit the root's visibility).
            Visibility::Hidden,
            ChildOf(set.root_entity),
            Name::new(format!("Tiles3dTile({} #{tile})", set.label)),
        ))
        .id();
    let group = set.twin_id.as_ref().map(|tid| TwinMeshGroup { twin_id: tid.clone() });
    for item in items {
        let child = match item {
            DecodedItem::Mesh(prim) => {
                // T8: refine click-selection to the picked feature when the
                // tile carries per-feature ids AND it's anchored to a twin
                // (world-layer tilesets have no twin to resolve to).
                let feature_table = match (prim.features, set.twin_id.as_ref()) {
                    (Some(f), Some(tid)) => Some(TileFeatureTable {
                        feature_of_triangle: f.feature_of_triangle,
                        node_of_feature: f.node_of_feature,
                        anchor_twin: tid.clone(),
                    }),
                    _ => None,
                };
                // Plain OPAQUE StandardMaterial: no `discard`, no alpha blend, so
                // the GPU keeps early-Z depth testing — dense, overlapping P3DT
                // photogrammetry would otherwise pay full overdraw on every
                // occluded fragment. (The dithered LOD cross-fade prototype that
                // lived here forced late-Z for the whole pipeline and tanked the
                // frame rate; reverted — LODs pop, but the view runs.)
                let material = StandardMaterial {
                    base_color: Color::LinearRgba(LinearRgba::new(
                        prim.material.base_color[0],
                        prim.material.base_color[1],
                        prim.material.base_color[2],
                        prim.material.base_color[3],
                    )),
                    base_color_texture: prim.material.base_color_image.map(|img| images.add(img)),
                    metallic: prim.material.metallic,
                    perceptual_roughness: prim.material.roughness,
                    // Photogrammetry (P3DT) ships baked lighting.
                    unlit: prim.material.unlit,
                    cull_mode: if prim.material.double_sided {
                        None
                    } else {
                        Some(bevy::render::render_resource::Face::Back)
                    },
                    ..default()
                };
                let mut child = commands.spawn((
                    Mesh3d(meshes.add(prim.mesh)),
                    MeshMaterial3d(materials.add(material)),
                    Transform::from_matrix(prim.transform),
                    ChildOf(tile_root),
                ));
                if let Some(ft) = feature_table {
                    child.insert(ft);
                }
                child.id()
            }
            DecodedItem::Points { transform, points } => commands
                .spawn((
                    crate::plugins::point_cloud::cloud_components(
                        clouds.add(bevy_pointcloud::point_cloud::PointCloud { points }),
                        shared_point_material,
                    ),
                    Transform::from_matrix(transform),
                    ChildOf(tile_root),
                ))
                .id(),
            DecodedItem::Splat { transform, gaussians } => commands
                .spawn((
                    crate::plugins::splat::splat_components(splats.add(
                        bevy_gaussian_splatting::PlanarGaussian3d::from(gaussians),
                    )),
                    Transform::from_matrix(transform),
                    ChildOf(tile_root),
                ))
                .id(),
        };
        if let Some(group) = &group {
            commands.entity(child).insert(group.clone());
        }
    }
    tile_root
}

/// Keep the `true`-flagged entries of `v`, preserving order — the index-aligned
/// twin of [`TileTree::retain`] for the per-tile side arrays.
fn gather<T: Clone>(v: &[T], keep: &[bool]) -> Vec<T> {
    v.iter().zip(keep).filter(|&(_, &k)| k).map(|(x, _)| x.clone()).collect()
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
            if matches!(set.slots[x], TileSlot::Ready { .. } | TileSlot::InFlight { .. }) {
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

/// Run the selection pass per tileset, apply the render cut as visibility,
/// schedule loads by priority (recomputed every frame, out-of-cut requests
/// aborted), and evict stale residents.
#[allow(clippy::too_many_arguments)]
fn drive_tiles3d(
    config: Res<Tiles3dConfig>,
    channel: Res<Tiles3dChannel>,
    origin: Res<ProjectOrigin>,
    mut sets: ResMut<Tiles3dSets>,
    mut credits: ResMut<TilesetCredits>,
    camera: Query<(&Camera, &GlobalTransform, &Projection, &Frustum), With<PanOrbitCamera>>,
    transforms: Query<&GlobalTransform>,
    mut vis_q: Query<&mut Visibility, With<Tiles3dTile>>,
    mut tile_transforms: Query<&mut Transform, With<Tiles3dTile>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Tiles3dSets { sets, frame, next_generation, .. } = &mut *sets;
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
    let Ok((cam, cam_gt, proj, frustum)) = camera.single() else { return };

    let fov_y = match proj {
        Projection::Perspective(p) => p.fov as f64,
        _ => std::f64::consts::FRAC_PI_4,
    };
    let viewport_h = cam.logical_viewport_size().map(|v| v.y as f64).unwrap_or(1080.0);
    let k_px = viewport_h / (2.0 * (fov_y * 0.5).tan()).max(1e-6);
    let cam_pos_world = cam_gt.translation().as_dvec3();
    let cam_forward_world = Vec3::from(cam_gt.forward()).as_dvec3();

    let mut any_in_flight = false;
    let mut google_visible = false;
    let mut ground_covering = false;
    let mut compacted_this_frame = false;
    for set in sets.iter_mut() {
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
                let Some(o) = origin.get() else {
                    // No ENU datum yet — hold the set entirely.
                    continue;
                };
                if *built != Some(o) {
                    // ORIGIN REBASE (basemap's model, exact-recompute form):
                    // re-place every resident tile from absolutes in f64.
                    *built = Some(o);
                    for (i, slot) in set.slots.iter_mut().enumerate() {
                        let TileSlot::Ready { entity } = *slot else { continue };
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
                (
                    crate::plugins::spatial_source::enu::world_from_ecef(o),
                    1.0,
                    Some(WGS84_EQUATORIAL_RADIUS_M),
                )
            }
        };
        let set_from_world = world_from_set.inverse();
        let cam_pos = set_from_world.transform_point3(cam_pos_world);
        let cam_forward = set_from_world
            .transform_vector3(cam_forward_world)
            .normalize_or(DVec3::NEG_Z);
        // Camera height above the planet surface (globe sets only): the falloff
        // measures from here, so flying high keeps the nadir tile sharp.
        let cam_height_m =
            planet_radius.map(|r| (cam_pos.length() - r).max(0.0)).unwrap_or(0.0);
        let params = SelectParams {
            cam_pos,
            cam_forward,
            k_px,
            sse_threshold_px: config.sse_threshold_px,
            detail_falloff_m: config.detail_falloff_m,
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
            // HORIZON CULL (globe sets): a tile entirely behind the planet's
            // limb for the camera's altitude is never refined, grafted, or
            // loaded — looking at the horizon no longer drags the whole far
            // hemisphere into the tree. Altitude-adaptive for free: the higher
            // the camera, the farther the limb, so more becomes visible. Runs
            // in the set frame (planet at the origin), where `cam_pos` and the
            // tile volume already live.
            if let Some(planet_r) = planet_radius
                && beyond_horizon(cam_pos, center, radius, planet_r)
            {
                return true;
            }
            // Frustum test happens in world space: local volume → world.
            let world_center = world_from_set.transform_point3(center);
            let sphere = Sphere {
                center: Vec3A::from(world_center.as_vec3()),
                // Inflate the test sphere 25%: keep tiles whose extent sits just
                // past the frustum edge so they don't pop out (and stop loading)
                // as the view rotates/tilts — the "tiles vanish at this exact
                // angle" finding. Cheap now that compaction bounds the tree.
                radius: (radius * set_scale * 1.25) as f32,
            };
            // `intersect_far = false`, like the basemap: distant tiles coarsen
            // via SSE; clipping handles the rest.
            !frustum.intersects_sphere(&sphere, false)
        };

        let sel = traversal::select(tree, &tiles_content, &set.history, &culled, params);

        // Eviction clock: everything the pass wanted stays fresh.
        for (i, &touched) in sel.touched.iter().enumerate() {
            if touched {
                set.last_touched[i] = *frame;
            }
        }

        // Apply the render cut as per-tile-root visibility — the selected tiles
        // show, everything else hides (children inherit the root's visibility).
        let mut want_visible = vec![false; tree.len()];
        for &t in &sel.render {
            want_visible[t] = true;
        }
        for (i, slot) in set.slots.iter().enumerate() {
            if let TileSlot::Ready { entity } = slot
                && let Ok(mut vis) = vis_q.get_mut(*entity)
            {
                let want =
                    if want_visible[i] { Visibility::Visible } else { Visibility::Hidden };
                if *vis != want {
                    *vis = want;
                }
            }
        }

        // First painted cut: strip the anchor's placeholder cube geometry
        // (same contract as `bind_spawned_scenes` for whole-file scenes —
        // remove `Mesh3d`, keep the entity as the transform anchor).
        if !set.placeholder_cleared && !sel.render.is_empty() && set.twin_id.is_some() {
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

        // Issue new requests in priority order under the concurrency cap.
        let mut in_flight =
            set.slots.iter().filter(|s| matches!(s, TileSlot::InFlight { .. })).count();
        for req in &sel.loads {
            if budget_exhausted || in_flight >= config.max_concurrent_loads {
                break;
            }
            if !matches!(set.slots[req.tile], TileSlot::NotLoaded) {
                continue; // already in flight, ready, or failed
            }
            let Some(uri) = tree.nodes[req.tile].content_uri.clone() else { continue };
            let generation = *next_generation;
            *next_generation += 1;
            set.slots[req.tile] = TileSlot::InFlight { generation };
            in_flight += 1;
            let abort = fetch::register_abort(generation);
            let source = set.source.clone();
            let tx = channel.tx.clone();
            let set_id = set.id;
            let georeferenced = matches!(set.frame, SetFrame::Ecef { .. });
            fetch::spawn_io(async move {
                // Fetch + decode entirely inside the task (wasm: every IO step
                // awaits a JS future and yields; decode is small-tile CPU).
                // External tilesets are detected by CONTENT, not URI — P3DT
                // serves subtree JSON and GLBs from the same extensionless
                // /files/<id> namespace.
                let result = match source.read_entry_cached(&uri, Some(&abort)).await {
                    Ok(bytes) if looks_like_external_tileset(&bytes) => {
                        schema::parse_tileset(&bytes)
                            .map(|ts| TileOutput::Subtree(Box::new(ts)))
                            .map_err(|e| format!("parse external tileset: {e}"))
                    }
                    Ok(bytes) => content::decode_tile(&bytes, georeferenced)
                        .await
                        .map(|tile| TileOutput::Content(Box::new(tile))),
                    Err(e) => Err(e.to_string()),
                };
                fetch::unregister_abort(generation);
                // Receiver gone (plugin torn down) is fine — drop silently.
                let _ = tx.send(Tiles3dMsg::TileContent { set_id, generation, result });
            });
        }

        // Eviction: out-of-cut residents past the grace window, then the
        // oldest extras over the hard budget (memory wins over reuse).
        let mut resident: Vec<(usize, u64)> = Vec::new();
        for (i, slot) in set.slots.iter().enumerate() {
            if matches!(slot, TileSlot::Ready { .. }) {
                resident.push((i, set.last_touched[i]));
            }
        }
        let mut evict: Vec<usize> = resident
            .iter()
            .filter(|(i, seen)| {
                !want_visible[*i] && frame.saturating_sub(*seen) > config.grace_frames
            })
            .map(|(i, _)| *i)
            .collect();
        if resident.len() - evict.len() > config.max_resident_tiles {
            let mut extras: Vec<(usize, u64)> = resident
                .iter()
                .filter(|(i, _)| !want_visible[*i] && !evict.contains(i))
                .copied()
                .collect();
            extras.sort_by_key(|(_, seen)| *seen);
            let over = resident.len() - evict.len() - config.max_resident_tiles;
            evict.extend(extras.iter().take(over).map(|(i, _)| *i));
        }
        for i in evict {
            if let TileSlot::Ready { entity } = set.slots[i] {
                commands.entity(entity).despawn();
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
                    "tiles3d: {}: cut {} tile(s) at depth {dmin}..{dmax}",
                    set.label, cut.0
                );
            }
        }
        set.history.absorb(&sel, tree.len());
        google_visible |= set.is_live() && !sel.render.is_empty();
        ground_covering |=
            matches!(set.frame, SetFrame::Ecef { .. }) && !sel.render.is_empty();
        any_in_flight |=
            set.slots.iter().any(|s| matches!(s, TileSlot::InFlight { .. }));
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

    // Keep the reactive loop awake while content streams — without this the
    // idle 200 ms tick would crawl through the decode queue (the same lesson
    // as `keep_awake_while_loading` in the asset loader).
    if any_in_flight {
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

    #[test]
    fn horizon_cull_hides_the_far_side_keeps_the_near() {
        let r = WGS84_EQUATORIAL_RADIUS_M;
        // Camera 1 km above the north pole, looking down the globe.
        let cam = DVec3::new(0.0, 0.0, r + 1_000.0);
        let small = 50.0; // a tile-sized bounding sphere
        // Directly below → visible.
        assert!(!beyond_horizon(cam, DVec3::new(0.0, 0.0, r), small, r));
        // Far side of the planet (south pole) → occluded.
        assert!(beyond_horizon(cam, DVec3::new(0.0, 0.0, -r), small, r));
        // Just past the geometric horizon (≈113 km along the surface at 1 km
        // altitude) → occluded; well inside it → visible.
        let d_h = (2.0 * r * 1_000.0_f64).sqrt(); // ~112.9 km straight-line
        let ang_far = (d_h * 1.5) / r; // past the limb
        let far = DVec3::new(r * ang_far.sin(), 0.0, r * ang_far.cos());
        assert!(beyond_horizon(cam, far, small, r), "past the limb is culled");
        let ang_near = (d_h * 0.3) / r;
        let near = DVec3::new(r * ang_near.sin(), 0.0, r * ang_near.cos());
        assert!(!beyond_horizon(cam, near, small, r), "inside the limb stays");
        // Whole-globe root volume (centre at the planet centre) is never culled.
        assert!(!beyond_horizon(cam, DVec3::ZERO, r, r));
    }

    #[test]
    fn archive_spec_detection_ignores_query_strings() {
        assert!(is_archive_spec("assets/fixtures/tiles3d-demo.3tz"));
        assert!(is_archive_spec("https://x.blob.core.windows.net/a/demo.3tz?se=2026&sig=abc"));
        assert!(is_archive_spec("https://x/a/demo.3tz#frag"));
        assert!(!is_archive_spec("assets/fixtures/tiles3d-demo/tileset.json"));
        assert!(!is_archive_spec("https://x/a/tileset.json?sas=1"));
    }

    #[test]
    fn external_tileset_detection_is_content_based() {
        let tileset = br#"{"asset":{"version":"1.1"},"geometricError":1e100,
            "root":{"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]},"geometricError":0}}"#;
        assert!(looks_like_external_tileset(tileset));
        assert!(looks_like_external_tileset(b"  \n{\"geometricError\": 1, \"root\": {}}"));
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
        assert_eq!(uri_dir_prefix("a/b/c.json?session=x"), Some("a/b/".to_string()));
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
        let source = TilesetSource::Exploded(ExplodedBase::Dir("assets/fixtures/tiles3d-demo".into()));
        let bytes = block_on(source.read_entry("tileset.json")).expect("fixture tileset.json");
        let tileset = schema::parse_tileset(&bytes).expect("parse");
        assert!(!geo::tileset_is_georeferenced(&tileset), "fixture is local-metres");
        let tree = TileTree::build(&tileset, ZUP_TO_BEVY, TreeFrame::Local).expect("build");
        assert_eq!(tree.len(), 21, "1 root + 4 children + 16 leaves");
        // Mixed volume kinds present.
        let spheres = tree
            .nodes
            .iter()
            .filter(|n| matches!(n.volume, traversal::WorldVolume::Sphere { .. }))
            .count();
        assert!(spheres > 0 && spheres < tree.len(), "mixed box/sphere volumes");
        // Every content GLB decodes.
        for node in &tree.nodes {
            let uri = node.content_uri.as_ref().expect("all fixture tiles carry content");
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
