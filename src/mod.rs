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

use std::sync::Arc;

use bevy::camera::primitives::{Frustum, Sphere};
use bevy::camera::Projection;
use bevy::math::{DMat4, DVec3, Vec3A};
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use bevy_panorbit_camera::PanOrbitCamera;

pub mod archive;
pub mod content;
pub mod fetch;
pub mod schema;
pub mod traversal;

use archive::Archive3tz;
use content::DecodedItem;
use fetch::{ByteSource, ExplodedBase, TilesetSource};
use traversal::{History, SelectParams, TileContent, TileTree, ZUP_TO_BEVY};

use crate::plugins::scene_layout::TwinMeshGroup;

/// Committed demo tileset (see `examples/gen_tiles3d_fixture.rs`). The path
/// doubles as a relative URL under Trunk (assets are `copy-dir`'d) and a
/// relative file path for native runs from the crate root.
const FIXTURE_SPEC: &str = "assets/fixtures/tiles3d-demo/tileset.json";

#[derive(Resource, Debug, Clone)]
pub struct Tiles3dConfig {
    /// Refine while a tile's screen-space error exceeds this (px).
    pub sse_threshold_px: f64,
    /// Max tile fetch+decode tasks in flight per tileset. Basemap's proven
    /// starting point; retune against CesiumJS's 50/18 once Front Door
    /// HTTP/2 carries tile traffic (D10).
    pub max_concurrent_loads: usize,
    /// Main-thread time box: max decoded tiles turned into entities per frame.
    pub max_spawns_per_frame: usize,
    /// Frames an out-of-cut tile stays resident before eviction (zoom
    /// in-and-back reuse, mirrors basemap).
    pub grace_frames: u64,
    /// Hard cap on resident (spawned) tiles per tileset.
    pub max_resident_tiles: usize,
}

impl Default for Tiles3dConfig {
    fn default() -> Self {
        Self {
            sse_threshold_px: traversal::DEFAULT_SSE_THRESHOLD_PX,
            max_concurrent_loads: 16,
            max_spawns_per_frame: 4,
            grace_frames: 180,
            max_resident_tiles: 512,
        }
    }
}

/// Attach a streaming tileset under an anchor entity (D6 resolver routing —
/// sent by the asset loader for `"3dtiles"` renditions).
#[derive(Message, Debug, Clone)]
pub struct Tiles3dAttach {
    /// Entity the tileset root parents under (twin entity / preview root).
    /// Tile placement = anchor's world transform × `local` × tile transforms.
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
    pub tile: usize,
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
    /// caller composes the root entity's `GlobalTransform`.
    pub fn root_volume_for_anchor(&self, anchor: Entity) -> Option<(Entity, Vec3, f32)> {
        let set = self.sets.iter().find(|s| s.anchor == Some(anchor))?;
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
        tile: usize,
        generation: u64,
        result: Result<Vec<DecodedItem>, String>,
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
            .add_message::<Tiles3dAttach>()
            .add_message::<Tiles3dDetach>()
            .add_systems(Startup, init_dev_tileset)
            .add_systems(
                Update,
                (apply_attach_detach, receive_tiles3d, drive_tiles3d).chain(),
            );
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
    spawn_tileset_open(spec, None, channel.tx.clone());
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
    attach: Option<AttachTarget>,
    tx: crossbeam_channel::Sender<Tiles3dMsg>,
) {
    fetch::spawn_io(async move {
        let result = open_tileset(&spec).await;
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

async fn open_tileset(spec: &str) -> Result<(TilesetSource, Box<schema::Tileset>), String> {
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
                    // Tilesets are local-metres Z-up at their own origin; the
                    // anchor entity (twin/preview, when present) carries the
                    // ENU/world placement, so the tree itself only converts
                    // Z-up → Bevy. Dev sets sit at the world origin.
                    let anchor = attach.as_ref().map(|a| a.anchor);
                    match TileTree::build(&tileset, ZUP_TO_BEVY) {
                        Ok(tree) => {
                            let n = tree.len();
                            let id = sets.next_set_id;
                            sets.next_set_id += 1;
                            let mut root = commands.spawn((
                                Name::new(format!("Tiles3d({label})")),
                                attach.as_ref().map(|a| a.local).unwrap_or_default(),
                                Visibility::default(),
                            ));
                            if let Some(a) = &attach {
                                root.insert(ChildOf(a.anchor));
                            }
                            let root_entity = root.id();
                            let mut history = History::default();
                            history.resize(n);
                            info!("tiles3d: {label}: {n} tiles");
                            sets.sets.push(ActiveTileset {
                                id,
                                label,
                                tree,
                                source,
                                slots: vec![TileSlot::NotLoaded; n],
                                history,
                                last_touched: vec![0; n],
                                root_entity,
                                anchor: attach.as_ref().map(|a| a.anchor),
                                twin_id: attach.and_then(|a| a.twin_id),
                                placeholder_cleared: false,
                                last_cut: None,
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
            Tiles3dMsg::TileContent { set_id, tile, generation, result } => {
                let Some(set) = sets.sets.iter_mut().find(|s| s.id == set_id) else {
                    continue;
                };
                // Cancelled (slot left InFlight) or reissued (generation
                // bumped) → drop the stale payload.
                let TileSlot::InFlight { generation: current } = set.slots[tile] else {
                    continue;
                };
                if current != generation {
                    continue;
                }
                match result {
                    Ok(items) => {
                        spawned += 1;
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
    items: Vec<DecodedItem>,
) -> Entity {
    let node = &set.tree.nodes[tile];
    let tile_root = commands
        .spawn((
            Tiles3dTile { set_id: set.id, tile },
            Transform::from_matrix(node.world_from_content.as_mat4()),
            // Spawned hidden; `drive_tiles3d`'s cut reveals it — a freshly
            // loaded tile must never flash over the coarser one it replaces.
            Visibility::Hidden,
            ChildOf(set.root_entity),
            Name::new(format!("Tiles3dTile({} #{tile})", set.label)),
        ))
        .id();
    let group = set.twin_id.as_ref().map(|tid| TwinMeshGroup { twin_id: tid.clone() });
    for item in items {
        let child = match item {
            DecodedItem::Mesh(prim) => {
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
                    cull_mode: if prim.material.double_sided {
                        None
                    } else {
                        Some(bevy::render::render_resource::Face::Back)
                    },
                    ..default()
                };
                commands
                    .spawn((
                        Mesh3d(meshes.add(prim.mesh)),
                        MeshMaterial3d(materials.add(material)),
                        Transform::from_matrix(prim.transform),
                        ChildOf(tile_root),
                    ))
                    .id()
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

// ── Per-frame manager ────────────────────────────────────────────────────────

/// Run the selection pass per tileset, apply the render cut as visibility,
/// schedule loads by priority (recomputed every frame, out-of-cut requests
/// aborted), and evict stale residents.
#[allow(clippy::too_many_arguments)]
fn drive_tiles3d(
    config: Res<Tiles3dConfig>,
    channel: Res<Tiles3dChannel>,
    mut sets: ResMut<Tiles3dSets>,
    camera: Query<(&Camera, &GlobalTransform, &Projection, &Frustum), With<PanOrbitCamera>>,
    transforms: Query<&GlobalTransform>,
    mut vis_q: Query<&mut Visibility, With<Tiles3dTile>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Tiles3dSets { sets, frame, next_generation, .. } = &mut *sets;
    *frame += 1;

    // GC: an anchored set whose root entity died (anchor despawned — the
    // hierarchy took the root and every tile entity with it) is torn down
    // here; its in-flight requests abort and late results drop harmlessly.
    // (`pending_anchors` / `failed_anchors` keep dead Entity ids — harmless:
    // entity generations never repeat, and the per-id cost is 8 bytes.)
    sets.retain(|set| {
        if set.anchor.is_none() || transforms.contains(set.root_entity) {
            return true;
        }
        info!("tiles3d: {}: anchor gone — dropping tileset", set.label);
        abort_in_flight(set);
        false
    });
    if sets.is_empty() {
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
    for set in sets.iter_mut() {
        // The set's frame: world_from_set = anchor chain × correction (the
        // root entity's GlobalTransform — last frame's propagation, fine for
        // streaming decisions). Selection runs in set-local coordinates:
        // distances and geometric errors share the tileset's metres, so SSE
        // is exact under rigid/uniform anchor transforms — no per-frame
        // volume rebuilds for moving twins.
        let world_from_set = transforms
            .get(set.root_entity)
            .map(|gt| gt.to_matrix().as_dmat4())
            .unwrap_or(DMat4::IDENTITY);
        let set_from_world = world_from_set.inverse();
        let set_scale = traversal::max_scale(&world_from_set).max(1e-12);
        let cam_pos = set_from_world.transform_point3(cam_pos_world);
        let cam_forward = set_from_world
            .transform_vector3(cam_forward_world)
            .normalize_or(DVec3::NEG_Z);
        let params = SelectParams {
            cam_pos,
            cam_forward,
            k_px,
            sse_threshold_px: config.sse_threshold_px,
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
            // Frustum test happens in world space: local volume → world.
            let (center, radius) = tree.nodes[i].volume.bounding_sphere();
            let world_center = world_from_set.transform_point3(center);
            let sphere = Sphere {
                center: Vec3A::from(world_center.as_vec3()),
                radius: (radius * set_scale) as f32,
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

        // Apply the render cut as per-tile-root visibility.
        let mut want_visible = vec![false; tree.len()];
        for &t in &sel.render {
            want_visible[t] = true;
        }
        for (i, slot) in set.slots.iter().enumerate() {
            if let TileSlot::Ready { entity } = slot
                && let Ok(mut vis) = vis_q.get_mut(*entity)
            {
                let want = if want_visible[i] { Visibility::Visible } else { Visibility::Hidden };
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
        // Issue new requests in priority order under the concurrency cap.
        let mut in_flight =
            set.slots.iter().filter(|s| matches!(s, TileSlot::InFlight { .. })).count();
        for req in &sel.loads {
            if in_flight >= config.max_concurrent_loads {
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
            let (set_id, tile) = (set.id, req.tile);
            fetch::spawn_io(async move {
                // Fetch + decode entirely inside the task (wasm: every IO step
                // awaits a JS future and yields; decode is small-tile CPU).
                let result = match source.read_entry_cached(&uri, Some(&abort)).await {
                    Ok(bytes) => content::decode_glb(&bytes),
                    Err(e) => Err(e.to_string()),
                };
                fetch::unregister_abort(generation);
                // Receiver gone (plugin torn down) is fine — drop silently.
                let _ = tx.send(Tiles3dMsg::TileContent { set_id, tile, generation, result });
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
        any_in_flight |=
            set.slots.iter().any(|s| matches!(s, TileSlot::InFlight { .. }));
    }

    // Keep the reactive loop awake while content streams — without this the
    // idle 200 ms tick would crawl through the decode queue (the same lesson
    // as `keep_awake_while_loading` in the asset loader).
    if any_in_flight {
        redraw.write(RequestRedraw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::tasks::block_on;

    #[test]
    fn archive_spec_detection_ignores_query_strings() {
        assert!(is_archive_spec("assets/fixtures/tiles3d-demo.3tz"));
        assert!(is_archive_spec("https://x.blob.core.windows.net/a/demo.3tz?se=2026&sig=abc"));
        assert!(is_archive_spec("https://x/a/demo.3tz#frag"));
        assert!(!is_archive_spec("assets/fixtures/tiles3d-demo/tileset.json"));
        assert!(!is_archive_spec("https://x/a/tileset.json?sas=1"));
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
        let tree = TileTree::build(&tileset, ZUP_TO_BEVY).expect("build");
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
        let tree = TileTree::build(&tileset, ZUP_TO_BEVY).expect("build");
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
