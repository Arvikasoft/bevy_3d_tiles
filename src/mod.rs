//! 3D Tiles 1.1 streaming plugin (BEVY-3D-TILES-PLAN, phase T0).
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
//! * [`fetch`] — byte sources (HTTP range / file / memory) + the
//!   never-block-the-executor task spawning discipline.
//! * [`content`] — tile GLB → Bevy mesh/material decode (T0: mesh only).
//! * this module — ECS wiring: per-frame selection, the request scheduler
//!   (priorities recomputed each frame, out-of-cut requests cancelled),
//!   time-boxed content spawning, visibility cut, eviction.
//!
//! **T0 scope**: the plugin only activates through a dev trigger — native
//! `TT_TILES3D=fixture|<path>|<url>` env var, wasm `?tiles3d=fixture|<url>`
//! query param. Resolver integration (D6 `"3dtiles"` renditions) lands with
//! T1; nothing here touches the twin/preview asset paths yet.
//!
//! Cancellation note: a request that falls out of the cut is cancelled by
//! state (its slot leaves `InFlight`, so the landed result is dropped via the
//! generation guard and the concurrency slot frees immediately); the network
//! transfer itself runs to completion. AbortController wiring is a T1
//! follow-up alongside the Front Door HTTP/2 work (D10).

use std::sync::Arc;

use bevy::camera::primitives::{Frustum, Sphere};
use bevy::camera::Projection;
use bevy::math::Vec3A;
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use bevy_panorbit_camera::PanOrbitCamera;

pub mod archive;
pub mod content;
pub mod fetch;
pub mod schema;
pub mod traversal;

use archive::Archive3tz;
use content::DecodedPrimitive;
use fetch::{ByteSource, ExplodedBase, TilesetSource};
use traversal::{History, SelectParams, TileContent, TileTree, ZUP_TO_BEVY};

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
    /// cancelled-then-reissued tile drops the stale payload.
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
    /// Last logged render-cut shape `(tiles, min_depth, max_depth)` —
    /// transitions are the observable trace of LOD swaps (dev trigger only).
    last_cut: Option<(usize, u32, u32)>,
}

/// Live tilesets + scheduler counters.
#[derive(Resource, Default)]
pub struct Tiles3dSets {
    sets: Vec<ActiveTileset>,
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
}

/// Async-task → ECS messages.
enum Tiles3dMsg {
    TilesetOpened {
        label: String,
        /// Boxed: a parsed tileset tree dwarfs the per-tile variant.
        result: Result<(TilesetSource, Box<schema::Tileset>), String>,
    },
    TileContent {
        set_id: u64,
        tile: usize,
        generation: u64,
        result: Result<Vec<DecodedPrimitive>, String>,
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
            .add_systems(Startup, init_dev_tileset)
            .add_systems(Update, (receive_tiles3d, drive_tiles3d).chain());
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
    spawn_tileset_open(spec, channel.tx.clone());
}

/// Open a tileset (async): resolve the source kind, fetch + parse the root
/// `tileset.json`, report back on the channel.
fn spawn_tileset_open(spec: String, tx: crossbeam_channel::Sender<Tiles3dMsg>) {
    fetch::spawn_io(async move {
        let result = open_tileset(&spec).await;
        let _ = tx.send(Tiles3dMsg::TilesetOpened { label: spec, result });
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
        .read_entry("tileset.json")
        .await
        .map_err(|e| format!("fetch tileset.json: {e}"))?;
    let tileset = schema::parse_tileset(&bytes).map_err(|e| format!("parse tileset.json: {e}"))?;
    Ok((source, Box::new(tileset)))
}

// ── ECS drain: tilesets + decoded tile content ───────────────────────────────

/// Drain async results into the ECS, time-boxed: at most
/// `max_spawns_per_frame` content spawns per frame (§7's main-thread budget);
/// the rest stay queued in the channel for the next frame.
fn receive_tiles3d(
    channel: Res<Tiles3dChannel>,
    config: Res<Tiles3dConfig>,
    mut sets: ResMut<Tiles3dSets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
) {
    let mut spawned = 0usize;
    while spawned < config.max_spawns_per_frame {
        let Ok(msg) = channel.rx.try_recv() else { break };
        match msg {
            Tiles3dMsg::TilesetOpened { label, result } => match result {
                Ok((source, tileset)) => {
                    // T0: tilesets are local-metres at the world origin
                    // (fixture / dev URLs). Twin-anchored ENU placement
                    // composes in with D6 resolver integration (T1).
                    match TileTree::build(&tileset, ZUP_TO_BEVY) {
                        Ok(tree) => {
                            let n = tree.len();
                            let id = sets.next_set_id;
                            sets.next_set_id += 1;
                            let root_entity = commands
                                .spawn((
                                    Name::new(format!("Tiles3d({label})")),
                                    Transform::IDENTITY,
                                    Visibility::default(),
                                ))
                                .id();
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
                                last_cut: None,
                            });
                        }
                        Err(e) => error!("tiles3d: {label}: unusable tileset: {e}"),
                    }
                }
                Err(e) => error!("tiles3d: {label}: {e}"),
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
                    Ok(prims) => {
                        spawned += 1;
                        let entity = spawn_tile_content(
                            &mut commands,
                            &mut meshes,
                            &mut materials,
                            &mut images,
                            set,
                            tile,
                            prims,
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

/// Spawn one tile's decoded primitives under a hidden tile-root entity. The
/// render cut flips the root's visibility; children inherit.
fn spawn_tile_content(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    set: &ActiveTileset,
    tile: usize,
    prims: Vec<DecodedPrimitive>,
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
    for prim in prims {
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
        commands.spawn((
            Mesh3d(meshes.add(prim.mesh)),
            MeshMaterial3d(materials.add(material)),
            Transform::from_matrix(prim.transform),
            ChildOf(tile_root),
        ));
    }
    tile_root
}

// ── Per-frame manager ────────────────────────────────────────────────────────

/// Run the selection pass per tileset, apply the render cut as visibility,
/// schedule loads by priority (recomputed every frame, out-of-cut requests
/// cancelled), and evict stale residents.
fn drive_tiles3d(
    config: Res<Tiles3dConfig>,
    channel: Res<Tiles3dChannel>,
    mut sets: ResMut<Tiles3dSets>,
    camera: Query<(&Camera, &GlobalTransform, &Projection, &Frustum), With<PanOrbitCamera>>,
    mut vis_q: Query<&mut Visibility, With<Tiles3dTile>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Tiles3dSets { sets, frame, next_generation, .. } = &mut *sets;
    *frame += 1;
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
    let params = SelectParams {
        cam_pos: cam_gt.translation().as_dvec3(),
        cam_forward: Vec3::from(cam_gt.forward()).as_dvec3(),
        k_px,
        sse_threshold_px: config.sse_threshold_px,
    };

    let mut any_in_flight = false;
    for set in sets.iter_mut() {
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
            let sphere =
                Sphere { center: Vec3A::from(center.as_vec3()), radius: radius as f32 };
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

        // Scheduler. Cancel-by-state first: an in-flight tile that fell out
        // of this frame's wanted loads frees its slot now; its landed payload
        // is dropped by the InFlight/generation guard in `receive_tiles3d`.
        let mut wanted = vec![false; tree.len()];
        for req in &sel.loads {
            wanted[req.tile] = true;
        }
        for (i, slot) in set.slots.iter_mut().enumerate() {
            if matches!(slot, TileSlot::InFlight { .. }) && !wanted[i] {
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
            let source = set.source.clone();
            let tx = channel.tx.clone();
            let (set_id, tile) = (set.id, req.tile);
            fetch::spawn_io(async move {
                // Fetch + decode entirely inside the task (wasm: every IO step
                // awaits a JS future and yields; decode is small-tile CPU).
                let result = match source.read_entry(&uri).await {
                    Ok(bytes) => content::decode_glb(&bytes),
                    Err(e) => Err(e.to_string()),
                };
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
            let prims = content::decode_glb(&glb).expect("decode");
            assert!(!prims.is_empty(), "{uri} has geometry");
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
}
