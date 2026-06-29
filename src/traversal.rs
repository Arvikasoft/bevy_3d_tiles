//! Tile-tree model + the per-frame selection algorithm (BEVY-3D-TILES-PLAN §7).
//!
//! cesium-native's documented algorithm, adapted to what the basemap streamer
//! already proved, generalized from quadtree arithmetic to tileset-defined
//! children:
//!
//! 1. DFS from the root; frustum-culled children are skipped.
//! 2. `sse = geometricError / max(distance_to_bounding_volume, ε) × k_px`.
//!    Camera **inside** the volume ⇒ distance 0 ⇒ refine — this is what kills
//!    the whole-file decision-5 degeneracy: the camera is inside the *root*
//!    but outside most *leaves*.
//! 3. `sse > threshold` → refine into the tileset's children; else render
//!    this tile. Leaves always render. The threshold is **distance-relaxed**
//!    (`threshold × (1 + dist/detail_falloff_m)`): far terrain stops
//!    subdividing so a grazing horizon view doesn't pull the whole visible
//!    hemisphere to full LOD (and graft+stream thousands of far subtrees).
//! 4. **Zoom-out protection**: a tile that refined last frame but isn't
//!    renderable yet keeps refining — its descendants stay visible until the
//!    coarser tile loads.
//! 5. **Kicking (zoom-in protection)**: a refined tile whose selected
//!    descendants are not all renderable *and* none rendered last frame
//!    renders itself instead; the descendants stay in the load queue.
//!    (Basemap solved this with per-quadrant masks; mesh tiles are opaque
//!    GLBs, so frame-history kicking is the general mechanism.)
//! 6. Load priorities — Urgent (camera inside volume), then Normal (current
//!    cut), then Preload (ancestors of the cut) — recomputed every frame; the
//!    scheduler in `mod.rs` diffs against in-flight requests and drops the
//!    ones that fell out of the cut.
//!
//! Everything here is pure data + math (no ECS, no IO) so the decision logic
//! is unit-tested without rendering.

use bevy::math::{DMat4, DVec3, DVec4};

use super::geo;
use super::schema::{self, Refine, VolumeKind};

/// Refine when a tile's screen-space error exceeds this many pixels.
/// CesiumJS's default `maximumScreenSpaceError` — D10 calibration baseline.
pub const DEFAULT_SSE_THRESHOLD_PX: f64 = 16.0;

/// Distances under this count as "camera inside the volume" → infinite SSE.
const INSIDE_EPS: f64 = 1e-9;

/// A bounding volume in **Bevy world space** (f64 — tileset frames can be
/// planetary-magnitude before f32 conversion at render time).
#[derive(Debug, Clone, Copy)]
pub enum WorldVolume {
    Sphere { center: DVec3, radius: f64 },
    /// Center + three half-axis vectors (orientation × half-extent).
    Obb { center: DVec3, half_axes: [DVec3; 3] },
}

impl WorldVolume {
    /// Distance from `p` to the volume surface; `0.0` when inside.
    pub fn distance_to(&self, p: DVec3) -> f64 {
        match self {
            WorldVolume::Sphere { center, radius } => (p - *center).length() - radius,
            WorldVolume::Obb { center, half_axes } => {
                // Project into the box frame, clamp, measure the residual.
                let d = p - *center;
                let mut closest = *center;
                for axis in half_axes {
                    let len = axis.length();
                    if len < 1e-12 {
                        continue;
                    }
                    let unit = *axis / len;
                    closest += unit * d.dot(unit).clamp(-len, len);
                }
                (p - closest).length()
            }
        }
        .max(0.0)
    }

    /// Enclosing sphere (for frustum culling).
    pub fn bounding_sphere(&self) -> (DVec3, f64) {
        match self {
            WorldVolume::Sphere { center, radius } => (*center, *radius),
            WorldVolume::Obb { center, half_axes } => {
                let r2: f64 = half_axes.iter().map(|a| a.length_squared()).sum();
                (*center, r2.sqrt())
            }
        }
    }

    fn transformed(&self, m: &DMat4) -> WorldVolume {
        match self {
            WorldVolume::Sphere { center, radius } => WorldVolume::Sphere {
                center: m.transform_point3(*center),
                radius: radius * max_scale(m),
            },
            WorldVolume::Obb { center, half_axes } => WorldVolume::Obb {
                center: m.transform_point3(*center),
                half_axes: half_axes.map(|a| m.transform_vector3(a)),
            },
        }
    }
}

/// Largest column scale of the 3×3 part — the factor a sphere radius grows by
/// under `m` (exact for uniform/rigid transforms, conservative otherwise).
pub fn max_scale(m: &DMat4) -> f64 {
    m.x_axis
        .truncate()
        .length()
        .max(m.y_axis.truncate().length())
        .max(m.z_axis.truncate().length())
}

/// 3D Tiles' tileset frame is right-handed **Z-up** (x east, y north, z up
/// for local-metres tilesets); Bevy is Y-up with north = −Z. Maps
/// `(x, y, z)_zup → (x, z, −y)_bevy`.
pub const ZUP_TO_BEVY: DMat4 = DMat4::from_cols(
    DVec4::new(1.0, 0.0, 0.0, 0.0),
    DVec4::new(0.0, 0.0, -1.0, 0.0),
    DVec4::new(0.0, 1.0, 0.0, 0.0),
    DVec4::new(0.0, 0.0, 0.0, 1.0),
);

/// glTF content is Y-up; the 3D Tiles spec mandates rotating it into the
/// tileset's Z-up frame: `(x, y, z)_gltf → (x, −z, y)_zup`. Composed with
/// [`ZUP_TO_BEVY`] this is the identity — Y-up content renders directly in
/// Bevy when the tile transform chain is identity, by construction.
pub const YUP_TO_ZUP: DMat4 = DMat4::from_cols(
    DVec4::new(1.0, 0.0, 0.0, 0.0),
    DVec4::new(0.0, 0.0, 1.0, 0.0),
    DVec4::new(0.0, -1.0, 0.0, 0.0),
    DVec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Which frame a tile tree's coordinates live in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeFrame {
    /// Local-metres Z-up tileset (our tilers / the dev fixture): the build's
    /// `world_from_tileset` (usually [`ZUP_TO_BEVY`]) is final, placement
    /// rides the anchor entity. `region` volumes degrade to the parent's.
    Local,
    /// Georeferenced tileset (T4 — Google P3DT, national open data): tree
    /// coordinates are **ECEF** (EPSG:4978) f64; `region` volumes convert via
    /// [`geo::region_to_ecef_volume`]. Build with `world_from_tileset =
    /// identity`; the per-frame ENU placement happens in `drive_tiles3d`.
    Ecef,
}

/// One flattened tile. Index 0 is the root.
#[derive(Debug, Clone)]
pub struct TileNode {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Tree depth (root = 0) — for debug/cut-composition reporting.
    pub depth: u32,
    pub geometric_error: f64,
    pub refine: Refine,
    pub content_uri: Option<String>,
    /// Bounding volume in the tree frame (tile transforms pre-composed;
    /// Bevy world space for [`TreeFrame::Local`], ECEF for [`TreeFrame::Ecef`]).
    pub volume: WorldVolume,
    /// Full content placement: `world_from_tileset × cumulative tile
    /// transforms × glTF-Y-up rotation`. The spawned tile entity's transform
    /// (for ECEF trees: composed against the ENU placement first, in f64).
    pub world_from_content: DMat4,
    /// Cumulative tile-frame transform WITHOUT the glTF rotation — the parent
    /// matrix external tileset roots compose under when grafted here.
    pub world_from_tile: DMat4,
}

/// Flattened tileset tree, ready for per-frame traversal.
#[derive(Debug, Clone, Default)]
pub struct TileTree {
    pub nodes: Vec<TileNode>,
}

impl TileTree {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Flatten a parsed tileset. `world_from_tileset` places the tileset's
    /// frame in Bevy world space (for local-metres tilesets at the project
    /// origin: [`ZUP_TO_BEVY`]; for [`TreeFrame::Ecef`] pass the identity —
    /// the tree stays in ECEF and placement happens per frame).
    pub fn build(
        tileset: &schema::Tileset,
        world_from_tileset: DMat4,
        frame: TreeFrame,
    ) -> Result<TileTree, String> {
        let mut tree = TileTree::default();
        build_node(
            &mut tree,
            &tileset.root,
            None,
            0,
            Refine::Replace,
            world_from_tileset,
            None,
            frame,
        )?;
        Ok(tree)
    }

    /// Graft an external tileset (a tile whose `content.uri` named another
    /// `tileset.json`) under node `at`: the external root becomes a child of
    /// `at`, composing under `at`'s cumulative transform and inheriting its
    /// refine, per the 3D Tiles external-tileset semantics. The caller clears
    /// `at`'s `content_uri` (the content was *consumed* by the graft).
    /// Returns the new child's index; every appended node sits at indices
    /// `>=` that value, so per-tile side arrays just resize.
    pub fn graft(
        &mut self,
        at: usize,
        external: &schema::Tileset,
        frame: TreeFrame,
    ) -> Result<usize, String> {
        let parent_world = self.nodes[at].world_from_tile;
        let inherited_refine = self.nodes[at].refine;
        let parent_volume = Some(self.nodes[at].volume);
        let depth = self.nodes[at].depth + 1;
        let idx = build_node(
            self,
            &external.root,
            Some(at),
            depth,
            inherited_refine,
            parent_world,
            parent_volume,
            frame,
        )?;
        self.nodes[at].children.push(idx);
        Ok(idx)
    }

    /// Compact the tree to the nodes flagged `true` in `keep`, in place,
    /// returning the old→new index map (`usize::MAX` for a pruned node).
    ///
    /// `keep` MUST be ancestor-closed: every kept node's parent is kept. The
    /// compactor prunes whole subtrees (rooted at a grafted child), which
    /// guarantees this — `debug_assert`ed here. Per-tile side arrays the caller
    /// holds outside the tree (slots, last_touched, …) must be gathered with the
    /// SAME mask so their indices stay aligned with `nodes` afterwards.
    pub fn retain(&mut self, keep: &[bool]) -> Vec<usize> {
        let n = self.nodes.len();
        debug_assert_eq!(keep.len(), n);
        let mut new_index = vec![usize::MAX; n];
        let mut next = 0;
        for (i, &k) in keep.iter().enumerate() {
            if k {
                new_index[i] = next;
                next += 1;
            }
        }
        if next == n {
            return new_index; // nothing pruned — leave `nodes` untouched
        }
        let old = std::mem::take(&mut self.nodes);
        let mut compacted = Vec::with_capacity(next);
        for (i, mut node) in old.into_iter().enumerate() {
            if !keep[i] {
                continue;
            }
            if let Some(p) = node.parent {
                debug_assert!(keep[p], "retain: kept node {i} has a pruned parent {p}");
                node.parent = Some(new_index[p]);
            }
            node.children.retain(|c| keep[*c]);
            for c in &mut node.children {
                *c = new_index[*c];
            }
            compacted.push(node);
        }
        self.nodes = compacted;
        new_index
    }
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    tree: &mut TileTree,
    tile: &schema::Tile,
    parent: Option<usize>,
    depth: u32,
    inherited_refine: Refine,
    parent_world: DMat4,
    parent_volume: Option<WorldVolume>,
    frame: TreeFrame,
) -> Result<usize, String> {
    // The tile's own transform applies to its bounding volume AND content,
    // and composes down the tree (3D Tiles §"tile transforms").
    let local = tile
        .transform
        .map(|t| DMat4::from_cols_array(&t))
        .unwrap_or(DMat4::IDENTITY);
    let world = parent_world * local;

    let volume = match tile.bounding_volume.kind() {
        Some(VolumeKind::Sphere([cx, cy, cz, r])) => {
            WorldVolume::Sphere { center: DVec3::new(cx, cy, cz), radius: r }.transformed(&world)
        }
        Some(VolumeKind::Box(b)) => WorldVolume::Obb {
            center: DVec3::new(b[0], b[1], b[2]),
            half_axes: [
                DVec3::new(b[3], b[4], b[5]),
                DVec3::new(b[6], b[7], b[8]),
                DVec3::new(b[9], b[10], b[11]),
            ],
        }
        .transformed(&world),
        // Regions are EPSG:4979 absolutes — per spec they are NOT affected
        // by tile transforms, so the conversion ignores `world`.
        Some(VolumeKind::Region(r)) if frame == TreeFrame::Ecef => {
            geo::region_to_ecef_volume(&r)
        }
        Some(VolumeKind::Region(_)) | None => {
            // A region volume in a LOCAL tree has no defined placement (our
            // tilers never emit them); it (or a missing volume) degrades to
            // the parent's, keeping the tree traversable. The root has no
            // parent to inherit.
            parent_volume.ok_or_else(|| {
                "root bounding volume must be box or sphere in a local-frame tileset \
                 (regions need the georeferenced ECEF path)"
                    .to_string()
            })?
        }
    };

    let refine = tile.refine.unwrap_or(inherited_refine);
    let idx = tree.nodes.len();
    tree.nodes.push(TileNode {
        parent,
        children: Vec::new(),
        depth,
        geometric_error: tile.geometric_error,
        refine,
        content_uri: tile.content.as_ref().map(|c| c.uri.clone()),
        volume,
        world_from_content: world * YUP_TO_ZUP,
        world_from_tile: world,
    });

    let mut children = Vec::with_capacity(tile.children.len());
    for child in &tile.children {
        children.push(build_node(
            tree,
            child,
            Some(idx),
            depth + 1,
            refine,
            world,
            Some(volume),
            frame,
        )?);
    }
    tree.nodes[idx].children = children;
    Ok(idx)
}

// ── Per-frame selection ──────────────────────────────────────────────────────

/// Camera-derived parameters for one selection pass.
#[derive(Debug, Clone, Copy)]
pub struct SelectParams {
    pub cam_pos: DVec3,
    /// Unit view direction (screen-center load-priority weighting).
    pub cam_forward: DVec3,
    /// Pinhole focal length in pixels: `viewport_h / (2·tan(fovy/2))`.
    pub k_px: f64,
    /// Refine while `sse > threshold` (px).
    pub sse_threshold_px: f64,
    /// Distance-relaxed detail falloff (metres). The effective refine threshold
    /// grows with how far the tile sits BEYOND the camera's own height —
    /// `threshold × (1 + max(0, dist − cam_height_m) / detail_falloff_m)` — so
    /// terrain toward the horizon stops subdividing instead of pulling the whole
    /// visible hemisphere to full LOD (the live P3DT "tilt → 98 k-tile tree"
    /// finding), while the tile directly under a high top-down view stays sharp
    /// (its `dist ≈ cam_height_m`, so it keeps the base threshold). `0` disables.
    pub detail_falloff_m: f64,
    /// Camera height above the planet surface (metres) for globe sets — the
    /// reference distance the falloff measures FROM, so altitude alone never
    /// coarsens the view (only reaching toward the horizon does). `0` for
    /// non-globe sets, where the falloff measures raw distance as before.
    pub cam_height_m: f64,
}

/// Screen-space error of a tile with geometric error `ge` at `dist` metres.
/// `f64::INFINITY` when the camera is inside the volume (`dist ≈ 0`).
pub fn screen_space_error(ge: f64, dist: f64, k_px: f64) -> f64 {
    if dist <= INSIDE_EPS {
        f64::INFINITY
    } else {
        ge * k_px / dist
    }
}

/// Load-priority tiers, highest first. `Ord`: `Urgent < Descend < Normal <
/// Preload` so an ascending sort puts the most important requests first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Camera inside the tile's bounding volume.
    Urgent,
    /// Refinement frontier: a childless tile the camera still wants finer than.
    /// Loading it either grafts a deeper subtree (descending toward sharp) or
    /// paints the best-available coarse content there — so ranking it above
    /// settled `Normal` content makes the view reach target LOD before the
    /// periphery fills (faster "my view is sharp" at startup).
    Descend,
    /// In the current render cut at its chosen LOD.
    Normal,
    /// Ancestors of the cut — zoom-out insurance, loaded when idle.
    Preload,
}

#[derive(Debug, Clone, Copy)]
pub struct LoadRequest {
    pub tile: usize,
    pub priority: Priority,
    /// Tie-break within a tier: distance weighted toward screen center
    /// (smaller = sooner).
    pub key: f64,
}

/// Result of one selection pass.
#[derive(Debug, Default)]
pub struct Selection {
    /// Tiles whose content should be visible this frame.
    pub render: Vec<usize>,
    /// Wanted content loads, sorted by `(priority, key)`. Includes tiles
    /// already in flight — the scheduler diffs; anything in flight but NOT
    /// here has fallen out of the cut and gets cancelled.
    pub loads: Vec<LoadRequest>,
    /// Tiles that refined this frame (input history for the next pass).
    pub refined: Vec<bool>,
    /// Tiles that are part of this frame's wanted set (render ∪ loads ∪
    /// traversed interior) — the eviction grace clock resets for these.
    pub touched: Vec<bool>,
    /// Whether the root subtree painted its whole footprint. `false` ⇒ a
    /// genuine traversal HOLE: some on-screen area has no Ready tile to render
    /// (vs. a render-side problem where `covered` is true but pixels are wrong).
    pub covered: bool,
}

/// Frame history needed by zoom-out protection + kicking.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub rendered: Vec<bool>,
    pub refined: Vec<bool>,
}

impl History {
    pub fn resize(&mut self, n: usize) {
        self.rendered.resize(n, false);
        self.refined.resize(n, false);
    }

    /// Roll a selection into the history for the next frame.
    pub fn absorb(&mut self, sel: &Selection, n: usize) {
        self.rendered.clear();
        self.rendered.resize(n, false);
        for &t in &sel.render {
            self.rendered[t] = true;
        }
        self.refined = sel.refined.clone();
    }
}

/// Per-tile content readiness, as the traversal sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileContent {
    /// No content URI — an interior/grouping tile; nothing to load or draw.
    None,
    /// Content exists but isn't displayable yet (not requested or in flight).
    Pending,
    /// Content decoded + spawned; can be made visible this frame.
    Ready,
    /// Terminal decode/fetch failure — never displayable, never re-queued.
    Failed,
}

impl TileContent {
    /// "Nothing further to wait for": ready, content-free, or terminally
    /// failed. The coverage test for kicking.
    fn settled(self) -> bool {
        !matches!(self, TileContent::Pending)
    }

    fn loadable(self) -> bool {
        matches!(self, TileContent::Pending)
    }
}

struct Ctx<'a, F: Fn(usize) -> bool> {
    tree: &'a TileTree,
    content: &'a [TileContent],
    history: &'a History,
    culled: &'a F,
    params: SelectParams,
}

struct VisitOut {
    /// The subtree paints its whole footprint (or has nothing to paint).
    covered: bool,
    /// Some tile of the selected subtree cut was rendered last frame.
    any_rendered_last: bool,
}

/// Run one selection pass. `content[i]` mirrors `tree.nodes[i]`;
/// `culled(i)` = tile `i`'s bounding sphere is outside the frustum.
pub fn select<F: Fn(usize) -> bool>(
    tree: &TileTree,
    content: &[TileContent],
    history: &History,
    culled: &F,
    params: SelectParams,
) -> Selection {
    let n = tree.nodes.len();
    debug_assert_eq!(content.len(), n);
    let mut sel = Selection {
        render: Vec::new(),
        loads: Vec::new(),
        refined: vec![false; n],
        touched: vec![false; n],
        covered: false,
    };
    if n == 0 {
        return sel;
    }
    let ctx = Ctx { tree, content, history, culled, params };
    sel.covered = visit(&ctx, 0, &mut sel).covered;

    // Preload tier: ancestors of the rendered cut that have unloaded content.
    // REPLACE refinement means a loaded ancestor gives instant zoom-out (and a
    // kick target) — basemap's "ancestors trickle in last" tier, generalized.
    // EVERY ancestor is marked touched (not just loadable ones): a READY
    // parent chain must stay resident, or kicks land on a much coarser
    // ancestor and the view collapses several levels while children stream.
    let mut queued = vec![false; n];
    for req in &sel.loads {
        queued[req.tile] = true;
    }
    for i in 0..sel.render.len() {
        let mut at = tree.nodes[sel.render[i]].parent;
        while let Some(p) = at {
            sel.touched[p] = true;
            if ctx.content[p].loadable() && !queued[p] {
                queued[p] = true;
                let dist = tree.nodes[p].volume.distance_to(params.cam_pos);
                sel.loads.push(LoadRequest {
                    tile: p,
                    priority: Priority::Preload,
                    key: load_key(&ctx, p, dist),
                });
            }
            at = tree.nodes[p].parent;
        }
    }

    sel.loads.sort_by(|a, b| {
        (a.priority, a.key)
            .partial_cmp(&(b.priority, b.key))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sel
}

/// Priority tie-break key: distance, weighted away from the screen edge —
/// `dist × (2 − cos θ)` where θ is the angle off the view axis. Center tiles
/// load first among equals (§7.6).
fn load_key<F: Fn(usize) -> bool>(ctx: &Ctx<'_, F>, tile: usize, dist: f64) -> f64 {
    let (center, _) = ctx.tree.nodes[tile].volume.bounding_sphere();
    let to = center - ctx.params.cam_pos;
    let cos = if to.length_squared() > 1e-12 {
        to.normalize().dot(ctx.params.cam_forward).clamp(-1.0, 1.0)
    } else {
        1.0
    };
    dist * (2.0 - cos)
}

fn push_load<F: Fn(usize) -> bool>(ctx: &Ctx<'_, F>, sel: &mut Selection, tile: usize, dist: f64) {
    if ctx.content[tile].loadable() {
        let priority = if dist <= INSIDE_EPS { Priority::Urgent } else { Priority::Normal };
        sel.loads.push(LoadRequest { tile, priority, key: load_key(ctx, tile, dist) });
    }
}

fn visit<F: Fn(usize) -> bool>(ctx: &Ctx<'_, F>, i: usize, sel: &mut Selection) -> VisitOut {
    let node = &ctx.tree.nodes[i];
    sel.touched[i] = true;

    // Floor the SSE distance at the camera's height above the ground. A coarse
    // ground tile's bounding box is TALL (its big geometric error inflates the
    // vertical extent), so at 10-15 km the camera sits INSIDE it ⇒
    // `distance_to` returns 0 ⇒ infinite SSE ⇒ the tile is forced to refine
    // into street-level children that (rightly) aren't loaded from that height
    // ⇒ a permanent hole right under the camera (the live "tiles below vanish
    // in the 10-15 km band" finding). But the tile's CONTENT is the ground,
    // ~cam_height below, so the real viewing distance is never less than that —
    // flooring there caps the nadir at an altitude-appropriate LOD that IS
    // loaded, so it renders instead of holing. `cam_height_m == 0` for non-globe
    // sets ⇒ no change (genuine camera-inside still gives infinite SSE).
    let dist = node.volume.distance_to(ctx.params.cam_pos).max(ctx.params.cam_height_m);
    let sse = screen_space_error(node.geometric_error, dist, ctx.params.k_px);

    // Distance-relaxed refine threshold: detail falls off with distance so a
    // grazing horizon view doesn't refine the whole visible hemisphere to full
    // LOD (and then graft+stream thousands of far subtrees it can't keep up
    // with). Near tiles (`dist ≪ falloff`) keep the base threshold; far tiles
    // need a proportionally larger error to subdivide, so the cut — and the
    // descent that triggers grafting — stays bounded to roughly the near view.
    // (`dist` is floored at the camera height above, so a high top-down view
    // refines to an altitude-appropriate level under the camera, never infinitely.)
    let threshold = if ctx.params.detail_falloff_m > 0.0 {
        // Measure the falloff from the camera's HEIGHT, not its position: the
        // tile right under a high top-down view has `dist ≈ cam_height_m` ⇒
        // `extra ≈ 0` ⇒ base threshold (stays sharp), while tiles reaching
        // toward the horizon (`dist ≫ height`) relax and stop subdividing.
        let extra = (dist - ctx.params.cam_height_m).max(0.0);
        ctx.params.sse_threshold_px * (1.0 + extra / ctx.params.detail_falloff_m)
    } else {
        ctx.params.sse_threshold_px
    };

    // Contentless tiles with children ALWAYS refine — there is nothing to
    // render at this level (structural interiors, and tiles whose external
    // tileset was grafted away), so stopping per SSE would paint a hole.
    // (Distance falloff can't stop here, but the content-bearing parent that
    // led here was already distance-gated, so far interiors are never reached.)
    let mut wants_refine =
        !node.children.is_empty() && (sse > threshold || node.content_uri.is_none());
    // Zoom-out protection (§7.4): this tile became the desired cut but its
    // content isn't here yet — keep refining (descendants stay visible) while
    // the coarser content loads.
    if !wants_refine
        && !node.children.is_empty()
        && ctx.history.refined[i]
        && !ctx.content[i].settled()
    {
        wants_refine = true;
        push_load(ctx, sel, i, dist);
    }

    if !wants_refine {
        // Render this tile (leaves always land here).
        push_load(ctx, sel, i, dist);
        // Refinement frontier: a childless tile the camera still wants finer
        // than is the edge we descend from. Upgrade its load to `Descend` so
        // the tree extends toward the view (and the nearest detail paints)
        // before slots go to already-good-enough content elsewhere.
        if node.children.is_empty()
            && sse > threshold
            && let Some(req) = sel.loads.last_mut()
            && req.tile == i
            && req.priority == Priority::Normal
        {
            req.priority = Priority::Descend;
        }
        if ctx.content[i] == TileContent::Ready {
            sel.render.push(i);
        }
        return VisitOut {
            covered: ctx.content[i].settled(),
            any_rendered_last: ctx.history.rendered[i],
        };
    }

    sel.refined[i] = true;

    if node.refine == Refine::Add {
        // Additive refinement: the parent renders alongside its children —
        // no replacement, so no kicking either.
        push_load(ctx, sel, i, dist);
        if ctx.content[i] == TileContent::Ready {
            sel.render.push(i);
        }
        let mut any_last = ctx.history.rendered[i];
        for &c in &node.children {
            if (ctx.culled)(c) {
                continue;
            }
            any_last |= visit(ctx, c, sel).any_rendered_last;
        }
        return VisitOut { covered: ctx.content[i].settled(), any_rendered_last: any_last };
    }

    // REPLACE refinement.
    let checkpoint = sel.render.len();
    let mut all_covered = true;
    let mut any_last = false;
    for &c in &node.children {
        if (ctx.culled)(c) {
            continue;
        }
        let v = visit(ctx, c, sel);
        all_covered &= v.covered;
        any_last |= v.any_rendered_last;
    }
    if all_covered {
        return VisitOut { covered: true, any_rendered_last: any_last };
    }
    // Some selected descendant isn't paintable yet. Resolution ORDER is
    // load-bearing (each clause earned by a live-P3DT regression):
    //
    // 1. Some of this subtree's FINER level was on screen last frame → KEEP
    //    IT. Hold the ready children that are already showing and let the
    //    pending siblings stream into a transient gap; this tile is queued as
    //    a backstop but NOT rendered, so coarse photogrammetry never overlaps
    //    the fine it would occlude. This is the "don't unload what the user is
    //    already looking at" rule — and because it returns *covered*, one
    //    freshly-grafted pending tile can no longer cascade up through the
    //    contentless graft-interiors and collapse every on-screen sibling to
    //    coarse (the live "spin → everything blinks coarse" finding). MUST run
    //    before the atomic swap, or a Ready ancestor wins and wipes the cut.
    if any_last {
        push_load(ctx, sel, i, dist);
        return VisitOut { covered: true, any_rendered_last: true };
    }
    // 2. Nothing finer has ever shown here (initial load, or zoom-in from this
    //    coarse tile) and it's renderable → ATOMIC SWAP: paint this tile's own
    //    content in place of the partial cut below. Coarse-first, never an
    //    overlap (coarse sits ABOVE fine and occludes it), never a hole.
    if ctx.content[i] == TileContent::Ready {
        sel.render.truncate(checkpoint);
        sel.render.push(i);
        return VisitOut { covered: true, any_rendered_last: ctx.history.rendered[i] };
    }
    // 3. Nothing finer shown AND nothing to swap to (contentless/pending):
    //    drop the partial selections and report uncovered so the nearest READY
    //    ancestor paints the whole footprint coarse-first. Load the fallback
    //    URGENTLY: this is a genuine HOLE (no Ready ancestor covers here yet),
    //    and the worst case is right under the camera, where the nadir refines
    //    hardest and its fine children stream slowest. At Normal priority the
    //    coarse backdrop queued behind all that fine detail and the gap
    //    persisted; Urgent makes the coarse backdrop win the race, so the view
    //    fills coarse-first and refines on top instead of holing. Every tile on
    //    the uncovered ancestor path lands here, so the SHALLOWEST loadable one
    //    (the quickest, smallest coarse tile) paints the footprint first.
    sel.render.truncate(checkpoint);
    if ctx.content[i].loadable() {
        sel.loads.push(LoadRequest {
            tile: i,
            priority: Priority::Urgent,
            key: load_key(ctx, i, dist),
        });
    }
    VisitOut { covered: false, any_rendered_last: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schema::Refine;

    /// k for a 1080-px viewport at 45° vertical FOV.
    fn k_1080() -> f64 {
        1080.0 / (2.0 * (45f64.to_radians() / 2.0).tan())
    }

    fn params(cam: DVec3) -> SelectParams {
        SelectParams {
            cam_pos: cam,
            cam_forward: (DVec3::ZERO - cam).normalize_or(DVec3::NEG_Z),
            k_px: k_1080(),
            sse_threshold_px: DEFAULT_SSE_THRESHOLD_PX,
            // Disabled in most tests so the SSE assertions stay exact; the
            // falloff has its own dedicated test below.
            detail_falloff_m: 0.0,
            cam_height_m: 0.0,
        }
    }

    fn sphere(center: DVec3, radius: f64) -> WorldVolume {
        WorldVolume::Sphere { center, radius }
    }

    /// root(ge 16, r 30) → 4 children (ge 4, r 9) → 4 leaves each (ge 0, r 4).
    /// All tiles carry content. Returns the tree; children are nodes 1–4,
    /// leaves 5–20 (children of node c start at 1+4+(c-1)*4).
    fn fixture_tree() -> TileTree {
        let mut tree = TileTree::default();
        tree.nodes.push(TileNode {
            parent: None,
            children: vec![],
            depth: 0,
            geometric_error: 16.0,
            refine: Refine::Replace,
            content_uri: Some("root.glb".into()),
            volume: sphere(DVec3::ZERO, 30.0),
            world_from_content: DMat4::IDENTITY,
            world_from_tile: DMat4::IDENTITY,
        });
        let quad = [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)];
        for (cx, cz) in quad {
            let c = tree.nodes.len();
            tree.nodes.push(TileNode {
                parent: Some(0),
                children: vec![],
                depth: 1,
                geometric_error: 4.0,
                refine: Refine::Replace,
                content_uri: Some(format!("c{c}.glb")),
                volume: sphere(DVec3::new(cx, 0.0, cz), 9.0),
                world_from_content: DMat4::IDENTITY,
                world_from_tile: DMat4::IDENTITY,
            });
            tree.nodes[0].children.push(c);
        }
        for c in 1..=4 {
            let (cx, cz) = quad[c - 1];
            for (lx, lz) in quad {
                let l = tree.nodes.len();
                tree.nodes.push(TileNode {
                    parent: Some(c),
                    children: vec![],
                    depth: 2,
                    geometric_error: 0.0,
                    refine: Refine::Replace,
                    content_uri: Some(format!("l{l}.glb")),
                    volume: sphere(DVec3::new(cx + lx * 0.25, 0.0, cz + lz * 0.25), 4.0),
                    world_from_content: DMat4::IDENTITY,
                    world_from_tile: DMat4::IDENTITY,
                });
                let cc = tree.nodes[c].children.clone();
                tree.nodes[c].children = [cc, vec![l]].concat();
            }
        }
        tree
    }

    fn all(content: TileContent, n: usize) -> Vec<TileContent> {
        vec![content; n]
    }

    fn no_cull(_: usize) -> bool {
        false
    }

    #[test]
    fn sse_math() {
        // 16 m error at 1000 m through a 1304-px focal length ≈ 20.9 px.
        let k = k_1080();
        let sse = screen_space_error(16.0, 1000.0, k);
        assert!((sse - 16.0 * k / 1000.0).abs() < 1e-9);
        assert_eq!(screen_space_error(16.0, 0.0, k), f64::INFINITY);
        assert_eq!(screen_space_error(0.0, 100.0, k), 0.0);
    }

    #[test]
    fn distance_falloff_stops_far_refinement() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history =
            History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        // ~970 m out: root SSE ≈ 21 px > the 16 px base threshold.
        let cam = DVec3::new(0.0, 0.0, 1000.0);

        // No falloff → the root refines into its children (pre-fix behaviour).
        let near = select(&tree, &content, &history, &no_cull, params(cam));
        assert_eq!(near.render, vec![1, 2, 3, 4], "without falloff the root refines");

        // 500 m falloff → the root's threshold relaxes to ≈47 px (> its 21 px
        // SSE), so it renders coarse and its far subtree is never descended —
        // no graft/stream storm toward the horizon.
        let relaxed = SelectParams { detail_falloff_m: 500.0, ..params(cam) };
        let far = select(&tree, &content, &history, &no_cull, relaxed);
        assert_eq!(far.render, vec![0], "falloff keeps the distant tile coarse");
    }

    #[test]
    fn high_camera_inside_tall_volume_renders_coarse_not_a_hole() {
        // The fixture root (ge 16, r 30) is centred at the origin, so a camera
        // AT the origin sits INSIDE it ⇒ `distance_to == 0` ⇒ infinite SSE ⇒
        // forced refinement to the finest leaves. That's the tall-bounding-box
        // trap behind the live "tiles directly below vanish in the 10-15 km
        // band" finding: a coarse ground tile's box reaches up to the camera, so
        // the nadir demands street-level tiles that aren't loaded from altitude.
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history =
            History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        let cam = DVec3::ZERO; // inside the root volume

        // No height floor (the bug): infinite SSE forces refinement to the leaves.
        let unfloored = select(&tree, &content, &history, &no_cull, params(cam));
        assert!(
            !unfloored.render.is_empty() && unfloored.render.iter().all(|&t| t >= 5),
            "without the floor an inside camera over-refines to the leaves: {:?}",
            unfloored.render
        );

        // 2 km height floor: root SSE ≈ 10 px < the 16 px threshold, so the
        // coarse root renders — altitude-appropriate, no over-refine, no hole.
        let floored = SelectParams { cam_height_m: 2000.0, ..params(cam) };
        let sel = select(&tree, &content, &history, &no_cull, floored);
        assert_eq!(sel.render, vec![0], "the height floor renders the coarse tile, not a hole");
    }

    #[test]
    fn refine_frontier_loads_at_descend_priority() {
        // A childless tile the camera wants finer than (here SSE ≫ threshold)
        // is the refinement frontier — its load outranks settled Normal
        // content but still yields to Urgent.
        let mut tree = TileTree::default();
        tree.nodes.push(TileNode {
            parent: None,
            children: vec![],
            depth: 0,
            geometric_error: 16.0,
            refine: Refine::Replace,
            content_uri: Some("x.glb".into()),
            volume: sphere(DVec3::ZERO, 30.0),
            world_from_content: DMat4::IDENTITY,
            world_from_tile: DMat4::IDENTITY,
        });
        let content = vec![TileContent::Pending];
        let history = History { rendered: vec![false; 1], refined: vec![false; 1] };
        // 10 m off the surface: SSE ≫ threshold, but the camera is NOT inside.
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 0.0, 40.0)));
        let req = sel.loads.iter().find(|r| r.tile == 0).expect("queued");
        assert_eq!(req.priority, Priority::Descend);
        // And Descend outranks Normal / Preload in the sort.
        assert!(Priority::Urgent < Priority::Descend && Priority::Descend < Priority::Normal);
    }

    #[test]
    fn far_camera_renders_root_only() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        // Root sse = 16·k/d; d ≈ 3000−30 → sse ≈ 7 px < 16 → no refine.
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 0.0, 3000.0)));
        assert_eq!(sel.render, vec![0]);
        assert!(sel.loads.is_empty());
    }

    #[test]
    fn near_camera_selects_leaves() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        // Inside the root sphere → refine; children near → refine; leaves render.
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        assert_eq!(sel.render.len(), 16);
        assert!(sel.render.iter().all(|&t| t >= 5), "only leaves: {:?}", sel.render);
        assert!(sel.refined[0]);
        assert!((1..=4).all(|c| sel.refined[c]));
    }

    #[test]
    fn unloaded_leaves_kick_to_ready_parent() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        for slot in content.iter_mut().skip(5) {
            *slot = TileContent::Pending;
        }
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        // Children render in place of their unloaded leaves…
        assert_eq!(sel.render, vec![1, 2, 3, 4]);
        // …and every leaf stays in the load queue at Normal/Urgent priority.
        let leaf_loads: Vec<usize> =
            sel.loads.iter().filter(|r| r.tile >= 5).map(|r| r.tile).collect();
        assert_eq!(leaf_loads.len(), 16);
        assert!(
            sel.loads.iter().all(|r| r.priority != Priority::Preload),
            "cut loads must not be Preload"
        );
    }

    #[test]
    fn kick_cascades_when_parent_unloaded_too() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Pending, tree.len());
        content[0] = TileContent::Ready;
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        // Nothing below the root can paint → the kick cascades to the root.
        assert_eq!(sel.render, vec![0]);
        // Children + leaves all queued.
        assert!(sel.loads.len() >= 20);
    }

    #[test]
    fn partial_refinement_keeps_onscreen_children_no_coarse_collapse() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        // Child 1's leaves: two ready (and previously rendered), two pending.
        let kids = tree.nodes[1].children.clone();
        content[kids[2]] = TileContent::Pending;
        content[kids[3]] = TileContent::Pending;
        let mut history =
            History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        history.rendered[kids[0]] = true;
        history.rendered[kids[1]] = true;
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        // The on-screen fine leaves KEEP rendering — a streaming sibling must
        // never collapse them back to the coarse parent…
        assert!(sel.render.contains(&kids[0]) && sel.render.contains(&kids[1]));
        // …and the parent does NOT render (no coarse-over-fine overlap); the
        // two pending footprints are a transient gap, not a coarse swap.
        assert!(!sel.render.contains(&1), "no coarse collapse over visible fine");
        // The pending two stay queued so the gap fills with full detail.
        let queued: Vec<usize> = sel.loads.iter().map(|r| r.tile).collect();
        assert!(queued.contains(&kids[2]) && queued.contains(&kids[3]));
    }

    /// A pending tile under a CONTENTLESS parent (P3DT's grafted interiors)
    /// whose siblings were on screen must NOT wipe them: the cascade brakes,
    /// the rendered siblings stay, the pending one is a small hole until it
    /// lands. (Pre-brake, the uncovered signal climbed through the
    /// contentless chain and swapped huge footprints — or the whole tileset
    /// — to coarse/black on rotation.)
    #[test]
    fn contentless_parent_with_rendered_siblings_brakes_the_cascade() {
        let mut tree = fixture_tree();
        tree.nodes[1].content_uri = None; // consumed/grafted interior
        let mut content = all(TileContent::Ready, tree.len());
        content[1] = TileContent::None;
        let kids = tree.nodes[1].children.clone();
        content[kids[3]] = TileContent::Pending; // one new sibling streaming
        let mut history =
            History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        for &k in &kids[0..3] {
            history.rendered[k] = true;
        }
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        // The three on-screen leaves keep rendering; no ancestor swap.
        for &k in &kids[0..3] {
            assert!(sel.render.contains(&k), "rendered sibling {k} kept");
        }
        assert!(!sel.render.contains(&0), "root does not wipe the region");
        // The pending sibling stays queued.
        assert!(sel.loads.iter().any(|r| r.tile == kids[3]));
        // The other quadrants' leaves are untouched.
        assert_eq!(sel.render.len(), 3 + 12);
    }

    /// Ready ancestors of the rendered cut stay `touched` — evicting them
    /// would make every kick collapse several levels instead of one.
    #[test]
    fn cut_ancestors_stay_touched_for_residency() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        assert_eq!(sel.render.len(), 16, "leaves render");
        for anc in 0..=4 {
            assert!(sel.touched[anc], "ancestor {anc} kept resident");
        }
    }

    #[test]
    fn zoom_out_keeps_children_until_parent_loads() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        content[0] = TileContent::Pending; // the coarse target isn't in yet
        let mut history =
            History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        history.refined[0] = true; // we were refined through the root last frame
        for c in 1..=4 {
            history.rendered[c] = true;
        }
        // Far camera → SSE wants the root alone, but it has no content yet.
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 0.0, 3000.0)));
        assert_eq!(sel.render, vec![1, 2, 3, 4], "children stay visible");
        assert!(sel.loads.iter().any(|r| r.tile == 0), "root load queued");
        // Once the root lands, the very next pass collapses to it.
        content[0] = TileContent::Ready;
        let mut h2 = History::default();
        h2.absorb(&sel, tree.len());
        let sel2 = select(&tree, &content, &h2, &no_cull, params(DVec3::new(0.0, 0.0, 3000.0)));
        assert_eq!(sel2.render, vec![0]);
    }

    #[test]
    fn camera_inside_volume_is_urgent() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        let leaf = 5;
        content[leaf] = TileContent::Pending;
        let history = History { rendered: vec![true; tree.len()], refined: vec![false; tree.len()] };
        // Camera inside leaf 5's sphere (center (-10.25,0,-10.25), r 4).
        let cam = DVec3::new(-10.0, 0.0, -10.0);
        let sel = select(&tree, &content, &history, &no_cull, params(cam));
        let req = sel.loads.iter().find(|r| r.tile == leaf).expect("leaf queued");
        assert_eq!(req.priority, Priority::Urgent);
        // Urgent sorts first.
        assert_eq!(sel.loads.first().unwrap().priority, Priority::Urgent);
    }

    #[test]
    fn ancestors_of_cut_become_preload() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        for slot in content.iter_mut().take(5) {
            *slot = TileContent::Pending;
        }
        // History: leaves rendered last frame so no kicking interferes.
        let mut history =
            History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        for leaf in 5..tree.len() {
            history.rendered[leaf] = true;
        }
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        assert_eq!(sel.render.len(), 16);
        // Root + children queued as Preload (they're not in the cut).
        for anc in 0..=4 {
            let req = sel.loads.iter().find(|r| r.tile == anc).expect("ancestor queued");
            assert_eq!(req.priority, Priority::Preload, "tile {anc}");
        }
        // …and Preload sorts after the cut's own (empty here) tier.
        assert!(sel.loads.windows(2).all(|w| w[0].priority <= w[1].priority));
    }

    #[test]
    fn culled_children_are_skipped() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        // Cull child 1 and its subtree (the traversal only tests children at
        // their parent, so culling node 1 prunes nodes 5–8 implicitly).
        let culled = |i: usize| i == 1;
        let sel = select(&tree, &content, &history, &culled, params(DVec3::new(0.0, 5.0, 0.0)));
        assert_eq!(sel.render.len(), 12, "3 visible children × 4 leaves");
        assert!(sel.render.iter().all(|&t| !(5..=8).contains(&t) && t != 1));
        // Culled subtrees request nothing.
        assert!(sel.loads.iter().all(|r| r.tile != 1));
    }

    #[test]
    fn add_refinement_renders_parent_and_children() {
        let mut tree = fixture_tree();
        for node in &mut tree.nodes {
            node.refine = Refine::Add;
        }
        let content = all(TileContent::Ready, tree.len());
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        // ADD: every level renders together.
        assert!(sel.render.contains(&0));
        assert!((1..=4).all(|c| sel.render.contains(&c)));
        assert_eq!(sel.render.len(), 21);
    }

    #[test]
    fn contentless_interior_tile_covers_via_children() {
        // root (no content) → 2 ready leaves: root renders nothing, leaves
        // paint, nothing queues for the root.
        let mut tree = TileTree::default();
        tree.nodes.push(TileNode {
            parent: None,
            children: vec![1, 2],
            depth: 0,
            geometric_error: 16.0,
            refine: Refine::Replace,
            content_uri: None,
            volume: sphere(DVec3::ZERO, 10.0),
            world_from_content: DMat4::IDENTITY,
            world_from_tile: DMat4::IDENTITY,
        });
        for i in 0..2 {
            tree.nodes.push(TileNode {
                parent: Some(0),
                children: vec![],
                depth: 1,
                geometric_error: 0.0,
                refine: Refine::Replace,
                content_uri: Some(format!("l{i}.glb")),
                volume: sphere(DVec3::new(i as f64 * 4.0 - 2.0, 0.0, 0.0), 5.0),
                world_from_content: DMat4::IDENTITY,
                world_from_tile: DMat4::IDENTITY,
            });
        }
        let content =
            vec![TileContent::None, TileContent::Ready, TileContent::Ready];
        let history = History { rendered: vec![false; 3], refined: vec![false; 3] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 1.0, 0.0)));
        assert_eq!(sel.render, vec![1, 2]);
        assert!(sel.loads.is_empty());

        // FAR camera: SSE alone would stop at the contentless root and paint
        // a hole — empty tiles must refine through to their children (the
        // grafted-subtree shape: the consumed tile keeps its volume but has
        // nothing to draw).
        let sel_far =
            select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 0.0, 5000.0)));
        assert_eq!(sel_far.render, vec![1, 2], "refines through the empty root");
    }

    /// A contentless interior whose children are ALL pending must report its
    /// footprint uncovered, so the ancestor kick backfills it (pre-fix it
    /// claimed coverage and left a hole while a grafted subtree streamed).
    #[test]
    fn pending_subtree_under_contentless_interior_kicks_to_ancestor() {
        let mut tree = fixture_tree();
        // Make child 1 contentless (a consumed/grafted interior).
        tree.nodes[1].content_uri = None;
        let mut content = all(TileContent::Ready, tree.len());
        content[1] = TileContent::None;
        for &leaf in tree.nodes[1].children.clone().iter() {
            content[leaf] = TileContent::Pending;
        }
        let history = History { rendered: vec![false; tree.len()], refined: vec![false; tree.len()] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 5.0, 0.0)));
        // Child 1's subtree is unpainted → the ROOT kicks and renders itself.
        assert_eq!(sel.render, vec![0], "root backfills the unpainted quadrant");
        // The pending leaves stay queued.
        let queued: Vec<usize> = sel.loads.iter().map(|r| r.tile).collect();
        for &leaf in &tree.nodes[1].children {
            assert!(queued.contains(&leaf), "leaf {leaf} queued");
        }
    }

    #[test]
    fn obb_distance_and_sphere() {
        let obb = WorldVolume::Obb {
            center: DVec3::ZERO,
            half_axes: [DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.0, 3.0, 0.0), DVec3::new(0.0, 0.0, 4.0)],
        };
        assert_eq!(obb.distance_to(DVec3::new(1.0, 1.0, 1.0)), 0.0); // inside
        assert!((obb.distance_to(DVec3::new(5.0, 0.0, 0.0)) - 3.0).abs() < 1e-9);
        let (_, r) = obb.bounding_sphere();
        assert!((r - (4.0f64 + 9.0 + 16.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn tree_build_composes_transforms_and_inherits() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 64,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 30] },
                "geometricError": 16,
                "refine": "REPLACE",
                "children": [{
                    "boundingVolume": { "box": [0,0,0, 5,0,0, 0,5,0, 0,0,1] },
                    "geometricError": 4,
                    "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 10,0,3,1],
                    "children": [{
                        "boundingVolume": { "region": [0,0,0,0,0,0] },
                        "geometricError": 0
                    }]
                }]
            }
        }"#;
        let ts = schema::parse_tileset(json.as_bytes()).unwrap();
        let tree = TileTree::build(&ts, ZUP_TO_BEVY, TreeFrame::Local).unwrap();
        assert_eq!(tree.len(), 3);
        // Child volume center: zup (10,0,3) → bevy (10, 3, 0).
        let WorldVolume::Obb { center, .. } = tree.nodes[1].volume else {
            panic!("expected obb")
        };
        assert!((center - DVec3::new(10.0, 3.0, 0.0)).length() < 1e-9);
        // Region grandchild inherits the (transformed) parent volume…
        let WorldVolume::Obb { center: gc, .. } = tree.nodes[2].volume else {
            panic!("expected inherited obb")
        };
        assert_eq!(gc, center);
        // …and the REPLACE refine.
        assert_eq!(tree.nodes[2].refine, Refine::Replace);
        // Content placement: world_from_content = zup→bevy × T × yup→zup; a
        // glTF point at origin lands at the tile's translation in Bevy space.
        let p = tree.nodes[1].world_from_content.transform_point3(DVec3::ZERO);
        assert!((p - DVec3::new(10.0, 3.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn region_root_rejected_in_local_frame_accepted_in_ecef() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 1,
            "root": {
                "boundingVolume": { "region": [-1.32, 0.69, -1.31, 0.70, 0, 90] },
                "geometricError": 0
            }
        }"#;
        let ts = schema::parse_tileset(json.as_bytes()).unwrap();
        assert!(TileTree::build(&ts, ZUP_TO_BEVY, TreeFrame::Local).is_err());
        let tree = TileTree::build(&ts, DMat4::IDENTITY, TreeFrame::Ecef).unwrap();
        // The region landed as a real ECEF volume — its centre sits at
        // planetary magnitude, not at the local origin.
        let (center, radius) = tree.nodes[0].volume.bounding_sphere();
        assert!(center.length() > 6_000_000.0, "center = {center:?}");
        assert!(radius > 100.0 && radius < 100_000.0, "radius = {radius}");
    }

    #[test]
    fn graft_external_tileset_under_content_tile() {
        // Host: root → child with a translated transform whose content is an
        // external tileset.json.
        let host = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 64,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 100] },
                "geometricError": 32,
                "refine": "REPLACE",
                "children": [{
                    "boundingVolume": { "sphere": [0, 0, 0, 40] },
                    "geometricError": 16,
                    "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 100,0,0,1],
                    "content": { "uri": "sub/tileset.json" }
                }]
            }
        }"#;
        // External: a root with its own child, positioned in the HOST tile's
        // frame (the +100 X translation must compose through).
        let external = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 16,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 40] },
                "geometricError": 8,
                "content": { "uri": "sub/root.glb" },
                "children": [{
                    "boundingVolume": { "sphere": [5, 0, 0, 10] },
                    "geometricError": 0,
                    "content": { "uri": "sub/leaf.glb" }
                }]
            }
        }"#;
        let host_ts = schema::parse_tileset(host.as_bytes()).unwrap();
        let ext_ts = schema::parse_tileset(external.as_bytes()).unwrap();
        let mut tree = TileTree::build(&host_ts, ZUP_TO_BEVY, TreeFrame::Local).unwrap();
        assert_eq!(tree.len(), 2);

        let new_root = tree.graft(1, &ext_ts, TreeFrame::Local).unwrap();
        tree.nodes[1].content_uri = None; // consumed by the graft
        assert_eq!(new_root, 2);
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.nodes[1].children, vec![2]);
        assert_eq!(tree.nodes[2].parent, Some(1));
        assert_eq!(tree.nodes[2].depth, 2);
        // Inherited REPLACE refine (external carries none).
        assert_eq!(tree.nodes[2].refine, Refine::Replace);
        // The host tile's +100 X (zup) transform composes into the grafted
        // volumes: zup (100,0,0) → bevy (100,0,0).
        let (c2, _) = tree.nodes[2].volume.bounding_sphere();
        assert!((c2 - DVec3::new(100.0, 0.0, 0.0)).length() < 1e-9, "c2 = {c2:?}");
        let (c3, _) = tree.nodes[3].volume.bounding_sphere();
        assert!((c3 - DVec3::new(105.0, 0.0, 0.0)).length() < 1e-9, "c3 = {c3:?}");
        // Content placement composes the same chain.
        let p = tree.nodes[3].world_from_content.transform_point3(DVec3::ZERO);
        assert!((p - DVec3::new(100.0, 0.0, 0.0)).length() < 1e-9, "p = {p:?}");
    }

    /// A bare node for the structural `retain` tests (geometry irrelevant).
    fn bare(parent: Option<usize>, children: Vec<usize>) -> TileNode {
        TileNode {
            parent,
            children,
            depth: 0,
            geometric_error: 1.0,
            refine: Refine::Replace,
            content_uri: None,
            volume: sphere(DVec3::ZERO, 1.0),
            world_from_content: DMat4::IDENTITY,
            world_from_tile: DMat4::IDENTITY,
        }
    }

    #[test]
    fn retain_drops_pruned_subtrees_and_renumbers_siblings() {
        // root(0) → [1, 2, 3]; prune the odd children, keep root + 2.
        let mut tree = TileTree::default();
        tree.nodes.push(bare(None, vec![1, 2, 3]));
        tree.nodes.push(bare(Some(0), vec![]));
        tree.nodes.push(bare(Some(0), vec![]));
        tree.nodes.push(bare(Some(0), vec![]));

        let map = tree.retain(&[true, false, true, false]);
        assert_eq!(map[0], 0);
        assert_eq!(map[2], 1);
        assert_eq!(map[1], usize::MAX);
        assert_eq!(map[3], usize::MAX);
        assert_eq!(tree.len(), 2);
        // root keeps only old-node-2, renumbered to new index 1.
        assert_eq!(tree.nodes[0].children, vec![1]);
        assert_eq!(tree.nodes[1].parent, Some(0));
        assert!(tree.nodes[1].children.is_empty());
    }

    #[test]
    fn retain_remaps_a_kept_chain() {
        // root(0) → 1 → 2 (a chain), plus root → 3. Prune node 3; the chain
        // survives and renumbers contiguously, parent/child links intact.
        let mut tree = TileTree::default();
        tree.nodes.push(bare(None, vec![1, 3]));
        tree.nodes.push(bare(Some(0), vec![2]));
        tree.nodes.push(bare(Some(1), vec![]));
        tree.nodes.push(bare(Some(0), vec![]));

        let map = tree.retain(&[true, true, true, false]);
        assert_eq!(tree.len(), 3);
        assert_eq!(map[3], usize::MAX);
        assert_eq!(tree.nodes[0].children, vec![1]); // old [1,3] → [1]
        assert_eq!(tree.nodes[1].parent, Some(0));
        assert_eq!(tree.nodes[1].children, vec![2]);
        assert_eq!(tree.nodes[2].parent, Some(1));
    }

    #[test]
    fn retain_noop_when_all_kept() {
        let mut tree = TileTree::default();
        tree.nodes.push(bare(None, vec![1]));
        tree.nodes.push(bare(Some(0), vec![]));
        let map = tree.retain(&[true, true]);
        assert_eq!(map, vec![0, 1]);
        assert_eq!(tree.len(), 2);
    }
}
