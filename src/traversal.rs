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
//!    this tile. Leaves always render.
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
    /// Bounding volume in Bevy world space (tile transforms pre-composed).
    pub volume: WorldVolume,
    /// Full content placement: `world_from_tileset × cumulative tile
    /// transforms × glTF-Y-up rotation`. The spawned tile entity's transform.
    pub world_from_content: DMat4,
}

impl TileNode {
    pub fn has_content(&self) -> bool {
        self.content_uri.is_some()
    }
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
    /// origin: [`ZUP_TO_BEVY`], optionally × the twin's ENU placement).
    pub fn build(
        tileset: &schema::Tileset,
        world_from_tileset: DMat4,
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
        )?;
        Ok(tree)
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
        Some(VolumeKind::Region(_)) | None => {
            // Region volumes are EPSG:4979 — placing them needs the
            // georeferenced ECEF→ENU path (T4, Google P3DT). Until then a
            // region (or missing) volume degrades to the parent's, keeping
            // the tree traversable; the root has no parent to inherit.
            parent_volume.ok_or_else(|| {
                "root bounding volume must be box or sphere (region placement lands with T4)"
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
    });

    let mut children = Vec::with_capacity(tile.children.len());
    for child in &tile.children {
        children.push(build_node(tree, child, Some(idx), depth + 1, refine, world, Some(volume))?);
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

/// Load-priority tiers, highest first. `Ord`: `Urgent < Normal < Preload`
/// so an ascending sort puts the most important requests first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Camera inside the tile's bounding volume.
    Urgent,
    /// In the current render cut.
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
    };
    if n == 0 {
        return sel;
    }
    let ctx = Ctx { tree, content, history, culled, params };
    visit(&ctx, 0, &mut sel);

    // Preload tier: ancestors of the rendered cut that have unloaded content.
    // REPLACE refinement means a loaded ancestor gives instant zoom-out (and a
    // kick target) — basemap's "ancestors trickle in last" tier, generalized.
    let mut queued = vec![false; n];
    for req in &sel.loads {
        queued[req.tile] = true;
    }
    for i in 0..sel.render.len() {
        let mut at = tree.nodes[sel.render[i]].parent;
        while let Some(p) = at {
            if ctx.content[p].loadable() && !queued[p] {
                queued[p] = true;
                sel.touched[p] = true;
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

    let dist = node.volume.distance_to(ctx.params.cam_pos);
    let sse = screen_space_error(node.geometric_error, dist, ctx.params.k_px);

    let mut wants_refine = !node.children.is_empty() && sse > ctx.params.sse_threshold_px;
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
    if any_last {
        // Part of the refined cut was already on screen last frame — keep it
        // (removing it to kick would regress detail); the missing pieces stay
        // queued and pop in as they land.
        return VisitOut { covered: true, any_rendered_last: any_last };
    }
    // Kick (§7.5): none of the selected descendants have ever shown — drop
    // them from the render list (their loads stay queued) and paint this
    // tile's own content instead until they arrive.
    sel.render.truncate(checkpoint);
    if ctx.content[i] == TileContent::Ready {
        sel.render.push(i);
        return VisitOut { covered: true, any_rendered_last: ctx.history.rendered[i] };
    }
    push_load(ctx, sel, i, dist);
    // Not renderable itself either — report uncovered so OUR ancestor kicks.
    VisitOut {
        covered: ctx.content[i] == TileContent::None && node.content_uri.is_none(),
        any_rendered_last: false,
    }
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
    fn partial_refinement_kept_when_some_rendered_last_frame() {
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
        // The two ready leaves stay on screen (no kick of child 1)…
        assert!(sel.render.contains(&kids[0]) && sel.render.contains(&kids[1]));
        assert!(!sel.render.contains(&1));
        // …while the pending two remain queued.
        let queued: Vec<usize> = sel.loads.iter().map(|r| r.tile).collect();
        assert!(queued.contains(&kids[2]) && queued.contains(&kids[3]));
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
            });
        }
        let content =
            vec![TileContent::None, TileContent::Ready, TileContent::Ready];
        let history = History { rendered: vec![false; 3], refined: vec![false; 3] };
        let sel = select(&tree, &content, &history, &no_cull, params(DVec3::new(0.0, 1.0, 0.0)));
        assert_eq!(sel.render, vec![1, 2]);
        assert!(sel.loads.is_empty());
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
        let tree = TileTree::build(&ts, ZUP_TO_BEVY).unwrap();
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
    fn region_root_is_rejected_until_t4() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 1,
            "root": {
                "boundingVolume": { "region": [-1.32, 0.69, -1.31, 0.70, 0, 90] },
                "geometricError": 0
            }
        }"#;
        let ts = schema::parse_tileset(json.as_bytes()).unwrap();
        assert!(TileTree::build(&ts, ZUP_TO_BEVY).is_err());
    }
}
