//! CPU pick copy of a tile mesh primitive, and per-feature hiding (0.5).
//!
//! Tile meshes are `RENDER_WORLD`-only by default since 0.5: after their first
//! upload their vertex data is no longer in `Assets<Mesh>`. [`TilePickMesh`] is
//! the compact CPU copy that replaces it for raycasts: positions, the PRISTINE
//! index buffer, and which `EXT_mesh_features` features are hidden.
//!
//! [`HiddenTileFeatures`] hides features by resolved owner id. A hidden
//! feature's triangles become degenerate IN PLACE on the GPU (`[a, b, c]` →
//! `[a, a, a]`): one index-only buffer write per changed primitive, no mesh
//! rebuild, no re-extract, no vertex re-upload, and every pass that draws the
//! mesh (any material, shadows, prepass, outlines) sees the same cut. Triangle
//! ordinals never move, so [`crate::TileFeaturePick`] lookups stay valid; the
//! pick copy skips hidden triangles by ordinal instead of holding a cut copy.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError, RwLock};

use bevy::prelude::*;

/// CPU geometry of one tile mesh primitive: positions, the PRISTINE indices and
/// which features are hidden. Triangle ordinals never move.
///
/// The crate inserts it on every tile mesh entity. It is shared by `Arc`
/// between the tile cache and every entity spawned from it, so cloning it is a
/// refcount bump, and a [`HiddenTileFeatures`] change is visible through every
/// clone at once. Read geometry here, not from `Assets<Mesh>`: the mesh asset
/// is `RENDER_WORLD`-only unless the set was attached with
/// [`crate::Tiles3dAttach::main_world_meshes`], and even then its indices stay
/// pristine while hidden features are cut on the GPU only.
#[derive(Component, Clone)]
pub struct TilePickMesh(pub(crate) Arc<PickMesh>);

pub(crate) struct PickMesh {
    positions: Box<[[f32; 3]]>,
    /// Pristine. `None` only for a non-indexed primitive without features
    /// (feature primitives are always indexed since 0.5).
    indices: Option<Arc<[u32]>>,
    /// Triangle ordinal → local feature id: the same `Arc` as the entity's
    /// [`crate::TileFeaturePick::feature_of_triangle`].
    feature_of_triangle: Option<Arc<[u32]>>,
    hidden: RwLock<Hidden>,
}

/// One bit per local feature id (empty = nothing hidden), and the triangle
/// count that leaves visible.
struct Hidden {
    features: Box<[u64]>,
    visible_tris: usize,
}

fn bit(mask: &[u64], fid: u32) -> bool {
    mask.get(fid as usize / 64)
        .is_some_and(|w| w >> (fid % 64) & 1 == 1)
}

impl TilePickMesh {
    pub(crate) fn new(
        positions: Vec<[f32; 3]>,
        indices: Option<Arc<[u32]>>,
        feature_of_triangle: Option<Arc<[u32]>>,
    ) -> Self {
        let tris = match &indices {
            Some(ix) => ix.len() / 3,
            None => positions.len() / 3,
        };
        Self(Arc::new(PickMesh {
            positions: positions.into(),
            indices,
            feature_of_triangle,
            hidden: RwLock::new(Hidden {
                features: Box::default(),
                visible_tris: tris,
            }),
        }))
    }

    /// A standalone pick copy, for geometry the streamer did not build (tests,
    /// a host's own tile-like entities): `hidden` lists the local feature ids
    /// to treat as hidden. Independent of every crate-built copy; the crate
    /// keeps its own copies' masks in step with [`HiddenTileFeatures`].
    pub fn from_parts(
        positions: Vec<[f32; 3]>,
        indices: Option<Vec<u32>>,
        feature_of_triangle: Option<Arc<[u32]>>,
        hidden: &[u32],
    ) -> Self {
        let pick = Self::new(positions, indices.map(Into::into), feature_of_triangle);
        if let Some(&max) = hidden.iter().max() {
            let mut mask = vec![0u64; max as usize / 64 + 1];
            for &f in hidden {
                mask[f as usize / 64] |= 1 << (f % 64);
            }
            pick.set_hidden(mask.into());
        }
        pick
    }

    /// Mesh-local vertex positions.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.0.positions
    }

    /// Triangles still VISIBLE (hidden features excluded). Cached: updated when
    /// the hidden mask changes, O(1) to read.
    pub fn triangle_count(&self) -> usize {
        self.read().visible_tris
    }

    /// Every triangle of the primitive, hidden ones included.
    pub fn pristine_triangle_count(&self) -> usize {
        match &self.0.indices {
            Some(ix) => ix.len() / 3,
            None => self.0.positions.len() / 3,
        }
    }

    /// Calls `f(pristine_ordinal, [a, b, c])` for every visible triangle, in
    /// index order (sequential triples for a non-indexed primitive). The
    /// ordinal indexes [`crate::TileFeaturePick::feature_of_triangle`].
    /// Holds the mask's read lock for the walk; don't call back into the crate.
    pub fn for_each_visible_triangle(&self, mut f: impl FnMut(usize, [u32; 3])) {
        let hidden = self.read();
        let mask = &hidden.features;
        let p = &self.0;
        for t in 0..self.pristine_triangle_count() {
            if !mask.is_empty()
                && let Some(fot) = &p.feature_of_triangle
                && fot.get(t).is_some_and(|&fid| bit(mask, fid))
            {
                continue;
            }
            let tri = match &p.indices {
                Some(ix) => [ix[t * 3], ix[t * 3 + 1], ix[t * 3 + 2]],
                None => {
                    let v = (t * 3) as u32;
                    [v, v + 1, v + 2]
                }
            };
            f(t, tri);
        }
    }

    /// Nearest visible hit of a ray in mesh-local space: `(t, pristine
    /// ordinal)`, `t` in units of `local_dir`. Möller–Trumbore over every
    /// visible triangle (no BVH); out-of-range indices are skipped.
    pub fn raycast(&self, local_origin: Vec3, local_dir: Vec3) -> Option<(f32, usize)> {
        let pos = self.positions();
        let mut best: Option<(f32, usize)> = None;
        self.for_each_visible_triangle(|t, [a, b, c]| {
            let (Some(a), Some(b), Some(c)) = (
                pos.get(a as usize),
                pos.get(b as usize),
                pos.get(c as usize),
            ) else {
                return;
            };
            if let Some(d) = moller_trumbore(
                local_origin,
                local_dir,
                (*a).into(),
                (*b).into(),
                (*c).into(),
            ) && best.is_none_or(|(bd, _)| d < bd)
            {
                best = Some((d, t));
            }
        });
        best
    }

    /// Bytes of this copy: positions + pristine indices (the feature table is
    /// shared with [`crate::TileFeaturePick`] and not counted here).
    pub(crate) fn cpu_bytes(&self) -> u64 {
        (self.0.positions.len() * 12 + self.0.indices.as_ref().map_or(0, |i| i.len() * 4)) as u64
    }

    pub(crate) fn pristine_indices(&self) -> Option<&Arc<[u32]>> {
        self.0.indices.as_ref()
    }

    /// Store `mask` (one bit per local feature id, normalized: empty = nothing
    /// hidden). Returns the index buffer the GPU must now hold — the pristine
    /// `Arc` itself when nothing is hidden, else a transient cut — or `None`
    /// when the mask did not change or there is no index buffer to rewrite.
    pub(crate) fn set_hidden(&self, mask: Box<[u64]>) -> Option<Arc<[u32]>> {
        let p = &self.0;
        if self.read().features == mask {
            return None;
        }
        let pristine = self.pristine_triangle_count();
        let hidden_tris = match (&p.feature_of_triangle, mask.is_empty()) {
            (Some(fot), false) => fot
                .iter()
                .take(pristine)
                .filter(|&&fid| bit(&mask, fid))
                .count(),
            _ => 0,
        };
        let gpu = p.indices.as_ref().map(|ix| match hidden_tris {
            0 => Arc::clone(ix),
            _ => cut(ix, p.feature_of_triangle.as_deref().unwrap_or(&[]), &mask),
        });
        let mut h = p.hidden.write().unwrap_or_else(PoisonError::into_inner);
        h.features = mask;
        h.visible_tris = pristine - hidden_tris;
        gpu
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Hidden> {
        self.0.hidden.read().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The GPU index buffer under `mask`: every triangle whose feature is hidden
/// collapses IN PLACE to `[a, a, a]` (zero area, so it rasterizes nothing and
/// keeps the vertex-cache locality of its neighbours). Ordinals never move.
fn cut(indices: &[u32], feature_of_triangle: &[u32], mask: &[u64]) -> Arc<[u32]> {
    let mut out = indices.to_vec();
    for (t, tri) in out.chunks_exact_mut(3).enumerate() {
        if feature_of_triangle
            .get(t)
            .is_some_and(|&fid| bit(mask, fid))
        {
            tri[1] = tri[0];
            tri[2] = tri[0];
        }
    }
    out.into()
}

/// Möller–Trumbore; `None` for a miss, a hit behind the origin, or a
/// degenerate (zero-area) triangle.
fn moller_trumbore(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    const EPSILON: f32 = 1e-7;
    let (e1, e2) = (v1 - v0, v2 - v0);
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < EPSILON {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    (t > EPSILON).then_some(t)
}

/// Resolved owner ids ([`crate::TileFeatureResolver`] output, the strings in
/// [`crate::TileFeaturePick::owner_of_feature`]) whose `EXT_mesh_features`
/// features are hidden, in every tileset. Replace-set: write the whole set; on
/// a change the crate re-cuts the affected resident tiles the same frame, and a
/// tile streaming in later arrives cut.
///
/// Only feature primitives are covered (the ones with a `TileFeaturePick`).
/// Hiding a whole featureless tile is still the host's (remove its `Mesh3d` or
/// hide its entity).
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct HiddenTileFeatures(pub HashSet<String>);

/// Owner ids interned to ordinals, plus which ordinals are hidden, so a change
/// costs one bit test per resident feature instead of a string-set probe.
///
// ponytail: never shrinks — bounded by the distinct owners streamed this
// session (tens of thousands of short strings on a large site). Rebuild from
// the resident caches if a long multi-site session ever shows it.
#[derive(Default)]
pub(crate) struct FeatureOwners {
    ix: HashMap<String, u32>,
    names: Vec<String>,
    hidden: Vec<bool>,
    /// Owners `< synced` have a current `hidden` bit; later ones were interned
    /// since the last [`FeatureOwners::sync`].
    synced: usize,
}

impl FeatureOwners {
    pub(crate) fn intern(&mut self, owner: &str) -> u32 {
        if let Some(&i) = self.ix.get(owner) {
            return i;
        }
        let i = self.names.len() as u32;
        self.ix.insert(owner.to_owned(), i);
        self.names.push(owner.to_owned());
        self.hidden.push(false);
        i
    }

    /// Bring the hidden bits up to `set`: all of them when it changed, else
    /// only the owners interned since the last call.
    pub(crate) fn sync(&mut self, set: &HashSet<String>, changed: bool) {
        let from = if changed { 0 } else { self.synced };
        for i in from..self.names.len() {
            self.hidden[i] = set.contains(&self.names[i]);
        }
        self.synced = self.names.len();
    }

    /// The per-feature mask of a primitive whose features have these owners.
    /// Normalized: empty when nothing is hidden.
    pub(crate) fn mask(&self, owner_ix: &[u32]) -> Box<[u64]> {
        let hidden = |o: u32| self.hidden.get(o as usize).copied().unwrap_or(false);
        if !owner_ix.iter().any(|&o| hidden(o)) {
            return Box::default();
        }
        let mut m = vec![0u64; owner_ix.len().div_ceil(64)];
        for (f, &o) in owner_ix.iter().enumerate() {
            if hidden(o) {
                m[f / 64] |= 1 << (f % 64);
            }
        }
        m.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four triangles, features `[0, 1, 2, 1]`, one quad strip of vertices.
    fn four_tris() -> TilePickMesh {
        let positions = (0..12).map(|i| [i as f32, (i % 3) as f32, 0.0]).collect();
        TilePickMesh::new(
            positions,
            Some((0..12).collect::<Vec<u32>>().into()),
            Some(vec![0, 1, 2, 1].into()),
        )
    }

    fn hide(fids: &[u32], features: usize) -> Box<[u64]> {
        let mut m = vec![0u64; features.div_ceil(64)];
        for &f in fids {
            m[f as usize / 64] |= 1 << (f % 64);
        }
        m.into()
    }

    #[test]
    fn hidden_triangles_collapse_in_place_and_keep_ordinals() {
        let pick = four_tris();
        let gpu = pick
            .set_hidden(hide(&[1], 3))
            .expect("a changed mask writes");
        assert_eq!(
            &*gpu,
            &[0, 1, 2, 3, 3, 3, 6, 7, 8, 9, 9, 9],
            "feature 1's triangles degenerate where they stand"
        );
        assert_eq!(gpu.len(), 12, "no triangle moves, so no ordinal moves");
        let mut seen = Vec::new();
        pick.for_each_visible_triangle(|t, tri| seen.push((t, tri)));
        assert_eq!(
            seen,
            vec![(0, [0, 1, 2]), (2, [6, 7, 8])],
            "pristine ordinals"
        );
        assert_eq!(
            pick.pristine_indices().map(|i| i.to_vec()),
            Some((0..12).collect()),
            "the pick copy stays pristine"
        );
    }

    #[test]
    fn unknown_feature_is_never_hidden() {
        let pick = TilePickMesh::new(
            vec![[0.0; 3]; 6],
            Some(vec![0, 1, 2, 3, 4, 5].into()),
            Some(vec![0, 7].into()),
        );
        // A mask that only knows features 0..2 says nothing about feature 7.
        pick.set_hidden(hide(&[1], 2));
        assert_eq!(pick.triangle_count(), 2);
        // A short feature table leaves the trailing triangle with no feature.
        let pick = TilePickMesh::new(
            vec![[0.0; 3]; 6],
            Some(vec![0, 1, 2, 3, 4, 5].into()),
            Some(vec![0].into()),
        );
        let gpu = pick.set_hidden(hide(&[0], 1)).unwrap();
        assert_eq!(&*gpu, &[0, 0, 0, 3, 4, 5], "the untabled triangle stays");
        assert_eq!(pick.triangle_count(), 1);
    }

    #[test]
    fn all_visible_mask_queues_the_pristine_arc() {
        let pick = four_tris();
        pick.set_hidden(hide(&[2], 3));
        let gpu = pick.set_hidden(Box::default()).expect("un-hide writes");
        assert!(
            Arc::ptr_eq(&gpu, pick.pristine_indices().unwrap()),
            "no copy"
        );
        // A mask that hides no triangle is all-visible too.
        let pick = four_tris();
        let gpu = pick.set_hidden(hide(&[40], 41)).unwrap();
        assert!(Arc::ptr_eq(&gpu, pick.pristine_indices().unwrap()));
    }

    #[test]
    fn triangle_count_drops_by_exactly_the_hidden_features_triangles() {
        let pick = four_tris();
        assert_eq!(
            (pick.triangle_count(), pick.pristine_triangle_count()),
            (4, 4)
        );
        pick.set_hidden(hide(&[1], 3));
        assert_eq!(pick.triangle_count(), 2, "feature 1 owns two triangles");
        pick.set_hidden(hide(&[0, 1, 2], 3));
        assert_eq!(pick.triangle_count(), 0);
        pick.set_hidden(Box::default());
        assert_eq!(pick.triangle_count(), 4, "un-hide returns them all");
        assert_eq!(pick.pristine_triangle_count(), 4);
    }

    #[test]
    fn unchanged_mask_writes_nothing() {
        let pick = four_tris();
        assert!(
            pick.set_hidden(Box::default()).is_none(),
            "already all visible"
        );
        assert!(pick.set_hidden(hide(&[1], 3)).is_some());
        assert!(pick.set_hidden(hide(&[1], 3)).is_none());
    }

    #[test]
    fn raycast_skips_hidden_triangles() {
        // Two stacked unit triangles facing +Z: the near one (z = 1) is
        // feature 0, the far one (z = 0) feature 1.
        let pick = TilePickMesh::new(
            vec![
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            Some(vec![0, 1, 2, 3, 4, 5].into()),
            Some(vec![0, 1].into()),
        );
        let (o, d) = (Vec3::new(0.2, 0.2, 5.0), Vec3::NEG_Z);
        assert_eq!(pick.raycast(o, d).map(|(_, t)| t), Some(0), "near first");
        pick.set_hidden(hide(&[0], 2));
        let (dist, tri) = pick.raycast(o, d).expect("passes through to the far one");
        assert_eq!(tri, 1, "resolved by its pristine ordinal");
        assert!((dist - 5.0).abs() < 1e-5);
        pick.set_hidden(hide(&[0, 1], 2));
        assert_eq!(pick.raycast(o, d), None);
    }

    #[test]
    fn non_indexed_featureless_primitive_walks_sequential_triples() {
        let pick = TilePickMesh::new(vec![[0.0; 3]; 7], None, None);
        let mut seen = Vec::new();
        pick.for_each_visible_triangle(|t, tri| seen.push((t, tri)));
        assert_eq!(
            seen,
            vec![(0, [0, 1, 2]), (1, [3, 4, 5])],
            "partial tail dropped"
        );
        assert!(
            pick.set_hidden(hide(&[0], 1)).is_none(),
            "nothing to rewrite"
        );
    }

    #[test]
    fn from_parts_hides_the_listed_features() {
        let pick = TilePickMesh::from_parts(
            vec![[0.0; 3]; 9],
            Some((0..9).collect()),
            Some(vec![0, 70, 0].into()),
            &[70],
        );
        assert_eq!(
            (pick.triangle_count(), pick.pristine_triangle_count()),
            (2, 3)
        );
        let mut seen = Vec::new();
        pick.for_each_visible_triangle(|t, _| seen.push(t));
        assert_eq!(seen, vec![0, 2]);
    }

    #[test]
    fn owners_intern_once_and_sync_new_ones() {
        let mut owners = FeatureOwners::default();
        let (a, b) = (owners.intern("a"), owners.intern("b"));
        assert_eq!(owners.intern("a"), a);
        let set: HashSet<String> = ["b".to_string(), "c".to_string()].into();
        owners.sync(&set, true);
        assert_eq!(&*owners.mask(&[a, b, a]), &[0b010]);
        // Interned after the change: picked up by the next unchanged sync.
        let c = owners.intern("c");
        owners.sync(&set, false);
        assert_eq!(&*owners.mask(&[c]), &[1]);
        assert!(
            owners.mask(&[a]).is_empty(),
            "nothing hidden normalizes to empty"
        );
    }
}
