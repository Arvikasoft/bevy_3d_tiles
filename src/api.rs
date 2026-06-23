//! Public, backend-agnostic API surface for `bevy_3d_tiles`.
//!
//! These four types are the seam that keeps the crate free of any application
//! (or SpacetimeDB) coupling. The host app supplies them; the crate reads them.
//! In the TurboTwin reference app (`bevy-client`) thin adapter systems map these
//! to the twin-specific machinery (`ProjectOrigin`, `TwinMeshGroup`, the section
//! map, the orbit camera) — but a standalone viewer can ignore all of them and
//! still stream local/relative tilesets.

use std::sync::Arc;

use bevy::math::DMat4;
use bevy::prelude::*;

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
pub struct TileFeatureResolver(
    pub Option<Arc<dyn Fn(&str, &[&str]) -> Vec<String> + Send + Sync>>,
);

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
