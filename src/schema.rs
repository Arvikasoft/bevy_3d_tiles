//! Serde model for 3D Tiles 1.1 `tileset.json` (BEVY-3D-TILES-PLAN T0).
//!
//! Deliberately minimal: the fields the runtime traversal needs (D1 — we emit
//! and consume explicit tilesets with glTF content only). Unknown fields are
//! ignored so externally-produced tilesets (extensions, metadata, implicit
//! tiling hints) still parse; anything we can't *render* is handled at tree
//! build, not at parse.
//!
//! Spec: 3D Tiles 1.1 (OGC 22-025r4). Key shapes:
//! * `boundingVolume` — exactly one of `box` (12 numbers: center + 3
//!   half-axes), `sphere` (4: center + radius), `region` (6: west/south/east/
//!   north radians + min/max height, EPSG:4979).
//! * `transform` — column-major 4×4, applies to the tile's own
//!   `boundingVolume` and `content`, composes down the tree.
//! * `refine` — `"REPLACE"` | `"ADD"`, inherited from the parent when absent
//!   (required on the root; we default a missing root refine to REPLACE).
//! * `content.uri` — relative to the tileset root (legacy `url` accepted).

use serde::Deserialize;

/// Top-level `tileset.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tileset {
    pub asset: Asset,
    /// Error (metres) when the whole tileset is not rendered at all.
    pub geometric_error: f64,
    pub root: Tile,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub version: String,
}

/// One tile in the explicit tree.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tile {
    pub bounding_volume: BoundingVolume,
    /// Error (metres) when this tile is rendered INSTEAD of its children.
    pub geometric_error: f64,
    #[serde(default)]
    pub refine: Option<Refine>,
    /// Column-major 4×4 affine transform (16 numbers), identity when absent.
    #[serde(default)]
    pub transform: Option<[f64; 16]>,
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(default)]
    pub children: Vec<Tile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Refine {
    #[serde(rename = "REPLACE")]
    Replace,
    #[serde(rename = "ADD")]
    Add,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    /// 3D Tiles 1.1 field name. Pre-1.0-final tilesets used `url`; accept both
    /// (alias) since external data in the wild still carries it.
    #[serde(alias = "url")]
    pub uri: String,
}

/// Exactly-one-of in the spec; modelled as three options + [`BoundingVolume::kind`]
/// so a malformed multi-volume tile degrades instead of failing the parse.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingVolume {
    /// `[cx,cy,cz, x-half-axis(3), y-half-axis(3), z-half-axis(3)]`.
    #[serde(rename = "box", default)]
    pub obb: Option<[f64; 12]>,
    /// `[cx, cy, cz, radius]`.
    #[serde(default)]
    pub sphere: Option<[f64; 4]>,
    /// `[west, south, east, north (radians), minHeight, maxHeight (m)]`.
    #[serde(default)]
    pub region: Option<[f64; 6]>,
}

/// The resolved volume variant, in the spec's priority order (box, sphere,
/// region — the order is arbitrary since the spec demands exactly one).
#[derive(Debug, Clone, Copy)]
pub enum VolumeKind {
    Box([f64; 12]),
    Sphere([f64; 4]),
    Region([f64; 6]),
}

impl BoundingVolume {
    pub fn kind(&self) -> Option<VolumeKind> {
        if let Some(b) = self.obb {
            Some(VolumeKind::Box(b))
        } else if let Some(s) = self.sphere {
            Some(VolumeKind::Sphere(s))
        } else {
            self.region.map(VolumeKind::Region)
        }
    }
}

/// Parse a `tileset.json` byte buffer.
pub fn parse_tileset(bytes: &[u8]) -> Result<Tileset, serde_json::Error> {
    serde_json::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "asset": { "version": "1.1" },
        "geometricError": 64,
        "root": {
            "boundingVolume": { "sphere": [0, 0, 0, 30] },
            "geometricError": 16,
            "refine": "REPLACE",
            "content": { "uri": "content/root.glb" },
            "children": [
                {
                    "boundingVolume": {
                        "box": [10, 10, 0, 10, 0, 0, 0, 10, 0, 0, 0, 2]
                    },
                    "geometricError": 4,
                    "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,3,1],
                    "content": { "uri": "content/ne.glb" },
                    "children": []
                },
                {
                    "boundingVolume": {
                        "region": [-1.32, 0.69, -1.31, 0.70, 0, 90]
                    },
                    "geometricError": 0,
                    "content": { "url": "content/legacy.glb" }
                }
            ]
        }
    }"#;

    #[test]
    fn parses_full_shape() {
        let ts = parse_tileset(FIXTURE.as_bytes()).expect("parse");
        assert_eq!(ts.asset.version, "1.1");
        assert_eq!(ts.geometric_error, 64.0);
        assert_eq!(ts.root.refine, Some(Refine::Replace));
        assert!(
            matches!(ts.root.bounding_volume.kind(), Some(VolumeKind::Sphere(s)) if s[3] == 30.0)
        );
        assert_eq!(ts.root.content.as_ref().unwrap().uri, "content/root.glb");
        assert_eq!(ts.root.children.len(), 2);

        let ne = &ts.root.children[0];
        assert!(matches!(
            ne.bounding_volume.kind(),
            Some(VolumeKind::Box(_))
        ));
        let t = ne.transform.expect("transform");
        assert_eq!(t[14], 3.0); // column-major translation z
        // refine absent on the child — inherited at tree build, None here.
        assert!(ne.refine.is_none());
    }

    #[test]
    fn legacy_url_field_aliases_to_uri() {
        let ts = parse_tileset(FIXTURE.as_bytes()).expect("parse");
        let legacy = &ts.root.children[1];
        assert_eq!(legacy.content.as_ref().unwrap().uri, "content/legacy.glb");
        assert!(matches!(
            legacy.bounding_volume.kind(),
            Some(VolumeKind::Region(_))
        ));
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let json = r#"{
            "asset": { "version": "1.1", "tilesetVersion": "x" },
            "geometricError": 1,
            "extensionsUsed": ["3DTILES_metadata"],
            "root": {
                "boundingVolume": { "sphere": [0,0,0,1] },
                "geometricError": 0,
                "metadata": { "class": "tile" }
            }
        }"#;
        let ts = parse_tileset(json.as_bytes()).expect("parse");
        assert!(ts.root.children.is_empty());
        assert!(ts.root.content.is_none());
        assert!(ts.root.refine.is_none());
    }

    #[test]
    fn missing_volume_yields_none_kind() {
        // A tile with an empty boundingVolume object parses but resolves to no
        // volume; tree build degrades it (inherits the parent's volume).
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 1,
            "root": { "boundingVolume": {}, "geometricError": 0 }
        }"#;
        let ts = parse_tileset(json.as_bytes()).expect("parse");
        assert!(ts.root.bounding_volume.kind().is_none());
    }
}
