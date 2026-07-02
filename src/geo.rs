//! Georeferenced-tileset geometry (BEVY-3D-TILES T4).
//!
//! External tilesets (Google P3DT, national open data) arrive in EPSG:4978
//! (ECEF metres) — `region` bounding volumes in EPSG:4979 geodetic radians,
//! box/sphere volumes under ECEF tile transforms. Trees for those tilesets
//! are built **in the ECEF frame** ([`super::traversal::TreeFrame::Ecef`]):
//! volumes stay planetary-magnitude f64, and the per-frame placement into the
//! host's world frame happens in `drive_tiles3d` against the host-supplied
//! [`crate::EcefOrigin`] transform — composed in f64, cast to f32 only after
//! the magnitudes cancel (the altitude-anchored-origin jitter lesson).

use bevy::math::DVec3;

use super::geodesy::{WGS84_EQUATORIAL_RADIUS_M, geodetic_to_ecef};

use super::schema::{self, VolumeKind};
use super::traversal::WorldVolume;

/// Regions spanning more than this many radians of latitude or longitude get
/// the conservative whole-globe sphere instead of a sampled OBB (the grid
/// sampling under-covers strongly curved patches). ~28°.
const REGION_OBB_MAX_SPAN_RAD: f64 = 0.5;

/// ENU basis unit vectors (east, north, up) in ECEF at geodetic
/// `(lat, lon)` **radians**.
fn enu_basis_rad(lat: f64, lon: f64) -> (DVec3, DVec3, DVec3) {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    (east, north, up)
}

/// 3D Tiles `region` volume (`[west, south, east, north (rad), minH, maxH]`,
/// EPSG:4979 — NOT affected by tile transforms) → an ECEF [`WorldVolume`].
///
/// Small/medium regions: an OBB in the ENU frame at the region's centre,
/// sized by projecting a 3×3 geodetic grid at both height bounds (the grid
/// captures the ellipsoidal bulge of the patch interior that corners alone
/// miss). Oversized or degenerate (antimeridian-crossing) regions degrade to
/// the conservative globe-bounding sphere — never culled, always refined
/// through, exactly what a planet-spanning root wants.
pub fn region_to_ecef_volume(region: &[f64; 6]) -> WorldVolume {
    let [west, south, east, north, min_h, max_h] = *region;
    let span_lon = east - west;
    let span_lat = north - south;
    let spans_usable = span_lon.is_finite()
        && span_lat.is_finite()
        && span_lon > 0.0 // antimeridian wrap arrives as west > east
        && span_lat > 0.0
        && span_lon <= REGION_OBB_MAX_SPAN_RAD
        && span_lat <= REGION_OBB_MAX_SPAN_RAD;
    if !spans_usable {
        return WorldVolume::Sphere {
            center: DVec3::ZERO,
            radius: WGS84_EQUATORIAL_RADIUS_M + max_h.max(0.0),
        };
    }

    let (clat, clon) = ((south + north) * 0.5, (west + east) * 0.5);
    let (e, n, u) = enu_basis_rad(clat, clon);
    let (cx, cy, cz) = geodetic_to_ecef(clat.to_degrees(), clon.to_degrees(), 0.0);
    let center0 = DVec3::new(cx, cy, cz);

    // Project a 3×3 × {minH, maxH} grid into the centre ENU frame; AABB it.
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for i in 0..3 {
        for j in 0..3 {
            let lat = south + span_lat * (i as f64) / 2.0;
            let lon = west + span_lon * (j as f64) / 2.0;
            for h in [min_h, max_h] {
                let (x, y, z) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
                let d = DVec3::new(x, y, z) - center0;
                let p = DVec3::new(d.dot(e), d.dot(n), d.dot(u));
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    let half = (hi - lo) * 0.5;
    let mid = (hi + lo) * 0.5;
    WorldVolume::Obb {
        center: center0 + e * mid.x + n * mid.y + u * mid.z,
        half_axes: [e * half.x, n * half.y, u * half.z],
    }
}

/// Whether a parsed tileset is georeferenced (ECEF frame) rather than one of
/// our local-metres products: a `region` root volume is definitionally
/// EPSG:4979, and a box/sphere root whose (root-transformed) centre sits at
/// planetary magnitude can only be ECEF — local-metres sites are km-scale.
pub fn tileset_is_georeferenced(ts: &schema::Tileset) -> bool {
    const PLANETARY_M: f64 = 2_000_000.0;
    let root_transform = ts
        .root
        .transform
        .map(|t| bevy::math::DMat4::from_cols_array(&t))
        .unwrap_or(bevy::math::DMat4::IDENTITY);
    match ts.root.bounding_volume.kind() {
        Some(VolumeKind::Region(_)) => true,
        Some(VolumeKind::Sphere([cx, cy, cz, _])) => {
            root_transform
                .transform_point3(DVec3::new(cx, cy, cz))
                .length()
                > PLANETARY_M
        }
        Some(VolumeKind::Box(b)) => {
            root_transform
                .transform_point3(DVec3::new(b[0], b[1], b[2]))
                .length()
                > PLANETARY_M
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eugene, OR-ish region (the autzen neighbourhood): ~0.01 rad across.
    const SMALL_REGION: [f64; 6] = [-2.1496, 0.7686, -2.1478, 0.7696, 0.0, 120.0];

    #[test]
    fn small_region_obb_contains_its_corners_and_centre() {
        let vol = region_to_ecef_volume(&SMALL_REGION);
        let WorldVolume::Obb { .. } = vol else {
            panic!("expected OBB")
        };
        let [west, south, east, north, min_h, max_h] = SMALL_REGION;
        for (lat, lon) in [
            (south, west),
            (south, east),
            (north, west),
            (north, east),
            ((south + north) / 2.0, (west + east) / 2.0),
        ] {
            for h in [min_h, max_h, (min_h + max_h) / 2.0] {
                let (x, y, z) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
                let d = vol.distance_to(DVec3::new(x, y, z));
                assert!(d < 1.0, "({lat}, {lon}, {h}) outside by {d} m");
            }
        }
    }

    #[test]
    fn small_region_obb_is_tight() {
        // A km-scale patch: the enclosing sphere radius must be the same
        // order, not planetary.
        let vol = region_to_ecef_volume(&SMALL_REGION);
        let (_, r) = vol.bounding_sphere();
        assert!(r < 10_000.0, "radius {r} m not tight");
        // Distance from far away is sane: a point 1000 km above the centre.
        let (clat, clon) = (
            (SMALL_REGION[1] + SMALL_REGION[3]) / 2.0,
            (SMALL_REGION[0] + SMALL_REGION[2]) / 2.0,
        );
        let (x, y, z) = geodetic_to_ecef(clat.to_degrees(), clon.to_degrees(), 1_000_000.0);
        let d = vol.distance_to(DVec3::new(x, y, z));
        assert!((d - 1_000_000.0).abs() < 10_000.0, "d = {d}");
    }

    #[test]
    fn huge_region_degrades_to_globe_sphere() {
        // The P3DT root: the whole earth.
        let vol = region_to_ecef_volume(&[
            -std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            0.0,
            9000.0,
        ]);
        let WorldVolume::Sphere { center, radius } = vol else {
            panic!("expected sphere")
        };
        assert_eq!(center, DVec3::ZERO);
        assert!(radius >= WGS84_EQUATORIAL_RADIUS_M);
        // A camera on the surface is INSIDE → distance 0 → always refines.
        let (x, y, z) = geodetic_to_ecef(45.0, 10.0, 500.0);
        assert_eq!(vol.distance_to(DVec3::new(x, y, z)), 0.0);
    }

    #[test]
    fn georeference_detection() {
        let geo = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 1e7,
            "root": {
                "boundingVolume": { "region": [-3.14, -1.57, 3.14, 1.57, 0, 9000] },
                "geometricError": 1e6
            }
        }"#;
        assert!(tileset_is_georeferenced(
            &schema::parse_tileset(geo.as_bytes()).unwrap()
        ));

        // ECEF box without a region (PDOK-style: root transform carries the
        // planetary translation).
        let ecef_box = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 1000,
            "root": {
                "boundingVolume": { "box": [0,0,0, 500,0,0, 0,500,0, 0,0,50] },
                "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 3890000.0, 333000.0, 5030000.0, 1],
                "geometricError": 500
            }
        }"#;
        assert!(tileset_is_georeferenced(
            &schema::parse_tileset(ecef_box.as_bytes()).unwrap()
        ));

        // Our local-metres tilers: small box at the origin.
        let local = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 64,
            "root": {
                "boundingVolume": { "box": [0,0,0, 50,0,0, 0,50,0, 0,0,10] },
                "geometricError": 16
            }
        }"#;
        assert!(!tileset_is_georeferenced(
            &schema::parse_tileset(local.as_bytes()).unwrap()
        ));
    }
}
