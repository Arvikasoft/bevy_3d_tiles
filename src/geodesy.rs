//! Minimal WGS84 geodesy — the only math `bevy_3d_tiles` needs to georeference
//! ECEF/region tilesets. Pure textbook formulas (no external crate, no STDB
//! dependency); inlined from TurboTwin's `turbotwin_sdk_rs::enu` so the crate is
//! self-contained. Besides geodetic→ECEF and the radius constant used in
//! horizon culling, [`world_from_ecef`] builds the ECEF→world matrix a
//! standalone host feeds into [`crate::EcefOrigin`] (a host with its own
//! world frame — like TurboTwin's project origin — computes its own instead).

use bevy::math::{DMat3, DMat4, DVec3, DVec4};

/// WGS84 semi-major axis (equatorial radius) `a`, in metres.
pub const WGS84_EQUATORIAL_RADIUS_M: f64 = 6_378_137.0;

/// WGS84 flattening `f`.
pub const WGS84_FLATTENING: f64 = 1.0 / 298.257_223_563;

/// WGS84 first eccentricity squared, `e² = f·(2 − f)`.
pub const WGS84_ECC_SQ: f64 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING);

/// WGS84 geodetic → ECEF (Earth-Centred, Earth-Fixed) Cartesian.
///
/// `lat`/`lon` in **degrees**, ellipsoidal height `h` in **metres**.
/// Returns `(X, Y, Z)` in metres. Closed form, exact.
pub fn geodetic_to_ecef(lat: f64, lon: f64, h: f64) -> (f64, f64, f64) {
    let (sin_lat, cos_lat) = lat.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon.to_radians().sin_cos();
    // Prime-vertical radius of curvature N.
    let n = WGS84_EQUATORIAL_RADIUS_M / (1.0 - WGS84_ECC_SQ * sin_lat * sin_lat).sqrt();
    let x = (n + h) * cos_lat * cos_lon;
    let y = (n + h) * cos_lat * sin_lon;
    let z = (n * (1.0 - WGS84_ECC_SQ) + h) * sin_lat;
    (x, y, z)
}

/// ECEF → Bevy-world rotation for the ENU frame anchored at `(lat, lon)`
/// degrees: rows are (east, up, −north), i.e. east → +X, up → +Y,
/// north → −Z (the default camera looking down −Z faces north).
fn bevy_frame_basis(lat: f64, lon: f64) -> DMat3 {
    let (sin_lat, cos_lat) = lat.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon.to_radians().sin_cos();
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    DMat3::from_cols(east, up, -north).transpose()
}

/// Rigid ECEF → Bevy-world transform for the ENU frame anchored at
/// `(lat, lon)` degrees / ellipsoidal height `elev` metres:
/// `world = B · (ecef − ecef₀)` with `B` = [`bevy_frame_basis`].
///
/// This is the matrix georeferenced tilesets (ECEF/EPSG:4978 coordinates —
/// including Google Photorealistic 3D Tiles) compose against; hand it to the
/// streamer via [`crate::EcefOrigin`]. Keep it f64: compose with content
/// transforms in f64 and cast the *product* to f32 — planetary magnitudes
/// cancel in f64, not in f32.
pub fn world_from_ecef(lat: f64, lon: f64, elev: f64) -> DMat4 {
    let b = bevy_frame_basis(lat, lon);
    let (x0, y0, z0) = geodetic_to_ecef(lat, lon, elev);
    let t = -(b * DVec3::new(x0, y0, z0));
    DMat4::from_cols(
        DVec4::from((b.x_axis, 0.0)),
        DVec4::from((b.y_axis, 0.0)),
        DVec4::from((b.z_axis, 0.0)),
        DVec4::from((t, 1.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const WGS84_POLAR_RADIUS_M: f64 = WGS84_EQUATORIAL_RADIUS_M * (1.0 - WGS84_FLATTENING);

    #[test]
    fn world_from_ecef_maps_origin_to_zero_and_up_to_y() {
        let (lat, lon, elev) = (44.0584, -123.0684, 111.0);
        let m = world_from_ecef(lat, lon, elev);
        // The anchor itself lands at the world origin.
        let (x0, y0, z0) = geodetic_to_ecef(lat, lon, elev);
        let w = m * DVec4::new(x0, y0, z0, 1.0);
        assert!(w.truncate().length() < 1e-6, "origin -> {w:?}");
        // A point straight above the anchor lands on +Y.
        let (x1, y1, z1) = geodetic_to_ecef(lat, lon, elev + 100.0);
        let up = m * DVec4::new(x1, y1, z1, 1.0);
        assert!((up.y - 100.0).abs() < 1e-6, "up -> {up:?}");
        assert!(up.x.abs() < 1e-6 && up.z.abs() < 1e-6, "up -> {up:?}");
    }

    #[test]
    fn ecef_known_points() {
        // (0,0,0) → (a, 0, 0).
        let (x, y, z) = geodetic_to_ecef(0.0, 0.0, 0.0);
        assert!((x - WGS84_EQUATORIAL_RADIUS_M).abs() < 1e-3, "x = {x}");
        assert!(y.abs() < 1e-3 && z.abs() < 1e-3, "y = {y}, z = {z}");

        // North pole → (0, 0, b).
        let (x, y, z) = geodetic_to_ecef(90.0, 0.0, 0.0);
        assert!(x.abs() < 1e-3 && y.abs() < 1e-3, "x = {x}, y = {y}");
        assert!((z - WGS84_POLAR_RADIUS_M).abs() < 1e-3, "z = {z}");
    }
}
