//! Minimal WGS84 geodesy — the only math `bevy_3d_tiles` needs to georeference
//! ECEF/region tilesets. Pure textbook formulas (no external crate, no STDB
//! dependency); inlined from TurboTwin's `turbotwin_sdk_rs::enu` so the crate is
//! self-contained. The full ENU origin projection (and the `world_from_ecef`
//! transform) lives in the host app, which feeds the result in via
//! [`crate::EcefOrigin`]; here we only need geodetic→ECEF + the radius constant
//! used in horizon culling.

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

#[cfg(test)]
mod tests {
    use super::*;

    const WGS84_POLAR_RADIUS_M: f64 = WGS84_EQUATORIAL_RADIUS_M * (1.0 - WGS84_FLATTENING);

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
