//! Missing vertex normals, computed where the geometry is prepared instead of
//! on the consumer's frame thread.
//!
//! A hand port of bevy_mesh 0.19's `Mesh::compute_normals` (smooth, corner-angle
//! weighted when indexed; flat when not), together with the glam 0.32 vector
//! math it calls, so the result is bit-identical: f32 throughout, the same
//! operation order, no fused multiply-add. `bevy_3d_tiles` pins it against the
//! real `Mesh::compute_normals` on seeded meshes, so a bevy bump that changes
//! the algorithm fails a test instead of shading differently. A `glam`
//! dependency would need the same pin and add a math crate to a crate that has
//! none, which is the choice `extract.rs` made for its transform math too.

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// glam `Vec3::dot`: `(x·x') + (y·y') + (z·z')`, summed left to right.
fn dot(a: V3, b: V3) -> f32 {
    (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])
}

/// glam `Vec3::cross`.
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - b[1] * a[2],
        a[2] * b[0] - b[2] * a[0],
        a[0] * b[1] - b[0] * a[1],
    ]
}

/// glam `Vec3::try_normalize`: `length().recip()`, accepted only when finite
/// and positive.
fn try_normalize(v: V3) -> Option<V3> {
    let rcp = 1.0 / dot(v, v).sqrt();
    (rcp.is_finite() && rcp > 0.0).then(|| [v[0] * rcp, v[1] * rcp, v[2] * rcp])
}

/// bevy_mesh `triangle_normal`: `(b − a) × (c − a)`, normalized or zero.
fn triangle_normal(a: V3, b: V3, c: V3) -> V3 {
    try_normalize(cross(sub(b, a), sub(c, a))).unwrap_or([0.0; 3])
}

/// glam's `acos_approx_f32` (DirectXMath `XMScalarAcos`), coefficient for
/// coefficient. NOT `f32::acos`: glam's `Vec3::angle_between` uses this
/// polynomial on both its std and libm backends.
#[allow(clippy::excessive_precision)]
fn acos_approx(v: f32) -> f32 {
    let nonnegative = v >= 0.0;
    let x = v.abs();
    let mut omx = 1.0 - x;
    if omx < 0.0 {
        omx = 0.0;
    }
    let root = omx.sqrt();
    #[allow(clippy::approx_constant)]
    let mut result =
        ((((((-0.001_262_491_1 * x + 0.006_670_09) * x - 0.017_088_126) * x + 0.030_891_88) * x
            - 0.050_174_303)
            * x
            + 0.088_978_99)
            * x
            - 0.214_598_8)
            * x
            + 1.570_796_3;
    result *= root;
    if nonnegative {
        result
    } else {
        core::f32::consts::PI - result
    }
}

/// The weight of one corner in bevy's smooth normals: the angle between its two
/// edges, or 0 when the corner is too short to have a stable angle (bevy's
/// `len²(u) · len²(v) > f32::EPSILON` guard).
fn corner_weight(u: V3, v: V3) -> f32 {
    if dot(u, u) * dot(v, v) > f32::EPSILON {
        // glam `angle_between`.
        acos_approx(dot(u, v) / (dot(u, u) * dot(v, v)).sqrt())
    } else {
        0.0
    }
}

/// The normals `Mesh::compute_normals` would give this triangle list:
///
/// * `indices: Some` — smooth: every triangle's unit normal, weighted by each
///   corner's angle, summed per vertex, then normalized (zero where nothing
///   accumulated). A partial index tail is ignored. The result has one normal
///   per position.
/// * `indices: None` — flat: each whole triangle's normal three times. A
///   trailing partial triangle gets none, so 3n+1 or 3n+2 positions give 3n
///   normals, exactly as bevy does.
///
/// A triangle with an out-of-range index is skipped (bevy panics on it; the
/// off-thread extraction declines such content before it gets here).
pub fn compute_normals(positions: &[[f32; 3]], indices: Option<&[u32]>) -> Vec<[f32; 3]> {
    let Some(indices) = indices else {
        return positions
            .chunks_exact(3)
            .flat_map(|t| [triangle_normal(t[0], t[1], t[2]); 3])
            .collect();
    };
    let mut acc = vec![[0.0f32; 3]; positions.len()];
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [t[0] as usize, t[1] as usize, t[2] as usize];
        let (Some(&pa), Some(&pb), Some(&pc)) =
            (positions.get(a), positions.get(b), positions.get(c))
        else {
            continue;
        };
        let (ab, ba, bc) = (sub(pb, pa), sub(pa, pb), sub(pc, pb));
        let (cb, ca, ac) = (sub(pb, pc), sub(pa, pc), sub(pc, pa));
        let weights = [
            corner_weight(ab, ac),
            corner_weight(ba, bc),
            corner_weight(ca, cb),
        ];
        let n = triangle_normal(pa, pb, pc);
        // `normals[v] += normal * weight`, in corner order a, b, c (a repeated
        // index accumulates twice, as in bevy).
        for (v, w) in [a, b, c].into_iter().zip(weights) {
            let s = &mut acc[v];
            *s = [s[0] + n[0] * w, s[1] + n[1] * w, s[2] + n[2] * w];
        }
    }
    acc.into_iter()
        .map(|v| try_normalize(v).unwrap_or([0.0; 3]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat normals keep bevy's length rule: whole triangles only.
    #[test]
    fn flat_normals_cover_whole_triangles_only() {
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for extra in 0..3 {
            let mut p = tri.repeat(2);
            p.extend(std::iter::repeat_n([5.0, 5.0, 5.0], extra));
            let n = compute_normals(&p, None);
            assert_eq!(n.len(), 6, "{} positions", p.len());
            assert!(n.iter().all(|v| *v == [0.0, 0.0, 1.0]));
        }
    }

    /// An out-of-range index skips its triangle instead of panicking.
    #[test]
    fn out_of_range_index_skips_the_triangle() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let n = compute_normals(&p, Some(&[0, 1, 2, 0, 1, 9]));
        assert_eq!(n, compute_normals(&p, Some(&[0, 1, 2])));
        assert!(n.iter().all(|v| v[2] > 0.99), "{n:?}");
    }
}
