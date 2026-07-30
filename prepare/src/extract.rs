//! Off-thread geometry extraction — the S5 half of the offthread-decode plan.
//!
//! [`crate::prepare_tile`] stops at "plain glTF bytes": the consumer still
//! pays a `gltf::Gltf::from_slice` parse plus a per-primitive attribute
//! collect on its own thread (measured 2026-07-30 on bevy 0.19: 6-9 ms per
//! tile on the wasm main thread, worst 137 ms). This module moves both. It
//! reads the SAME parsed document `prepare_tile` already holds — no second
//! parse anywhere — and emits plain typed vertex buffers, leaving the
//! consumer only `Mesh::insert_attribute` + the GPU upload, which cannot move
//! (plan §5).
//!
//! It is deliberately narrow. [`extract_tile_meshes`] **declines**
//! (`Ok(None)`) anything it cannot reproduce byte-identically to the
//! consumer's own `gltf`-crate decode — any texture, non-triangle content,
//! integer/quantized attributes, a surviving required extension — and a
//! declined tile takes the S4 route (prepared GLB back, decoded inline) and
//! renders exactly as it did before. Identical geometry through every route,
//! or no route at all.

use serde_json::Value;

use crate::{DecodeError, read_accessor};

/// Node-graph recursion cap. Tile bytes are untrusted network input and a
/// cyclic `children` chain would otherwise recurse until the (Worker) stack
/// dies; a tile this deep declines and decodes inline instead.
const MAX_NODE_DEPTH: usize = 256;

/// Column-major 4×4, the glTF `matrix` layout (and `Mat4::from_cols_array`).
type Mat4 = [f32; 16];

const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// Material inputs of one glTF material, in exactly the set the consumer's
/// inline decode reads (`decode_material`) — factors and flags, no textures.
/// A document with any image declines, so there is nothing else to carry.
///
/// [`Default`] is the **glTF** default material (metallic 1.0 — NOT the
/// consumer's `DecodedMaterial::default()`), because that is what a primitive
/// with no `material` decodes to through the `gltf` crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtractedMaterial {
    /// Linear RGBA `pbrMetallicRoughness.baseColorFactor`.
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
    /// `KHR_materials_unlit` present on this material.
    pub unlit: bool,
}

impl Default for ExtractedMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            double_sided: false,
            unlit: false,
        }
    }
}

/// One TRIANGLES primitive, flattened out of the node graph: plain typed
/// vertex buffers plus the identity a consumer needs to rebuild a mesh and
/// attach feature picking.
///
/// Every buffer is a `Vec` of a POD array type, so an off-thread transport can
/// write them as byte ranges (offset + length) into one transferable and read
/// them back without touching the values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtractedPrimitive {
    /// Node-chain transform flattened to the tile's own (glTF Y-up) frame —
    /// `root × … × node`, composed in f32 in the same order and with the same
    /// operand association the `gltf` crate and `glam` use, so it is
    /// bit-identical to what the inline route computes.
    pub transform: Mat4,
    /// glTF mesh index — with `prim_ix`, the key of the `EXT_mesh_features`
    /// side-band ([`crate::PreparedFeatures::vertex_ids`]).
    pub mesh_ix: u64,
    /// Primitive index within its mesh.
    pub prim_ix: u64,
    /// Index into [`ExtractedMeshes::materials`]; `None` = the glTF default
    /// material ([`ExtractedMaterial::default`]).
    pub material: Option<usize>,
    pub positions: Vec<[f32; 3]>,
    /// `None` = the tile omitted NORMAL; the consumer smooth-computes, exactly
    /// as it does inline.
    pub normals: Option<Vec<[f32; 3]>>,
    /// `TEXCOORD_0`.
    pub uvs: Option<Vec<[f32; 2]>>,
    /// `COLOR_0`, always RGBA.
    pub colors: Option<Vec<[f32; 4]>>,
    /// Widened to u32 (from the accessor's u8/u16/u32) like the `gltf`
    /// crate's `into_u32`; `None` = non-indexed.
    pub indices: Option<Vec<u32>>,
}

/// The geometry of one tile, extracted off-thread: every TRIANGLES primitive
/// of the default scene in traversal order (a node's own primitives before its
/// children's — the inline route's order), plus the document's materials.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtractedMeshes {
    pub primitives: Vec<ExtractedPrimitive>,
    pub materials: Vec<ExtractedMaterial>,
}

/// Extract every renderable primitive of an ALREADY-PARSED, already-prepared
/// tile document (post-meshopt/basisu/RTC rewrite — [`crate::prepare_tile`]'s
/// document, mid-flight) into plain typed buffers.
///
/// * `Ok(Some(_))` — the consumer builds meshes straight from the buffers and
///   never parses the glTF at all.
/// * `Ok(None)` — **declined**: content this cannot reproduce byte-identically
///   (any image/texture, a surviving `extensionsRequired`, non-TRIANGLES
///   primitives, sparse or non-`FLOAT` vertex attributes, a node graph deeper
///   than [`MAX_NODE_DEPTH`]). Not an error — the caller falls back to the
///   glTF-bytes route.
/// * `Err(_)` — the document is malformed (an attribute accessor that does not
///   read). The caller falls back too; the inline decode surfaces it with full
///   diagnostics.
pub fn extract_tile_meshes(
    json: &Value,
    bin: Option<&[u8]>,
) -> Result<Option<ExtractedMeshes>, DecodeError> {
    // Textures need `Image` decode/transcode on the consumer's side, which is
    // bevy-typed and stays there (plan §5); a surviving required extension is
    // something no pass here handled, and the `gltf` crate would reject it.
    let non_empty = |key: &str| json[key].as_array().is_some_and(|a| !a.is_empty());
    if non_empty("images") || non_empty("textures") || non_empty("extensionsRequired") {
        return Ok(None);
    }

    let Some(materials) = extract_materials(json) else {
        return Ok(None);
    };
    let mut out = ExtractedMeshes {
        primitives: Vec::new(),
        materials,
    };

    // Default scene, resolved exactly like `Document::default_scene()` then
    // `.or(scenes().next())`: the `scene` index if present, else scene 0.
    let Some(scene_ix) = opt_index(&json["scene"]) else {
        return Ok(None);
    };
    // To serde `scenes` is a `Vec<Scene>` and `scene` an `Index`, and
    // `Scene::nodes` carries NO `#[serde(default)]` — it is required. So only
    // two shapes here legally draw nothing: a document with neither `scenes`
    // nor `scene` (the empty content tile) and a scene whose `nodes` is empty.
    // Every other shape errors the inline route, and returning an empty tile
    // instead would render a blank where the tile should have failed.
    let Value::Array(scenes) = &json["scenes"] else {
        return Ok((json["scenes"].is_null() && scene_ix.is_none()).then_some(out));
    };
    let Some(scene) = scenes.get(scene_ix.unwrap_or(0) as usize) else {
        // An empty `scenes` list has no scene 0 to default to, which is legal;
        // an explicit `scene` naming nothing is an out-of-range `Index`.
        return Ok(scene_ix.is_none().then_some(out));
    };
    let Some(roots) = scene["nodes"].as_array() else {
        return Ok(None);
    };
    for root in roots {
        let Some(ix) = root.as_u64() else {
            return Ok(None);
        };
        if !extract_node(json, bin, ix as usize, IDENTITY, 0, &mut out.primitives)? {
            return Ok(None);
        }
    }
    // An out-of-range `material` index fails the `gltf` crate's
    // `validate_minimally`, i.e. the inline route errors the whole tile.
    // Substituting the default material here would RENDER it, in the wrong
    // colour — the one divergence the fallback lattice must never have.
    if out
        .primitives
        .iter()
        .any(|p| p.material.is_some_and(|ix| ix >= out.materials.len()))
    {
        return Ok(None);
    }
    Ok(Some(out))
}

/// An OPTIONAL JSON index. `None` = **decline**: present but not an unsigned
/// integer is malformed, the `gltf` crate rejects the document, and falling
/// through as "absent" would render the tile differently from the inline
/// route instead of erroring with it. `Some(None)` = absent, `Some(Some(ix))`
/// = read.
fn opt_index(v: &Value) -> Option<Option<u64>> {
    match v {
        Value::Null => Some(None),
        v => v.as_u64().map(Some),
    }
}

/// [`opt_index`] for a float, resolving to `default` when absent.
fn opt_f32(v: &Value, default: f32) -> Option<f32> {
    match v {
        Value::Null => Some(default),
        v => v.as_f64().map(|x| x as f32),
    }
}

/// Every material of the document, in index order. `None` = decline (a
/// material shape this cannot reproduce).
fn extract_materials(json: &Value) -> Option<Vec<ExtractedMaterial>> {
    let Some(materials) = json["materials"].as_array() else {
        // Absent is "no materials"; present-but-not-an-array fails serde, so it
        // declines rather than rendering everything with the default material.
        return json["materials"].is_null().then(Vec::new);
    };
    let mut out = Vec::with_capacity(materials.len());
    for m in materials {
        let pbr = &m["pbrMetallicRoughness"];
        // A texture reference is unreachable (the document-level image check
        // declined first), so only the factors are read — the same set the
        // consumer's `decode_material` reads.
        let mut mat = ExtractedMaterial {
            double_sided: m["doubleSided"].as_bool().unwrap_or(false),
            unlit: !m["extensions"]["KHR_materials_unlit"].is_null(),
            ..ExtractedMaterial::default()
        };
        if let Some(c) = pbr["baseColorFactor"].as_array() {
            let v: Vec<f32> = c
                .iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect();
            mat.base_color = <[f32; 4]>::try_from(v).ok()?;
        }
        mat.metallic = opt_f32(&pbr["metallicFactor"], mat.metallic)?;
        mat.roughness = opt_f32(&pbr["roughnessFactor"], mat.roughness)?;
        out.push(mat);
    }
    Some(out)
}

/// Walk one node: its own mesh primitives first, then its children — the
/// inline route's order, which is the order the consumer spawns entities in.
/// `Ok(false)` = decline.
fn extract_node(
    json: &Value,
    bin: Option<&[u8]>,
    node_ix: usize,
    parent: Mat4,
    depth: usize,
    out: &mut Vec<ExtractedPrimitive>,
) -> Result<bool, DecodeError> {
    if depth > MAX_NODE_DEPTH {
        return Ok(false);
    }
    let node = &json["nodes"][node_ix];
    if node.is_null() {
        return Ok(false); // dangling index — the gltf crate rejects the file
    }
    let Some(local) = node_matrix(node) else {
        return Ok(false);
    };
    let global = mat4_mul(&parent, &local);

    let Some(mesh) = opt_index(&node["mesh"]) else {
        return Ok(false);
    };
    if let Some(mesh_ix) = mesh {
        let Some(prims) = json["meshes"][mesh_ix as usize]["primitives"].as_array() else {
            return Ok(false);
        };
        for (prim_ix, prim) in prims.iter().enumerate() {
            // Non-TRIANGLES content (POINTS/LINES, and the splat/point
            // renderers behind them) is the consumer's business.
            // Absent `mode` is glTF's default 4/TRIANGLES; present-but-not-an-
            // integer declines rather than defaulting to 4 (see `opt_index`).
            match opt_index(&prim["mode"]) {
                Some(None) | Some(Some(4)) => {}
                _ => return Ok(false),
            }
            let Some(extracted) =
                extract_primitive(json, bin, prim, global, mesh_ix, prim_ix as u64)?
            else {
                return Ok(false);
            };
            out.push(extracted);
        }
    }

    // `children` is an `Option<Vec<Index<Node>>>`: absent is legal, present-but-
    // not-an-array fails serde. Ignoring it would drop the whole subtree and
    // render a partial scene where the inline route errors.
    if !node["children"].is_null() {
        let Some(children) = node["children"].as_array() else {
            return Ok(false);
        };
        for child in children {
            let Some(ix) = child.as_u64() else {
                return Ok(false);
            };
            if !extract_node(json, bin, ix as usize, global, depth + 1, out)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// One TRIANGLES primitive → typed buffers. `Ok(None)` = decline.
fn extract_primitive(
    json: &Value,
    bin: Option<&[u8]>,
    prim: &Value,
    transform: Mat4,
    mesh_ix: u64,
    prim_ix: u64,
) -> Result<Option<ExtractedPrimitive>, DecodeError> {
    let attrs = &prim["attributes"];
    // A primitive whose geometry hangs off an extension (Draco) has no plain
    // POSITION accessor to read.
    let Some(pos_ix) = attrs["POSITION"].as_u64() else {
        return Ok(None);
    };
    let Some(positions) = float_attribute::<3>(json, bin, pos_ix as usize, "VEC3")? else {
        return Ok(None);
    };

    // Optional attributes. Present-but-unreadable DECLINES — it must never
    // fall through as absent: a missing NORMAL means "smooth-compute", which
    // is a different mesh from one whose normals we simply failed to read.
    let Some(normals) = optional_float::<3>(json, bin, attrs, "NORMAL", "VEC3")? else {
        return Ok(None);
    };
    let Some(uvs) = optional_float::<2>(json, bin, attrs, "TEXCOORD_0", "VEC2")? else {
        return Ok(None);
    };
    let Some(colors) = optional_float::<4>(json, bin, attrs, "COLOR_0", "VEC4")? else {
        return Ok(None);
    };
    let indices = match &prim["indices"] {
        Value::Null => None,
        v => match v.as_u64() {
            Some(ix) => match read_indices(json, bin, ix as usize)? {
                Some(v) => Some(v),
                None => return Ok(None),
            },
            None => return Ok(None),
        },
    };

    let Some(material) = opt_index(&prim["material"]) else {
        return Ok(None);
    };

    Ok(Some(ExtractedPrimitive {
        transform,
        mesh_ix,
        prim_ix,
        material: material.map(|i| i as usize),
        positions,
        normals,
        uvs,
        colors,
        indices,
    }))
}

/// An OPTIONAL `FLOAT` vertex attribute, with the decline arm folded in:
/// `Ok(None)` = decline the tile, `Ok(Some(None))` = the attribute is absent,
/// `Ok(Some(Some(v)))` = read.
#[allow(clippy::type_complexity)]
fn optional_float<const N: usize>(
    json: &Value,
    bin: Option<&[u8]>,
    attrs: &Value,
    name: &str,
    dims: &str,
) -> Result<Option<Option<Vec<[f32; N]>>>, DecodeError> {
    match &attrs[name] {
        Value::Null => Ok(Some(None)),
        v => match v.as_u64() {
            Some(ix) => Ok(float_attribute::<N>(json, bin, ix as usize, dims)?.map(Some)),
            None => Ok(None),
        },
    }
}

/// Read a `FLOAT` vertex attribute. `Ok(None)` = decline: any other component
/// type (normalized u8/u16, quantized) or a sparse accessor is a conversion
/// whose rounding would have to match the `gltf` crate's exactly, and "close
/// enough" is not a thing this seam can offer.
fn float_attribute<const N: usize>(
    json: &Value,
    bin: Option<&[u8]>,
    acc_ix: usize,
    dims: &str,
) -> Result<Option<Vec<[f32; N]>>, DecodeError> {
    let acc = &json["accessors"][acc_ix];
    if acc["componentType"].as_u64() != Some(5126)
        || acc["type"].as_str() != Some(dims)
        || !acc["sparse"].is_null()
    {
        return Ok(None);
    }
    Ok(Some(read_accessor::<N>(json, bin, acc_ix)?))
}

/// Read an index accessor widened to u32 (u8/u16/u32 — the same set the `gltf`
/// crate's `ReadIndices::into_u32` accepts). `Ok(None)` = decline.
fn read_indices(
    json: &Value,
    bin: Option<&[u8]>,
    acc_ix: usize,
) -> Result<Option<Vec<u32>>, DecodeError> {
    let acc = &json["accessors"][acc_ix];
    if acc["type"].as_str() != Some("SCALAR") || !acc["sparse"].is_null() {
        return Ok(None);
    }
    let width = match acc["componentType"].as_u64() {
        Some(5121) => 1usize,
        Some(5123) => 2,
        Some(5125) => 4,
        _ => return Ok(None),
    };
    let count = acc["count"]
        .as_u64()
        .ok_or("index accessor without count")? as usize;
    let bv_ix = acc["bufferView"]
        .as_u64()
        .ok_or("index accessor without bufferView")? as usize;
    let bv = &json["bufferViews"][bv_ix];
    if bv["buffer"].as_u64() != Some(0) {
        return Err("index bufferView must reference buffer 0 (BIN chunk)".into());
    }
    let bin = bin.ok_or("index accessor references the BIN chunk but the GLB has none")?;
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let bv_len = bv["byteLength"]
        .as_u64()
        .ok_or("index bufferView without byteLength")? as usize;
    let stride = bv["byteStride"]
        .as_u64()
        .map(|s| s as usize)
        .unwrap_or(width);
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let view = bin
        .get(bv_offset..bv_offset + bv_len)
        .ok_or("index bufferView out of BIN bounds")?;

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = acc_offset + i * stride;
        let bytes = view
            .get(at..at + width)
            .ok_or_else(|| format!("index accessor {acc_ix} element {i} out of bounds"))?;
        out.push(match width {
            1 => u32::from(bytes[0]),
            2 => u32::from(u16::from_le_bytes(bytes.try_into().unwrap())),
            _ => u32::from_le_bytes(bytes.try_into().unwrap()),
        });
    }
    Ok(Some(out))
}

// ── f32 transform math ───────────────────────────────────────────────────────
//
// Hand-rolled (this crate has no math dependency and must not grow one), but
// NOT freely written: every operation below mirrors the `gltf` crate's
// `Transform::matrix()` and `glam`'s `Mat4 * Mat4` operand order and
// association, because the consumer's inline route composes tile transforms
// with exactly those.
//
// The agreement is to within an ulp, NOT bit-for-bit on every backend. Two
// known sources: (1) JSON numbers arrive through `serde_json` as f64 and are
// narrowed here, where the `gltf` crate parses them straight to f32, so a
// decimal within half an ulp of an f32 boundary double-rounds differently;
// (2) glam's SCALAR backend fuses its `mul_add` chain where the plain
// `a*x + b*y + …` below does not, so a target with FMA can differ in the last
// bit (glam's wasm32 simd128 backend — where this actually ships — is not
// fused and agrees). A tile takes ONE route end to end, so the worst case is
// an ulp-scale seam between an extracted tile and a declined neighbour: far
// below the RTC/planetary-rebase scale anything renders at.

/// `a * b`, column-major.
fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0f32; 16];
    for col in 0..4 {
        let (x, y, z, w) = (b[col * 4], b[col * 4 + 1], b[col * 4 + 2], b[col * 4 + 3]);
        for row in 0..4 {
            out[col * 4 + row] = a[row] * x + a[4 + row] * y + a[8 + row] * z + a[12 + row] * w;
        }
    }
    out
}

/// A node's local transform: `matrix` verbatim, else `T × R × S` from
/// translation/rotation/scale (glTF defaults when absent). `None` = decline —
/// a malformed `matrix`/TRS is a placement the inline route would never draw
/// (the `gltf` crate fails the file), so guessing one renders the tile in the
/// wrong place instead of erroring with it.
fn node_matrix(node: &Value) -> Option<Mat4> {
    // Present-but-not-an-array must decline too, not fall through to the TRS
    // branch: `Node::matrix` is an `Option<[f32; 16]>`, so `"matrix": 5` fails
    // serde and the inline route never draws the node, where composing TRS from
    // three absent keys would place it at the ORIGIN and draw it.
    if !node["matrix"].is_null() {
        let m = node["matrix"].as_array()?;
        if m.len() != 16 {
            return None;
        }
        let mut out = [0f32; 16];
        for (o, v) in out.iter_mut().zip(m) {
            *o = v.as_f64()? as f32;
        }
        return Some(out);
    }
    let t = read_floats::<3>(node, "translation", [0.0; 3])?;
    let r = read_floats::<4>(node, "rotation", [0.0, 0.0, 0.0, 1.0])?;
    let s = read_floats::<3>(node, "scale", [1.0; 3])?;

    let mut translation = IDENTITY;
    translation[12..15].copy_from_slice(&t);
    let mut scale = IDENTITY;
    scale[0] = s[0];
    scale[5] = s[1];
    scale[10] = s[2];
    Some(mat4_mul(
        &mat4_mul(&translation, &quaternion_matrix(r)),
        &scale,
    ))
}

/// Rotation matrix of an xyzw quaternion, term for term as the `gltf` crate
/// builds it.
fn quaternion_matrix([x, y, z, s]: [f32; 4]) -> Mat4 {
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx2, xy2, xz2) = (x2 * x, x2 * y, x2 * z);
    let (yy2, yz2, zz2) = (y2 * y, y2 * z, z2 * z);
    let (sx2, sy2, sz2) = (x2 * s, y2 * s, z2 * s);
    [
        1.0 - yy2 - zz2,
        xy2 + sz2,
        xz2 - sy2,
        0.0,
        xy2 - sz2,
        1.0 - xx2 - zz2,
        yz2 + sx2,
        0.0,
        xz2 + sy2,
        yz2 - sx2,
        1.0 - xx2 - yy2,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// A fixed-width float array property, `default` when absent. `None` =
/// decline (wrong length, or a non-numeric element) — same reasoning as
/// [`node_matrix`]: silently returning the default is a wrong placement.
fn read_floats<const N: usize>(node: &Value, key: &str, default: [f32; N]) -> Option<[f32; N]> {
    let Some(arr) = node[key].as_array() else {
        return node[key].is_null().then_some(default);
    };
    if arr.len() != N {
        return None;
    }
    let mut out = default;
    for (o, v) in out.iter_mut().zip(arr) {
        *o = v.as_f64()? as f32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity every route starts from must survive composition exactly.
    #[test]
    fn identity_composition_is_exact() {
        let m = node_matrix(&serde_json::json!({ "translation": [10.0, 0.0, 0.0] })).unwrap();
        assert_eq!(mat4_mul(&IDENTITY, &m), m);
        let mut want = IDENTITY;
        want[12] = 10.0;
        assert_eq!(m, want);
    }

    /// A 90° rotation about Y, composed with a translation and a scale, in the
    /// `gltf` crate's order — guards the operand order, not the formula: a
    /// `S × R × T` mix-up still round-trips the axes but puts the translation
    /// through the rotation.
    #[test]
    fn trs_composes_translate_rotate_scale() {
        let m = node_matrix(&serde_json::json!({
            "translation": [1.0, 2.0, 3.0],
            "rotation": [0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2],
            "scale": [2.0, 2.0, 2.0],
        }))
        .unwrap();
        // Translation column is untouched by R and S in T×R×S.
        assert_eq!(&m[12..15], &[1.0, 2.0, 3.0]);
        // +X maps to -Z, scaled by 2.
        assert!(
            (m[0] - 0.0).abs() < 1e-6 && (m[2] + 2.0).abs() < 1e-6,
            "{m:?}"
        );
    }

    #[test]
    fn declines_textured_documents() {
        let json = serde_json::json!({
            "images": [{ "mimeType": "image/png", "bufferView": 0 }],
            "scenes": [{ "nodes": [0] }],
        });
        assert!(extract_tile_meshes(&json, None).unwrap().is_none());
    }

    #[test]
    fn declines_surviving_required_extension() {
        let json = serde_json::json!({ "extensionsRequired": ["KHR_something_new"] });
        assert!(extract_tile_meshes(&json, None).unwrap().is_none());
    }

    /// A document with no scene is legal and renders nothing — it must not be
    /// mistaken for a decline (that would send every empty tile on a pointless
    /// second decode).
    #[test]
    fn empty_document_extracts_to_no_primitives() {
        let out = extract_tile_meshes(&serde_json::json!({}), None)
            .unwrap()
            .expect("accepted");
        assert!(out.primitives.is_empty() && out.materials.is_empty());
    }

    /// A minimal one-triangle document, parameterised by the bits the decline
    /// lattice turns on.
    fn doc(prim: serde_json::Value) -> Value {
        serde_json::json!({
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [prim] }],
            "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                          { "bufferView": 0, "componentType": 5123, "count": 3, "type": "VEC3" }],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }],
        })
    }

    /// POINTS/LINES content belongs to the consumer's own renderers, and a
    /// non-`FLOAT` attribute would need the `gltf` crate's exact normalization
    /// rounding — both decline rather than guess.
    #[test]
    fn declines_non_triangle_and_non_float_attributes() {
        let bin = vec![0u8; 36];
        let points = doc(serde_json::json!({ "mode": 0, "attributes": { "POSITION": 0 } }));
        assert!(
            extract_tile_meshes(&points, Some(&bin)).unwrap().is_none(),
            "POINTS declines"
        );
        let quantized = doc(serde_json::json!({ "attributes": { "POSITION": 1 } }));
        assert!(
            extract_tile_meshes(&quantized, Some(&bin))
                .unwrap()
                .is_none(),
            "u16 POSITION declines"
        );
        // The same document with a FLOAT position and the default mode is the
        // control: it must NOT decline, or the two asserts above prove nothing.
        let ok = doc(serde_json::json!({ "attributes": { "POSITION": 0 } }));
        let out = extract_tile_meshes(&ok, Some(&bin))
            .unwrap()
            .expect("plain triangles extract");
        assert_eq!(out.primitives.len(), 1);
        assert_eq!(out.primitives[0].positions.len(), 3);
    }

    /// Malformed shapes the `gltf` crate REJECTS (so the inline route errors
    /// the tile) must decline here, never be guessed into a tile that renders
    /// with the wrong material, the wrong placement, or as the wrong topology.
    #[test]
    fn malformed_shapes_decline_instead_of_guessing() {
        let bin = vec![0u8; 36];
        let prim = serde_json::json!({ "attributes": { "POSITION": 0 } });
        for (what, mut json) in [
            (
                "material index out of range",
                doc(serde_json::json!({
                    "attributes": { "POSITION": 0 }, "material": 7,
                })),
            ),
            (
                "non-integer mode",
                doc(serde_json::json!({
                    "attributes": { "POSITION": 0 }, "mode": "TRIANGLES",
                })),
            ),
            (
                "non-integer material",
                doc(serde_json::json!({
                    "attributes": { "POSITION": 0 }, "material": "red",
                })),
            ),
        ] {
            json["materials"] = serde_json::json!([{}]);
            assert!(
                extract_tile_meshes(&json, Some(&bin)).unwrap().is_none(),
                "{what} must decline"
            );
        }
        // Node-level shapes: a 15-element `matrix` and a 3-element `rotation`
        // both used to fall through to a silently DIFFERENT transform.
        for (what, node) in [
            (
                "15-element matrix",
                serde_json::json!({ "mesh": 0, "matrix": vec![0.0f32; 15] }),
            ),
            (
                "3-element rotation",
                serde_json::json!({ "mesh": 0, "rotation": [0.0, 0.0, 1.0] }),
            ),
            (
                "non-numeric translation",
                serde_json::json!({ "mesh": 0, "translation": ["a", 0, 0] }),
            ),
            ("non-integer mesh", serde_json::json!({ "mesh": "cube" })),
            // Not an array at all: this used to fall through to the TRS branch
            // and place the node at the origin.
            (
                "non-array matrix",
                serde_json::json!({ "mesh": 0, "matrix": 5 }),
            ),
            (
                "non-array children",
                serde_json::json!({ "mesh": 0, "children": 3 }),
            ),
        ] {
            let mut json = doc(prim.clone());
            json["nodes"] = serde_json::json!([node]);
            assert!(
                extract_tile_meshes(&json, Some(&bin)).unwrap().is_none(),
                "{what} must decline"
            );
        }
        // Document-level shapes. `scene` is an `Index`, `scenes` a `Vec<Scene>`,
        // `materials` a `Vec<Material>`, and `Scene::nodes` is REQUIRED — each
        // of these errors the inline route, so succeeding with an empty tile
        // would render a blank instead of failing.
        for (what, json) in [
            ("non-integer scene", serde_json::json!({ "scene": "main" })),
            (
                "out-of-range scene",
                serde_json::json!({ "scene": 5, "scenes": [{ "nodes": [] }] }),
            ),
            (
                "scene named with no scenes",
                serde_json::json!({ "scene": 0 }),
            ),
            ("non-array scenes", serde_json::json!({ "scenes": 3 })),
            (
                "scene without `nodes`",
                serde_json::json!({ "scenes": [{}] }),
            ),
            ("non-array materials", serde_json::json!({ "materials": 3 })),
        ] {
            assert!(
                extract_tile_meshes(&json, None).unwrap().is_none(),
                "{what} must decline"
            );
        }
        // Controls for the two shapes that legally draw nothing — without these
        // the loop above would pass on a function that declined everything.
        for (what, json) in [
            ("empty `scenes` list", serde_json::json!({ "scenes": [] })),
            (
                "scene with an empty `nodes`",
                serde_json::json!({ "scenes": [{ "nodes": [] }] }),
            ),
        ] {
            assert!(
                extract_tile_meshes(&json, None).unwrap().is_some(),
                "{what} must still extract"
            );
        }
    }

    #[test]
    fn material_defaults_are_the_gltf_defaults() {
        let json = serde_json::json!({ "materials": [{}, {
            "doubleSided": true,
            "extensions": { "KHR_materials_unlit": {} },
            "pbrMetallicRoughness": { "baseColorFactor": [0.5, 0.25, 0.0, 1.0], "metallicFactor": 0.0 },
        }] });
        let mats = extract_materials(&json).expect("materials");
        assert_eq!(mats[0], ExtractedMaterial::default());
        assert_eq!(mats[0].metallic, 1.0, "glTF default metallic is 1.0");
        assert_eq!(mats[1].base_color, [0.5, 0.25, 0.0, 1.0]);
        assert!(mats[1].double_sided && mats[1].unlit);
        assert_eq!((mats[1].metallic, mats[1].roughness), (0.0, 1.0));
    }
}
