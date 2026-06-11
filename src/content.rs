//! Tile content decode: GLB bytes → Bevy mesh/material data (T0: mesh tiles
//! only; point + splat tile content land with T2/T3 and feed the existing
//! `PointCloud` / `PlanarGaussian3d` renderers per D5).
//!
//! Decodes via the `gltf` crate (no `import` feature — that would pull the
//! `image` crate; embedded textures decode through Bevy's `Image::from_buffer`
//! with the png/jpeg features the GLB twin pipeline already enables). Runs
//! entirely inside the loader task, off the frame loop: the output is plain
//! `Send` data (`Mesh`, `Image`) that the ECS drain turns into entities.
//!
//! Tile GLBs are self-contained by construction (D1/D3: our tilers emit
//! GLB-with-BIN-chunk). External buffer/image URIs are rejected with a clear
//! error rather than fetched — a tile that needs side files defeats the
//! one-blob range-read design.

use bevy::asset::RenderAssetUsages;
use bevy::image::{CompressedImageFormats, Image, ImageSampler, ImageType};
use bevy::math::Mat4;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

/// One decoded glTF primitive, positioned by its node's global transform
/// (glTF Y-up frame — the spawned tile entity applies
/// [`super::traversal::TileNode::world_from_content`] above it).
pub struct DecodedPrimitive {
    pub transform: Mat4,
    pub mesh: Mesh,
    pub material: DecodedMaterial,
}

/// Material inputs resolved at decode time; turned into a `StandardMaterial`
/// at spawn (asset insertion needs ECS access).
pub struct DecodedMaterial {
    /// Linear RGBA base color factor.
    pub base_color: [f32; 4],
    pub base_color_image: Option<Image>,
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
}

impl Default for DecodedMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            base_color_image: None,
            metallic: 0.0,
            roughness: 1.0,
            double_sided: false,
        }
    }
}

/// Decode a GLB (or self-contained glTF JSON) tile into renderable primitives.
pub fn decode_glb(bytes: &[u8]) -> Result<Vec<DecodedPrimitive>, String> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| format!("gltf parse: {e}"))?;
    let doc = gltf.document;
    let blob = gltf.blob;

    let mut out = Vec::new();
    let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) else {
        return Ok(out); // empty content tile — legal, renders nothing
    };
    for node in scene.nodes() {
        decode_node(&node, Mat4::IDENTITY, blob.as_deref(), &mut out)?;
    }
    Ok(out)
}

/// Resolve a glTF buffer: GLB BIN chunk only (tiles are self-contained).
fn resolve_buffer<'b>(buffer: &gltf::Buffer<'_>, blob: Option<&'b [u8]>) -> Option<&'b [u8]> {
    match buffer.source() {
        gltf::buffer::Source::Bin => blob,
        gltf::buffer::Source::Uri(_) => None,
    }
}

fn decode_node(
    node: &gltf::Node<'_>,
    parent: Mat4,
    blob: Option<&[u8]>,
    out: &mut Vec<DecodedPrimitive>,
) -> Result<(), String> {
    let global = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                // Points/lines tiles are T2/T3 content types; skip quietly so
                // a mixed-content tile still shows its triangles.
                continue;
            }
            out.push(decode_primitive(&primitive, global, blob)?);
        }
    }
    for child in node.children() {
        decode_node(&child, global, blob, out)?;
    }
    Ok(())
}

fn decode_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: Mat4,
    blob: Option<&[u8]>,
) -> Result<DecodedPrimitive, String> {
    let reader = primitive.reader(|buffer| resolve_buffer(&buffer, blob));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("primitive has no POSITION (or buffer is an external URI)")?
        .collect();
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
    let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|tc| tc.into_f32().collect());
    let colors: Option<Vec<[f32; 4]>> =
        reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
    let indices: Option<Vec<u32>> = reader.read_indices().map(|ix| ix.into_u32().collect());

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        // MAIN_WORLD + RENDER_WORLD: the camera-focus/selection raycasts read
        // mesh vertices on the main world (the basemap panic lesson). The CPU
        // copy is a T2 memory-budget follow-up, not a T0 risk.
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    if let Some(uvs) = uvs {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    if let Some(colors) = colors {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    }
    if let Some(indices) = indices {
        mesh.insert_indices(Indices::U32(indices));
    }
    match normals {
        Some(n) => mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n),
        // Tiler output may omit normals to save bytes; smooth-compute them
        // (decode-task CPU, not frame time).
        None => mesh.compute_normals(),
    }

    let material = decode_material(&primitive.material(), blob)?;
    Ok(DecodedPrimitive { transform, mesh, material })
}

fn decode_material(
    material: &gltf::Material<'_>,
    blob: Option<&[u8]>,
) -> Result<DecodedMaterial, String> {
    let pbr = material.pbr_metallic_roughness();
    let mut out = DecodedMaterial {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        double_sided: material.double_sided(),
        base_color_image: None,
    };
    if let Some(info) = pbr.base_color_texture() {
        let image = info.texture().source();
        match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                let buf = resolve_buffer(&view.buffer(), blob)
                    .ok_or("texture bufferView points at an external buffer")?;
                let bytes = buf
                    .get(view.offset()..view.offset() + view.length())
                    .ok_or("texture bufferView out of bounds")?;
                let decoded = Image::from_buffer(
                    bytes,
                    ImageType::MimeType(mime_type),
                    CompressedImageFormats::NONE,
                    true, // base color is sRGB
                    ImageSampler::Default,
                    // GPU-only: tile textures are never read back on the CPU.
                    RenderAssetUsages::RENDER_WORLD,
                )
                .map_err(|e| format!("texture decode ({mime_type}): {e}"))?;
                out.base_color_image = Some(decoded);
            }
            gltf::image::Source::Uri { .. } => {
                return Err("external texture URIs unsupported in tile content".into());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic GLB: one triangle with COLOR_0, no normals.
    /// (The fixture generator writes real tiles with the same layout.)
    fn tiny_glb() -> Vec<u8> {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let colors: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 1.0]; 3];
        let indices: [u16; 4] = [0, 1, 2, 0]; // padded to 4-byte alignment

        let mut bin: Vec<u8> = Vec::new();
        for p in positions.iter().flatten() {
            bin.extend_from_slice(&p.to_le_bytes());
        }
        for c in colors.iter().flatten() {
            bin.extend_from_slice(&c.to_le_bytes());
        }
        let idx_offset = bin.len();
        for i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }

        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0, "translation": [0.0, 2.0, 0.0] }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "COLOR_0": 1 },
                "indices": 2,
                "mode": 4
            }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" },
                { "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 48 },
                { "buffer": 0, "byteOffset": idx_offset, "byteLength": 6 }
            ],
            "buffers": [{ "byteLength": bin.len() }]
        });
        let mut json_bytes = serde_json::to_vec(&json).unwrap();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let mut glb = Vec::new();
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        glb
    }

    #[test]
    fn decodes_positions_colors_and_node_transform() {
        let prims = decode_glb(&tiny_glb()).expect("decode");
        assert_eq!(prims.len(), 1);
        let p = &prims[0];
        // Node translation carried into the primitive transform.
        assert_eq!(p.transform.w_axis.y, 2.0);
        assert_eq!(
            p.mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(),
            3
        );
        // Normals were computed (absent in the GLB).
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(p.mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        assert_eq!(p.material.base_color, [1.0; 4]);
        assert!(p.material.base_color_image.is_none());
    }

    #[test]
    fn garbage_bytes_error_cleanly() {
        assert!(decode_glb(b"not a glb").is_err());
    }
}
