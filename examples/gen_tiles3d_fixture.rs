//! Deterministic generator for the committed 3D Tiles fixture
//! (BEVY-3D-TILES-PLAN T0 unit-test bed + dev-viewer gate).
//!
//! ```bash
//! cargo run --example gen_tiles3d_fixture   # from bevy-client/
//! ```
//!
//! Emits `assets/fixtures/tiles3d-demo/` (exploded: tileset.json + 21 GLBs)
//! and `assets/fixtures/tiles3d-demo.3tz` (the same bytes packed through
//! `archive::write_3tz`, so the ranged reader's integration test exercises a
//! byte-identical archive). Output is fully deterministic — no timestamps,
//! no randomness — so reruns are no-ops in git.
//!
//! The tileset is a 40×40 m wavy terrain patch in a local-metres Z-up frame
//! (x east, y north, z up), three levels deep with REPLACE refinement:
//!
//! * root — sphere volume, 4×4 grid, reddish, geometricError 16
//! * 4 children — box volumes, 8×8 grids, amber, geometricError 4
//! * 16 leaves — alternating box/sphere volumes, 16×16 grids, green,
//!   geometricError 0
//!
//! All levels sample the SAME analytic surface at increasing density, so an
//! LOD swap is visible as both a silhouette change and a color flip. The NE
//! child subtree carries a `transform` (+3 m up) with its content + volumes
//! authored in the shifted local frame — exercising the runtime's transform
//! composition: if the math is right the quadrant lands 3 m above the rest.
//!
//! **Georeferenced mode (T4 verification, not committed):**
//!
//! ```bash
//! cargo run --example gen_tiles3d_fixture -- --geo <lon> <lat> <h> [out_dir]
//! ```
//!
//! Emits the same patch as an externally-shaped georeferenced tileset to
//! `out_dir` (default `/tmp/tiles3d-demo-geo`): a host `tileset.json` whose
//! root carries a **region** bounding volume and whose content is an
//! **external tileset** `sub/tileset.json`, which in turn places the patch
//! via an ENU→ECEF root **transform** (the PDOK / Cesium-mirror shape).
//! Exercises region volumes, external-tileset grafting, georeference
//! detection, and the ECEF→ENU placement path in one artifact.

use std::fs;
use std::path::Path;

use bevy_3d_tiles::archive::write_3tz;

const OUT_DIR: &str = "assets/fixtures/tiles3d-demo";
const OUT_3TZ: &str = "assets/fixtures/tiles3d-demo.3tz";

/// The shared analytic surface (height in metres at east/north metres).
fn surface(e: f64, n: f64) -> f64 {
    3.0 * (e * 0.25).sin() * (n * 0.25).cos()
}

/// Per-level vertex colors (linear RGBA).
const LEVEL_COLORS: [[f32; 4]; 3] = [
    [0.85, 0.25, 0.20, 1.0], // root: red
    [0.90, 0.70, 0.20, 1.0], // children: amber
    [0.30, 0.80, 0.35, 1.0], // leaves: green
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--geo") {
        let at = args.iter().position(|a| a == "--geo").unwrap();
        let lon: f64 = args[at + 1].parse().expect("--geo <lon> <lat> <h>");
        let lat: f64 = args[at + 2].parse().expect("--geo <lon> <lat> <h>");
        let h: f64 = args[at + 3].parse().expect("--geo <lon> <lat> <h>");
        let out = args.get(at + 4).cloned().unwrap_or_else(|| "/tmp/tiles3d-demo-geo".into());
        return gen_geo_fixture(lon, lat, h, &out);
    }

    let content_dir = Path::new(OUT_DIR).join("content");
    fs::create_dir_all(&content_dir).expect("create fixture dirs");

    // (path, bytes) pairs, accumulated for the 3tz pack.
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

    // Root: whole patch, coarse grid, world frame (no transform).
    let root_glb = tile_glb(-20.0, 20.0, -20.0, 20.0, 4, 0.0, LEVEL_COLORS[0]);
    entries.push(("content/root.glb".into(), root_glb));

    // Quadrants in fixed order: sw, se, nw, ne.
    let quads: [(&str, f64, f64); 4] =
        [("sw", -10.0, -10.0), ("se", 10.0, -10.0), ("nw", -10.0, 10.0), ("ne", 10.0, 10.0)];
    let mut children_json = Vec::new();
    for (qi, (qname, ce, cn)) in quads.iter().enumerate() {
        // The NE subtree is authored in a +3 m-shifted local frame; its tile
        // transform puts it back. Everything else is in the tileset frame.
        let is_ne = *qname == "ne";
        let z_off = if is_ne { 3.0 } else { 0.0 };

        let child_glb =
            tile_glb(ce - 10.0, ce + 10.0, cn - 10.0, cn + 10.0, 8, z_off, LEVEL_COLORS[1]);
        entries.push((format!("content/c_{qname}.glb"), child_glb));

        let mut leaves_json = Vec::new();
        for (li, (le, ln)) in
            [(-5.0, -5.0), (5.0, -5.0), (-5.0, 5.0), (5.0, 5.0)].iter().enumerate()
        {
            let (lce, lcn) = (ce + le, cn + ln);
            let leaf_glb =
                tile_glb(lce - 5.0, lce + 5.0, lcn - 5.0, lcn + 5.0, 16, z_off, LEVEL_COLORS[2]);
            let uri = format!("content/l_{qname}{li}.glb");
            entries.push((uri.clone(), leaf_glb));

            // Mixed volume kinds: alternate box and enclosing sphere.
            let volume = if (qi + li) % 2 == 0 {
                serde_json::json!({ "box": [lce, lcn, -z_off, 5.0,0.0,0.0, 0.0,5.0,0.0, 0.0,0.0,4.0] })
            } else {
                let r = (5.0f64 * 5.0 + 5.0 * 5.0 + 4.0 * 4.0).sqrt();
                serde_json::json!({ "sphere": [lce, lcn, -z_off, r] })
            };
            leaves_json.push(serde_json::json!({
                "boundingVolume": volume,
                "geometricError": 0.0,
                "content": { "uri": uri }
            }));
        }

        let mut child = serde_json::json!({
            "boundingVolume": { "box": [*ce, *cn, -z_off, 10.0,0.0,0.0, 0.0,10.0,0.0, 0.0,0.0,4.0] },
            "geometricError": 4.0,
            "content": { "uri": format!("content/c_{qname}.glb") },
            "children": leaves_json
        });
        if is_ne {
            child["transform"] = serde_json::json!([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 3.0, 1.0
            ]);
        }
        children_json.push(child);
    }

    let tileset = serde_json::json!({
        "asset": { "version": "1.1" },
        "geometricError": 64.0,
        "root": {
            "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 30.0] },
            "geometricError": 16.0,
            "refine": "REPLACE",
            "content": { "uri": "content/root.glb" },
            "children": children_json
        }
    });
    let tileset_bytes = serde_json::to_vec_pretty(&tileset).expect("tileset json");

    // Exploded layout.
    fs::write(Path::new(OUT_DIR).join("tileset.json"), &tileset_bytes).expect("write tileset");
    for (path, bytes) in &entries {
        fs::write(Path::new(OUT_DIR).join(path), bytes).expect("write tile");
    }

    // Packed twin: tileset.json deflated, GLBs stored (already-dense data).
    let mut files: Vec<(&str, &[u8], bool)> = vec![("tileset.json", &tileset_bytes, true)];
    for (path, bytes) in &entries {
        files.push((path.as_str(), bytes.as_slice(), false));
    }
    let archive = write_3tz(&files, b"");
    fs::write(OUT_3TZ, &archive).expect("write 3tz");

    let total: usize = entries.iter().map(|(_, b)| b.len()).sum();
    println!(
        "fixture: {} GLBs ({} KiB) + tileset.json ({} B) → {OUT_DIR}/ and {OUT_3TZ} ({} KiB)",
        entries.len(),
        total / 1024,
        tileset_bytes.len(),
        archive.len() / 1024
    );
}

/// Georeferenced fixture (see module docs): host tileset with a region root
/// + external-tileset content; sub tileset with an ENU→ECEF root transform.
fn gen_geo_fixture(lon: f64, lat: f64, h: f64, out: &str) {
    use bevy_3d_tiles::geodesy::geodetic_to_ecef;

    let out_dir = Path::new(out);
    fs::create_dir_all(out_dir.join("sub/content")).expect("create geo fixture dirs");

    // The same 3-level patch, but authored for the SUB tileset (content URIs
    // are sub-tileset-relative).
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    let root_glb = tile_glb(-20.0, 20.0, -20.0, 20.0, 4, 0.0, LEVEL_COLORS[0]);
    entries.push(("content/root.glb".into(), root_glb));
    let quads: [(&str, f64, f64); 4] =
        [("sw", -10.0, -10.0), ("se", 10.0, -10.0), ("nw", -10.0, 10.0), ("ne", 10.0, 10.0)];
    let mut children_json = Vec::new();
    for (qname, ce, cn) in quads.iter() {
        let child_glb =
            tile_glb(ce - 10.0, ce + 10.0, cn - 10.0, cn + 10.0, 8, 0.0, LEVEL_COLORS[1]);
        entries.push((format!("content/c_{qname}.glb"), child_glb));
        let mut leaves_json = Vec::new();
        for (li, (le, ln)) in
            [(-5.0, -5.0), (5.0, -5.0), (-5.0, 5.0), (5.0, 5.0)].iter().enumerate()
        {
            let (lce, lcn) = (ce + le, cn + ln);
            let leaf_glb =
                tile_glb(lce - 5.0, lce + 5.0, lcn - 5.0, lcn + 5.0, 16, 0.0, LEVEL_COLORS[2]);
            let uri = format!("content/l_{qname}{li}.glb");
            entries.push((uri.clone(), leaf_glb));
            leaves_json.push(serde_json::json!({
                "boundingVolume": { "box": [lce, lcn, 0.0, 5.0,0.0,0.0, 0.0,5.0,0.0, 0.0,0.0,4.0] },
                "geometricError": 0.0,
                "content": { "uri": uri }
            }));
        }
        children_json.push(serde_json::json!({
            "boundingVolume": { "box": [*ce, *cn, 0.0, 10.0,0.0,0.0, 0.0,10.0,0.0, 0.0,0.0,4.0] },
            "geometricError": 4.0,
            "content": { "uri": format!("content/c_{qname}.glb") },
            "children": leaves_json
        }));
    }

    // ENU→ECEF: columns = east, north, up unit vectors + the site's ECEF
    // position (column-major 4×4, the standard Cesium-mirror root transform).
    let (sin_lat, cos_lat) = lat.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon.to_radians().sin_cos();
    let east = [-sin_lon, cos_lon, 0.0];
    let north = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat];
    let up = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat];
    let (x0, y0, z0) = geodetic_to_ecef(lat, lon, h);
    let enu_to_ecef = [
        east[0], east[1], east[2], 0.0,
        north[0], north[1], north[2], 0.0,
        up[0], up[1], up[2], 0.0,
        x0, y0, z0, 1.0,
    ];

    let sub_tileset = serde_json::json!({
        "asset": { "version": "1.1" },
        "geometricError": 64.0,
        "root": {
            "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 30.0] },
            "geometricError": 16.0,
            "refine": "REPLACE",
            "transform": enu_to_ecef,
            "content": { "uri": "content/root.glb" },
            "children": children_json
        }
    });

    // Host root: region volume (geodetic radians) around the patch; its
    // content is the external sub tileset.
    let dlat = 30.0 / 6_378_137.0;
    let dlon = 30.0 / (6_378_137.0 * cos_lat);
    let (lat_r, lon_r) = (lat.to_radians(), lon.to_radians());
    let host_tileset = serde_json::json!({
        "asset": { "version": "1.1" },
        "geometricError": 256.0,
        "root": {
            "boundingVolume": {
                "region": [lon_r - dlon, lat_r - dlat, lon_r + dlon, lat_r + dlat,
                           h - 10.0, h + 10.0]
            },
            "geometricError": 64.0,
            "refine": "REPLACE",
            "content": { "uri": "sub/tileset.json" }
        }
    });

    fs::write(
        out_dir.join("sub/tileset.json"),
        serde_json::to_vec_pretty(&sub_tileset).expect("sub tileset json"),
    )
    .expect("write sub tileset");
    for (path, bytes) in &entries {
        fs::write(out_dir.join("sub").join(path), bytes).expect("write geo tile");
    }
    fs::write(
        out_dir.join("tileset.json"),
        serde_json::to_vec_pretty(&host_tileset).expect("host tileset json"),
    )
    .expect("write host tileset");
    println!(
        "geo fixture: {} GLBs at ({lon}, {lat}, {h}) → {out}/tileset.json \
         (region root → external sub/tileset.json → ENU→ECEF transform)",
        entries.len()
    );
}

/// One tile: a `div × div` grid over `[e0,e1] × [n0,n1]` sampling [`surface`],
/// authored in glTF Y-up (`x = east, y = up − z_off, z = −north`), with flat
/// per-level vertex colors. No normals (the runtime computes them — that path
/// is part of what the fixture tests).
fn tile_glb(e0: f64, e1: f64, n0: f64, n1: f64, div: usize, z_off: f64, color: [f32; 4]) -> Vec<u8> {
    let stride = div + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(stride * stride);
    for j in 0..=div {
        for i in 0..=div {
            let e = e0 + (e1 - e0) * i as f64 / div as f64;
            let n = n0 + (n1 - n0) * j as f64 / div as f64;
            let u = surface(e, n) - z_off;
            positions.push([e as f32, u as f32, -n as f32]);
        }
    }
    let mut indices: Vec<u16> = Vec::with_capacity(div * div * 6);
    for j in 0..div {
        for i in 0..div {
            let a = (j * stride + i) as u16;
            let b = a + 1;
            let c = a + stride as u16;
            let d = c + 1;
            // CCW from +Y (up).
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    write_glb(&positions, color, &indices)
}

/// Minimal binary glTF writer: one mesh, one primitive, POSITION + COLOR_0 +
/// u16 indices, a rough non-metal double-sided material.
fn write_glb(positions: &[[f32; 3]], color: [f32; 4], indices: &[u16]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let (mut pmin, mut pmax) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in positions {
        for k in 0..3 {
            pmin[k] = pmin[k].min(p[k]);
            pmax[k] = pmax[k].max(p[k]);
        }
        for v in p {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    let colors_offset = bin.len();
    for _ in positions {
        for v in color {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    let idx_offset = bin.len();
    for i in indices {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    let idx_len = bin.len() - idx_offset;

    let json = serde_json::json!({
        "asset": { "version": "2.0", "generator": "turbotwin gen_tiles3d_fixture" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": [{
            "attributes": { "POSITION": 0, "COLOR_0": 1 },
            "indices": 2,
            "material": 0,
            "mode": 4
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0
            },
            "doubleSided": true
        }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": positions.len(),
              "type": "VEC3", "min": pmin, "max": pmax },
            { "bufferView": 1, "componentType": 5126, "count": positions.len(), "type": "VEC4" },
            { "bufferView": 2, "componentType": 5123, "count": indices.len(), "type": "SCALAR" }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": colors_offset, "byteStride": 12,
              "target": 34962 },
            { "buffer": 0, "byteOffset": colors_offset, "byteLength": idx_offset - colors_offset,
              "byteStride": 16, "target": 34962 },
            { "buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963 }
        ],
        "buffers": [{ "byteLength": bin.len() }]
    });
    let mut json_bytes = serde_json::to_vec(&json).expect("glb json");
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let mut glb = Vec::with_capacity(28 + json_bytes.len() + bin.len());
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
