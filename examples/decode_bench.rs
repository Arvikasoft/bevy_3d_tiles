//! Native decode-path bench over real tile GLBs.
//!
//! The wasm client decodes tiles on the main thread (no workers), so per-tile
//! `decode_glb` time IS the frame hitch (wasm runs ~2-3x slower than native).
//! Point this at a tiler output directory to see the distribution:
//!
//!     cargo run --release --example decode_bench -- /path/to/tiles/content [N]
//!
//! N = how many of the largest tiles to bench (default 24; root chain r*.glb
//! is always included).

use std::time::Instant;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: decode_bench <glb-dir> [N]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(24);

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "glb"))
        .collect();
    files.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)));

    // Largest N + the root chain (what a fresh open decodes first).
    let mut bench: Vec<_> = files.iter().take(n).cloned().collect();
    for p in &files {
        let stem = p.file_stem().unwrap().to_string_lossy();
        if (stem == "r" || stem.len() <= 5) && !bench.contains(p) {
            bench.push(p.clone());
        }
    }

    let mut rows = Vec::new();
    let mut total_ms = 0.0;
    for p in &bench {
        let bytes = std::fs::read(p).expect("read");
        let t0 = Instant::now();
        let items = bevy_3d_tiles::content::decode_glb(&bytes).expect("decode");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;
        let (mut tris, mut prims) = (0usize, 0usize);
        for it in &items {
            // DecodedItem is single-variant without the points/splats features.
            #[allow(irrefutable_let_patterns)]
            if let bevy_3d_tiles::content::DecodedItem::Mesh(m) = it {
                prims += 1;
                tris += m.mesh.indices().map_or(0, |ix| ix.len() / 3);
            }
        }
        rows.push((p.file_name().unwrap().to_string_lossy().to_string(), bytes.len(), ms, tris, prims));
    }
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("{:<18} {:>9} {:>9} {:>9} {:>6}", "tile", "bytes", "ms", "tris", "prims");
    for (name, len, ms, tris, prims) in &rows {
        println!("{:<18} {:>9} {:>9.1} {:>9} {:>6}", name, len, ms, tris, prims);
    }
    println!(
        "\n{} tiles, total {:.0}ms, mean {:.1}ms — multiply by ~2-3x for wasm main-thread cost",
        rows.len(),
        total_ms,
        total_ms / rows.len() as f64
    );
}
