# bevy_3d_tiles

**An [OGC 3D Tiles 1.1](https://docs.ogc.org/cs/22-025r4/22-025r4.html)
streaming renderer for [Bevy](https://bevyengine.org)** — the tiled-LOD
format used by Cesium, Google Photorealistic 3D Tiles, and most large-scale
photogrammetry/BIM/GIS pipelines. Native and WebGPU/wasm.

Extracted from [TurboTwin](https://turbotwin.cloud)'s production digital-twin
viewer, where it streams multi-hundred-MB site meshes, LiDAR point clouds,
and gaussian-splat captures in the browser.

**Community:** [Discord — #bevy-3d-tiles](https://discord.gg/SPqnj4pdAE) for
questions and dev chat · [GitHub issues](https://github.com/Arvikasoft/bevy_3d_tiles/issues)
for bugs and feature requests.

## What it does

- **3D Tiles 1.1 traversal** — per-tile `geometricError` screen-space-error
  selection with replacement refinement, zoom-out protection, frame-history
  kicking (no holes while streaming), Urgent/Normal/Preload request
  priorities recomputed per frame, and cancellation of out-of-cut fetches.
- **Packed `.3tz` archives streamed over HTTP range requests** — one blob per
  asset, no unpacking, no server compute: a tail scan finds the
  `@3dtilesIndex1@`, then ~2 range-GETs per tile. As far as we know no other
  runtime (including Cesium's) streams `.3tz` from a URL.
- **Exploded `tileset.json` tilesets** too, of course — local paths or URLs,
  including external-tileset grafting (`content.uri` → sub-tileset.json).
- **glTF tile content**: meshes, `POINTS` point clouds (`points` feature →
  [`bevy_pointcloud_x`](https://github.com/Arvikasoft/bevy_pointcloud_x)),
  and `KHR_gaussian_splatting` splat tiles (`splats` feature →
  [`bevy_gaussian_splatting`](https://github.com/mosure/bevy_gaussian_splatting),
  with `COLOR_0` point fallback). The splat extension is decoded from its
  Release-Candidate spec — expect follow-ups if ratification shifts it.
- **Compressed content**: `EXT_meshopt_compression` (pure-Rust decoder — no C
  toolchain, wasm-friendly), `KHR_texture_basisu`/KTX2 (BC7 on desktop,
  clean untextured fallback where GPU block formats are absent), and Draco
  *read* for foreign tilesets (browser shim).
- **Feature metadata + picking**: `EXT_mesh_features` +
  `EXT_structural_metadata` decode into a per-tile triangle→feature table, so
  a raycast hit resolves to the source-model node — click a pump in a
  10M-triangle tiled plant and know which pump.
- **Georeferenced (ECEF) tilesets**: `region`/planetary volumes detected and
  built in f64, placed through a host-supplied `EcefOrigin` (helper:
  [`geodesy::world_from_ecef`]) — including **Google Photorealistic 3D
  Tiles** with the full session protocol, attribution aggregation, cache
  bypass, and a client-side daily request cap (see the ToS note below).

## What it deliberately does not do

Raster overlays, quantized-mesh terrain, vector/voxel tiles, time-dynamic
tiles, Cesium ion / iTwin clients, implicit tiling (explicit tilesets are
fine to ~100M points), legacy `b3dm`/`pnts`/`i3dm` content (deprecated in
1.1). If you need those, [cesium-native](https://github.com/CesiumGS/cesium-native)
is the reference implementation.

## Quickstart

```rust,no_run
use bevy::prelude::*;
use bevy_3d_tiles::{Tiles3dAttach, Tiles3dCamera, Tiles3dPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Tiles3dPlugin)
        .add_systems(Startup, |mut commands: Commands,
                               mut attach: MessageWriter<Tiles3dAttach>| {
            commands.spawn((
                Camera3d::default(),
                Transform::from_xyz(60.0, 45.0, 90.0).looking_at(Vec3::ZERO, Vec3::Y),
                Tiles3dCamera, // ← SSE is computed against this camera
            ));
            let anchor = commands.spawn((Transform::IDENTITY, Visibility::default())).id();
            attach.write(Tiles3dAttach {
                anchor,
                url: "https://example.com/asset.3tz".into(), // or …/tileset.json
                local: Transform::IDENTITY,
                owner_id: None,
                label: "my tileset".into(),
                p3dt: None,
                sse_threshold_px: None, // per-set SSE override; None = Tiles3dConfig default
            });
        })
        .run();
}
```

Try it now — a small fixture tileset ships in the repo:

```sh
cargo run --example local_tileset                 # bundled 3-level demo tileset
cargo run --example local_tileset -- <path-or-url>
GOOGLE_MAPS_API_KEY=… cargo run --example google_p3dt   # photorealistic Earth
```

Dev trigger (works in any host app): `TT_TILES3D=fixture|<path>|<url>` on
native, `?tiles3d=…` on wasm.

## Host integration (the seams)

The crate is backend-agnostic: it knows nothing about your data model. Five
optional seams wire it into a host app:

| Seam | What the host does with it |
|---|---|
| `EcefOrigin` (Resource) | supply the ECEF→world matrix for georeferenced sets ([`geodesy::world_from_ecef`] for the common case) |
| `Tiles3dCamera` (marker) | tag the camera SSE selection follows |
| `TileOwner` (Component) | read it back — every spawned tile entity carries the attach's `owner_id`, so selection/highlight map to your domain |
| `TileFeatureResolver` (Resource) | map `EXT_mesh_features` node paths to your own sub-entity ids |
| `PointTileMaterial` (Resource, `points`) | own the point material (sizing/shading) |

All have inert defaults — a standalone viewer can ignore every one of them.

## Cargo features

| Feature | Default | Pulls | For |
|---|---|---|---|
| *(none)* | ✓ | — | mesh tiles, .3tz, KTX2/meshopt/Draco, ECEF, P3DT |
| `points` | – | `bevy_pointcloud_x` | glTF `POINTS` tile content |
| `splats` | – | `bevy_gaussian_splatting` | `KHR_gaussian_splatting` tile content |

## WASM notes

- Fetching, Cache-Storage CAS, abort plumbing, and executor discipline
  (never block the single-threaded executor) are handled internally.
- **KTX2 tile textures** on wasm transcode through a lazy-loaded JS shim
  (`window.__tt_ktx2_transcode`, backed by KTX-Software's `libktx_read.wasm`);
  **Draco-compressed foreign tilesets** use `window.__tt_draco_decode`
  (Google's decoder, lazy-loaded). Copy the `wasm/` shim snippet + assets
  from this repo into your `index.html`/dist. Without the shims you still
  render — KTX2 tiles fall back to untextured, Draco tiles fail cleanly.
  (Native builds need neither: bevy's `basis-universal` transcodes KTX2.)
- Serve tiles with CORS exposing `Content-Range` (Azure gotcha: an
  `ExposedHeaders: *` wildcard does NOT include it) and HTTP/2 if you can —
  a tile cut is many small ranged GETs.

## Google Photorealistic 3D Tiles — ToS

The loader implements the session protocol, **never caches or persists
Google tiles**, aggregates per-tile copyright into `TilesetCredits`, and
enforces a client-side `daily_request_cap`. What remains YOUR job under
Google's Map Tiles API terms: show the Google logo + the aggregated
attribution lines whenever tiles are visible, and bring your own API key
(requests are billed to it). See `examples/google_p3dt.rs`.

## Bevy compatibility

| `bevy_3d_tiles` | Bevy |
|---|---|
| 0.1 | 0.18 |

Bevy 0.19 support is planned for 0.2 (waiting on the render-crate ecosystem).

## Battle-tested

This is not a weekend renderer — it shipped in production first and was
extracted second. The fix history it carries: traversal holes (parent
backfill, empty-tile refine-through), kick-cascade braking, SSE in physical
pixels on high-DPI, no-collapse-while-streaming protection, tree compaction
for long-lived grafted tilesets (and its crash fix), texture wrap/mipmap
correctness on tiling textures, Azure Blob's silent suffix-range rejection,
and a dithered LOD cross-fade that was measured and *removed* (discard
killed early-Z — the simple swap won).

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache License 2.0](LICENSE-APACHE), at your option. The demo fixture under
`assets/fixtures/` is generated by `cargo run --example gen_tiles3d_fixture`
and carries no third-party content.
