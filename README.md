# bevy_3d_tiles

**An [OGC 3D Tiles 1.1](https://docs.ogc.org/cs/22-025r4/22-025r4.html)
streaming renderer for [Bevy](https://bevyengine.org)** — the tiled-LOD
format used by Cesium, Google Photorealistic 3D Tiles, and most large-scale
photogrammetry/BIM/GIS pipelines. Native and WebGPU/wasm.

Extracted from [TurboTwin](https://turbotwin.cloud)'s production digital-twin
viewer, where it streams multi-hundred-MB site meshes, LiDAR point clouds,
and gaussian-splat captures in the browser.

**[▶ Live demo](https://www.arvikasoft.se/bevy-3d-tiles)** — the [`viewer/`](viewer/)
example compiled to wasm, streaming Swiss federal buildings and Tokyo's
PLATEAU model; paste any CORS-enabled tileset URL to view your own.

**Community:** [Discord — #bevy-3d-tiles](https://discord.gg/SPqnj4pdAE) for
questions and dev chat · [GitHub issues](https://github.com/Arvikasoft/bevy_3d_tiles/issues)
for bugs and feature requests.

## What it does

- **3D Tiles 1.1 traversal** — per-tile `geometricError` screen-space-error
  selection with replacement refinement, zoom-out protection, frame-history
  kicking (no holes while streaming), Urgent/Normal/Preload request
  priorities recomputed per frame, and cancellation of out-of-cut fetches.
- **Packed `.3tz` archives streamed over HTTP range requests** — one blob per
  asset, no unpacking, no server compute. Opening costs a single parallel
  round-trip pair (suffix: EOCD + central directory + `@3dtilesIndex1@`;
  speculative head: a front-packed `tileset.json` + root tile render with
  **zero further requests**), and each other tile is exactly one range-GET —
  its byte span is derived from the index, so header and data arrive
  together. As far as we know no other runtime (including Cesium's) streams
  `.3tz` from a URL.
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
- **Legacy `b3dm` containers** — unwrapped to their embedded glTF with
  feature-table `RTC_CENTER`/`CESIUM_RTC` placement, because the big open
  fleets (swisstopo's swissBUILDINGS3D, Japan's PLATEAU) still serve 1.0
  tilesets. Batch tables are not surfaced (no per-feature picking on b3dm).

## What it deliberately does not do

Raster overlays, quantized-mesh terrain, vector/voxel tiles, time-dynamic
tiles, Cesium ion / iTwin clients, implicit tiling (explicit tilesets are
fine to ~100M points), legacy `pnts`/`i3dm`/`cmpt` content (deprecated in
1.1; plain `b3dm` IS supported, see above). If you need those,
[cesium-native](https://github.com/CesiumGS/cesium-native)
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
                label: "my tileset".into(),
                // local transform, owner id, P3DT session, per-set SSE override
                ..default()
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
(cd viewer && trunk serve)                        # the web viewer (live-demo source)
```

Dev trigger (works in any host app): `TT_TILES3D=fixture|<path>|<url>` on
native, `?tiles3d=…` on wasm.

## Host integration (the seams)

The crate is backend-agnostic: it knows nothing about your data model. These
optional seams wire it into a host app:

| Seam | What the host does with it |
|---|---|
| `EcefOrigin` (Resource) | supply the ECEF→world matrix for georeferenced sets ([`geodesy::world_from_ecef`] for the common case) |
| `Tiles3dCamera` (marker) | tag the camera SSE selection follows |
| `TileOwner` (Component) | read it back — every spawned tile entity carries the attach's `owner_id`, so selection/highlight map to your domain |
| `TileFeatureResolver` (Resource) | map `EXT_mesh_features` node paths to your own sub-entity ids |
| `TilePickMesh` (Component) | read it back — the CPU geometry of every tile mesh entity (positions, visible triangles with stable ordinals, a ray-restricted walk, `raycast()`), since tile meshes are `RENDER_WORLD`-only |
| `HiddenTileFeatures` (Resource) | hide `EXT_mesh_features` features by resolved owner id; affected tiles are re-cut on the GPU in place |
| `TileSseMultiplier` (Component) | per-set refine-threshold dial on the anchor — coarsen ground/background sets without touching the twins |
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
enforces a client-side `daily_request_cap` (counted in billable
root/session-opening requests — sessioned tile traffic is unmetered by
Google and never charged). What remains YOUR job under
Google's Map Tiles API terms: show the Google logo + the aggregated
attribution lines whenever tiles are visible, and bring your own API key
(requests are billed to it). See `examples/google_p3dt.rs`.

## Bevy compatibility

| `bevy_3d_tiles` | Bevy |
|---|---|
| 0.3 – 0.5 | 0.19 |
| 0.1 – 0.2 | 0.18 |

## Upgrading

### 0.4.x → 0.5.0

A performance release for large multi-tileset scenes (measured on a large
mining site with ~450 resident tiles). Breaking changes:

- **Tile meshes are `RenderAssetUsages::RENDER_WORLD`-only.** After their first
  upload `Assets<Mesh>` holds no vertex data for them: `Mesh::attribute()`,
  `indices()` and `morph_targets()` panic, the `try_*` variants return `Err`,
  and bevy's `MeshPickingPlugin` skips them. Read tile geometry from the new
  `TilePickMesh` component on every tile mesh entity instead (positions, the
  visible triangles with their stable ordinals, `triangle_count()`,
  `raycast()`). Attach with `Tiles3dAttach { main_world_meshes: true, .. }` for
  a set whose meshes you must read in full (an outline through
  `build_submesh`, a physics proxy, bevy_picking); even then the main-world
  indices stay pristine while hidden features are cut on the GPU only, so pick
  through `TilePickMesh`.
- **bevy_mod_outline ≤ 0.13 panics on a `RENDER_WORLD` tile entity that carries
  an `OutlineVolume`** (its pipeline key calls `morph_targets()`). Use
  `main_world_meshes: true` for those sets, or a bevy_mod_outline that reads
  `try_morph_targets()`.
- **`build_submesh` needs a `main_world_meshes` source**; on an extracted mesh it
  returns an empty mesh instead of panicking.
- **Every tile mesh entity spawns with `Aabb` + `NoAutoAabb`**, computed at
  decode, so `calculate_bounds` never recomputes tile bounds. Don't rely on it
  to (re)compute them.
- **Hide features with `HiddenTileFeatures`, not by editing tile meshes.** Insert
  the resolved owner ids (`TileFeaturePick::owner_of_feature` strings) to hide;
  every feature primitive drops those features' triangles — degenerate in place
  on the GPU (one index-only buffer write per affected primitive, under any
  material and in every pass), skipped by `TilePickMesh` — and triangle ordinals
  never change, so `TileFeaturePick` lookups stay valid. Hiding a whole
  featureless tile is still yours (remove its `Mesh3d`).
- **The `TilePrepareHook` closure receives `&[u8]`, not `Vec<u8>`.**
  `TilePrepareFn` is `for<'a> Fn(&'a [u8], bool) -> Pin<Box<dyn Future<…> + 'a>>`
  (plus `Send`/`Sync` on native): the future may borrow the fetched bytes, which
  the crate keeps for its inline fallback anyway, so handing the hook a copy was
  a full-tile allocation per request. Copy inside the hook if your future must
  own the bytes beyond its own lifetime.
- **`TileFeaturePick` fields are `Arc<[u32]>` and `Arc<[String]>`.** Reads
  (`.get()`, `.iter()`, indexing) compile unchanged; code that builds or
  replaces a table builds an `Arc` (`vec.into()`). Every primitive of a tile now
  shares one owner table, and the `TileFeatureResolver` runs once per tile
  instead of once per primitive.
- **`Tiles3dConfig::max_feature_submeshes` is removed** (it has had no effect
  since 0.1.6).
- **`Tiles3dAttach` implements `Default`** (anchored to `Entity::PLACEHOLDER`).
  Prefer `..default()` so future fields don't break your literals.
- **`Tiles3dDecodeStats`** gains `primitives`, `textured_primitives` and
  `material_keys` (distinct untextured PBR factor sets per decoded tile,
  summed), so a struct literal of it needs the new fields.
- **`TileFeatures::feature_of_vertex` is removed.** The per-vertex feature ids
  have always also been on the decoded mesh as `Mesh::ATTRIBUTE_UV_1`
  (`[fid, 0]`); read them there (on `DecodedPrimitive::mesh`, before spawn).
- **Non-indexed feature primitives are now indexed** (U32 `0..n`). The
  triangles and their ordinals are unchanged; every `EXT_mesh_features`
  primitive is now addressable by index.
- **`DecodedPrimitive` gains `bounds`**: the position AABB `[min, max]` in the
  primitive's own frame, computed off-thread on the extracted route.
- **`bevy_3d_tiles_prepare` 0.3.** `ExtractedPrimitive` gains `feature_uv1`,
  `feature_of_triangle` and `bounds`, so a hook that builds one by struct
  literal must fill them; `None` (or `..Default::default()`) keeps the 0.2
  behaviour, where the crate derives the tables from
  `PreparedFeatures::vertex_ids`. `prepare_tile_extracting` now builds the
  feature tables (and the synthesized indices) on the preparing thread, so the
  main thread only moves them onto the mesh, and `PreparedFeatures::vertex_ids`
  is empty when `meshes` is `Some`. `feature_tables()`, `bounds_of()` and
  `ExtractedPrimitive::set_feature_ids()` are the one implementation every
  route calls.
- **Extraction accepts textured tiles** (prepare 0.3). `ExtractedMeshes` gains
  `textures` and `ExtractedMaterial` gains `base_color_texture`, so a hook
  that builds them by struct literal must fill them (empty / `None` for an
  untextured tile). Base-colour textures ride encoded and decode here exactly
  as inline (PNG/JPEG through `Image::from_buffer`, KTX2 through the transcode
  pass), and missing normals arrive filled (`prepare::compute_normals`, bevy's
  `Mesh::compute_normals` bit for bit). `extract_tile_meshes` takes an
  `ExtractOptions` (both on by default; the `prepare_tile_extracting*`
  functions keep their signatures, `prepare_tile_extracting_with` takes the
  options). `DecodedMaterial` gains `base_color_host`. A document that
  requires only `KHR_materials_unlit` extracts too (photorealistic layers
  require it on every tile). `PreparedTile` gains `extract_declined`, a short
  phrase naming why extraction declined a tile (`None` when it extracted;
  diagnostic text, not a value to match on), so a hook that builds one by struct literal sets it to `None`.
- **Host-decoded textures need a `TileTextureHook`.** A hook may decode a
  texture itself and hand the crate an opaque `TileImage::Host` token; the
  crate builds a data-less destination `Image` and calls the hook once per
  token with `Some(AssetId)`. Return `false` for a token you no longer hold and
  the tile is decoded again. A token whose tile will not spawn (cancelled,
  detached, waiting for the origin, failed, refused) comes back once with
  `None`: free it then. Without a hook such textures render untextured.
  `TileImage` and `ExtractOptions` are `#[non_exhaustive]`: match `TileImage`
  with a catch-all and build `ExtractOptions` from `Default`.
- **`TilePickMesh` has a ray-restricted walk.**
  `for_each_visible_triangle_on_ray(origin, dir, f)` visits only the runs of 64
  triangles whose padded bounds the ray touches, in the same order with the
  same ordinals and hidden mask, and `raycast()` uses it. A picker that runs its
  own triangle test over `for_each_visible_triangle` can switch for the same
  nearest hit (grazing rays aside; the test must reject a triangle with a NaN
  coordinate, as Möller–Trumbore does). The bounds are built on a copy's first
  ray (24 B per 64 triangles) and counted in `resident_cpu_bytes()` from the
  start, so that figure is an upper bound: it includes copies no ray has
  reached yet.
- **Tile `StandardMaterial`s are shared.** Every untextured primitive with the
  same PBR factors (base color, metallic, roughness, unlit, double-sided) uses
  ONE material, across tiles and tilesets, so their draws share a bind group and
  a landing or respawning tile creates no material. Mutating one in place now
  changes every tile that shares it: **clone before mutating**, and cache a
  replacement material per (base material, your key) rather than per entity
  (the `TileGeometry` doc example shows the pattern, with an `AssetEvent`
  prune). Textured primitives still own their material, as before.

Behavioral:

- `Tiles3dSets::resident_content_bytes()` keeps its meaning (decoded geometry
  bytes, so the same scene budgets the same cut) and is now O(1): the figure as
  of the end of the last `Tiles3dSet::Drive`. A reader ordered before Drive sees
  the previous frame's value. The new `resident_cpu_bytes()` reports what tile
  geometry costs in CPU memory now (pick copies, plus full meshes for
  `main_world_meshes` sets).
- The per-cut, per-graft and per-tileset-open log lines moved from `info` to
  `debug` (a moving camera changes the cut most frames, and on wasm every
  `info` line is a console write). Filter `bevy_3d_tiles=debug` to see them.
- With a `TilePrepareHook` that extracts (`prepare_tile_extracting*`), textured
  and normal-less content (photorealistic mesh layers) no longer comes back as
  a prepared GLB: the glTF parse, the attribute collect, the feature tables and
  the missing normals run on the preparing thread. Encoded base-colour
  textures still decode on the main thread unless a `TileTextureHook` takes
  them as host-decoded tokens. `ExtractOptions { textures: false, .. }` keeps
  textured content on the prepared route, as in 0.4.
- `TilePickMesh::raycast()` visits only the runs of 64 triangles whose padded
  bounds the ray touches, for the same nearest hit. A copy's first ray builds
  those bounds (under one full walk); every later ray costs a small fraction
  of one.

### 0.3.0 → 0.4.0

- **Breaking, and the only break:** `PreparedTile` gained a public field
  (`meshes`), so a hook that builds one with a struct literal must add
  `meshes: None` — which keeps today's behaviour exactly. Nothing else changed
  shape. (Minor bump, not patch, precisely because of that one line;
  `bevy_3d_tiles_prepare` goes 0.1 → 0.2 alongside it.)
- **The `TilePrepareHook` can now hand back decoded GEOMETRY, not just prepared
  glTF bytes.** `bevy_3d_tiles_prepare` 0.2 adds `prepare_tile_extracting` /
  `extract_tile_meshes`, which run the glTF parse and the per-primitive
  attribute collect on the hook's thread and return plain typed buffers
  (`ExtractedMeshes`); the crate then only builds `Mesh` objects and uploads
  them, which is the part that cannot leave the main thread. `prepare_tile`
  still fills `meshes` with `None`.
- Extraction **declines** (`meshes: None`) anything it cannot reproduce
  byte-identically to the in-engine decode — textured tiles, non-triangle
  content, quantized/integer vertex attributes, a surviving `extensionsRequired`
  — and those tiles take the prepared route (the 0.3 behaviour) with identical
  output.
- `DecodedTile::stage_ms[1]` (the glTF parse) reads **0** on the extracted
  route, and `[2]` measures the `Mesh` build alone.

### 0.2.4 → 0.3.0

- **Bevy 0.19** (wgpu 29). No `bevy_3d_tiles` API changed — every public type,
  system set, resource, and component is identical to 0.2.4. The bump is the
  whole release.
- Optional-feature deps move with it: `points` → `bevy_pointcloud_x` 0.2,
  `splats` → `bevy_gaussian_splatting` 8.
- The bevy dependency is now **exact-pinned** (`=0.19.0`) where 0.1–0.2 used a
  caret, matching the pin discipline of its consumers. Note this is *not* what
  makes `Assets<PointCloud>` typecheck across the `points` boundary — cargo
  unifies caret ranges too, and that only ever needed ONE source for
  `bevy_pointcloud_x`. It is here so a bevy patch release cannot enter the tree
  without someone deciding to. If you need `0.19.1`, patch or ask.

### 0.2.3 → 0.2.4

- **Hidden tiles no longer keep their entities.** A tile outside the render cut
  (a REPLACE-refined parent, or one waiting out the eviction grace window) is
  **despawned**; its decoded assets stay in `Assets<*>`, held by the slot, and a
  re-entering tile respawns from them — no fetch, no decode. Residency and
  `Tiles3dSets::resident_content_bytes()` are unchanged by design (the memory
  really is still resident); only eviction reclaims. What changes for a host:
  **a tile's `Entity` is no longer stable** — resolve tiles by identity, never by
  a cached `Entity`, and expect `Added<TileOwner>` / `Added<TileGeometry>` /
  `Added<TileFeaturePick>` (and `Added<Mesh3d>`) to fire again on every re-entry,
  which is what keeps host material/clip/section adapters correct.
- **New knob `Tiles3dConfig::max_respawns_per_frame`** (default 64, GLOBAL across
  sets) boxes those re-entries. It is separate from `max_spawns_per_frame` on
  purpose: that one boxes *decode* (wasm hosts lower it to 2–4), while a respawn
  reuses assets that are already decoded and uploaded. A despawn is held only for
  the tiles actually covering a selected tile that has not spawned yet — its
  ancestors and descendants — so a refining parent never leaves before its
  children arrive, and unrelated out-of-cut tiles still leave immediately.
- **Swap sequencing is unchanged from a viewer's seat.** A refinement still shows
  exactly one rung per frame: while a coarse parent is held and painting, its
  arrived children WAIT (spawned, hidden) instead of drawing on top of it, and
  they all flip visible on the same frame the parent is despawned — one command
  flush, so no gap and no coarse-over-fine overlap however long the respawn
  budget makes the children trickle. A tile with no painting ancestor to wait
  behind — cut entry from cache, where the whole chain is despawned — respawns
  VISIBLE on the frame it is selected. Coarsening keeps its one-frame
  coarse-over-fine overlap (the parent paints while the children it replaces are
  still up), which is deliberate: coarse over fine beats a gap.
- **New seam `TileSseMultiplier(f32)`** on the anchor entity — a live, relative
  dial on the set's refine threshold (`>1.0` = coarser cut). The
  "ground tilesets don't need twin-grade density" knob; composes with
  `Tiles3dAttach::sse_threshold_px` and the memory-pressure valve.

### 0.1.8 → 0.1.9

- **`build_submesh(mesh, tris)` is now public** — the on-demand half of the
  removed eager per-feature split: extract just the triangles you want (e.g.
  the clicked feature's, via `TileFeaturePick`) into a compact mesh for
  outline passes, physics proxies, or export. No behavioral change.

### 0.1.7 → 0.1.8

**Fixes a 0.1.7 regression** (0.1.7 is yanked): feature tiles WITHOUT texture
coordinates got `UV1` without `UV0`, a combination bevy 0.18's pbr shader
never handles (`pbr_fragment.wgsl` declares `uv` only under `VERTEX_UVS_A`
but references it under `VERTEX_UVS`) — pipeline creation failed and the
geometry silently vanished, for any `StandardMaterial`-derived material.
Untextured feature tiles now get zero-filled `UV0` alongside the feature-id
`UV1`. No API change.

### 0.1.6 → 0.1.7

**Feature tiles carry their feature ids as `ATTRIBUTE_UV_1`** (`[fid, 0]`,
the raw per-vertex `_FEATURE_ID_0` values), enabling the render-state
per-feature styling 0.1.6 pointed at: an
`ExtendedMaterial<StandardMaterial, _>` fragment extension reads
`in.uv_b.x` through the standard pipeline's `VERTEX_UVS_B` path — no custom
vertex shader — and tints/hides fragments per feature (the CesiumJS
`Cesium3DTileFeature.color` model). Swap materials on entities carrying
[`TileFeaturePick`] (or post-process via [`TileGeometry`]).

- `TileFeatures` gained `feature_of_vertex: Vec<f32>` (affects only code
  constructing it directly, i.e. tests).
- Feature tiles never carried a real `TEXCOORD_1` (the decoder always
  dropped it), so nothing is displaced. Featureless tiles are unchanged.

### 0.1.5 → 0.1.6

**Feature tiles no longer split into per-owner submeshes** — every primitive
spawns as ONE mesh (the Cesium model: batch ids + hit-time resolution, never
geometry splitting). The split cost seconds of main-thread hang per refine
wave on wasm even capped; pure-decode tilesets only micro-stutter.

- New component **`TileFeaturePick`** on feature-tile mesh entities:
  `owner_of_feature[feature_of_triangle[hit_triangle]]` is the same owner
  string the per-feature submeshes used to carry in `TileOwner`. A host
  raycaster that knows the hit triangle's index-buffer ordinal keeps
  per-feature *selection* exactly as before.
- Per-feature *hover/outline visuals* that keyed off per-owner entities need a
  render-state replacement (e.g. a feature-id tint in the material — the
  CesiumJS `Cesium3DTileFeature.color` model). Until then they degrade to
  whole-tile.
- `Tiles3dConfig.max_feature_submeshes` is vestigial (kept for struct-literal
  compatibility).

### 0.1.4 → 0.1.5

- **`Tiles3dConfig.memory_budget_bytes: u64`** (default `0` = off) — the
  memory-pressure valve. When the raw content bytes of all resident tiles
  exceed the budget, the effective SSE threshold inflates by the overshoot
  (clamped ×8): the cut coarsens instead of the client dying with
  "memory access out of bounds". wasm hosts should set a few hundred MB
  (decoded CPU+GPU cost runs ~2-4× raw bytes against a grows-only ~4 GiB
  address space). Config literals using `..Default::default()` need no
  change.

### 0.1.3 → 0.1.4

- Behavioral only: the speculative open head is 512 KiB (was 2 MiB) — sized
  for bandwidth, see `archive.rs`.

### 0.1.2 → 0.1.3

Two structs gained fields — struct-literal construction sites need a one-line
addition each:

- **`Tiles3dAttach.sse_threshold_px: Option<f64>`** — per-tileset
  screen-space-error threshold override; `None` keeps the app-global
  [`Tiles3dConfig`] value. Add `sse_threshold_px: None` to existing literals.
  Set it (e.g. `Some(24.0)`) for dense single-asset previews so they stop
  over-refining past the root while a globe basemap keeps the sharp default.
- **`Tiles3dConfig.max_feature_submeshes: usize`** (default 64) — ceiling on
  the per-feature submesh split at tile spawn. Unbounded splitting froze the
  wasm main thread for seconds on tiles whose "features" were hundreds of
  exporter part names; over the cap a tile spawns as one mesh (per-feature
  hover degrades on that tile, picking correctness is unaffected). Config
  literals built with `..Default::default()` need no change.

Behavioral (no API change): the `.3tz` open now issues its suffix and a 2 MiB
speculative head request in parallel and serves front-packed entries from the
head, taking a cold open from ~5–7 serial round trips to one parallel pair;
per-tile reads collapse to a single range-GET via index-derived spans. Foreign
archives that are not front-packed lose nothing — unused windows fall back to
the previous behavior. Pack archives with `tileset.json` first and the root
tile second (any preorder writer does this) to get the zero-request first
paint.

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
