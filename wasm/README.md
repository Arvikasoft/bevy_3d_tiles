# WASM runtime shims

Browser builds need two JS hooks the crate calls via `wasm-bindgen`
(native builds need neither):

| Window global | Used for | Backed by |
|---|---|---|
| `__tt_ktx2_transcode(ktx2Bytes, wantBc)` | `KHR_texture_basisu` / KTX2 tile textures → BC7 or RGBA8 | `libktx_read.wasm` (KTX-Software, Apache-2.0 — bundled here) |
| `__tt_draco_decode(bytes, attrIds)` | Draco *read* for foreign tilesets (e.g. offline Google P3DT content) | Google's glTF Draco decoder, lazy-loaded from the versioned gstatic CDN |

Setup: include `shims.js` in your `index.html` (a plain `<script>`, before the
wasm loads) and serve `libktx_read.js` + `libktx_read.wasm` from your dist
root. Both shims lazy-load on first use — tilesets without KTX2/Draco content
never pay for them.

Degradation without the shims is clean: KTX2 tiles render untextured
(base-color factor), Draco tiles fail with a logged error.
