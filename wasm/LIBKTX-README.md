# libktx_read (KTX-Software web build, vendored)

`libktx_read.{js,wasm}` is the **read/transcode-only** web build of
[KTX-Software](https://github.com/KhronosGroup/KTX-Software) **v4.4.2**
(Apache-2.0), from the release asset `KTX-Software-4.4.2-Web-libktx_read.zip`.

It is lazy-loaded by the `__tt_ktx2_transcode` shim in `index.html` to transcode
`KHR_texture_basisu` KTX2/UASTC tile textures → BC7 (or RGBA8) on the wasm
viewer — bevy's own basis transcoder is C++ and won't build for
`wasm32-unknown-unknown` (the locked "no C toolchain in the wasm build"
decision), so we transcode in JS exactly like the Draco shim. Native builds use
bevy's `basis-universal` feature instead (C++ compiles fine off-wasm).

To update: download the matching `-Web-libktx_read.zip` from the KTX-Software
release and replace both files (keep the version in sync with the `ktx` CLI in
`infra/blender-service/Dockerfile`).
