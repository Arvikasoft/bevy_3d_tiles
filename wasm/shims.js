// WASM shims for bevy_3d_tiles — include this script (and the libktx files
// next to it) in your index.html/dist. Extracted from the TurboTwin
// reference app. Two window globals:
//   __tt_draco_decode  — Draco read for foreign tilesets (lazy gstatic decoder)
//   __tt_ktx2_transcode — KTX2/Basis texture transcode via libktx_read.wasm
// Without them: KTX2 tiles render untextured, Draco tiles fail cleanly.
      // Draco decode shim for 3D Tiles content (tiles3d/draco.rs — Google
      // Photorealistic 3D Tiles ship Draco-compressed meshes). Lazy-loads
      // Google's official glTF-subset decoder (Apache-2.0) from the
      // versioned gstatic CDN on FIRST use — non-P3DT sessions never pay.
      // Contract: __tt_draco_decode(Uint8Array, Uint32Array of glTF
      // attribute unique ids) -> Promise<{indices: Uint32Array,
      // attributes: [{id, components, data: Float32Array}]}>.
      window.__tt_draco_decode = (function () {
        var DECODER_BASE = 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/';
        var modPromise = null;
        function loadModule() {
          if (!modPromise) {
            modPromise = new Promise(function (resolve, reject) {
              var s = document.createElement('script');
              s.src = DECODER_BASE + 'draco_wasm_wrapper_gltf.js';
              s.onload = function () {
                fetch(DECODER_BASE + 'draco_decoder_gltf.wasm')
                  .then(function (r) {
                    if (!r.ok) throw new Error('draco wasm fetch: ' + r.status);
                    return r.arrayBuffer();
                  })
                  .then(function (wasmBinary) {
                    return DracoDecoderModule({ wasmBinary: wasmBinary });
                  })
                  .then(resolve, reject);
              };
              s.onerror = function () {
                modPromise = null;
                reject(new Error('draco decoder script failed to load'));
              };
              document.head.appendChild(s);
            });
          }
          return modPromise;
        }
        return function (bytes, ids) {
          return loadModule().then(function (draco) {
            var buffer = new draco.DecoderBuffer();
            buffer.Init(bytes, bytes.length);
            var decoder = new draco.Decoder();
            var mesh = null;
            try {
              if (decoder.GetEncodedGeometryType(buffer) !== draco.TRIANGULAR_MESH) {
                throw new Error('not a triangular draco mesh');
              }
              mesh = new draco.Mesh();
              var status = decoder.DecodeBufferToMesh(buffer, mesh);
              if (!status.ok()) throw new Error('draco decode: ' + status.error_msg());
              var faces = mesh.num_faces();
              var indices = new Uint32Array(faces * 3);
              var ia = new draco.DracoInt32Array();
              for (var f = 0; f < faces; f++) {
                decoder.GetFaceFromMesh(mesh, f, ia);
                indices[f * 3] = ia.GetValue(0);
                indices[f * 3 + 1] = ia.GetValue(1);
                indices[f * 3 + 2] = ia.GetValue(2);
              }
              draco.destroy(ia);
              var points = mesh.num_points();
              var attributes = [];
              for (var k = 0; k < ids.length; k++) {
                var id = ids[k];
                var attr = decoder.GetAttributeByUniqueId(mesh, id);
                var comps = attr.num_components();
                var fa = new draco.DracoFloat32Array();
                // Float read applies the dequantization transforms.
                decoder.GetAttributeFloatForAllPoints(mesh, attr, fa);
                var data = new Float32Array(points * comps);
                for (var i = 0; i < data.length; i++) data[i] = fa.GetValue(i);
                draco.destroy(fa);
                attributes.push({ id: id, components: comps, data: data });
              }
              return { indices: indices, attributes: attributes };
            } finally {
              if (mesh) draco.destroy(mesh);
              draco.destroy(decoder);
              draco.destroy(buffer);
            }
          });
        };
      })();
      // KTX2 / Basis (UASTC) transcode shim for 3D Tiles textures
      // (tiles3d/ktx2.rs, BEVY-3D-TILES-PLAN T7). bevy's basis transcoder is
      // C++ and won't build for wasm (the "no C toolchain in the wasm build"
      // locked decision), so we transcode in JS via KTX-Software's
      // libktx_read.wasm (vendored, copy-filed to the dist root; lazy-loaded on
      // the FIRST KTX2 tile — PNG/JPEG tilesets never pay). Native builds use
      // bevy's `basis-universal` feature instead.
      // Contract: __tt_ktx2_transcode(Uint8Array ktx2, bool wantBc)
      //   -> Promise<{ format: "bc7" | "rgba8", width, height, data: Uint8Array }>
      // ("bc7"/"rgba8" are the sRGB variants — tile base color is always sRGB.)
      window.__tt_ktx2_transcode = (function () {
        var modPromise = null;
        function loadModule() {
          if (!modPromise) {
            modPromise = new Promise(function (resolve, reject) {
              var s = document.createElement('script');
              s.src = './libktx_read.js';
              s.onload = function () {
                var factory = window.createKtxReadModule || window.LIBKTX;
                if (typeof factory !== 'function') {
                  reject(new Error('libktx factory missing after load'));
                  return;
                }
                factory({ locateFile: function (p) { return './' + p; } }).then(resolve, reject);
              };
              s.onerror = function () {
                reject(new Error('libktx_read.js failed to load'));
              };
              document.head.appendChild(s);
            });
          }
          return modPromise;
        }
        return function (bytes, wantBc) {
          return loadModule().then(function (ktx) {
            var tex = new ktx.texture(bytes);
            try {
              if (tex.needsTranscoding) {
                var fmt = wantBc ? ktx.transcode_fmt.BC7_RGBA : ktx.transcode_fmt.RGBA32;
                var rc = tex.transcodeBasis(fmt, 0);
                // ktx_error_code: KTX_SUCCESS === 0 (rc is the embind enum val).
                if (rc && rc.value !== undefined && rc.value !== 0) {
                  throw new Error('transcodeBasis failed (' + rc.value + ')');
                }
              }
              // Concatenate ALL mip levels (level 0 = base) so the bevy side
              // can build a mipmapped Image — without mips, tiling textures
              // alias into grain under minification. Each getImage(level) is
              // already in the format's tight, GPU-upload layout, so appending
              // them in order matches what a mipmapped Image expects.
              var numLevels = tex.numLevels || 1;
              var parts = [];
              var total = 0;
              for (var lvl = 0; lvl < numLevels; lvl++) {
                var li = tex.getImage(lvl, 0, 0);
                var lc = new Uint8Array(li.length);
                lc.set(li); // copy out of wasm memory before the texture is freed
                parts.push(lc);
                total += lc.length;
              }
              var out = new Uint8Array(total);
              var off = 0;
              for (var i = 0; i < parts.length; i++) {
                out.set(parts[i], off);
                off += parts[i].length;
              }
              return {
                format: wantBc ? 'bc7' : 'rgba8',
                width: tex.baseWidth,
                height: tex.baseHeight,
                levels: numLevels,
                data: out,
              };
            } finally {
              if (typeof tex.delete === 'function') tex.delete();
            }
          });
        };
      })();
