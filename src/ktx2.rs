//! KTX2 / Basis (UASTC) tile-texture transcode for wasm (BEVY-3D-TILES T7).
//!
//! bevy's basis transcoder is C++ (`basis-universal-sys`) and won't build for
//! `wasm32-unknown-unknown` (no libc — the "no C toolchain in the wasm build"
//! locked decision, the same reason meshopt is a pure-Rust port). So on wasm we
//! transcode `KHR_texture_basisu` KTX2 textures through the
//! `window.__tt_ktx2_transcode` shim (see `index.html`), which lazy-loads
//! KTX-Software's vendored `libktx_read.wasm` and returns transcoded bytes
//! (KTX2 container + zstd + UASTC → BC7 or RGBA8, in one call). Native builds
//! use bevy's `basis-universal` feature directly (see `content.rs`), so this
//! module is wasm-only (declared `#[cfg(target_arch = "wasm32")]` in `mod.rs`).

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

/// Transcode a KTX2 base-color texture to a bevy `Image`. `want_bc` requests
/// BC7 (desktop WebGPU); otherwise RGBA8. Both are the sRGB variants — tile
/// base color is always sRGB.
pub async fn transcode(ktx2: &[u8], want_bc: bool) -> Result<Image, String> {
    fn err(label: &'static str) -> impl Fn(JsValue) -> String {
        move |e| format!("{label}: {e:?}")
    }

    let window = web_sys::window().ok_or("no window")?;
    let func = js_sys::Reflect::get(&window, &JsValue::from_str("__tt_ktx2_transcode"))
        .map_err(err("ktx2 shim lookup"))?;
    if !func.is_function() {
        return Err("__tt_ktx2_transcode shim missing — index.html out of date?".into());
    }
    let func = js_sys::Function::from(func);

    let bytes = js_sys::Uint8Array::new_with_length(ktx2.len() as u32);
    bytes.copy_from(ktx2);

    let promise: js_sys::Promise = func
        .call2(&JsValue::NULL, &bytes, &JsValue::from_bool(want_bc))
        .map_err(err("ktx2 shim call"))?
        .into();
    let result = JsFuture::from(promise).await.map_err(err("ktx2 transcode"))?;

    let get = |key: &str| {
        js_sys::Reflect::get(&result, &JsValue::from_str(key))
            .map_err(|e| format!("ktx2 result missing {key}: {e:?}"))
    };
    let format = get("format")?.as_string().ok_or("ktx2 format not a string")?;
    let width = get("width")?.as_f64().ok_or("ktx2 width not a number")? as u32;
    let height = get("height")?.as_f64().ok_or("ktx2 height not a number")? as u32;
    let data = js_sys::Uint8Array::new(&get("data")?).to_vec();

    let tex_format = match format.as_str() {
        "bc7" => TextureFormat::Bc7RgbaUnormSrgb,
        "rgba8" => TextureFormat::Rgba8UnormSrgb,
        other => return Err(format!("ktx2 shim returned unknown format {other}")),
    };
    Ok(Image::new(
        Extent3d { width, height, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        tex_format,
        RenderAssetUsages::RENDER_WORLD,
    ))
}
