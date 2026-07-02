//! Platform Draco decode for `KHR_draco_mesh_compression` tile content
//! (BEVY-3D-TILES T4 — Google Photorealistic 3D Tiles ship Draco-compressed
//! meshes; no published pure-Rust decoder exists yet, 2026-06 survey).
//!
//! wasm: calls the `window.__tt_draco_decode` shim (see `index.html`), which
//! lazy-loads Google's official `draco_decoder_gltf.wasm` (1.5.7, Apache-2.0)
//! from the versioned gstatic CDN on first use — the exact decoder CesiumJS
//! and three.js use for P3DT, so correctness rides Google's own releases.
//! Native: errors cleanly (P3DT is a browser-viewer surface; native dev runs
//! show everything else). Swap-in candidate when it matures: `draco-oxide`'s
//! pure-Rust decoder (reearth/draco-oxide PR #19).

use crate::content::DecodeError;

/// One decoded Draco mesh: triangle indices + dequantized float attributes,
/// in the same order as the requested glTF attribute unique ids.
pub struct DracoMesh {
    pub indices: Vec<u32>,
    /// `(unique_id, components_per_element, dequantized values)`.
    pub attributes: Vec<(u32, usize, Vec<f32>)>,
}

#[cfg(target_arch = "wasm32")]
pub async fn decode(compressed: &[u8], attr_ids: &[u32]) -> Result<DracoMesh, DecodeError> {
    use wasm_bindgen::JsValue;
    use wasm_bindgen_futures::JsFuture;

    fn err(label: &'static str) -> impl Fn(JsValue) -> DecodeError {
        move |e| DecodeError::draco(format!("{label}: {e:?}"))
    }

    let window = web_sys::window().ok_or_else(|| DecodeError::draco("no window"))?;
    let func = js_sys::Reflect::get(&window, &JsValue::from_str("__tt_draco_decode"))
        .map_err(err("draco shim lookup"))?;
    if !func.is_function() {
        return Err(DecodeError::draco(
            "__tt_draco_decode shim missing — index.html out of date?",
        ));
    }
    let func = js_sys::Function::from(func);

    let bytes = js_sys::Uint8Array::new_with_length(compressed.len() as u32);
    bytes.copy_from(compressed);
    let ids = js_sys::Uint32Array::new_with_length(attr_ids.len() as u32);
    ids.copy_from(attr_ids);

    let promise: js_sys::Promise = func
        .call2(&JsValue::NULL, &bytes, &ids)
        .map_err(err("draco shim call"))?
        .into();
    let result = JsFuture::from(promise).await.map_err(err("draco decode"))?;

    let get = |obj: &JsValue, key: &str| {
        js_sys::Reflect::get(obj, &JsValue::from_str(key))
            .map_err(|e| DecodeError::draco(format!("draco result missing {key}: {e:?}")))
    };
    let indices = js_sys::Uint32Array::new(&get(&result, "indices")?).to_vec();
    let attrs_js = js_sys::Array::from(&get(&result, "attributes")?);
    let mut attributes = Vec::with_capacity(attrs_js.length() as usize);
    for entry in attrs_js.iter() {
        let id = get(&entry, "id")?
            .as_f64()
            .ok_or_else(|| DecodeError::draco("draco attribute id not a number"))?
            as u32;
        let components = get(&entry, "components")?
            .as_f64()
            .ok_or_else(|| DecodeError::draco("draco attribute components not a number"))?
            as usize;
        let data = js_sys::Float32Array::new(&get(&entry, "data")?).to_vec();
        attributes.push((id, components, data));
    }
    Ok(DracoMesh {
        indices,
        attributes,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn decode(_compressed: &[u8], _attr_ids: &[u32]) -> Result<DracoMesh, DecodeError> {
    Err(DecodeError::draco(
        "Draco-compressed tile content needs the browser decoder — \
         render this tileset in the wasm viewer (native P3DT decode is not supported)",
    ))
}
