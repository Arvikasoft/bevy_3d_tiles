// Hashed-dither LOD cross-fade for 3D-Tiles meshes (BEVY-3D-TILES skip-LOD
// prototype). An `ExtendedMaterial<StandardMaterial, TileDither>` fragment:
// keep a `fade` fraction of pixels via an interleaved-gradient-noise hash, so a
// tile dissolves in (fade 0→1) / out (1→0) instead of popping. Structure mirrors
// Bevy 0.18's official `extended_material.wgsl` so the PBR path stays standard.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

struct TileDither {
    fade: f32,
}

@group(2) @binding(100)
var<uniform> tile_dither: TileDither;

// Jorge Jimenez interleaved-gradient noise — a fine, animation-stable hash over
// screen-space pixel coordinates. Returns [0, 1).
fn ign(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Fully opaque tiles skip the hash entirely (full quality, no edge cost);
    // mid-transition, keep the pixels whose hash falls under `fade`.
    if tile_dither.fade < 1.0 && ign(in.position.xy) >= tile_dither.fade {
        discard;
    }

    // Standard StandardMaterial → PbrInput → shading (unchanged from base).
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
