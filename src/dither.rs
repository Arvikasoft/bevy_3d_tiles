//! Hashed-dither LOD cross-fade for 3D-Tiles meshes (BEVY-3D-TILES skip-LOD
//! prototype). Each tile mesh renders with
//! `ExtendedMaterial<StandardMaterial, TileDither>`; `drive_tiles3d` sets a
//! per-tile fade *target* (1 = in the render cut, 0 = leaving) and
//! [`tick_tile_fade`] eases the *current* toward it, pushing it into the
//! material so an LOD swap dissolves instead of popping. The dither itself
//! (interleaved-gradient noise + `discard`) lives in `dither.wgsl`.
//!
//! NOTE (prototype): dither alone smooths the *pop*; it does not eliminate the
//! brief z-fight where a coarse parent and its fine children overlap during a
//! transition — that needs LOD-keyed depth bias (the full skip-LOD step).

use bevy::asset::embedded_asset;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use bevy::window::RequestRedraw;

/// Seconds for a tile to dissolve IN.
const FADE_SECS: f32 = 0.28;
/// Seconds a *left* tile stays opaque+visible before hiding — long enough for
/// the incoming finer tiles to finish dissolving over it (so the swap is
/// old→new, never new-over-empty). A touch longer than `FADE_SECS`.
const HOLD_SECS: f32 = 0.34;

/// `StandardMaterial` + a `fade` the render cut animates for dithered LOD
/// transitions. Each tile primitive owns its own instance (as the plain
/// `StandardMaterial` path already did), so the fade is per-tile.
pub type TileMat = ExtendedMaterial<StandardMaterial, TileDither>;

/// The `StandardMaterial` extension carrying the dither fade.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, Default)]
pub struct TileDither {
    /// Fraction of pixels kept: `0` = fully dithered away, `1` = fully opaque.
    #[uniform(100)]
    pub fade: f32,
}

impl MaterialExtension for TileDither {
    fn fragment_shader() -> ShaderRef {
        // Path derived by `embedded_asset!` (module_path! → `bevy_client`,
        // file! → `src/...`, so the hyphenated package dir never enters).
        "embedded://bevy_client/plugins/tiles3d/dither.wgsl".into()
    }
    // NOTE: we deliberately do NOT override `prepass_fragment_shader()`. The
    // depth/shadow prepass keeps the default StandardMaterial shader, so the
    // dither `discard` isn't applied there — a half-dissolved tile still casts
    // a full shadow for the ~0.28 s transition (minor, cosmetic). Overriding it
    // would route the shadow pipeline through the `#ifdef PREPASS_PIPELINE`
    // branch (untested here); not worth a second grey-screen for the prototype.
}

/// Per-tile-root fade state. `drive_tiles3d` writes `wanted` (is this tile in
/// the render cut this frame); [`tick_tile_fade`] dissolves the tile IN while
/// wanted and, once it leaves, keeps it opaque for `hold` seconds before hiding
/// — so a finer tile dissolves over the coarse one it replaces, not over empty.
#[derive(Component, Debug)]
pub struct TileFade {
    /// Dither fade [0,1]. Ramps UP when wanted; **never ramps down** — dithering
    /// a leaving tile out would punch holes to the background (the live finding).
    pub fade: f32,
    /// Whether the render cut wants this tile this frame.
    pub wanted: bool,
    /// Seconds a just-left tile stays opaque+visible before hiding.
    pub hold: f32,
}

impl Default for TileFade {
    fn default() -> Self {
        // New tiles start invisible and dissolve in when the cut wants them.
        Self { fade: 0.0, wanted: false, hold: 0.0 }
    }
}

/// Registers the dithered-tile material + its embedded shader + the fade driver.
/// Added from `lib.rs` (keeps it out of `Tiles3dPlugin::build`).
pub struct Tiles3dDitherPlugin;

impl Plugin for Tiles3dDitherPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "dither.wgsl");
        app.add_plugins(MaterialPlugin::<TileMat>::default())
            .add_systems(Update, tick_tile_fade);
    }
}

/// Ease every transitioning tile's fade toward its target, push it into the
/// child mesh materials, and hide fully-faded-out tiles. Keeps requesting
/// redraws while any dissolve is in flight — reactive winit would otherwise
/// freeze the animation between input events.
fn tick_tile_fade(
    time: Res<Time>,
    mut materials: ResMut<Assets<TileMat>>,
    mut tiles: Query<(&mut TileFade, &Children, &mut Visibility)>,
    mesh_mats: Query<&MeshMaterial3d<TileMat>>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    let dt = time.delta_secs();
    let step = (dt / FADE_SECS).max(0.0);
    let mut animating = false;
    for (mut fade, children, mut vis) in &mut tiles {
        let visible = if fade.wanted {
            // Dissolve in; keep the hold clock charged for when it leaves.
            if fade.fade < 1.0 {
                fade.fade = (fade.fade + step).min(1.0);
                animating = true;
            }
            fade.hold = HOLD_SECS;
            true
        } else if fade.hold > 0.0 {
            // Left the cut: stay opaque (DON'T dither out) to back the finer
            // tile dissolving in over us; hide once it's had time to cover.
            fade.hold -= dt;
            animating = true;
            true
        } else {
            false
        };

        let f = fade.fade;
        for &child in children {
            if let Ok(handle) = mesh_mats.get(child)
                && let Some(mat) = materials.get_mut(&handle.0)
                && mat.extension.fade != f
            {
                mat.extension.fade = f;
            }
        }
        let want = if visible { Visibility::Visible } else { Visibility::Hidden };
        if *vis != want {
            *vis = want;
        }
    }
    if animating {
        redraw.write(RequestRedraw);
    }
}
