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

/// Seconds for a full 0↔1 dissolve.
const FADE_SECS: f32 = 0.28;

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
    fn prepass_fragment_shader() -> ShaderRef {
        // Same shader handles the prepass branch via `#ifdef PREPASS_PIPELINE`,
        // so the discard applies there too (no depth halo from dithered pixels).
        "embedded://bevy_client/plugins/tiles3d/dither.wgsl".into()
    }
}

/// Per-tile-root fade state. `drive_tiles3d` writes `target`; [`tick_tile_fade`]
/// eases `current` toward it, writes it into the child mesh materials, and owns
/// the tile's `Visibility` (hidden once fully faded out).
#[derive(Component, Debug)]
pub struct TileFade {
    pub current: f32,
    pub target: f32,
}

impl Default for TileFade {
    fn default() -> Self {
        // New tiles start invisible and dissolve in when the cut wants them.
        Self { current: 0.0, target: 0.0 }
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
    let step = (time.delta_secs() / FADE_SECS).max(0.0);
    let mut animating = false;
    for (mut fade, children, mut vis) in &mut tiles {
        if (fade.current - fade.target).abs() <= f32::EPSILON {
            continue;
        }
        animating = true;
        fade.current = if fade.current < fade.target {
            (fade.current + step).min(fade.target)
        } else {
            (fade.current - step).max(fade.target)
        };
        let f = fade.current;
        for &child in children {
            if let Ok(handle) = mesh_mats.get(child)
                && let Some(mat) = materials.get_mut(&handle.0)
            {
                mat.extension.fade = f;
            }
        }
        // Skip drawing a fully-dissolved tile; reveal as soon as it has pixels.
        let want = if f > 0.0 { Visibility::Visible } else { Visibility::Hidden };
        if *vis != want {
            *vis = want;
        }
    }
    if animating {
        redraw.write(RequestRedraw);
    }
}
