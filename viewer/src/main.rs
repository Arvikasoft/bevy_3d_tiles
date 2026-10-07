//! The bevy_3d_tiles web viewer — live at <https://www.arvikasoft.se/bevy-3d-tiles>,
//! and the smallest real integration of the crate: one tileset, one orbit
//! camera, one light. The dataset picker lives in `index.html`; each pick
//! reloads the page with new query params, so this binary holds zero UI.
//!
//! Query params (wasm) / positional args (native):
//! * `tileset` — tileset.json or .3tz URL (default: the bundled fixture)
//! * `lat`, `lon`, `elev` — WGS84 anchor for georeferenced (ECEF) tilesets
//! * `dist` — initial camera distance in metres (default 1500)

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_3d_tiles::{EcefOrigin, Tiles3dAttach, Tiles3dCamera, Tiles3dPlugin, geodesy};

#[derive(Resource)]
struct Params {
    tileset: String,
    lat_lon: Option<(f64, f64)>,
    elev: f64,
    dist: f32,
}

const DEFAULT_DIST: f64 = 1500.0;

#[cfg(target_arch = "wasm32")]
fn params() -> Params {
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    let q = web_sys::UrlSearchParams::new_with_str(&search).ok();
    let get = |k: &str| q.as_ref().and_then(|q| q.get(k)).filter(|v| !v.is_empty());
    let num = |k: &str| get(k).and_then(|v| v.parse::<f64>().ok());
    Params {
        tileset: get("tileset").unwrap_or_else(|| "tiles3d-demo/tileset.json".into()),
        lat_lon: num("lat").zip(num("lon")),
        elev: num("elev").unwrap_or(0.0),
        dist: num("dist").unwrap_or(DEFAULT_DIST) as f32,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn params() -> Params {
    // viewer <tileset-url> [lat lon [elev [dist]]]
    let mut a = std::env::args().skip(1);
    let tileset = a
        .next()
        .unwrap_or_else(|| "../assets/fixtures/tiles3d-demo/tileset.json".into());
    let mut num = || a.next().and_then(|v| v.parse::<f64>().ok());
    Params {
        tileset,
        lat_lon: num().zip(num()),
        elev: num().unwrap_or(0.0),
        dist: num().unwrap_or(DEFAULT_DIST) as f32,
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    App::new()
        .insert_resource(params())
        .insert_resource(ClearColor(Color::srgb(0.85, 0.89, 0.94)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_3d_tiles viewer".into(),
                canvas: Some("#tiles3d-canvas".into()),
                fit_canvas_to_parent: true,
                prevent_default_event_handling: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Tiles3dPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, orbit_camera)
        .run();
}

fn setup(mut commands: Commands, params: Res<Params>, mut attach: MessageWriter<Tiles3dAttach>) {
    // Georeferenced (ECEF) sets place themselves through this matrix: the ENU
    // frame at the picked coordinate becomes the Bevy world (east +X, up +Y).
    if let Some((lat, lon)) = params.lat_lon {
        commands.insert_resource(EcefOrigin {
            world_from_ecef: Some(geodesy::world_from_ecef(lat, lon, params.elev)),
        });
    }
    commands.spawn((
        Camera3d::default(),
        Transform::IDENTITY,
        Tiles3dCamera,
        // Non-LUT tonemapper — the LUT ones need the tonemapping_luts feature
        // (~4 MB of embedded KTX2 in the wasm).
        Tonemapping::ReinhardLuminance,
        AmbientLight {
            brightness: 400.0,
            ..default()
        },
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.4, 0.0)),
    ));
    commands.insert_resource(Orbit {
        target: Vec3::ZERO,
        dist: params.dist,
        yaw: 0.5,
        pitch: 0.7,
    });

    let anchor = commands
        .spawn((Transform::IDENTITY, Visibility::default()))
        .id();
    attach.write(Tiles3dAttach {
        anchor,
        url: params.tileset.clone(),
        local: Transform::IDENTITY,
        owner_id: None,
        label: "web viewer".into(),
        p3dt: None,
        sse_threshold_px: None,
        ..default()
    });
}

/// Left-drag orbits, wheel zooms, right-drag (or shift+drag) pans the target.
#[derive(Resource)]
struct Orbit {
    target: Vec3,
    dist: f32,
    yaw: f32,
    pitch: f32,
}

fn orbit_camera(
    mut orbit: ResMut<Orbit>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut cam: Single<&mut Transform, With<Tiles3dCamera>>,
) {
    let delta: Vec2 = motion.read().map(|m| m.delta).sum();
    let scroll: f32 = wheel
        .read()
        .map(|w| match w.unit {
            MouseScrollUnit::Line => w.y * 40.0,
            MouseScrollUnit::Pixel => w.y,
        })
        .sum();

    let panning = buttons.pressed(MouseButton::Right)
        || (buttons.pressed(MouseButton::Left)
            && (keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight)));
    if panning && delta != Vec2::ZERO {
        let scale = orbit.dist * 0.0015;
        let (right, up) = (cam.right(), cam.up());
        orbit.target += (right * -delta.x + up * delta.y) * scale;
    } else if buttons.pressed(MouseButton::Left) && delta != Vec2::ZERO {
        orbit.yaw -= delta.x * 0.005;
        orbit.pitch = (orbit.pitch + delta.y * 0.005).clamp(0.05, 1.55);
    }
    if scroll != 0.0 {
        orbit.dist = (orbit.dist * (1.0 - scroll * 0.001)).clamp(5.0, 500_000.0);
    }

    let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
    cam.translation = orbit.target + rot * (Vec3::Z * orbit.dist);
    let target = orbit.target;
    cam.look_at(target, Vec3::Y);
}
