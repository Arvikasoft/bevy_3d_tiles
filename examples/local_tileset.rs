//! Minimal viewer: stream a 3D Tiles 1.1 tileset — a packed `.3tz` archive or
//! an exploded `tileset.json` — from a local path or URL.
//!
//! ```sh
//! # The bundled demo fixture (3 LOD levels, mixed box/sphere volumes):
//! cargo run --example local_tileset
//! # Any tileset:
//! cargo run --example local_tileset -- path/or/url/to/tileset.json
//! cargo run --example local_tileset -- https://example.com/asset.3tz
//! ```
//!
//! Move the camera by editing `setup` — this example has no controller on
//! purpose (zero extra dependencies); the streamer re-selects tiles whenever
//! the [`Tiles3dCamera`] moves.

use bevy::prelude::*;
use bevy_3d_tiles::{Tiles3dAttach, Tiles3dCamera, Tiles3dPlugin};

#[derive(Resource)]
struct TilesetSource(String);

fn main() {
    let source = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/fixtures/tiles3d-demo/tileset.json".into());
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Tiles3dPlugin)
        .insert_resource(TilesetSource(source))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    source: Res<TilesetSource>,
    mut attach: MessageWriter<Tiles3dAttach>,
) {
    // The camera the streamer computes screen-space error against.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(60.0, 45.0, 90.0).looking_at(Vec3::ZERO, Vec3::Y),
        Tiles3dCamera,
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.4, 0.0)),
    ));

    // Tiles parent under an anchor entity; its transform places the tileset
    // (twin/asset hosts drive this — here the world origin is fine).
    let anchor = commands
        .spawn((Transform::IDENTITY, Visibility::default()))
        .id();
    attach.write(Tiles3dAttach {
        anchor,
        url: source.0.clone(),
        local: Transform::IDENTITY,
        owner_id: None,
        label: "local_tileset example".into(),
        p3dt: None,
    });
}
