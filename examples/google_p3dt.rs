//! Google Photorealistic 3D Tiles — live, sessioned, attributed.
//!
//! ```sh
//! GOOGLE_MAPS_API_KEY=... cargo run --example google_p3dt              # Stockholm
//! GOOGLE_MAPS_API_KEY=... cargo run --example google_p3dt -- 48.8584 2.2945
//! ```
//!
//! What the crate handles for you (and the Map Tiles API ToS requires):
//! * **Session protocol** — one root fetch per session; the session token is
//!   extracted from the root response and threaded onto every tile request.
//! * **No persistence** — the tile cache (Cache-Storage CAS) is bypassed for
//!   live sources; Google tiles are never written anywhere.
//! * **Budget guardrail** — `daily_request_cap` hard-stops requests
//!   client-side (persisted per UTC day in `localStorage` on wasm).
//! * **Attribution aggregation** — per-tile `asset.copyright` lines land in
//!   the [`TilesetCredits`] resource. **You must display them, plus the
//!   Google logo, whenever tiles are on screen** (on wasm the crate drives a
//!   `#tt-google-logo` DOM overlay; this native example logs the lines —
//!   a real app renders them).

use bevy::prelude::*;
use bevy_3d_tiles::{
    EcefOrigin, P3dtParams, Tiles3dAttach, Tiles3dCamera, Tiles3dPlugin, TilesetCredits, geodesy,
};

#[derive(Resource)]
struct Site {
    lat: f64,
    lon: f64,
    api_key: String,
}

fn main() {
    let api_key = std::env::var("GOOGLE_MAPS_API_KEY")
        .expect("set GOOGLE_MAPS_API_KEY (Map Tiles API enabled)");
    let mut args = std::env::args().skip(1);
    let lat: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(59.3293);
    let lon: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(18.0686);

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Tiles3dPlugin)
        .insert_resource(Site { lat, lon, api_key })
        .add_systems(Startup, setup)
        .add_systems(Update, log_credits)
        .run();
}

fn setup(mut commands: Commands, site: Res<Site>, mut attach: MessageWriter<Tiles3dAttach>) {
    // ECEF tilesets place themselves through this matrix: the ENU frame at
    // the site becomes the Bevy world frame (east +X, up +Y, north −Z).
    commands.insert_resource(EcefOrigin {
        world_from_ecef: Some(geodesy::world_from_ecef(site.lat, site.lon, 0.0)),
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 500.0, 700.0).looking_at(Vec3::ZERO, Vec3::Y),
        Tiles3dCamera,
    ));
    // P3DT geometry is unlit (baked satellite lighting) — no light needed.

    // The anchor only scopes the tileset's lifecycle for ECEF sets (they do
    // not inherit its transform).
    let anchor = commands
        .spawn((Transform::IDENTITY, Visibility::default()))
        .id();
    attach.write(Tiles3dAttach {
        anchor,
        url: "https://tile.googleapis.com/v1/3dtiles/root.json".into(),
        local: Transform::IDENTITY,
        owner_id: None,
        label: "google_p3dt example".into(),
        p3dt: Some(P3dtParams {
            api_key: site.api_key.clone(),
            // Hard client-side stop — protects a demo key from runaway cost.
            daily_request_cap: 2_000,
        }),
        // Per-set SSE override; None = the app-global Tiles3dConfig default.
        sse_threshold_px: None,
    });
}

fn log_credits(credits: Res<TilesetCredits>) {
    if credits.is_changed() && !credits.lines.is_empty() {
        info!(
            "attribution (display in your UI!): {}",
            credits.lines.join(" · ")
        );
    }
}
