//! GPU side of [`crate::HiddenTileFeatures`]: a queued index buffer is written
//! straight into the mesh's slot in bevy's `MeshAllocator` index slab with one
//! `RenderQueue::write_buffer`. No mesh asset is touched, nothing re-extracts,
//! no vertex data moves.
//!
//! Ordering is the whole design. [`write_tile_indices`] runs in
//! `RenderSystems::PrepareAssets` AFTER `allocate_and_free_meshes` (this
//! frame's uploads have their slab slots and their copies are already queued,
//! so a cut written now lands on top of the pristine upload in the same submit:
//! a fresh tile is never drawn uncut) and BEFORE `prepare_assets::<RenderMesh>`
//! (which drains `ExtractedAssets.removed`/`.extracted`, the two lists the
//! bookkeeping needs).

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::mesh::RenderMesh;
use bevy::render::mesh::allocator::{MeshAllocator, allocate_and_free_meshes};
use bevy::render::render_asset::{ExtractedAssets, prepare_assets};
use bevy::render::renderer::RenderQueue;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};

/// Index buffers to write this frame: `(mesh, indices, reapply)`. Filled by
/// `apply_feature_visibility`, cleared in `First`. `reapply` is set only for a
/// CUT on a `main_world_meshes` set: those meshes still hold pristine data in
/// the main world, and a re-extract uploads it again, so the cut is re-written
/// after one. A `RENDER_WORLD` mesh can never re-extract with data.
#[derive(Resource, Default)]
pub(crate) struct TileIndexQueue(pub(crate) Vec<IndexWrite>);

/// `(mesh, indices, reapply)` — see [`TileIndexQueue`].
pub(crate) type IndexWrite = (AssetId<Mesh>, Arc<[u32]>, bool);

pub(crate) fn clear_tile_index_queue(mut queue: ResMut<TileIndexQueue>) {
    queue.0.clear();
}

/// Render-world state: writes not yet applied, and the cuts to re-apply when a
/// main-world mesh re-uploads.
#[derive(Resource, Default)]
pub(crate) struct TileIndexState {
    pending: HashMap<AssetId<Mesh>, Arc<[u32]>>,
    reapply: HashMap<AssetId<Mesh>, Arc<[u32]>>,
}

/// Wire the render half. Called from `Tiles3dPlugin::finish`, so it does not
/// depend on plugin order; absent `RenderApp` (headless) it does nothing.
pub(crate) fn register_render(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<TileIndexState>()
        .add_systems(ExtractSchedule, extract_tile_indices)
        .add_systems(Render, write_system_config());
}

fn write_system_config() -> bevy::ecs::schedule::ScheduleConfigs<bevy::ecs::system::ScheduleSystem>
{
    write_tile_indices
        .in_set(RenderSystems::PrepareAssets)
        .after(allocate_and_free_meshes)
        .before(prepare_assets::<RenderMesh>)
}

fn extract_tile_indices(queue: Extract<Res<TileIndexQueue>>, mut state: ResMut<TileIndexState>) {
    state.record(&queue.0);
}

fn write_tile_indices(
    extracted: Res<ExtractedAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    render_queue: Res<RenderQueue>,
    mut state: ResMut<TileIndexState>,
) {
    if state.pending.is_empty() && state.reapply.is_empty() {
        return;
    }
    state.flush(
        &extracted,
        |id| allocator.mesh_index_slice(id).map(|s| s.range.len()),
        |id, indices| {
            if let Some(slice) = allocator.mesh_index_slice(id) {
                // Tile meshes are U32 by construction (`build_tile_cache`), so an
                // element is 4 bytes.
                render_queue.write_buffer(
                    slice.buffer,
                    u64::from(slice.range.start) * 4,
                    bytemuck::cast_slice(indices),
                );
            }
        },
    );
}

impl TileIndexState {
    fn record(&mut self, queued: &[IndexWrite]) {
        for (id, indices, reapply) in queued {
            self.pending.insert(*id, Arc::clone(indices));
            if *reapply {
                self.reapply.insert(*id, Arc::clone(indices));
            } else {
                // Pristine (or a RENDER_WORLD mesh): nothing to restore later.
                self.reapply.remove(id);
            }
        }
    }

    /// The bookkeeping half of [`write_tile_indices`], with the allocator
    /// lookup (`slot_len`: the mesh's index slot length in elements, `None` =
    /// not allocated) and the GPU write as closures.
    fn flush(
        &mut self,
        extracted: &ExtractedAssets<RenderMesh>,
        slot_len: impl Fn(&AssetId<Mesh>) -> Option<usize>,
        mut write: impl FnMut(&AssetId<Mesh>, &[u32]),
    ) {
        // The mesh is gone: its slot is freed (or about to be).
        for id in &extracted.removed {
            self.pending.remove(id);
            self.reapply.remove(id);
        }
        // Re-uploaded with pristine main-world data: cut it again.
        for (id, _) in &extracted.extracted {
            if let Some(cut) = self.reapply.get(id) {
                self.pending.insert(*id, Arc::clone(cut));
            }
        }
        self.pending.retain(|id, indices| match slot_len(id) {
            // Not allocated yet: allocation and this write share a frame and a
            // queue, so the first frame it can be drawn is a cut one.
            None => true,
            Some(n) if n == indices.len() => {
                write(id, indices);
                false
            }
            Some(n) => {
                warn_once!(
                    "tiles3d: tile index slot holds {n} indices, the hidden-feature \
                     rewrite has {}; skipping (and any later mismatch)",
                    indices.len()
                );
                false
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::uuid::Uuid;

    fn id(n: u128) -> AssetId<Mesh> {
        AssetId::Uuid {
            uuid: Uuid::from_u128(n),
        }
    }

    fn arc(v: &[u32]) -> Arc<[u32]> {
        v.into()
    }

    fn empty_mesh() -> Mesh {
        Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        )
    }

    /// Runs one flush with every listed mesh allocated at `len` indices;
    /// returns what was written.
    fn flush(
        state: &mut TileIndexState,
        extracted: &ExtractedAssets<RenderMesh>,
        allocated: &[(AssetId<Mesh>, usize)],
    ) -> Vec<(AssetId<Mesh>, Vec<u32>)> {
        let mut written = Vec::new();
        state.flush(
            extracted,
            |m| allocated.iter().find(|(a, _)| a == m).map(|(_, n)| *n),
            |m, ix| written.push((*m, ix.to_vec())),
        );
        written
    }

    #[test]
    fn writes_wait_for_allocation() {
        let mut state = TileIndexState::default();
        let none = ExtractedAssets::<RenderMesh>::default();
        state.record(&[(id(1), arc(&[0, 0, 0]), false)]);
        assert!(flush(&mut state, &none, &[]).is_empty(), "no slot yet");
        assert!(state.pending.contains_key(&id(1)), "kept pending");
        let written = flush(&mut state, &none, &[(id(1), 3)]);
        assert_eq!(written, vec![(id(1), vec![0, 0, 0])]);
        assert!(state.pending.is_empty(), "written once");
        assert!(flush(&mut state, &none, &[(id(1), 3)]).is_empty());
    }

    #[test]
    fn length_mismatch_is_skipped() {
        let mut state = TileIndexState::default();
        let none = ExtractedAssets::<RenderMesh>::default();
        state.record(&[(id(1), arc(&[0, 0, 0]), false)]);
        assert!(flush(&mut state, &none, &[(id(1), 6)]).is_empty());
        assert!(state.pending.is_empty(), "dropped, not retried forever");
    }

    #[test]
    fn state_dropped_when_mesh_removed() {
        let mut state = TileIndexState::default();
        state.record(&[(id(1), arc(&[0, 0, 0]), true)]);
        let mut extracted = ExtractedAssets::<RenderMesh>::default();
        extracted.removed.insert(id(1));
        assert!(flush(&mut state, &extracted, &[(id(1), 3)]).is_empty());
        assert!(state.pending.is_empty() && state.reapply.is_empty());
    }

    #[test]
    fn reapply_requeued_when_mesh_re_extracted() {
        let mut state = TileIndexState::default();
        let none = ExtractedAssets::<RenderMesh>::default();
        state.record(&[(id(1), arc(&[0, 0, 0, 3, 4, 5]), true)]);
        assert_eq!(flush(&mut state, &none, &[(id(1), 6)]).len(), 1);
        // A main-world mesh re-uploads pristine data: the cut goes back on top.
        let mut extracted = ExtractedAssets::<RenderMesh>::default();
        extracted.extracted.push((id(1), empty_mesh()));
        let written = flush(&mut state, &extracted, &[(id(1), 6)]);
        assert_eq!(written, vec![(id(1), vec![0, 0, 0, 3, 4, 5])]);
        // Un-hidden (pristine queued without reapply): nothing to restore.
        state.record(&[(id(1), arc(&[0, 1, 2, 3, 4, 5]), false)]);
        flush(&mut state, &none, &[(id(1), 6)]);
        assert!(flush(&mut state, &extracted, &[(id(1), 6)]).is_empty());
    }

    #[test]
    fn render_world_set_keeps_no_reapply_copy() {
        let mut state = TileIndexState::default();
        state.record(&[(id(1), arc(&[0, 0, 0]), false)]);
        assert!(state.reapply.is_empty());
        let mut extracted = ExtractedAssets::<RenderMesh>::default();
        extracted.extracted.push((id(1), empty_mesh()));
        assert_eq!(flush(&mut state, &extracted, &[(id(1), 3)]).len(), 1);
        assert!(
            flush(&mut state, &extracted, &[(id(1), 3)]).is_empty(),
            "nothing re-queued on a later extraction"
        );
    }

    /// The order that makes the design work: after the allocator (slots exist,
    /// this frame's uploads are queued first) and before `prepare_assets`
    /// (which drains the `removed`/`extracted` lists the bookkeeping reads).
    #[test]
    fn write_tile_indices_runs_between_allocation_and_prepare() {
        use bevy::ecs::system::System;
        use std::any::TypeId;
        fn type_of<M>(s: impl IntoSystem<(), (), M>) -> TypeId {
            IntoSystem::into_system(s).system_type()
        }
        let mut world = World::new();
        let mut schedule = bevy::ecs::schedule::Schedule::new(Render);
        // As bevy registers them (render_asset.rs / mesh/allocator.rs), plus ours.
        schedule.add_systems((
            prepare_assets::<RenderMesh>.in_set(RenderSystems::PrepareAssets),
            write_system_config(),
            allocate_and_free_meshes
                .in_set(RenderSystems::PrepareAssets)
                .before(prepare_assets::<RenderMesh>),
        ));
        schedule.initialize(&mut world).expect("schedule builds");
        let order: Vec<TypeId> = schedule
            .systems()
            .unwrap()
            .map(|(_, s)| s.system_type())
            .collect();
        let at = |t: TypeId| order.iter().position(|o| *o == t).expect("scheduled");
        let alloc = at(type_of(allocate_and_free_meshes));
        let ours = at(type_of(write_tile_indices));
        let prepare = at(type_of(prepare_assets::<RenderMesh>));
        assert!(alloc < ours, "after the allocator");
        assert!(ours < prepare, "before prepare_assets drains the lists");
    }
}
