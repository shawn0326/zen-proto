use std::mem::size_of;

use super::{
    asset::MeshletPsoClass,
    config::MeshletBackend,
    frame::{BackendWorkHandles, MeshletGraphResources},
    gpu_types::GpuCounters,
    renderer::{MeshletRenderer, PreparedMeshletFrame},
};
use crate::{MeshRenderTargets, mesh::visibility::HiZPyramidDesc};
use zen_frame_graph::{
    BufferRange, ClearBufferOp, ColorAttachmentOps, DepthAttachmentOps, Frame, FrameGraphError,
    WriteContents,
};

const BINS: [MeshletPsoClass; 2] = [
    MeshletPsoClass::OpaqueBackface,
    MeshletPsoClass::OpaqueTwoSided,
];

fn clear_coarse_results_before_backend(
    _backend: MeshletBackend,
    enable_occlusion_culling: bool,
) -> bool {
    enable_occlusion_culling
}

fn record_final_compute_cull(_backend: MeshletBackend, enable_occlusion_culling: bool) -> bool {
    enable_occlusion_culling
}

#[derive(Clone, Copy)]
pub(crate) struct MeshletGraphRecorder<'frame> {
    renderer: &'frame MeshletRenderer,
}

impl<'frame> MeshletGraphRecorder<'frame> {
    pub(crate) const fn new(renderer: &'frame MeshletRenderer) -> Self {
        Self { renderer }
    }

    pub(crate) fn record(
        self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        prepared: &PreparedMeshletFrame,
    ) -> Result<(), FrameGraphError> {
        let pyramid = HiZPyramidDesc::new(prepared.extent.width, prepared.extent.height);
        let readback = prepared
            .readback_index
            .map(|index| self.renderer.readback_buffer(index));
        let resources = MeshletGraphResources::register(
            frame,
            self.renderer.scene(),
            self.renderer.backend(),
            prepared.enable_occlusion_culling.then_some(pyramid),
            readback,
        )?;

        frame.clear_buffers(
            "meshlet.clear-frame-counters",
            [ClearBufferOp::new(resources.counters, BufferRange::whole())],
        )?;
        self.record_classify(frame, &resources)?;
        self.record_prefix_scan(frame, &resources)?;
        self.record_candidate_scatter(frame, &resources)?;
        self.record_cull(frame, "meshlet.coarse-cull", &resources, false)?;
        self.record_occluder_depth(frame, targets, &resources)?;

        if prepared.enable_occlusion_culling {
            self.record_hiz(frame, targets, &resources, pyramid)?;
        }
        // Every backend consumes the same compute-produced visible list. Without Hi-Z the coarse
        // result is final; with Hi-Z, reset its visibility/stats fields and produce the final list.
        if clear_coarse_results_before_backend(
            self.renderer.backend(),
            prepared.enable_occlusion_culling,
        ) {
            Self::record_visible_counter_clear(frame, &resources)?;
        }
        if record_final_compute_cull(self.renderer.backend(), prepared.enable_occlusion_culling) {
            self.record_cull(frame, "meshlet.final-cull", &resources, true)?;
        }

        self.record_indirect_prepare(frame, &resources)?;
        self.record_backend_raster(frame, targets, &resources)?;

        let readback = if prepared.readback_index.is_some() {
            Some(self.record_stats_copy(frame, &resources)?)
        } else {
            None
        };
        if let Some(readback) = readback {
            frame.mark_readback(readback, BufferRange::whole())?;
        }
        Ok(())
    }

    fn record_classify(
        self,
        frame: &mut Frame<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass("meshlet.instance-classify-lod-count");
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_write(
            resources.classifications,
            BufferRange::whole(),
            WriteContents::Overwrite,
        )?;
        let _ = pass.storage_buffer_write(
            resources.counters,
            BufferRange::whole(),
            WriteContents::Preserve,
        )?;
        pass.finish_compute(move |mut context| {
            self.renderer.passes().encode_classify(
                context.device,
                &mut context.pass,
                self.renderer.scene(),
                &self.renderer.scene().frame_uniform,
            );
            Ok(())
        })?;
        Ok(())
    }

    fn record_prefix_scan(
        self,
        frame: &mut Frame<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass("meshlet.prefix-scan");
        pass.set_side_effect(false);
        for buffer in [resources.classifications, resources.counters] {
            let _ =
                pass.storage_buffer_write(buffer, BufferRange::whole(), WriteContents::Preserve)?;
        }
        let _ = pass.storage_buffer_write(
            resources.candidate_dispatch,
            BufferRange::whole(),
            WriteContents::Overwrite,
        )?;
        pass.finish_compute(move |mut context| {
            self.renderer.passes().encode_prefix_scan(
                context.device,
                &mut context.pass,
                self.renderer.scene(),
                &self.renderer.scene().frame_uniform,
            );
            Ok(())
        })?;
        Ok(())
    }

    fn record_candidate_scatter(
        self,
        frame: &mut Frame<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass("meshlet.candidate-scatter");
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_read(resources.classifications, BufferRange::whole())?;
        let _ = pass.storage_buffer_write(
            resources.candidates,
            BufferRange::whole(),
            WriteContents::Overwrite,
        )?;
        let _ = pass.storage_buffer_write(
            resources.counters,
            BufferRange::whole(),
            WriteContents::Preserve,
        )?;
        pass.finish_compute(move |mut context| {
            self.renderer.passes().encode_candidate_scatter(
                context.device,
                &mut context.pass,
                self.renderer.scene(),
                &self.renderer.scene().frame_uniform,
            );
            Ok(())
        })?;
        Ok(())
    }

    fn record_cull(
        self,
        frame: &mut Frame<'frame>,
        label: &'static str,
        resources: &MeshletGraphResources<'frame>,
        use_hiz: bool,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass(label);
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_read(resources.candidates, BufferRange::whole())?;
        for buffer in [resources.visible, resources.draw_args] {
            let _ =
                pass.storage_buffer_write(buffer, BufferRange::whole(), WriteContents::Overwrite)?;
        }
        let _ = pass.storage_buffer_write(
            resources.counters,
            BufferRange::whole(),
            WriteContents::Preserve,
        )?;
        let _ = pass.indirect_buffer(resources.candidate_dispatch, BufferRange::whole())?;
        let sampled = use_hiz
            .then(|| pass.sampled_texture(resources.hiz()?.texture))
            .transpose()?;
        pass.finish_compute(move |mut context| {
            let hiz = match sampled {
                Some(sampled) => context.resources.texture_view(sampled)?,
                None => self.renderer.passes().dummy_hiz_view(),
            };
            let uniform = if use_hiz {
                &self.renderer.scene().frame_uniform
            } else {
                &self.renderer.scene().coarse_frame_uniform
            };
            self.renderer.passes().encode_cull(
                context.device,
                &mut context.pass,
                self.renderer.scene(),
                uniform,
                hiz,
            );
            Ok(())
        })?;
        Ok(())
    }

    fn record_occluder_depth(
        self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.render_pass("meshlet.opaque-occluder-depth");
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_read(resources.visible, BufferRange::whole())?;
        let _ = pass.indirect_buffer(resources.draw_args, BufferRange::whole())?;
        let _ = pass.indirect_buffer(resources.counters, BufferRange::whole())?;
        let _ = pass.depth_attachment(targets.depth, DepthAttachmentOps::clear_store(1.0))?;
        pass.finish_render(move |mut context| {
            for bin in BINS {
                self.renderer.passes().encode_indexed_depth(
                    context.device,
                    &mut context.pass,
                    self.renderer.scene(),
                    self.renderer.bindless().bind_group(),
                    bin,
                );
            }
            Ok(())
        })?;
        Ok(())
    }

    fn record_hiz(
        self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        resources: &MeshletGraphResources<'frame>,
        pyramid: HiZPyramidDesc,
    ) -> Result<(), FrameGraphError> {
        let hiz = resources.hiz()?;
        frame.with_debug_group("Meshlet Current-frame Hi-Z", |frame| {
            let mut pass = frame.compute_pass("meshlet.hiz-depth-to-mip0");
            pass.set_side_effect(false);
            let source = pass.sampled_texture(targets.depth)?;
            let destination = pass.storage_texture_write(hiz.views[0], WriteContents::Overwrite)?;
            pass.finish_compute(move |mut context| {
                let source = context.resources.texture_view(source)?;
                let destination = context.resources.texture_view(destination)?;
                self.renderer.hiz_stage().encode_depth_to_mip0(
                    context.device,
                    &mut context.pass,
                    source,
                    destination,
                    pyramid,
                );
                Ok(())
            })?;

            for mip in 1..pyramid.mip_level_count() {
                let mut pass =
                    frame.compute_pass(format!("meshlet.hiz-mip{}-to-mip{mip}", mip - 1));
                pass.set_side_effect(false);
                let source = pass.sampled_texture(hiz.views[(mip - 1) as usize])?;
                let destination =
                    pass.storage_texture_write(hiz.views[mip as usize], WriteContents::Overwrite)?;
                pass.finish_compute(move |mut context| {
                    let source = context.resources.texture_view(source)?;
                    let destination = context.resources.texture_view(destination)?;
                    self.renderer.hiz_stage().encode_mip_to_mip(
                        context.device,
                        &mut context.pass,
                        source,
                        destination,
                        pyramid,
                        mip,
                    );
                    Ok(())
                })?;
            }
            Ok(())
        })
    }

    fn record_visible_counter_clear(
        frame: &mut Frame<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        frame.clear_buffers(
            "meshlet.clear-coarse-results",
            [
                ClearBufferOp::new(
                    resources.counters,
                    BufferRange::new(
                        GpuCounters::VISIBLE_BACKFACE_OFFSET,
                        2 * size_of::<u32>() as u64,
                    ),
                ),
                ClearBufferOp::new(
                    resources.counters,
                    BufferRange::new(
                        std::mem::offset_of!(GpuCounters, culled_frustum) as u64,
                        5 * size_of::<u32>() as u64,
                    ),
                ),
                ClearBufferOp::new(
                    resources.counters,
                    BufferRange::new(
                        GpuCounters::CONSERVATIVELY_VISIBLE_OFFSET,
                        size_of::<u32>() as u64,
                    ),
                ),
            ],
        )?;
        Ok(())
    }

    fn record_indirect_prepare(
        self,
        frame: &mut Frame<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass("meshlet.indirect-prepare");
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_write(
            resources.counters,
            BufferRange::whole(),
            WriteContents::Preserve,
        )?;
        match resources.backend {
            BackendWorkHandles::IndexedIndirect => {}
            BackendWorkHandles::MeshOnly {
                work_counts,
                dispatch,
            }
            | BackendWorkHandles::TaskMesh {
                work_counts,
                dispatch,
            } => {
                let _ = pass.storage_buffer_write(
                    work_counts,
                    BufferRange::whole(),
                    WriteContents::Overwrite,
                )?;
                let _ = pass.storage_buffer_write(
                    dispatch,
                    BufferRange::whole(),
                    WriteContents::Overwrite,
                )?;
            }
        }
        pass.finish_compute(move |mut context| {
            self.renderer.passes().encode_indirect_prepare(
                context.device,
                &mut context.pass,
                self.renderer.scene(),
                &self.renderer.scene().frame_uniform,
            );
            Ok(())
        })?;
        Ok(())
    }

    fn record_backend_raster(
        self,
        frame: &mut Frame<'frame>,
        targets: MeshRenderTargets<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.render_pass(format!(
            "meshlet.backend-raster.{}",
            self.renderer.backend().as_str()
        ));
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_read(resources.visible, BufferRange::whole())?;
        match self.renderer.backend() {
            MeshletBackend::IndexedIndirect => {
                debug_assert!(matches!(
                    resources.backend,
                    BackendWorkHandles::IndexedIndirect
                ));
                let _ = pass.indirect_buffer(resources.draw_args, BufferRange::whole())?;
                let _ = pass.indirect_buffer(resources.counters, BufferRange::whole())?;
            }
            MeshletBackend::MeshOnly => {
                let BackendWorkHandles::MeshOnly {
                    work_counts,
                    dispatch,
                } = resources.backend
                else {
                    unreachable!("meshlet backend handles must match the renderer backend")
                };
                let _ = pass.storage_buffer_read(work_counts, BufferRange::whole())?;
                let _ = pass.indirect_buffer(dispatch, BufferRange::whole())?;
            }
            MeshletBackend::TaskMesh => {
                let BackendWorkHandles::TaskMesh {
                    work_counts,
                    dispatch,
                } = resources.backend
                else {
                    unreachable!("meshlet backend handles must match the renderer backend")
                };
                let _ = pass.storage_buffer_read(work_counts, BufferRange::whole())?;
                let _ = pass.indirect_buffer(dispatch, BufferRange::whole())?;
            }
            MeshletBackend::Auto => unreachable!("renderer stores a resolved backend"),
        }
        let _ = pass.color_attachment(
            targets.color,
            ColorAttachmentOps::clear_store(wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            }),
        )?;
        let _ = pass.depth_attachment(targets.depth, DepthAttachmentOps::load_store())?;
        pass.finish_render(move |mut context| {
            for bin in BINS {
                match self.renderer.backend() {
                    MeshletBackend::IndexedIndirect => {
                        self.renderer.passes().encode_indexed_raster(
                            context.device,
                            &mut context.pass,
                            self.renderer.scene(),
                            self.renderer.bindless().bind_group(),
                            bin,
                        );
                    }
                    MeshletBackend::MeshOnly | MeshletBackend::TaskMesh => {
                        self.renderer.passes().encode_mesh_raster(
                            context.device,
                            &mut context.pass,
                            self.renderer.scene(),
                            self.renderer.bindless().bind_group(),
                            bin,
                        );
                    }
                    MeshletBackend::Auto => unreachable!("renderer stores a resolved backend"),
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    fn record_stats_copy(
        self,
        frame: &mut Frame<'frame>,
        resources: &MeshletGraphResources<'frame>,
    ) -> Result<zen_frame_graph::Buffer<'frame>, FrameGraphError> {
        let destination = resources
            .readback
            .ok_or_else(|| FrameGraphError::Internal {
                message: "meshlet stats copy recorded without a staging buffer".into(),
            })?;
        let mut pass = frame.copy_pass("meshlet.stats-readback");
        pass.set_side_effect(false);
        pass.copy_buffer_to_buffer(
            resources.counters,
            0,
            destination,
            0,
            size_of::<GpuCounters>() as u64,
        )?;
        pass.finish()?;
        Ok(destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backend_reuses_coarse_visibility_or_runs_the_same_final_compute_cull() {
        for backend in [
            MeshletBackend::IndexedIndirect,
            MeshletBackend::MeshOnly,
            MeshletBackend::TaskMesh,
        ] {
            assert!(!clear_coarse_results_before_backend(backend, false));
            assert!(!record_final_compute_cull(backend, false));
            assert!(clear_coarse_results_before_backend(backend, true));
            assert!(record_final_compute_cull(backend, true));
        }
    }
}
