use super::{config::MeshletBackend, gpu_scene::MeshletGpuScene};
use crate::mesh::visibility::HiZPyramidDesc;
use zen_frame_graph::{
    Buffer, BufferDesc, Frame, FrameGraphError, ImportBufferOptions, InitialContents, Texture,
    TextureView, UsagePolicy,
};

/// Logical FrameGraph handles for meshlet work passed between graph nodes.
pub(crate) struct MeshletGraphResources<'frame> {
    pub classifications: Buffer<'frame>,
    pub candidates: Buffer<'frame>,
    pub visible: Buffer<'frame>,
    pub draw_args: Buffer<'frame>,
    pub counters: Buffer<'frame>,
    pub backend: BackendWorkHandles<'frame>,
    pub candidate_dispatch: Buffer<'frame>,

    pub hiz: Option<MeshletHiZHandles<'frame>>,
    pub readback: Option<Buffer<'frame>>,
}

#[derive(Clone, Copy)]
pub(crate) enum BackendWorkHandles<'frame> {
    IndexedIndirect,
    MeshOnly {
        work_counts: Buffer<'frame>,
        dispatch: Buffer<'frame>,
    },
    TaskMesh {
        work_counts: Buffer<'frame>,
        dispatch: Buffer<'frame>,
    },
}

pub(crate) struct MeshletHiZHandles<'frame> {
    pub texture: Texture<'frame>,
    pub views: Vec<TextureView<'frame>>,
}

impl<'frame> MeshletGraphResources<'frame> {
    pub(crate) fn register(
        frame: &mut Frame<'frame>,
        scene: &MeshletGpuScene,
        backend: MeshletBackend,
        pyramid: Option<HiZPyramidDesc>,
        readback: Option<&wgpu::Buffer>,
    ) -> Result<Self, FrameGraphError> {
        let classifications = import_buffer(
            frame,
            "meshlet.classifications",
            &scene.classifications,
            false,
        )?;
        let candidates = import_buffer(frame, "meshlet.candidates", &scene.candidates, false)?;
        let visible = import_buffer(frame, "meshlet.visible-work", &scene.visible, false)?;
        let draw_args = import_buffer(frame, "meshlet.draw-args", &scene.draw_args, false)?;
        let counters = import_buffer(frame, "meshlet.counters", &scene.counters, false)?;
        let backend = match backend {
            MeshletBackend::IndexedIndirect => BackendWorkHandles::IndexedIndirect,
            MeshletBackend::MeshOnly => BackendWorkHandles::MeshOnly {
                work_counts: import_buffer(
                    frame,
                    "meshlet.backend-work-counts",
                    &scene.backend_work_counts,
                    false,
                )?,
                dispatch: import_buffer(
                    frame,
                    "meshlet.mesh-dispatch",
                    &scene.mesh_dispatch,
                    false,
                )?,
            },
            MeshletBackend::TaskMesh => BackendWorkHandles::TaskMesh {
                work_counts: import_buffer(
                    frame,
                    "meshlet.backend-work-counts",
                    &scene.backend_work_counts,
                    false,
                )?,
                dispatch: import_buffer(
                    frame,
                    "meshlet.task-dispatch",
                    &scene.task_dispatch,
                    false,
                )?,
            },
            MeshletBackend::Auto => unreachable!("renderer stores a resolved backend"),
        };
        let candidate_dispatch = import_buffer(
            frame,
            "meshlet.candidate-dispatch",
            &scene.candidate_dispatch,
            false,
        )?;
        let hiz = pyramid
            .map(|pyramid| {
                let texture = frame.create_texture(pyramid.texture_desc())?;
                let views = (0..pyramid.mip_level_count())
                    .map(|mip| frame.create_texture_view(texture, pyramid.mip_view_desc(mip)))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(MeshletHiZHandles { texture, views })
            })
            .transpose()?;
        let readback = readback
            .map(|buffer| import_buffer(frame, "meshlet.stats-readback", buffer, false))
            .transpose()?;

        Ok(Self {
            classifications,
            candidates,
            visible,
            draw_args,
            counters,
            backend,
            candidate_dispatch,
            hiz,
            readback,
        })
    }

    pub(crate) fn hiz(&self) -> Result<&MeshletHiZHandles<'frame>, FrameGraphError> {
        self.hiz.as_ref().ok_or_else(|| FrameGraphError::Internal {
            message: "meshlet Hi-Z resources requested while occlusion culling is disabled".into(),
        })
    }
}

fn import_buffer<'frame>(
    frame: &mut Frame<'frame>,
    label: impl Into<String>,
    native: &wgpu::Buffer,
    defined: bool,
) -> Result<Buffer<'frame>, FrameGraphError> {
    let handle = frame.import_buffer(
        BufferDesc {
            label: label.into(),
            size: native.size(),
            usage: UsagePolicy::Fixed(native.usage()),
        },
        ImportBufferOptions {
            initial_contents: if defined {
                InitialContents::Defined
            } else {
                InitialContents::Undefined
            },
            exposed_usage: Some(native.usage()),
        },
    )?;
    frame.bind_imported_buffer(handle, native)?;
    Ok(handle)
}
