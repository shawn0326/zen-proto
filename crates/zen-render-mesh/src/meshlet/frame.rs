use super::gpu_scene::MeshletGpuScene;
use crate::mesh::visibility::HiZPyramidDesc;
use zen_frame_graph::{
    Buffer, BufferDesc, Frame, FrameGraphError, ImportBufferOptions, ImportTextureOptions,
    InitialContents, Texture, TextureDesc, TextureSet, TextureSetDesc, TextureView, UsagePolicy,
};

/// Logical FrameGraph handles for the renderer-owned meshlet scene and work arenas.
///
/// Native bindless textures deliberately do not appear here one-by-one. `textures` represents the
/// complete, read-only residency table and is registered exactly once per frame.
pub(crate) struct MeshletGraphResources<'frame> {
    pub vertices: Buffer<'frame>,
    pub meshes: Buffer<'frame>,
    pub lods: Buffer<'frame>,
    pub meshlets: Buffer<'frame>,
    pub meshlet_vertices: Buffer<'frame>,
    pub micro_indices: Buffer<'frame>,
    pub fallback_indices: Buffer<'frame>,
    pub instances: Buffer<'frame>,
    pub materials: Buffer<'frame>,

    pub classifications: Buffer<'frame>,
    pub scan_blocks: Buffer<'frame>,
    pub lod_history: Buffer<'frame>,
    pub candidates: Buffer<'frame>,
    pub visible: Buffer<'frame>,
    pub draw_args: Buffer<'frame>,
    pub counters: Buffer<'frame>,
    pub backend_work_counts: Buffer<'frame>,
    pub mesh_dispatch: Buffer<'frame>,
    pub task_dispatch: Buffer<'frame>,
    pub candidate_dispatch: Buffer<'frame>,
    pub frame_uniform: Buffer<'frame>,
    pub coarse_frame_uniform: Buffer<'frame>,
    pub raster_uniform: Buffer<'frame>,

    pub textures: TextureSet<'frame>,
    pub dummy_hiz: Texture<'frame>,
    pub hiz: Texture<'frame>,
    pub hiz_views: Vec<TextureView<'frame>>,
    pub readback: Option<Buffer<'frame>>,
}

impl<'frame> MeshletGraphResources<'frame> {
    pub(crate) fn register(
        frame: &mut Frame<'frame>,
        scene: &MeshletGpuScene,
        dummy_hiz: &wgpu::Texture,
        bindless_texture_count: u32,
        pyramid: HiZPyramidDesc,
        readback: Option<&wgpu::Buffer>,
    ) -> Result<Self, FrameGraphError> {
        let vertices = import_buffer(frame, "meshlet.vertices", &scene.vertices, true)?;
        let meshes = import_buffer(frame, "meshlet.meshes", &scene.meshes, true)?;
        let lods = import_buffer(frame, "meshlet.lods", &scene.lods, true)?;
        let meshlets = import_buffer(frame, "meshlet.meshlets", &scene.meshlets, true)?;
        let meshlet_vertices = import_buffer(
            frame,
            "meshlet.meshlet-vertices",
            &scene.meshlet_vertices,
            true,
        )?;
        let micro_indices =
            import_buffer(frame, "meshlet.micro-indices", &scene.micro_indices, true)?;
        let fallback_indices = import_buffer(
            frame,
            "meshlet.fallback-indices",
            &scene.fallback_indices,
            true,
        )?;
        let instances = import_buffer(frame, "meshlet.instances", &scene.instances, true)?;
        let materials = import_buffer(frame, "meshlet.materials", &scene.materials, true)?;

        let classifications = import_buffer(
            frame,
            "meshlet.classifications",
            &scene.classifications,
            false,
        )?;
        // Prefix-scan scratch is fully rewritten for the active block range every frame. Native
        // buffers start zeroed, so preserving the unused capacity is valid and avoids claiming a
        // whole-buffer overwrite when the scene contains fewer than max_instances.
        let scan_blocks = import_buffer(
            frame,
            "meshlet.prefix-scan-blocks",
            &scene.scan_blocks,
            true,
        )?;
        // Native wgpu buffers are zero-initialized. Mark history defined so it can persist between
        // frames without an initialization-only graph branch.
        let lod_history = import_buffer(frame, "meshlet.lod-history", &scene.lod_history, true)?;
        let candidates = import_buffer(frame, "meshlet.candidates", &scene.candidates, false)?;
        let visible = import_buffer(frame, "meshlet.visible-work", &scene.visible, false)?;
        let draw_args = import_buffer(frame, "meshlet.draw-args", &scene.draw_args, false)?;
        let counters = import_buffer(frame, "meshlet.counters", &scene.counters, false)?;
        let backend_work_counts = import_buffer(
            frame,
            "meshlet.backend-work-counts",
            &scene.backend_work_counts,
            false,
        )?;
        let mesh_dispatch =
            import_buffer(frame, "meshlet.mesh-dispatch", &scene.mesh_dispatch, false)?;
        let task_dispatch =
            import_buffer(frame, "meshlet.task-dispatch", &scene.task_dispatch, false)?;
        let candidate_dispatch = import_buffer(
            frame,
            "meshlet.candidate-dispatch",
            &scene.candidate_dispatch,
            false,
        )?;
        let frame_uniform =
            import_buffer(frame, "meshlet.frame-uniform", &scene.frame_uniform, true)?;
        let coarse_frame_uniform = import_buffer(
            frame,
            "meshlet.coarse-frame-uniform",
            &scene.coarse_frame_uniform,
            true,
        )?;
        let raster_uniform =
            import_buffer(frame, "meshlet.raster-uniform", &scene.raster_uniform, true)?;

        let textures = frame.import_texture_set(TextureSetDesc::new(
            "meshlet.bindless-textures",
            bindless_texture_count,
        ))?;
        let dummy_hiz = import_texture(frame, "meshlet.dummy-hiz", dummy_hiz)?;
        let hiz = frame.create_texture(pyramid.texture_desc())?;
        let hiz_views = (0..pyramid.mip_level_count())
            .map(|mip| frame.create_texture_view(hiz, pyramid.mip_view_desc(mip)))
            .collect::<Result<Vec<_>, _>>()?;
        let readback = readback
            .map(|buffer| import_buffer(frame, "meshlet.stats-readback", buffer, false))
            .transpose()?;

        Ok(Self {
            vertices,
            meshes,
            lods,
            meshlets,
            meshlet_vertices,
            micro_indices,
            fallback_indices,
            instances,
            materials,
            classifications,
            scan_blocks,
            lod_history,
            candidates,
            visible,
            draw_args,
            counters,
            backend_work_counts,
            mesh_dispatch,
            task_dispatch,
            candidate_dispatch,
            frame_uniform,
            coarse_frame_uniform,
            raster_uniform,
            textures,
            dummy_hiz,
            hiz,
            hiz_views,
            readback,
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

fn import_texture<'frame>(
    frame: &mut Frame<'frame>,
    label: impl Into<String>,
    native: &wgpu::Texture,
) -> Result<Texture<'frame>, FrameGraphError> {
    let handle = frame.import_texture(
        TextureDesc {
            label: label.into(),
            size: native.size(),
            mip_level_count: native.mip_level_count(),
            sample_count: native.sample_count(),
            dimension: native.dimension(),
            format: native.format(),
            view_formats: Vec::new(),
            usage: UsagePolicy::Fixed(native.usage()),
        },
        ImportTextureOptions {
            initial_contents: InitialContents::Defined,
            exposed_usage: Some(native.usage()),
        },
    )?;
    frame.bind_imported_texture(handle, native)?;
    Ok(handle)
}
