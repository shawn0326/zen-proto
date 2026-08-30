use crate::mesh::{
    scene::MeshGpuScene,
    visibility::{HiZPyramidDesc, VisibilityHistory, VisibilityList},
};
use zen_frame_graph::{
    Buffer, BufferDesc, Frame, FrameGraphError, ImportBufferOptions, ImportTextureOptions,
    InitialContents, Texture, TextureDesc, TextureView, UsagePolicy,
};

#[derive(Clone, Copy)]
pub(crate) struct VisibilityListHandles<'frame> {
    pub visible_instances: Buffer<'frame>,
    pub visible_count: Buffer<'frame>,
    pub dispatch_args: Buffer<'frame>,
    pub draw_args: Buffer<'frame>,
    pub draw_count: Buffer<'frame>,
}

#[derive(Clone, Copy)]
pub struct MeshRenderTargets<'frame> {
    pub color: Texture<'frame>,
    pub depth: Texture<'frame>,
}

pub(crate) struct HiZHandles<'frame> {
    pub texture: Texture<'frame>,
    pub views: Vec<TextureView<'frame>>,
}

pub(crate) struct MeshGraphResources<'frame> {
    pub vertices: Buffer<'frame>,
    pub indices: Buffer<'frame>,
    pub mesh_table: Buffer<'frame>,
    pub materials: Buffer<'frame>,
    pub instances: Buffer<'frame>,
    pub scene_textures: Vec<Texture<'frame>>,
    pub main_cull_uniform: Buffer<'frame>,
    pub occlusion_uniform: Buffer<'frame>,
    pub camera_uniform: Buffer<'frame>,
    pub history: Buffer<'frame>,
    pub list_a: VisibilityListHandles<'frame>,
    pub list_b: VisibilityListHandles<'frame>,
    pub hiz: HiZHandles<'frame>,
    pub readback: Option<Buffer<'frame>>,
}

impl<'frame> MeshRenderTargets<'frame> {
    /// Creates Mesh render targets from handles already registered with the caller-owned graph.
    pub fn new(color: Texture<'frame>, depth: Texture<'frame>) -> Self {
        Self { color, depth }
    }
}

impl<'frame> MeshGraphResources<'frame> {
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors MeshRenderer-owned GPU resources"
    )]
    pub(crate) fn register(
        frame: &mut Frame<'frame>,
        resources: &MeshGpuScene,
        list_a: &VisibilityList,
        list_b: &VisibilityList,
        history: &VisibilityHistory,
        hiz: HiZPyramidDesc,
        main_cull_uniform: &wgpu::Buffer,
        occlusion_uniform: &wgpu::Buffer,
        camera_uniform: &wgpu::Buffer,
        readback: Option<&wgpu::Buffer>,
    ) -> Result<Self, FrameGraphError> {
        let meshes = resources.meshes();
        let vertices = import_buffer(frame, "meshes.vertices", meshes.vertex_buffer())?;
        let indices = import_buffer(frame, "meshes.indices", meshes.index_buffer())?;
        let mesh_table = import_buffer(frame, "meshes.table", meshes.mesh_table_buffer())?;
        let materials = import_buffer(frame, "materials", resources.materials().material_buffer())?;
        let instances = import_buffer(frame, "instances", resources.instances().instance_buffer())?;
        let scene_textures = resources
            .textures()
            .textures()
            .iter()
            .enumerate()
            .map(|(index, texture)| {
                import_texture(frame, format!("scene-texture-{index}"), texture, true)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let main_cull_uniform = import_buffer(frame, "main-cull.uniform", main_cull_uniform)?;
        let occlusion_uniform = import_buffer(frame, "occlusion.uniform", occlusion_uniform)?;
        let camera_uniform = import_buffer(frame, "draw.camera", camera_uniform)?;
        let history = import_buffer(frame, "visibility-history", history.buffer())?;
        let list_a = import_visibility_list(frame, list_a)?;
        let list_b = import_visibility_list(frame, list_b)?;
        let hiz_texture = frame.create_texture(hiz.texture_desc())?;
        let hiz_views = (0..hiz.mip_level_count())
            .map(|mip| frame.create_texture_view(hiz_texture, hiz.mip_view_desc(mip)))
            .collect::<Result<Vec<_>, _>>()?;
        let readback = readback
            .map(|buffer| import_buffer_with_contents(frame, "stats-readback", buffer, false))
            .transpose()?;

        Ok(Self {
            vertices,
            indices,
            mesh_table,
            materials,
            instances,
            scene_textures,
            main_cull_uniform,
            occlusion_uniform,
            camera_uniform,
            history,
            list_a,
            list_b,
            hiz: HiZHandles {
                texture: hiz_texture,
                views: hiz_views,
            },
            readback,
        })
    }
}

fn import_visibility_list<'frame>(
    frame: &mut Frame<'frame>,
    list: &VisibilityList,
) -> Result<VisibilityListHandles<'frame>, FrameGraphError> {
    Ok(VisibilityListHandles {
        visible_instances: import_buffer(
            frame,
            format!("{}.visible-instances", list.label()),
            list.visible_instances_buffer(),
        )?,
        visible_count: import_buffer(
            frame,
            format!("{}.visible-count", list.label()),
            list.visible_count_buffer(),
        )?,
        dispatch_args: import_buffer(
            frame,
            format!("{}.dispatch-args", list.label()),
            list.dispatch_args_buffer(),
        )?,
        draw_args: import_buffer(
            frame,
            format!("{}.draw-args", list.label()),
            list.draw_args_buffer(),
        )?,
        draw_count: import_buffer(
            frame,
            format!("{}.draw-count", list.label()),
            list.draw_count_buffer(),
        )?,
    })
}

fn import_buffer<'frame>(
    frame: &mut Frame<'frame>,
    label: impl Into<String>,
    native: &wgpu::Buffer,
) -> Result<Buffer<'frame>, FrameGraphError> {
    import_buffer_with_contents(frame, label, native, true)
}

fn import_buffer_with_contents<'frame>(
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
    defined: bool,
) -> Result<Texture<'frame>, FrameGraphError> {
    let handle = frame.import_texture(
        texture_desc(label, native),
        ImportTextureOptions {
            initial_contents: if defined {
                InitialContents::Defined
            } else {
                InitialContents::Undefined
            },
            exposed_usage: Some(native.usage()),
        },
    )?;
    frame.bind_imported_texture(handle, native)?;
    Ok(handle)
}

fn texture_desc(label: impl Into<String>, native: &wgpu::Texture) -> TextureDesc {
    TextureDesc {
        label: label.into(),
        size: native.size(),
        mip_level_count: native.mip_level_count(),
        sample_count: native.sample_count(),
        dimension: native.dimension(),
        format: native.format(),
        view_formats: vec![],
        usage: UsagePolicy::Fixed(native.usage()),
    }
}
