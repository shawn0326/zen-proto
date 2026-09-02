use crate::mesh::visibility::{HiZPyramidDesc, VisibilityHistory, VisibilityList};
use zenfg::{
    Buffer, BufferDesc, Frame, FrameGraphError, ImportBufferOptions, InitialContents, Texture,
    TextureView, UsagePolicy,
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
    pub history: Buffer<'frame>,
    pub list_a: VisibilityListHandles<'frame>,
    pub list_b: VisibilityListHandles<'frame>,
    pub hiz: Option<HiZHandles<'frame>>,
    pub readback: Option<Buffer<'frame>>,
}

impl<'frame> MeshRenderTargets<'frame> {
    /// Creates Mesh render targets from handles already registered with the caller-owned graph.
    pub fn new(color: Texture<'frame>, depth: Texture<'frame>) -> Self {
        Self { color, depth }
    }
}

impl<'frame> MeshGraphResources<'frame> {
    pub(crate) fn register(
        frame: &mut Frame<'frame>,
        list_a: &VisibilityList,
        list_b: &VisibilityList,
        history: &VisibilityHistory,
        hiz: Option<HiZPyramidDesc>,
        readback: Option<&wgpu::Buffer>,
    ) -> Result<Self, FrameGraphError> {
        let history = import_buffer(frame, "visibility-history", history.buffer())?;
        let list_a = import_visibility_list(frame, list_a)?;
        let list_b = import_visibility_list(frame, list_b)?;
        let hiz = hiz
            .map(|hiz| {
                let texture = frame.create_texture(hiz.texture_desc())?;
                let views = (0..hiz.mip_level_count())
                    .map(|mip| frame.create_texture_view(texture, hiz.mip_view_desc(mip)))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(HiZHandles { texture, views })
            })
            .transpose()?;
        let readback = readback
            .map(|buffer| import_buffer_with_contents(frame, "stats-readback", buffer, false))
            .transpose()?;

        Ok(Self {
            history,
            list_a,
            list_b,
            hiz,
            readback,
        })
    }

    pub(crate) fn hiz(&self) -> Result<&HiZHandles<'frame>, FrameGraphError> {
        self.hiz.as_ref().ok_or_else(|| FrameGraphError::Internal {
            message: "Hi-Z resources requested while occlusion culling is disabled".into(),
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
