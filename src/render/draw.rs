use crate::camera::Camera;
use crate::primitive::Primitive;
use crate::render::PrimitivesContext;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    // 为了 WGSL/std430 对齐简单，使用 vec4 存
    pub position: glam::Vec4,
    pub color: glam::Vec4,
}

pub struct DrawResources {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub index_buffer: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub camera_buffer: wgpu::Buffer,
}

impl DrawResources {
    pub fn update_camera_buffer(&self, queue: &wgpu::Queue, camera: &Camera) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&camera.view_projection()),
        );
    }
}

pub fn create_draw_resources(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    primitives: &PrimitivesContext,
) -> DrawResources {
    use wgpu::util::DeviceExt;

    // 先固定一个三角形：position + color
    let vertices: [Vertex; 3] = [
        Vertex {
            position: glam::Vec4::new(-0.5, -0.5, 0.0, 1.0),
            color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.5, -0.5, 0.0, 1.0),
            color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
        },
        Vertex {
            position: glam::Vec4::new(0.0, 0.5, 0.0, 1.0),
            color: glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        },
    ];

    let indices: [u16; 3] = [0, 1, 2];

    let vertex_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("draw.vertex_storage"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("draw.index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // camera（先做成静态的，和 cull 里保持一致）
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("draw.camera_buffer"),
        size: 2 * std::mem::size_of::<glam::Mat4>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("draw.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/draw.wgsl").into()),
    });

    // 绑定：顶点(storage)、实例(storage)、可见列表(storage)
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("draw.bgl"),
        entries: &[
            // vertices
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<Vertex>() as u64).unwrap(),
                    ),
                },
                count: None,
            },
            // instances (InstanceData)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<Primitive>() as u64).unwrap(),
                    ),
                },
                count: None,
            },
            // camera
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<glam::Mat4>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("draw.bind_group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_storage.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: primitives.instance_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: camera_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("draw.pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("draw.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            // 不使用 vertex attributes
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview: None,
        cache: None,
    });

    DrawResources {
        pipeline,
        bind_group,
        index_buffer,
        index_format: wgpu::IndexFormat::Uint16,
        camera_buffer,
    }
}
