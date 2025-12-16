use crate::camera::Camera;
use crate::mesh::Vertex;
use crate::primitive::Primitive;
use crate::render::{MeshesContext, PrimitivesContext};

pub struct DrawResources {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
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
    meshes: &MeshesContext,
    primitives: &PrimitivesContext,
) -> DrawResources {
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
                resource: meshes.vertex_buffer.as_entire_binding(),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
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
        camera_buffer,
    }
}
