use crate::camera::Camera;
use crate::mesh::{
    frame::{MeshGraphResources, MeshRenderTargets, VisibilityListHandles},
    scene::{Instance, Material, MeshGpuScene, VertexPacked},
    visibility::VisibilityList,
};
use zen_frame_graph::{
    BufferRange, ColorAttachmentOps, DepthAttachmentOps, Frame, FrameGraphError,
};

const UNIFORM_SIZE_BYTES: u32 = 256;
const MAX_UNIFORM_COUNT: u32 = 2;

pub struct MeshDrawPass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
}

impl MeshDrawPass {
    pub(crate) fn camera_buffer(&self) -> &wgpu::Buffer {
        &self.camera_buffer
    }

    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        resources: &MeshGpuScene,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw.camera.buffer"),
            size: (UNIFORM_SIZE_BYTES * MAX_UNIFORM_COUNT) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("draw.shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/mesh/draw.wgsl").into(),
            ),
        });

        // 绑定：顶点(storage)、实例(storage)、可见列表(storage)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw.bind_group_layout"),
            entries: &[
                // vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<VertexPacked>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // materials (MaterialData)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<Material>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // instances (InstanceData)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<Instance>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(UNIFORM_SIZE_BYTES as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw.bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.meshes().vertex_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.materials().material_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.instances().instance_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &camera_buffer,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(UNIFORM_SIZE_BYTES as u64).unwrap()),
                    }),
                },
            ],
        });

        let texture_storage = resources.textures();
        let max_texture_count = texture_storage
            .max_texture_count()
            .min(device.limits().max_binding_array_elements_per_shader_stage);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("draw.textures.bindless_bgl"),
                entries: &[
                    // bindless textures: texture_2d<f32> textures[]
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: std::num::NonZeroU32::new(max_texture_count),
                    },
                    // shared sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let view_refs: Vec<&wgpu::TextureView> = texture_storage.texture_views().iter().collect();
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw.textures.bindless_bg"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&view_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(texture_storage.sampler()),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("draw.pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&texture_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("draw.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
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
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            texture_bind_group,
            camera_buffer,
        }
    }
}

impl MeshDrawPass {
    pub fn update(&self, queue: &wgpu::Queue, camera: &Camera, offset: u64) {
        queue.write_buffer(
            &self.camera_buffer,
            offset * UNIFORM_SIZE_BYTES as u64,
            bytemuck::bytes_of(&camera.view_projection()),
        );
    }

    pub fn encode(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        index_buffer: &wgpu::Buffer,
        list: &VisibilityList,
        max_count: u32,
        offset: u32,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[offset * UNIFORM_SIZE_BYTES]);
        render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        render_pass.multi_draw_indexed_indirect_count(
            list.draw_args_buffer(),
            0,
            list.draw_count_buffer(),
            0,
            max_count,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a draw node connects its explicit logical and native inputs"
    )]
    pub(crate) fn record<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        label: impl Into<String>,
        targets: MeshRenderTargets<'frame>,
        resources: &MeshGraphResources<'frame>,
        handles: VisibilityListHandles<'frame>,
        list: &'frame VisibilityList,
        scene: &'frame MeshGpuScene,
        max_instance_count: u32,
        camera_uniform_index: u32,
        clear: bool,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.render_pass(label);
        pass.set_side_effect(false);
        for buffer in [resources.vertices, resources.materials, resources.instances] {
            let _ = pass.storage_buffer_read(buffer, BufferRange::whole())?;
        }
        let _ = pass.uniform_buffer(resources.camera_uniform, BufferRange::whole())?;
        for texture in &resources.scene_textures {
            let _ = pass.sampled_texture(*texture)?;
        }
        let _ = pass.index_buffer(resources.indices, BufferRange::whole())?;
        let _ = pass.indirect_buffer(handles.draw_args, BufferRange::whole())?;
        let _ = pass.indirect_buffer(handles.draw_count, BufferRange::whole())?;

        let color_ops = if clear {
            ColorAttachmentOps::clear_store(wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            })
        } else {
            ColorAttachmentOps::load_store()
        };
        let depth_ops = if clear {
            DepthAttachmentOps::clear_store(1.0)
        } else {
            DepthAttachmentOps::load_store()
        };
        let _ = pass.color_attachment(targets.color, color_ops)?;
        let _ = pass.depth_attachment(targets.depth, depth_ops)?;

        pass.finish_render(move |mut context| {
            self.encode(
                &mut context.pass,
                scene.meshes().index_buffer(),
                list,
                max_instance_count,
                camera_uniform_index,
            );
            Ok(())
        })?;
        Ok(())
    }
}
