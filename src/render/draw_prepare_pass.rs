use crate::render::{MeshStorage, PrimitiveStorage};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

pub struct DrawPreparePass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    indirect_args_buffer: wgpu::Buffer,
}

impl DrawPreparePass {
    pub fn new(
        device: &wgpu::Device,
        meshes: &MeshStorage,
        primitives: &PrimitiveStorage,
        visible_instances_buffer: &wgpu::Buffer,
        visible_count_buffer: &wgpu::Buffer,
        history_visibility_buffer: &wgpu::Buffer,
    ) -> Self {
        let instance_count = primitives.instance_count;

        let indirect_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.indirect_args_buffer"),
            size: instance_count as u64 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("draw_prepare.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/draw_prepare.wgsl").into(),
            ),
        });

        // draw_prepare.wgsl bindings:
        // 0 visible_instances (ro storage)
        // 1 instances (ro storage)
        // 2 mesh_table (ro storage)
        // 3 counters (ro storage)
        // 4 indirect_args (rw storage)
        // 5 history_visibility (ro storage)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull.draw_prepare_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull.draw_prepare_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull.draw_prepare_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull.draw_prepare_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visible_instances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: primitives.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meshes.mesh_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: history_visibility_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            indirect_args_buffer,
        }
    }

    pub fn encode_indirect(
        &self,
        encoder: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>,
        dispatch_args_buffer: &wgpu::Buffer,
    ) {
        let mut pass = encoder.scoped_compute_pass("DrawPrepare Pass");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups_indirect(dispatch_args_buffer, 0);
    }

    pub fn indirect_args_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_args_buffer
    }
}
