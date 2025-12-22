use crate::{
    camera::Camera,
    render::{MeshStorage, PrimitiveStorage},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrustumUniform {
    // 6 planes in world space: normal.xyz, d (Ax + By + Cz + D = 0)
    pub planes: [glam::Vec4; 6],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullParams {
    pub instance_count: u32,
    pub mesh_count: u32,
    pub enable_occlusion: u32,
    pub _pad1: u32,
}

pub struct MainCullPass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    frustum_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,

    visibility_history_buffer: wgpu::Buffer,

    visible_count_buffer_a: wgpu::Buffer,
    visible_instances_buffer_a: wgpu::Buffer,

    visible_count_buffer_b: wgpu::Buffer,
    visible_instances_buffer_b: wgpu::Buffer,
}

impl MainCullPass {
    pub fn new(device: &wgpu::Device, meshes: &MeshStorage, primitives: &PrimitiveStorage) -> Self {
        use wgpu::util::DeviceExt;

        let instance_count = primitives.instance_count;

        let frustum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.frustum_buffer"),
            size: std::mem::size_of::<FrustumUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = CullParams {
            instance_count,
            mesh_count: meshes.mesh_count,
            enable_occlusion: 1,
            _pad1: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cull.params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let visibility_history_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.visibility_history_buffer"),
            size: instance_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_count_buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.visible_count_buffer_a"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_instances_buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.visible_instances_buffer_a"),
            size: instance_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_count_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.visible_count_buffer_b"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_instances_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.visible_instances_buffer_b"),
            size: instance_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/main_cull.wgsl").into()),
        });

        // main_cull.wgsl bindings:
        // 0 instances (ro storage)
        // 1 mesh_table (ro storage)
        // 2 frustum (uniform)
        // 3 params (uniform)
        // 4 counters (rw storage)
        // 5 visible_instances (rw storage)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull.main_cull_bgl"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull.main_cull_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull.main_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull.main_cull_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: primitives.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meshes.mesh_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frustum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visibility_history_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: visible_count_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: visible_instances_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: visible_count_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: visible_instances_buffer_b.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            frustum_buffer,
            _params_buffer: params_buffer,
            visibility_history_buffer,
            visible_count_buffer_a,
            visible_instances_buffer_a,
            visible_count_buffer_b,
            visible_instances_buffer_b,
        }
    }

    pub fn update_frustum(&self, queue: &wgpu::Queue, camera: &Camera) {
        queue.write_buffer(
            &self.frustum_buffer,
            0,
            bytemuck::bytes_of(&camera.frustum()),
        );
    }

    pub fn reset_visible_count(&self, queue: &wgpu::Queue) {
        let zero4 = [0u8; 4];
        queue.write_buffer(&self.visible_count_buffer_a, 0, &zero4);
        queue.write_buffer(&self.visible_count_buffer_b, 0, &zero4);
    }

    pub fn enable_occlusion_culling(&self, queue: &wgpu::Queue, enable: bool) {
        let flag: u32 = if enable { 1 } else { 0 };
        queue.write_buffer(&self._params_buffer, 8, bytemuck::bytes_of(&flag));
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>,
        instance_count: u32,
    ) {
        let wg_size = 64;
        let group_count = (instance_count + wg_size - 1) / wg_size;

        let mut pass = encoder.scoped_compute_pass("MainCull Pass");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(group_count, 1, 1);
    }

    pub fn visibility_history_buffer(&self) -> &wgpu::Buffer {
        &self.visibility_history_buffer
    }

    pub fn visible_count_buffer_a(&self) -> &wgpu::Buffer {
        &self.visible_count_buffer_a
    }

    pub fn visible_instances_buffer_a(&self) -> &wgpu::Buffer {
        &self.visible_instances_buffer_a
    }

    pub fn visible_count_buffer_b(&self) -> &wgpu::Buffer {
        &self.visible_count_buffer_b
    }

    pub fn visible_instances_buffer_b(&self) -> &wgpu::Buffer {
        &self.visible_instances_buffer_b
    }
}
