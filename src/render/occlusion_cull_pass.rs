use crate::{
    camera::Camera,
    render::{visibility_history::VisibilityHistory, visibility_list::VisibilityList},
    resources::Resources,
};
use std::{cell::RefCell, collections::HashMap};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OcclusionCullUniform {
    view: glam::Mat4,
    proj: glam::Mat4,
    // x=width, y=height, z=bias, w=slack
    screen_bias: [f32; 4],
}

pub struct OcclusionCullPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_cache: RefCell<HashMap<u64, wgpu::BindGroup>>,
    uniform_buffer: wgpu::Buffer,
}

impl OcclusionCullPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_cull.uniform.buffer"),
            size: std::mem::size_of::<OcclusionCullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("occlusion_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/occlusion_cull.wgsl").into(),
            ),
        });

        // occlusion_cull.wgsl bindings:
        // 0 visible_instances (ro storage)
        // 1 instances (ro storage)
        // 2 mesh_table (ro storage)
        // 3 counters (ro storage)
        // 4 visibility_history (rw storage)
        // 5 hiz (sampled r32float, all mips)
        // 6 params (uniform)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion_cull.bind_group_layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("occlusion_cull.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("occlusion_cull.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group_cache: RefCell::new(HashMap::new()),
            uniform_buffer,
        }
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        resources: &Resources,
        visibility_history: &VisibilityHistory,
        hiz_view: &wgpu::TextureView,
        list: &VisibilityList,
    ) {
        let mut cache = self.bind_group_cache.borrow_mut();
        cache.entry(list.id()).or_insert_with(|| {
            let label = format!("occlusion_cull.{}.bind_group", list.label());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: list.visible_instances_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: resources.primitives.instance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: resources.meshes.mesh_table_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: list.visible_count_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: visibility_history.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            })
        });
    }

    pub fn clear_cache(&self) {
        self.bind_group_cache.borrow_mut().clear();
    }

    pub fn update(&self, queue: &wgpu::Queue, camera: &Camera, width: u32, height: u32) {
        let params = OcclusionCullUniform {
            view: camera.view(),
            proj: camera.projection(),
            screen_bias: [width as f32, height as f32, 0.0001, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>,
        list: &VisibilityList,
    ) {
        let bind_group_cache = self.bind_group_cache.borrow();
        let bind_group = bind_group_cache
            .get(&list.id())
            .expect("OcclusionCullPass: missing bind group; call prepare() before encode()");

        let mut pass = encoder.scoped_compute_pass("occlusion_cull.pass");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups_indirect(list.dispatch_args_buffer(), 0);
    }
}
