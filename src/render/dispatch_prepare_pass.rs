use crate::render::visibility_list::VisibilityList;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct DispatchPreparePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_cache: RefCell<HashMap<u64, wgpu::BindGroup>>,
}

impl DispatchPreparePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dispatch_prepare.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dispatch_prepare.wgsl").into(),
            ),
        });

        // dispatch_prepare.wgsl bindings:
        // 0 counters (ro storage)
        // 1 dispatch_args (rw storage)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dispatch_prepare.bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dispatch_prepare.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dispatch_prepare.pipeline"),
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
        }
    }

    pub fn prepare(&self, device: &wgpu::Device, list: &VisibilityList) {
        let mut cache = self.bind_group_cache.borrow_mut();
        cache.entry(list.id()).or_insert_with(|| {
            let label = format!("dispatch_prepare.{}.bind_group", list.label());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&label),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: list.visible_count_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: list.dispatch_args_buffer().as_entire_binding(),
                    },
                ],
            })
        });
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>,
        list: &VisibilityList,
    ) {
        let bind_group_cache = self.bind_group_cache.borrow();
        let bind_group = bind_group_cache
            .get(&list.id())
            .expect("DispatchPreparePass: missing bind group; call prepare() before encode()");

        let mut pass = encoder.scoped_compute_pass("dispatch_prepare.pass");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
