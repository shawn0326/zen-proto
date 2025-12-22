pub struct DispatchPreparePass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_args_buffer: wgpu::Buffer,
}

impl DispatchPreparePass {
    pub fn new(device: &wgpu::Device, visible_count_buffer: &wgpu::Buffer) -> Self {
        let dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull.dispatch_args_buffer"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dispatch_prepare.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/dispatch_prepare.wgsl").into(),
            ),
        });

        // dispatch_prepare.wgsl bindings:
        // 0 counters (ro storage)
        // 1 dispatch_args (rw storage)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull.dispatch_prepare_bgl"),
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
            label: Some("cull.dispatch_prepare_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull.dispatch_prepare_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull.dispatch_prepare_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dispatch_args_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            dispatch_args_buffer,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu_profiler::Scope<wgpu::CommandEncoder>) {
        let mut pass = encoder.scoped_compute_pass("DispatchPrepare Pass");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn dispatch_args_buffer(&self) -> &wgpu::Buffer {
        &self.dispatch_args_buffer
    }
}
