use crate::mesh::{
    frame::{MeshGraphResources, VisibilityListHandles},
    scene::MeshGpuScene,
    visibility::{VisibilityHistory, VisibilityList},
};
use std::cell::RefCell;
use std::collections::HashMap;
use zen_frame_graph::{BufferRange, Frame, FrameGraphError, WriteContents};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

pub struct IndirectDrawPreparePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group_cache: RefCell<HashMap<u64, wgpu::BindGroup>>,
}

impl IndirectDrawPreparePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("draw_prepare.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/mesh/draw_prepare.wgsl").into(),
            ),
        });

        // draw_prepare.wgsl bindings:
        // 0 visible_instances (ro storage)
        // 1 instances (ro storage)
        // 2 mesh_table (ro storage)
        // 3 counters (ro storage)
        // 4 indirect_args (rw storage)
        // 5 history_visibility (ro storage)
        // 6 draw_count (rw storage)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw_prepare.bind_group_layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("draw_prepare.pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("draw_prepare.pipeline"),
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

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        resources: &MeshGpuScene,
        visibility_history: &VisibilityHistory,
        list: &VisibilityList,
    ) {
        let mut cache = self.bind_group_cache.borrow_mut();
        cache.entry(list.id()).or_insert_with(|| {
            let label = format!("draw_prepare.{}.bind_group", list.label());
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
                        resource: resources.instances().instance_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: resources.meshes().mesh_table_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: list.visible_count_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: list.draw_args_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: visibility_history.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: list.draw_count_buffer().as_entire_binding(),
                    },
                ],
            })
        });
    }

    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>, list: &VisibilityList) {
        let bind_group_cache = self.bind_group_cache.borrow();
        let bind_group = bind_group_cache
            .get(&list.id())
            .expect("IndirectDrawPreparePass: missing bind group; call prepare() before encode()");

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups_indirect(list.dispatch_args_buffer(), 0);
    }

    pub(crate) fn record<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        label: impl Into<String>,
        resources: &MeshGraphResources<'frame>,
        handles: VisibilityListHandles<'frame>,
        list: &'frame VisibilityList,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass(label);
        pass.set_side_effect(false);
        for buffer in [
            handles.visible_instances,
            handles.visible_count,
            resources.history,
        ] {
            let _ = pass.storage_buffer_read(buffer, BufferRange::whole())?;
        }
        let _ = pass.indirect_buffer(handles.dispatch_args, BufferRange::whole())?;
        for buffer in [handles.draw_args, handles.draw_count] {
            let _ =
                pass.storage_buffer_write(buffer, BufferRange::whole(), WriteContents::Preserve)?;
        }
        pass.finish_compute(move |mut context| {
            self.encode(&mut context.pass, list);
            Ok(())
        })?;
        Ok(())
    }
}
