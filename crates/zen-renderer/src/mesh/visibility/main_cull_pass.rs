use crate::{
    camera::Camera,
    mesh::{
        frame::MeshFrameResources,
        scene::MeshSceneResources,
        visibility::{VisibilityHistory, VisibilityList},
    },
};
use std::cell::RefCell;
use zen_frame_graph::{BufferRange, Frame, FrameGraphError, WriteContents};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MainCullUniform {
    planes: [glam::Vec4; 6],
    max_instance_count: u32,
    mesh_count: u32,
    enable_occlusion: u32,
    _pad: u32,
}

pub struct MainCullPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: RefCell<Option<wgpu::BindGroup>>,
    uniform_buffer: wgpu::Buffer,
}

impl MainCullPass {
    pub(crate) fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("main_cull.uniform.buffer"),
            size: std::mem::size_of::<MainCullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/mesh/main_cull.wgsl").into(),
            ),
        });

        // main_cull.wgsl bindings:
        // 0 instances (ro storage)
        // 1 mesh_table (ro storage)
        // 2 uniform_buffer (uniform)
        // 3 visibility_history_buffer (rw storage)
        // 4 list_a.visible_count (rw storage)
        // 5 list_a.visible_instances (rw storage)
        // 6 list_b.visible_count (rw storage)
        // 7 list_b.visible_instances (rw storage)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("main_cull.bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main_cull.pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("main_cull.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group: RefCell::new(None),
            uniform_buffer,
        }
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        resources: &MeshSceneResources,
        visibility_history: &VisibilityHistory,
        list_a: &VisibilityList,
        list_b: &VisibilityList,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("main_cull.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.instances().instance_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.meshes().mesh_table_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visibility_history.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: list_a.visible_count_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: list_a.visible_instances_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: list_b.visible_count_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: list_b.visible_instances_buffer().as_entire_binding(),
                },
            ],
        });

        *self.bind_group.borrow_mut() = Some(bind_group);
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        resources: &MeshSceneResources,
        camera: &Camera,
        enable_occlusion: bool,
    ) {
        let uniform = MainCullUniform {
            planes: camera.frustum(),
            max_instance_count: resources.instances().instance_count(),
            mesh_count: resources.meshes().mesh_count(),
            enable_occlusion: if enable_occlusion { 1 } else { 0 },
            _pad: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>, max_instance_count: u32) {
        let wg_size = 64;
        let group_count = max_instance_count.div_ceil(wg_size);

        let bind_group = self.bind_group.borrow();
        let bind_group = bind_group
            .as_ref()
            .expect("MainCullPass: missing bind group; call prepare() before encode()");

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(group_count, 1, 1);
    }

    pub(crate) fn record<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        resources: &MeshFrameResources<'frame>,
        max_instance_count: u32,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass("main-cull");
        pass.set_side_effect(false);
        let _ = pass.storage_buffer_read(resources.instances, BufferRange::whole())?;
        let _ = pass.storage_buffer_read(resources.mesh_table, BufferRange::whole())?;
        let _ = pass.uniform_buffer(resources.main_cull_uniform, BufferRange::whole())?;
        for buffer in [
            resources.history,
            resources.list_a.visible_count,
            resources.list_a.visible_instances,
            resources.list_b.visible_count,
            resources.list_b.visible_instances,
        ] {
            let _ =
                pass.storage_buffer_write(buffer, BufferRange::whole(), WriteContents::Preserve)?;
        }
        pass.finish_compute(move |mut context| {
            self.encode(&mut context.pass, max_instance_count);
            Ok(())
        })?;
        Ok(())
    }
}
