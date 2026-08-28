use crate::{
    camera::Camera,
    mesh::{
        frame::{MeshFrameResources, VisibilityListHandles},
        scene::MeshSceneResources,
        visibility::{VisibilityHistory, VisibilityList},
    },
};
use zen_frame_graph::{BufferRange, Frame, FrameGraphError, WriteContents};

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
    uniform_buffer: wgpu::Buffer,
}

impl OcclusionCullPass {
    pub(crate) fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

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
                include_str!("../../../shaders/mesh/occlusion_cull.wgsl").into(),
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
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
            uniform_buffer,
        }
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
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        resources: &MeshSceneResources,
        visibility_history: &VisibilityHistory,
        hiz_view: &wgpu::TextureView,
        list: &VisibilityList,
    ) {
        let label = format!("occlusion_cull.{}.bind_group", list.label());
        let bind_group = self.create_bind_group(
            device,
            &label,
            list.visible_instances_buffer(),
            resources.instances().instance_buffer(),
            resources.meshes().mesh_table_buffer(),
            list.visible_count_buffer(),
            visibility_history.buffer(),
            hiz_view,
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups_indirect(list.dispatch_args_buffer(), 0);
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "occlusion culling connects explicit logical and native inputs"
    )]
    pub(crate) fn record<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        label: impl Into<String>,
        resources: &MeshFrameResources<'frame>,
        handles: VisibilityListHandles<'frame>,
        list: &'frame VisibilityList,
        scene: &'frame MeshSceneResources,
        history: &'frame VisibilityHistory,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass(label);
        pass.set_side_effect(false);
        for buffer in [
            handles.visible_instances,
            resources.instances,
            resources.mesh_table,
            handles.visible_count,
        ] {
            let _ = pass.storage_buffer_read(buffer, BufferRange::whole())?;
        }
        let _ = pass.storage_buffer_write(
            resources.history,
            BufferRange::whole(),
            WriteContents::Preserve,
        )?;
        let hiz_access = pass.sampled_texture(resources.hiz.texture)?;
        let _ = pass.uniform_buffer(resources.occlusion_uniform, BufferRange::whole())?;
        let _ = pass.indirect_buffer(handles.dispatch_args, BufferRange::whole())?;
        pass.finish_compute(move |mut context| {
            let hiz_view = context.resources.texture_view(hiz_access)?;
            self.encode(
                context.device,
                &mut context.pass,
                scene,
                history,
                hiz_view,
                list,
            );
            Ok(())
        })?;
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "matches the shader's fixed resource bindings"
    )]
    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        label: &str,
        visible_instances: &wgpu::Buffer,
        instances: &wgpu::Buffer,
        mesh_table: &wgpu::Buffer,
        visible_count: &wgpu::Buffer,
        visibility_history: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visible_instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visibility_history.as_entire_binding(),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::visibility::HiZPyramidDesc;
    use std::cell::Cell;
    use zen_frame_graph::{CompileOptions, FrameGraph, WriteContents};

    fn storage_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn node_local_bind_group_accepts_a_resolved_transient_full_pyramid_view() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let occlusion = OcclusionCullPass::new(&device);
        let visible_instances = storage_buffer(&device, "visible-instances");
        let instances = storage_buffer(&device, "instances");
        let mesh_table = storage_buffer(&device, "mesh-table");
        let visible_count = storage_buffer(&device, "visible-count");
        let history = storage_buffer(&device, "history");
        let callback_ran = Cell::new(false);

        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let pyramid = HiZPyramidDesc::new(8, 8);
        let hiz = frame.create_texture(pyramid.texture_desc()).unwrap();
        for mip in 0..pyramid.mip_level_count() {
            let view = frame
                .create_texture_view(hiz, pyramid.mip_view_desc(mip))
                .unwrap();
            let mut pass = frame.compute_pass(format!("write-hiz-mip-{mip}"));
            pass.set_side_effect(false);
            let _ = pass
                .storage_texture_write(view, WriteContents::Overwrite)
                .unwrap();
            pass.finish_compute(|_| Ok(())).unwrap();
        }

        let mut pass = frame.compute_pass("occlusion-consumer");
        pass.set_side_effect(true);
        let hiz_access = pass.sampled_texture(hiz).unwrap();
        pass.finish_compute(|ctx| {
            let hiz_view = ctx.resources.texture_view(hiz_access)?;
            let _bind_group = occlusion.create_bind_group(
                ctx.device,
                "occlusion-test",
                &visible_instances,
                &instances,
                &mesh_table,
                &visible_count,
                &history,
                hiz_view,
            );
            callback_ran.set(true);
            Ok(())
        })
        .unwrap();

        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(&queue)
            .unwrap();
        assert!(callback_ran.get());
        assert_eq!(graph.resource_pool_stats().acquire_count, 1);
    }
}
