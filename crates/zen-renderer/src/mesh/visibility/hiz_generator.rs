use super::HiZPyramidDesc;
use crate::mesh::frame::{FrameTargets, MeshFrameResources};
use zen_frame_graph::{Frame, FrameGraphError, WriteContents};

pub struct HiZGenerator {
    depth_to_mip0_pipeline: wgpu::ComputePipeline,
    depth_to_mip0_bgl: wgpu::BindGroupLayout,
    mip_to_mip_pipeline: wgpu::ComputePipeline,
    mip_to_mip_bgl: wgpu::BindGroupLayout,
}

impl HiZGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let depth_to_mip0_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_depth_to_mip0.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/mesh/hiz_depth_to_mip0.wgsl").into(),
            ),
        });

        let mip_to_mip_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_mip_to_mip.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/mesh/hiz_mip_to_mip.wgsl").into(),
            ),
        });

        let depth_to_mip0_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz.depth_to_mip0_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let depth_to_mip0_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hiz.depth_to_mip0_pipeline_layout"),
                bind_group_layouts: &[Some(&depth_to_mip0_bgl)],
                immediate_size: 0,
            });
        let depth_to_mip0_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hiz.depth_to_mip0_pipeline"),
                layout: Some(&depth_to_mip0_pipeline_layout),
                module: &depth_to_mip0_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mip_to_mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz.mip_to_mip_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let mip_to_mip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hiz.mip_to_mip_pipeline_layout"),
                bind_group_layouts: &[Some(&mip_to_mip_bgl)],
                immediate_size: 0,
            });
        let mip_to_mip_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hiz.mip_to_mip_pipeline"),
                layout: Some(&mip_to_mip_pipeline_layout),
                module: &mip_to_mip_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            depth_to_mip0_pipeline,
            depth_to_mip0_bgl,
            mip_to_mip_pipeline,
            mip_to_mip_bgl,
        }
    }

    pub fn encode_depth_to_mip0(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        source: &wgpu::TextureView,
        destination: &wgpu::TextureView,
        pyramid: HiZPyramidDesc,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz.depth_to_mip0_bg"),
            layout: &self.depth_to_mip0_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination),
                },
            ],
        });

        pass.set_pipeline(&self.depth_to_mip0_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(pyramid.width().div_ceil(8), pyramid.height().div_ceil(8), 1);
    }

    pub fn encode_mip_to_mip(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
        source: &wgpu::TextureView,
        destination: &wgpu::TextureView,
        pyramid: HiZPyramidDesc,
        destination_mip: u32,
    ) {
        let extent = pyramid.mip_extent(destination_mip);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz.mip_to_mip_bg"),
            layout: &self.mip_to_mip_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination),
                },
            ],
        });

        pass.set_pipeline(&self.mip_to_mip_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
    }

    pub(crate) fn record_depth_to_mip0<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        label: impl Into<String>,
        targets: FrameTargets<'frame>,
        resources: &MeshFrameResources<'frame>,
        pyramid: HiZPyramidDesc,
    ) -> Result<(), FrameGraphError> {
        let mut pass = frame.compute_pass(label);
        pass.set_side_effect(false);
        let source = pass.sampled_texture(targets.depth)?;
        let destination =
            pass.storage_texture_write(resources.hiz.views[0], WriteContents::Overwrite)?;
        pass.finish_compute(move |mut context| {
            let source = context.resources.texture_view(source)?;
            let destination = context.resources.texture_view(destination)?;
            self.encode_depth_to_mip0(
                context.device,
                &mut context.pass,
                source,
                destination,
                pyramid,
            );
            Ok(())
        })?;
        Ok(())
    }

    pub(crate) fn record_mip_to_mip<'frame>(
        &'frame self,
        frame: &mut Frame<'frame>,
        label: impl Into<String>,
        resources: &MeshFrameResources<'frame>,
        pyramid: HiZPyramidDesc,
        destination_mip: u32,
    ) -> Result<(), FrameGraphError> {
        let source_mip = destination_mip.checked_sub(1).ok_or_else(|| {
            FrameGraphError::InvalidResourceDescriptor {
                message: "Hi-Z mip-to-mip destination must be greater than zero".into(),
            }
        })?;
        let source_view = *resources
            .hiz
            .views
            .get(source_mip as usize)
            .ok_or_else(|| FrameGraphError::InvalidResourceDescriptor {
                message: format!("Hi-Z source mip {source_mip} is unavailable"),
            })?;
        let destination_view = *resources
            .hiz
            .views
            .get(destination_mip as usize)
            .ok_or_else(|| FrameGraphError::InvalidResourceDescriptor {
                message: format!("Hi-Z destination mip {destination_mip} is unavailable"),
            })?;

        let mut pass = frame.compute_pass(label);
        pass.set_side_effect(false);
        let source = pass.sampled_texture(source_view)?;
        let destination = pass.storage_texture_write(destination_view, WriteContents::Overwrite)?;
        pass.finish_compute(move |mut context| {
            let source = context.resources.texture_view(source)?;
            let destination = context.resources.texture_view(destination)?;
            self.encode_mip_to_mip(
                context.device,
                &mut context.pass,
                source,
                destination,
                pyramid,
                destination_mip,
            );
            Ok(())
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use zen_frame_graph::{
        CompileOptions, DepthAttachmentOps, FrameGraph, RootReason, TextureDesc, UsagePolicy,
        WriteContents,
    };

    fn execute_pyramid(
        graph: &mut FrameGraph,
        queue: &wgpu::Queue,
        generator: &HiZGenerator,
        pyramid: HiZPyramidDesc,
    ) {
        let mut frame = graph.begin_frame();
        let depth = frame
            .create_texture(TextureDesc {
                label: "test-depth-transient".into(),
                size: pyramid.mip_extent(0),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                view_formats: vec![],
                usage: UsagePolicy::Infer,
            })
            .unwrap();
        let hiz = frame.create_texture(pyramid.texture_desc()).unwrap();
        let views = (0..pyramid.mip_level_count())
            .map(|mip| {
                frame
                    .create_texture_view(hiz, pyramid.mip_view_desc(mip))
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let mut pass = frame.render_pass("depth-prepass");
        pass.set_side_effect(false);
        let _ = pass
            .depth_attachment(depth, DepthAttachmentOps::clear_store(1.0))
            .unwrap();
        pass.finish_render(|_| Ok(())).unwrap();

        let mut pass = frame.compute_pass("depth-to-mip0");
        pass.set_side_effect(false);
        let source = pass.sampled_texture(depth).unwrap();
        let destination = pass
            .storage_texture_write(views[0], WriteContents::Overwrite)
            .unwrap();
        pass.finish_compute(move |mut ctx| {
            let source = ctx.resources.texture_view(source)?;
            let destination = ctx.resources.texture_view(destination)?;
            generator.encode_depth_to_mip0(ctx.device, &mut ctx.pass, source, destination, pyramid);
            Ok(())
        })
        .unwrap();

        for mip in 1..pyramid.mip_level_count() {
            let mut pass = frame.compute_pass(format!("mip{}-to-mip{mip}", mip - 1));
            pass.set_side_effect(false);
            let source = pass.sampled_texture(views[(mip - 1) as usize]).unwrap();
            let destination = pass
                .storage_texture_write(views[mip as usize], WriteContents::Overwrite)
                .unwrap();
            pass.finish_compute(move |mut ctx| {
                let source = ctx.resources.texture_view(source)?;
                let destination = ctx.resources.texture_view(destination)?;
                generator.encode_mip_to_mip(
                    ctx.device,
                    &mut ctx.pass,
                    source,
                    destination,
                    pyramid,
                    mip,
                );
                Ok(())
            })
            .unwrap();
        }

        frame
            .mark_texture_root(
                views[pyramid.mip_level_count() as usize - 1],
                RootReason::DebugCapture,
            )
            .unwrap();
        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(queue)
            .unwrap();
    }

    #[test]
    fn transient_pyramid_executes_reuses_and_resets_on_resize() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let generator = HiZGenerator::new(&device);
        let mut graph = FrameGraph::with_device(&device);

        execute_pyramid(&mut graph, &queue, &generator, HiZPyramidDesc::new(13, 7));
        let cold = graph.resource_pool_stats();
        assert_eq!(cold.acquire_count, 2);
        assert_eq!(cold.created_count, 2);
        assert_eq!(cold.reuse_count, 0);
        assert_eq!(cold.retained_count, 2);

        execute_pyramid(&mut graph, &queue, &generator, HiZPyramidDesc::new(13, 7));
        let warm = graph.resource_pool_stats();
        assert_eq!(warm.acquire_count, 4);
        assert_eq!(warm.created_count, 2);
        assert_eq!(warm.reuse_count, 2);
        assert_eq!(warm.retained_count, 2);

        graph.clear_resource_pool();
        assert_eq!(graph.resource_pool_stats().retained_count, 0);
        execute_pyramid(&mut graph, &queue, &generator, HiZPyramidDesc::new(21, 11));
        let resized = graph.resource_pool_stats();
        assert_eq!(resized.acquire_count, 6);
        assert_eq!(resized.created_count, 4);
        assert_eq!(resized.reuse_count, 2);
        assert_eq!(resized.retained_count, 2);
    }

    #[test]
    fn unreferenced_pyramid_does_not_acquire_from_the_pool() {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut graph = FrameGraph::with_device(&device);
        let mut frame = graph.begin_frame();
        let pyramid = HiZPyramidDesc::new(8, 8);
        let hiz = frame.create_texture(pyramid.texture_desc()).unwrap();
        let producer_ran = Cell::new(false);
        for mip in 0..pyramid.mip_level_count() {
            let view = frame
                .create_texture_view(hiz, pyramid.mip_view_desc(mip))
                .unwrap();
            let mut pass = frame.compute_pass(format!("unused-write-{mip}"));
            pass.set_side_effect(false);
            let _ = pass
                .storage_texture_write(view, WriteContents::Overwrite)
                .unwrap();
            pass.finish_compute(|_| {
                producer_ran.set(true);
                Ok(())
            })
            .unwrap();
        }
        let consumer_ran = Cell::new(false);
        let mut pass = frame.compute_pass("unused-occlusion-consumer");
        pass.set_side_effect(false);
        let _ = pass.sampled_texture(hiz).unwrap();
        pass.finish_compute(|_| {
            consumer_ran.set(true);
            Ok(())
        })
        .unwrap();
        frame
            .compile(CompileOptions::default())
            .unwrap()
            .execute(&queue)
            .unwrap();
        assert!(!producer_ran.get());
        assert!(!consumer_ran.get());
        assert_eq!(graph.resource_pool_stats(), Default::default());
    }
}
