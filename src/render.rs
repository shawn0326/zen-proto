mod dispatch_prepare_pass;
mod draw_pass;
mod draw_prepare_pass;
mod hiz_generate_pass;
mod hiz_texture;
mod main_cull_pass;
mod occlusion_cull_pass;
mod visibility_list;

use crate::camera::Camera;
use crate::material::Material;
use crate::mesh::{Mesh, MeshStorage};
use crate::primitive::{Primitive, PrimitiveStorage};
use crate::resources::Resources;
use dispatch_prepare_pass::DispatchPreparePass;
use draw_pass::DrawPass;
use draw_prepare_pass::DrawPreparePass;
use hiz_generate_pass::HiZGeneratePass;
use hiz_texture::HiZTexture;
use main_cull_pass::MainCullPass;
use occlusion_cull_pass::OcclusionCullPass;
use visibility_list::VisibilityList;
use wgpu_profiler::GpuProfilerSettings;

pub struct RenderContext {
    surface: wgpu::Surface<'static>,
    surface_configuration: wgpu::SurfaceConfiguration,
    depth_stencil_texture: wgpu::Texture,
    hiz_texture: HiZTexture,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl RenderContext {
    pub async fn new(instance: &wgpu::Instance, surface: wgpu::Surface<'static>) -> Self {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        println!("{:?}", adapter.get_info());
        println!("{:?}", adapter.features());

        let required_features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::TIMESTAMP_QUERY;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: required_features,
                ..Default::default()
            })
            .await
            .unwrap();

        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_configuration);

        let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_stencil_texture"),
            size: wgpu::Extent3d {
                width: surface_configuration.width,
                height: surface_configuration.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let hiz_texture = HiZTexture::new(
            &device,
            surface_configuration.width,
            surface_configuration.height,
        );

        Self {
            surface,
            surface_configuration,
            depth_stencil_texture,
            hiz_texture,
            device,
            queue,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.surface_configuration.width == width && self.surface_configuration.height == height
        {
            return;
        }

        self.surface_configuration.width = width;
        self.surface_configuration.height = height;

        self.surface
            .configure(&self.device, &self.surface_configuration);

        self.depth_stencil_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_stencil_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.hiz_texture = HiZTexture::new(&self.device, width, height);
    }
}

pub struct DefaultRenderer {
    pub resources: Resources,
    list_a: VisibilityList,
    list_b: VisibilityList,
    pub main_cull_pass: MainCullPass,
    pub dispatch_prepare_pass_a: DispatchPreparePass,
    pub draw_prepare_pass_a: DrawPreparePass,
    pub dispatch_prepare_pass_b: DispatchPreparePass,
    pub draw_prepare_pass_b: DrawPreparePass,
    pub hiz_generate_pass: HiZGeneratePass,
    pub occlusion_cull_pass: OcclusionCullPass,
    pub draw_pass: DrawPass,
    pub draw_pass_debug: DrawPass,
    profiler: wgpu_profiler::GpuProfiler,
    need_print_gpu_profile: bool,
}

impl DefaultRenderer {
    pub fn new(
        context: &RenderContext,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
    ) -> DefaultRenderer {
        let resources = Resources::new(&context.device, meshes, materials, primitives);
        let list_a = VisibilityList::new(
            &context.device,
            "List_A",
            resources.primitives.instance_count,
        );
        let list_b = VisibilityList::new(
            &context.device,
            "List_B",
            resources.primitives.instance_count,
        );
        let main_cull_pass = MainCullPass::new(
            &context.device,
            &resources.meshes,
            &resources.primitives,
            &list_a,
            &list_b,
        );
        let dispatch_prepare_pass_a = DispatchPreparePass::new(&context.device, &list_a);
        let draw_prepare_pass_a = DrawPreparePass::new(
            &context.device,
            &resources.meshes,
            &resources.primitives,
            &list_a,
            main_cull_pass.visibility_history_buffer(),
        );
        let dispatch_prepare_pass_b = DispatchPreparePass::new(&context.device, &list_b);
        let draw_prepare_pass_b = DrawPreparePass::new(
            &context.device,
            &resources.meshes,
            &resources.primitives,
            &list_b,
            main_cull_pass.visibility_history_buffer(),
        );
        let draw_pass = DrawPass::new(
            &context.device,
            context.surface_configuration.format,
            &resources.meshes,
            &resources.materials,
            &resources.primitives,
        );
        let draw_pass_debug = DrawPass::new(
            &context.device,
            context.surface_configuration.format,
            &resources.meshes,
            &resources.materials,
            &resources.primitives,
        );

        let hiz_generate_pass = HiZGeneratePass::new(&context.device);
        let occlusion_cull_pass = OcclusionCullPass::new(&context.device);

        let profiler =
            wgpu_profiler::GpuProfiler::new(&context.device, GpuProfilerSettings::default())
                .unwrap();
        Self {
            resources,
            list_a,
            list_b,
            main_cull_pass,
            dispatch_prepare_pass_a,
            draw_prepare_pass_a,
            dispatch_prepare_pass_b,
            draw_prepare_pass_b,
            hiz_generate_pass,
            occlusion_cull_pass,
            draw_pass,
            draw_pass_debug,
            profiler,
            need_print_gpu_profile: false,
        }
    }

    pub fn render(
        &mut self,
        context: &RenderContext,
        camera: Camera,
        debug_camera: Option<Camera>,
        enable_occlusion_culling: bool,
    ) {
        let surface_texture = context.surface.get_current_texture().unwrap();

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut scope = self.profiler.scope("Frame", &mut encoder);

            self.list_a.reset(&context.queue);
            self.list_b.reset(&context.queue);

            let main_cull_pass = &self.main_cull_pass;
            main_cull_pass.prepare(&context.queue, &camera, enable_occlusion_culling);
            main_cull_pass.encode(&mut scope);

            self.dispatch_prepare_pass_a.encode(&mut scope);

            self.draw_prepare_pass_a
                .encode_indirect(&mut scope, &self.list_a);

            let draw_pass = &self.draw_pass;
            draw_pass.update_camera_buffer(&context.queue, &camera);
            draw_pass.encode(
                &mut scope,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &context
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                &self.list_a,
                self.resources.primitives.instance_count,
                true,
                true,
            );

            if enable_occlusion_culling {
                if self.hiz_generate_pass.needs_rebuild(&context.hiz_texture) {
                    let depth_for_hiz_view =
                        context
                            .depth_stencil_texture
                            .create_view(&wgpu::TextureViewDescriptor {
                                label: Some("depth_for_hiz_view"),
                                format: Some(wgpu::TextureFormat::Depth32Float),
                                dimension: Some(wgpu::TextureViewDimension::D2),
                                aspect: wgpu::TextureAspect::DepthOnly,
                                base_mip_level: 0,
                                mip_level_count: Some(1),
                                base_array_layer: 0,
                                array_layer_count: Some(1),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                            });
                    self.hiz_generate_pass.rebuild_bind_groups(
                        &context.device,
                        &depth_for_hiz_view,
                        &context.hiz_texture,
                    );
                }

                self.hiz_generate_pass
                    .encode(&mut scope, &context.hiz_texture);

                self.dispatch_prepare_pass_b.encode(&mut scope);

                // Occlusion cull List B: update visibility_history based on Hi-Z.
                // (History for List A will be handled later.)
                self.occlusion_cull_pass.update_params(
                    &context.queue,
                    &camera,
                    context.surface_configuration.width,
                    context.surface_configuration.height,
                    0.0001,
                    0.0,
                );
                self.occlusion_cull_pass.encode_indirect(
                    &context.device,
                    &mut scope,
                    &self.list_b,
                    &self.resources.primitives.instance_buffer,
                    &self.resources.meshes.mesh_table_buffer,
                    main_cull_pass.visibility_history_buffer(),
                    context.hiz_texture.sampled_full_view(),
                );

                self.draw_prepare_pass_b
                    .encode_indirect(&mut scope, &self.list_b);

                draw_pass.encode(
                    &mut scope,
                    &surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    &context
                        .depth_stencil_texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    &self.resources.meshes.index_buffer,
                    &self.list_b,
                    self.resources.primitives.instance_count,
                    false,
                    false,
                );

                self.hiz_generate_pass
                    .encode(&mut scope, &context.hiz_texture);

                // Occlusion cull List B: update visibility_history based on Hi-Z.
                // (History for List A will be handled later.)
                self.occlusion_cull_pass.encode_indirect(
                    &context.device,
                    &mut scope,
                    &self.list_a,
                    &self.resources.primitives.instance_buffer,
                    &self.resources.meshes.mesh_table_buffer,
                    main_cull_pass.visibility_history_buffer(),
                    context.hiz_texture.sampled_full_view(),
                );
            }

            if let Some(debug_camera) = debug_camera {
                self.draw_pass_debug
                    .update_camera_buffer(&context.queue, &debug_camera);

                self.draw_pass_debug.encode(
                    &mut scope,
                    &surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    &context
                        .depth_stencil_texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    &self.resources.meshes.index_buffer,
                    &self.list_a,
                    self.resources.primitives.instance_count,
                    true,
                    true,
                );

                self.draw_pass_debug.encode(
                    &mut scope,
                    &surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    &context
                        .depth_stencil_texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    &self.resources.meshes.index_buffer,
                    &self.list_b,
                    self.resources.primitives.instance_count,
                    false,
                    false,
                );
            }
        }

        self.profiler.resolve_queries(&mut encoder);

        context.queue.submit(Some(encoder.finish()));

        surface_texture.present();

        self.profiler.end_frame().ok();

        if self.need_print_gpu_profile {
            self.print_gpu_profile(context);
        }
    }

    fn print_gpu_profile(&mut self, context: &RenderContext) {
        if let Some(profiling_data) = self
            .profiler
            .process_finished_frame(context.queue.get_timestamp_period())
        {
            wgpu_profiler::chrometrace::write_chrometrace(
                std::path::Path::new("mytrace.json"),
                &profiling_data,
            )
            .unwrap();
        }

        self.need_print_gpu_profile = false;
    }

    pub fn request_print_gpu_profile(&mut self) {
        self.need_print_gpu_profile = true;
    }
}
