mod dispatch_prepare_pass;
mod draw_pass;
mod draw_prepare_pass;
mod hiz_generate_pass;
mod main_cull_pass;
mod occlusion_cull_pass;

use crate::camera::Camera;
use crate::material::{Material, MaterialStorage};
use crate::mesh::{Mesh, MeshStorage};
use crate::primitive::{Primitive, PrimitiveStorage};
use dispatch_prepare_pass::DispatchPreparePass;
use draw_pass::DrawPass;
use draw_prepare_pass::DrawPreparePass;
use hiz_generate_pass::HiZGeneratePass;
use main_cull_pass::MainCullPass;
use occlusion_cull_pass::OcclusionCullPass;

pub struct HiZTexture {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    mip_level_count: u32,
    sampled_full_view: wgpu::TextureView,
    sampled_views: Vec<wgpu::TextureView>,
    storage_views: Vec<wgpu::TextureView>,
}

impl HiZTexture {
    fn calc_mip_level_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height).max(1);
        // WebGPU mip sizes follow integer right-shift (floor division by 2).
        // Total levels = floor(log2(max_dim)) + 1.
        32 - max_dim.leading_zeros()
    }

    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_level_count = Self::calc_mip_level_count(width, height);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sampled_full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_full_sampled_view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_level_count),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        let mut sampled_views = Vec::with_capacity(mip_level_count as usize);
        let mut storage_views = Vec::with_capacity(mip_level_count as usize);
        for mip in 0..mip_level_count {
            sampled_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip_sampled_view"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            }));

            storage_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz_mip_storage_view"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            }));
        }

        Self {
            texture,
            width,
            height,
            mip_level_count,
            sampled_full_view,
            sampled_views,
            storage_views,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    pub fn sampled_view(&self, mip: u32) -> &wgpu::TextureView {
        &self.sampled_views[mip as usize]
    }

    pub fn sampled_full_view(&self) -> &wgpu::TextureView {
        &self.sampled_full_view
    }

    pub fn storage_view(&self, mip: u32) -> &wgpu::TextureView {
        &self.storage_views[mip as usize]
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

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

        let required_features =
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::INDIRECT_FIRST_INSTANCE;

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

pub struct RenderResources {
    pub meshes: MeshStorage,
    pub materials: MaterialStorage,
    pub primitives: PrimitiveStorage,
}

impl RenderResources {
    pub fn new(
        device: &wgpu::Device,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
    ) -> Self {
        let meshes = MeshStorage::from_meshes(device, meshes);
        let materials = MaterialStorage::from_materials(device, materials);
        let primitives = PrimitiveStorage::from_primitives(device, primitives);

        Self {
            meshes,
            materials,
            primitives,
        }
    }
}

pub struct DefaultRenderer {
    pub resources: RenderResources,
    pub main_cull_pass: MainCullPass,
    pub dispatch_prepare_pass_a: DispatchPreparePass,
    pub draw_prepare_pass_a: DrawPreparePass,
    pub dispatch_prepare_pass_b: DispatchPreparePass,
    pub draw_prepare_pass_b: DrawPreparePass,
    pub hiz_generate_pass: HiZGeneratePass,
    pub occlusion_cull_pass: OcclusionCullPass,
    pub draw_pass: DrawPass,
    pub draw_pass_debug: DrawPass,
}

impl DefaultRenderer {
    pub fn new(
        context: &RenderContext,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
    ) -> DefaultRenderer {
        let resources = RenderResources::new(&context.device, meshes, materials, primitives);
        let main_cull_pass =
            MainCullPass::new(&context.device, &resources.meshes, &resources.primitives);
        let dispatch_prepare_pass_a =
            DispatchPreparePass::new(&context.device, main_cull_pass.visible_count_buffer_a());
        let draw_prepare_pass_a = DrawPreparePass::new(
            &context.device,
            &resources.meshes,
            &resources.primitives,
            main_cull_pass.visible_instances_buffer_a(),
            main_cull_pass.visible_count_buffer_a(),
            main_cull_pass.visibility_history_buffer(),
        );
        let dispatch_prepare_pass_b =
            DispatchPreparePass::new(&context.device, main_cull_pass.visible_count_buffer_b());
        let draw_prepare_pass_b = DrawPreparePass::new(
            &context.device,
            &resources.meshes,
            &resources.primitives,
            main_cull_pass.visible_instances_buffer_b(),
            main_cull_pass.visible_count_buffer_b(),
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
        Self {
            resources,
            main_cull_pass,
            dispatch_prepare_pass_a,
            draw_prepare_pass_a,
            dispatch_prepare_pass_b,
            draw_prepare_pass_b,
            hiz_generate_pass,
            occlusion_cull_pass,
            draw_pass,
            draw_pass_debug,
        }
    }

    pub fn render(
        &self,
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

        let main_cull_pass = &self.main_cull_pass;
        main_cull_pass.update_frustum(&context.queue, &camera);
        main_cull_pass.reset_visible_count(&context.queue);
        main_cull_pass.enable_occlusion_culling(&context.queue, enable_occlusion_culling);
        main_cull_pass.encode(&mut encoder, self.resources.primitives.instance_count);

        self.dispatch_prepare_pass_a.encode(&mut encoder);

        self.draw_prepare_pass_a.encode_indirect(
            &mut encoder,
            self.dispatch_prepare_pass_a.dispatch_args_buffer(),
        );

        let draw_pass = &self.draw_pass;
        draw_pass.update_camera_buffer(&context.queue, &camera);
        draw_pass.encode(
            &mut encoder,
            &surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &context
                .depth_stencil_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &self.resources.meshes.index_buffer,
            self.draw_prepare_pass_a.indirect_args_buffer(),
            main_cull_pass.visible_count_buffer_a(),
            self.resources.primitives.instance_count,
            true,
            true,
        );

        if enable_occlusion_culling {
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
            self.hiz_generate_pass.encode(
                &context.device,
                &mut encoder,
                &depth_for_hiz_view,
                &context.hiz_texture,
            );

            self.dispatch_prepare_pass_b.encode(&mut encoder);

            // Occlusion cull List B: update visibility_history based on Hi-Z.
            // (History for List A will be handled later.)
            self.occlusion_cull_pass.update_params(
                &context.queue,
                &camera,
                context.surface_configuration.width,
                context.surface_configuration.height,
                0.00001,
                0.0,
            );
            self.occlusion_cull_pass.encode_indirect(
                &context.device,
                &mut encoder,
                self.dispatch_prepare_pass_b.dispatch_args_buffer(),
                main_cull_pass.visible_instances_buffer_b(),
                main_cull_pass.visible_count_buffer_b(),
                &self.resources.primitives.instance_buffer,
                &self.resources.meshes.mesh_table_buffer,
                main_cull_pass.visibility_history_buffer(),
                context.hiz_texture.sampled_full_view(),
            );

            self.draw_prepare_pass_b.encode_indirect(
                &mut encoder,
                self.dispatch_prepare_pass_b.dispatch_args_buffer(),
            );

            draw_pass.encode(
                &mut encoder,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &context
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                self.draw_prepare_pass_b.indirect_args_buffer(),
                main_cull_pass.visible_count_buffer_b(),
                self.resources.primitives.instance_count,
                false,
                false,
            );

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
            self.hiz_generate_pass.encode(
                &context.device,
                &mut encoder,
                &depth_for_hiz_view,
                &context.hiz_texture,
            );

            // Occlusion cull List B: update visibility_history based on Hi-Z.
            // (History for List A will be handled later.)
            self.occlusion_cull_pass.encode_indirect(
                &context.device,
                &mut encoder,
                self.dispatch_prepare_pass_a.dispatch_args_buffer(),
                main_cull_pass.visible_instances_buffer_a(),
                main_cull_pass.visible_count_buffer_a(),
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
                &mut encoder,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &context
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                self.draw_prepare_pass_a.indirect_args_buffer(),
                main_cull_pass.visible_count_buffer_a(),
                self.resources.primitives.instance_count,
                true,
                true,
            );

            self.draw_pass_debug.encode(
                &mut encoder,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &context
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                self.draw_prepare_pass_b.indirect_args_buffer(),
                main_cull_pass.visible_count_buffer_b(),
                self.resources.primitives.instance_count,
                false,
                false,
            );
        }

        context.queue.submit(Some(encoder.finish()));

        surface_texture.present();
    }
}
