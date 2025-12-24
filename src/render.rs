mod dispatch_prepare_pass;
mod draw_pass;
mod draw_prepare_pass;
mod hiz_generate_pass;
mod hiz_texture;
mod main_cull_pass;
mod occlusion_cull_pass;
mod visibility_history;
mod visibility_list;

use crate::camera::Camera;
use crate::material::Material;
use crate::mesh::Mesh;
use crate::primitive::Primitive;
use crate::resources::Resources;
use dispatch_prepare_pass::DispatchPreparePass;
use draw_pass::DrawPass;
use draw_prepare_pass::DrawPreparePass;
use hiz_generate_pass::HiZGeneratePass;
use hiz_texture::HiZTexture;
use main_cull_pass::MainCullPass;
use occlusion_cull_pass::OcclusionCullPass;
use visibility_history::VisibilityHistory;
use visibility_list::VisibilityList;

pub async fn request_device_and_target(
    instance: &wgpu::Instance,
    surface: wgpu::Surface<'static>,
) -> (wgpu::Device, wgpu::Queue, RenderTarget) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    println!("{:?}", adapter.get_info());
    println!("{:?}", surface.get_capabilities(&adapter).formats);

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
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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

    let target = RenderTarget {
        surface,
        surface_configuration,
        depth_stencil_texture,
        hiz_texture,
    };

    (device, queue, target)
}

pub struct RenderTarget {
    surface: wgpu::Surface<'static>,
    surface_configuration: wgpu::SurfaceConfiguration,
    depth_stencil_texture: wgpu::Texture,
    hiz_texture: HiZTexture,
}

impl RenderTarget {
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.surface_configuration.width == width && self.surface_configuration.height == height
        {
            return;
        }

        self.surface_configuration.width = width;
        self.surface_configuration.height = height;

        self.surface.configure(device, &self.surface_configuration);

        self.depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
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

        self.hiz_texture = HiZTexture::new(device, width, height);
    }
}

pub struct DefaultRenderer {
    resources: Resources,

    list_a: VisibilityList,
    list_b: VisibilityList,

    visibility_history: VisibilityHistory,

    main_cull_pass: MainCullPass,
    dispatch_prepare_pass: DispatchPreparePass,
    draw_prepare_pass: DrawPreparePass,
    occlusion_cull_pass: OcclusionCullPass,
    draw_pass: DrawPass,
    hiz_generate_pass: HiZGeneratePass,
}

impl DefaultRenderer {
    pub fn new(
        device: &wgpu::Device,
        target: &RenderTarget,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
    ) -> DefaultRenderer {
        // buffers

        let resources = Resources::new(device, meshes, materials, primitives);

        let list_a = VisibilityList::new(device, "list_a", resources.primitives.instance_count);
        let list_b = VisibilityList::new(device, "list_b", resources.primitives.instance_count);

        let visibility_history =
            VisibilityHistory::new(device, resources.primitives.instance_count);

        // passes

        let main_cull_pass = MainCullPass::new(device);
        main_cull_pass.prepare(device, &resources, &visibility_history, &list_a, &list_b);

        let dispatch_prepare_pass = DispatchPreparePass::new(device);
        dispatch_prepare_pass.prepare(device, &list_a);
        dispatch_prepare_pass.prepare(device, &list_b);

        let draw_prepare_pass = DrawPreparePass::new(device);
        draw_prepare_pass.prepare(device, &resources, &visibility_history, &list_a);
        draw_prepare_pass.prepare(device, &resources, &visibility_history, &list_b);

        let occlusion_cull_pass = OcclusionCullPass::new(device);
        let hiz_view = target.hiz_texture.sampled_full_view();
        occlusion_cull_pass.prepare(device, &resources, &visibility_history, hiz_view, &list_a);
        occlusion_cull_pass.prepare(device, &resources, &visibility_history, hiz_view, &list_b);

        let draw_pass = DrawPass::new(device, target.surface_configuration.format, &resources);

        let hiz_generate_pass = HiZGeneratePass::new(device);

        // assemble

        Self {
            resources,
            list_a,
            list_b,
            visibility_history,
            main_cull_pass,
            dispatch_prepare_pass,
            draw_prepare_pass,
            occlusion_cull_pass,
            draw_pass,
            hiz_generate_pass,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &RenderTarget,
        camera: Camera,
        debug_camera: Option<Camera>,
        enable_occlusion_culling: bool,
    ) {
        let resources = &self.resources;
        let max_instance_count = resources.primitives.instance_count;

        let surface_texture = target.surface.get_current_texture().unwrap();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        self.list_a.reset(queue);
        self.list_b.reset(queue);

        self.main_cull_pass
            .update(queue, resources, &camera, enable_occlusion_culling);
        self.main_cull_pass.encode(&mut encoder, max_instance_count);

        self.dispatch_prepare_pass
            .encode(&mut encoder, &self.list_a);
        self.draw_prepare_pass.encode(&mut encoder, &self.list_a);

        self.draw_pass.update(&queue, &camera, 0);
        self.draw_pass.encode(
            &mut encoder,
            &surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &target
                .depth_stencil_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &self.resources.meshes.index_buffer,
            &self.list_a,
            max_instance_count,
            true,
            true,
            0,
        );

        if enable_occlusion_culling {
            // TODO move this to hiz_generate_pass
            if self.hiz_generate_pass.needs_rebuild(&target.hiz_texture) {
                let depth_for_hiz_view =
                    target
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
                    device,
                    &depth_for_hiz_view,
                    &target.hiz_texture,
                );

                let hiz_view = target.hiz_texture.sampled_full_view();

                self.occlusion_cull_pass.clear_cache();
                self.occlusion_cull_pass.prepare(
                    device,
                    &resources,
                    &self.visibility_history,
                    hiz_view,
                    &self.list_a,
                );
                self.occlusion_cull_pass.prepare(
                    device,
                    &resources,
                    &self.visibility_history,
                    hiz_view,
                    &self.list_b,
                );
            }

            self.hiz_generate_pass
                .encode(&mut encoder, &target.hiz_texture);

            self.dispatch_prepare_pass
                .encode(&mut encoder, &self.list_b);

            // Occlusion cull List B: update visibility_history based on Hi-Z.
            // (History for List A will be handled later.)
            self.occlusion_cull_pass.update(
                &queue,
                &camera,
                target.surface_configuration.width,
                target.surface_configuration.height,
            );
            self.occlusion_cull_pass.encode(&mut encoder, &self.list_b);

            self.draw_prepare_pass.encode(&mut encoder, &self.list_b);

            self.draw_pass.encode(
                &mut encoder,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &target
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                &self.list_b,
                max_instance_count,
                false,
                false,
                0,
            );

            self.hiz_generate_pass
                .encode(&mut encoder, &target.hiz_texture);

            // Occlusion cull List B: update visibility_history based on Hi-Z.
            // (History for List A will be handled later.)
            self.occlusion_cull_pass.encode(&mut encoder, &self.list_a);
        }

        if let Some(debug_camera) = debug_camera {
            self.draw_pass.update(&queue, &debug_camera, 1);

            self.draw_pass.encode(
                &mut encoder,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &target
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                &self.list_a,
                max_instance_count,
                true,
                true,
                1,
            );

            self.draw_pass.encode(
                &mut encoder,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &target
                    .depth_stencil_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                &self.resources.meshes.index_buffer,
                &self.list_b,
                max_instance_count,
                false,
                false,
                1,
            );
        }

        queue.submit(Some(encoder.finish()));

        surface_texture.present();
    }
}
