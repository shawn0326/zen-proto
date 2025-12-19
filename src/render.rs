mod dispatch_prepare_pass;
mod draw_pass;
mod draw_prepare_pass;
mod main_cull_pass;

use crate::camera::Camera;
use crate::material::{Material, MaterialStorage};
use crate::mesh::{Mesh, MeshStorage};
use crate::primitive::{Primitive, PrimitiveStorage};
use dispatch_prepare_pass::DispatchPreparePass;
use draw_pass::DrawPass;
use draw_prepare_pass::DrawPreparePass;
use main_cull_pass::MainCullPass;

pub struct RenderContext {
    surface: wgpu::Surface<'static>,
    surface_configuration: wgpu::SurfaceConfiguration,
    depth_stencil_texture: wgpu::Texture,
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
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        Self {
            surface,
            surface_configuration,
            depth_stencil_texture,
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
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
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
    pub dispatch_prepare_pass: DispatchPreparePass,
    pub draw_prepare_pass: DrawPreparePass,
    pub draw_pass: DrawPass,
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
        let dispatch_prepare_pass =
            DispatchPreparePass::new(&context.device, main_cull_pass.visible_count_buffer());
        let draw_prepare_pass = DrawPreparePass::new(
            &context.device,
            &resources.meshes,
            &resources.primitives,
            main_cull_pass.visible_instances_buffer(),
            main_cull_pass.visible_count_buffer(),
        );
        let draw_pass = DrawPass::new(
            &context.device,
            context.surface_configuration.format,
            &resources.meshes,
            &resources.materials,
            &resources.primitives,
        );
        Self {
            resources,
            main_cull_pass,
            dispatch_prepare_pass,
            draw_prepare_pass,
            draw_pass,
        }
    }

    pub fn render(&self, context: &RenderContext, camera: Camera, debug_camera: Option<Camera>) {
        let surface_texture = context.surface.get_current_texture().unwrap();

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Step 1: 视锥裁剪
        let main_cull_pass = &self.main_cull_pass;
        main_cull_pass.update_frustum(&context.queue, &camera);
        main_cull_pass.reset_visible_count(&context.queue);
        main_cull_pass.encode(&mut encoder, self.resources.primitives.instance_count);

        // Step 2: 准备 Dispatch 参数
        self.dispatch_prepare_pass.encode(&mut encoder);

        // Step 3: 准备 DrawIndirect 参数
        self.draw_prepare_pass.encode_indirect(
            &mut encoder,
            self.dispatch_prepare_pass.dispatch_args_buffer(),
        );

        // Step 4: 绘制
        let draw_pass = &self.draw_pass;
        let camera_to_use = debug_camera.as_ref().unwrap_or(&camera);
        draw_pass.update_camera_buffer(&context.queue, camera_to_use);
        draw_pass.encode(
            &mut encoder,
            &surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &context
                .depth_stencil_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &self.resources.meshes.index_buffer,
            self.draw_prepare_pass.indirect_args_buffer(),
            main_cull_pass.visible_count_buffer(),
            self.resources.primitives.instance_count,
        );

        context.queue.submit(Some(encoder.finish()));

        surface_texture.present();
    }
}
