mod dispatch_prepare_pass;
mod draw_pass;
mod draw_prepare_pass;
mod main_cull_pass;

use crate::camera::Camera;
use crate::material::{Material, MaterialsContext};
use crate::mesh::{Mesh, MeshesContext};
use crate::primitive::{Primitive, PrimitivesContext};
use dispatch_prepare_pass::DispatchPreparePass;
use draw_pass::DrawPass;
use draw_prepare_pass::DrawPreparePass;
use main_cull_pass::MainCullPass;

pub struct RenderContext {
    pub meshes: MeshesContext,
    pub materials: MaterialsContext,
    pub primitives: PrimitivesContext,
    pub main_cull_pass: MainCullPass,
    pub dispatch_prepare_pass: DispatchPreparePass,
    pub draw_prepare_pass: DrawPreparePass,
    pub draw_pass: DrawPass,
}

impl RenderContext {
    pub fn new(
        renderer: &Renderer,
        meshes: &[Mesh],
        materials: &[Material],
        primitives: &[Primitive],
    ) -> RenderContext {
        let meshes = MeshesContext::from_meshes(&renderer.device, meshes);
        let materials = MaterialsContext::from_materials(&renderer.device, materials);
        let primitives = PrimitivesContext::from_primitives(&renderer.device, primitives);
        let main_cull_pass = MainCullPass::new(&renderer.device, &meshes, &primitives);
        let dispatch_prepare_pass =
            DispatchPreparePass::new(&renderer.device, main_cull_pass.visible_count_buffer());
        let draw_prepare_pass = DrawPreparePass::new(
            &renderer.device,
            &meshes,
            &primitives,
            main_cull_pass.visible_instances_buffer(),
            main_cull_pass.visible_count_buffer(),
        );
        let draw_pass = DrawPass::new(
            &renderer.device,
            renderer.surface_configuration.format,
            &meshes,
            &materials,
            &primitives,
        );
        Self {
            meshes,
            materials,
            primitives,
            main_cull_pass,
            dispatch_prepare_pass,
            draw_prepare_pass,
            draw_pass,
        }
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    surface_configuration: wgpu::SurfaceConfiguration,
    depth_stencil_texture: wgpu::Texture,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Renderer {
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

        Renderer {
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

    pub fn render(
        &mut self,
        camera: Camera,
        debug_camera: Option<Camera>,
        render_context: &RenderContext,
    ) {
        let surface_texture = self.surface.get_current_texture().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Step 1: 视锥裁剪
        let main_cull_pass = &render_context.main_cull_pass;
        main_cull_pass.update_frustum(&self.queue, &camera);
        main_cull_pass.reset_visible_count(&self.queue);
        main_cull_pass.encode(&mut encoder, render_context.primitives.instance_count);

        // Step 2: 准备 Dispatch 参数
        render_context.dispatch_prepare_pass.encode(&mut encoder);

        // Step 3: 准备 DrawIndirect 参数
        render_context.draw_prepare_pass.encode_indirect(
            &mut encoder,
            render_context.dispatch_prepare_pass.dispatch_args_buffer(),
        );

        // Step 4: 绘制
        let draw_pass = &render_context.draw_pass;
        let camera_to_use = debug_camera.as_ref().unwrap_or(&camera);
        draw_pass.update_camera_buffer(&self.queue, camera_to_use);
        draw_pass.encode(
            &mut encoder,
            &surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &self
                .depth_stencil_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            &render_context.meshes.index_buffer,
            render_context.draw_prepare_pass.indirect_args_buffer(),
            main_cull_pass.visible_count_buffer(),
            render_context.primitives.instance_count,
        );

        self.queue.submit(Some(encoder.finish()));

        surface_texture.present();
    }
}
