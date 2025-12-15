mod cull;
mod draw;

use crate::camera::Camera;
use crate::primitive::Primitive;
use cull::*;
use draw::*;

pub struct PrimitivesContext {
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}

impl PrimitivesContext {
    pub fn from_primitives(device: &wgpu::Device, primitives: &[Primitive]) -> Self {
        use wgpu::util::DeviceExt;

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("primitives.instance_buffer"),
            contents: bytemuck::cast_slice(primitives),
            usage: wgpu::BufferUsages::STORAGE,
        });

        PrimitivesContext {
            instance_buffer,
            instance_count: primitives.len() as u32,
        }
    }
}

pub struct RenderContext {
    pub primitives: PrimitivesContext,
    pub cull_resources: CullResources,
    pub draw_resources: DrawResources,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    surface_configuration: wgpu::SurfaceConfiguration,
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

        Renderer {
            surface,
            surface_configuration,
            device,
            queue,
        }
    }

    pub fn create_context(&self, primitives: &[Primitive]) -> RenderContext {
        let primitives = PrimitivesContext::from_primitives(&self.device, primitives);
        let cull_resources = create_cull_resources(&self.device, &primitives);
        let draw_resources =
            create_draw_resources(&self.device, self.surface_configuration.format, &primitives);
        RenderContext {
            primitives,
            cull_resources,
            draw_resources,
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
    }

    pub fn render(
        &mut self,
        cull_camera: Camera,
        draw_camera: Camera,
        render_context: &RenderContext,
    ) {
        let surface_texture = self.surface.get_current_texture().unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            render_context
                .cull_resources
                .update_frustum(&self.queue, cull_camera.view_projection());
            render_context
                .cull_resources
                .reset_indirect_buffers(&self.queue);

            let wg_size = 64;
            let group_count = (render_context.primitives.instance_count + wg_size - 1) / wg_size;

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Frustum Culling Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&render_context.cull_resources.cull_pipeline);
            pass.set_bind_group(0, &render_context.cull_resources.cull_bind_group, &[]);
            pass.dispatch_workgroups(group_count, 1, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            render_context
                .draw_resources
                .update_camera_buffer(&self.queue, &draw_camera.view_projection());

            render_pass.set_pipeline(&render_context.draw_resources.pipeline);
            render_pass.set_bind_group(0, &render_context.draw_resources.bind_group, &[]);
            render_pass.set_index_buffer(
                render_context.draw_resources.index_buffer.slice(..),
                render_context.draw_resources.index_format,
            );

            render_pass.multi_draw_indexed_indirect_count(
                &render_context.cull_resources.indirect_args_buffer,
                0,
                &render_context.cull_resources.indirect_count_buffer,
                0,
                render_context.primitives.instance_count,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        surface_texture.present();
    }
}
