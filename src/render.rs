mod cull;
mod draw;

use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshTableEntry {
    pub index_count: u32,   // number of indices
    pub first_index: u32,   // offset in the global index buffer (in indices)
    pub base_vertex: i32,   // offset in the global vertex buffer (in vertices)
    pub _pad: u32,          // pad to 16 bytes for WGSL/storage friendliness
    pub sphere: glam::Vec4, // bounding sphere (xyz: center, w: radius)
}

pub struct MeshesContext {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub mesh_table_buffer: wgpu::Buffer,
}

impl MeshesContext {
    pub fn from_meshes(device: &wgpu::Device, meshes: &[Mesh]) -> Self {
        use wgpu::util::DeviceExt;

        let total_vertices: usize = meshes.iter().map(|m| m.vertices.len()).sum();
        let total_indices: usize = meshes.iter().map(|m| m.indices.len()).sum();

        let mut all_vertices: Vec<Vertex> = Vec::with_capacity(total_vertices);
        let mut all_indices: Vec<u16> = Vec::with_capacity(total_indices);
        let mut mesh_table: Vec<MeshTableEntry> = Vec::with_capacity(meshes.len());

        for mesh in meshes {
            let base_vertex = all_vertices.len() as i32;
            let first_index = all_indices.len() as u32;
            let index_count = mesh.indices.len() as u32;

            // 不改写 u16 索引值；用 base_vertex 来做顶点偏移（更安全，避免 u16 溢出）
            all_vertices.extend_from_slice(&mesh.vertices);
            all_indices.extend_from_slice(&mesh.indices);

            mesh_table.push(MeshTableEntry {
                index_count,
                first_index,
                base_vertex,
                _pad: 0,
                sphere: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshes.vertex_buffer"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshes.index_buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mesh_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("meshes.mesh_table_buffer"),
            contents: bytemuck::cast_slice(&mesh_table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        MeshesContext {
            vertex_buffer,
            index_buffer,
            mesh_table_buffer,
        }
    }
}

pub struct RenderContext {
    pub meshes: MeshesContext,
    pub primitives: PrimitivesContext,
    pub cull_resources: CullResources,
    pub draw_resources: DrawResources,
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

    pub fn create_context(&self, meshes: &[Mesh], primitives: &[Primitive]) -> RenderContext {
        let meshes = MeshesContext::from_meshes(&self.device, meshes);
        let primitives = PrimitivesContext::from_primitives(&self.device, primitives);
        let cull_resources = create_cull_resources(&self.device, &meshes, &primitives);
        let draw_resources = create_draw_resources(
            &self.device,
            self.surface_configuration.format,
            &meshes,
            &primitives,
        );
        RenderContext {
            meshes,
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
                .update_frustum(&self.queue, &cull_camera);
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self
                        .depth_stencil_texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            render_context
                .draw_resources
                .update_camera_buffer(&self.queue, &draw_camera);

            render_pass.set_pipeline(&render_context.draw_resources.pipeline);
            render_pass.set_bind_group(0, &render_context.draw_resources.bind_group, &[]);
            render_pass.set_index_buffer(
                render_context.meshes.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
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
