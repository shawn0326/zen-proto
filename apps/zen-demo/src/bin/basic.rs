use rand::Rng;
use std::path::Path;
use std::sync::Arc;
use winit::window::Window;
use zen_demo::{
    Example,
    orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    run,
    surface_state::SurfaceState,
};
use zen_renderer::{
    FrameInput, GpuTimingReport, Renderer,
    camera::{Camera, PerspectiveProjection},
    device::request_device_and_queue,
    mesh::{Instance, Material, Mesh, MeshFrameInput, MeshRenderer, Texture},
};

struct Demo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: SurfaceState,
    renderer: Renderer,
    projection: PerspectiveProjection,
    camera: Camera,
    debug_camera: Camera,
    camera_controller: OrbitCameraController,
    use_debug_camera: bool,
    enable_occlusion_culling: bool,

    frame_index: u64,
}

impl Example for Demo {
    async fn init(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let size = window.inner_size();
        let surface = instance.create_surface(window).unwrap();
        let (device, queue) = request_device_and_queue(&instance, &surface).await;
        let surface = SurfaceState::new(&device, surface, size.width, size.height);
        let projection = PerspectiveProjection {
            aspect: surface.width() as f32 / surface.height() as f32,
            ..Default::default()
        };
        let camera = Camera::new(
            glam::Mat4::look_at_rh(
                glam::vec3(0.0, 0.0, 10.0),
                glam::vec3(0.0, 0.0, 0.0),
                glam::vec3(0.0, 1.0, 0.0),
            )
            .inverse(),
            projection,
        );
        let debug_camera = Camera::new(
            glam::Mat4::look_at_rh(
                glam::vec3(-150.0, 150.0, 150.0),
                glam::vec3(0.0, 0.0, 0.0),
                glam::vec3(0.0, 1.0, 0.0),
            )
            .inverse(),
            projection,
        );
        // for cull camera control
        let camera_controller = OrbitCameraController::new(OrbitCameraControllerOptions {
            target: glam::vec3(0.0, 0.0, 0.0),
            position: Some(glam::vec3(0.0, 0.0, 10.0)),
            ..Default::default()
        });

        let meshes = vec![
            Mesh::create_triangle(),
            Mesh::create_box(),
            Mesh::create_sphere(6),
        ];

        let textures = vec![
            Texture::white_1x1(),
            load_texture_from_assets("uv_grid_opengl.jpg"),
        ];
        let textures_count = textures.len() as u32;

        let mut materials = vec![];
        let mut rng = rand::rng();
        for i in 0..20 {
            let hue = rng.random::<f32>() * 360.0;
            let saturation = 1.0_f32;
            let lightness = 0.5_f32;

            // Convert HSL to RGB
            let c = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
            let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
            let m = lightness - c / 2.0;

            let (r, g, b) = match hue as u32 {
                0..60 => (c, x, 0.0),
                60..120 => (x, c, 0.0),
                120..180 => (0.0, c, x),
                180..240 => (0.0, x, c),
                240..300 => (x, 0.0, c),
                _ => (c, 0.0, x),
            };

            materials.push(Material {
                albedo_factor: glam::Vec4::new(r + m, g + m, b + m, 1.0),
                // emissive rgb + ao_strength in w
                emissive_ao: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
                // albedo/emissive/ao; fall back to white (0)
                tex_ids: [i % textures_count, 0, 0, 0],
            });
        }

        let instance_count = 100_0000u32;
        let mut instances = Vec::with_capacity(instance_count as usize);
        let mut rng = rand::rng();
        for i in 0..instance_count {
            let translation = rng.random::<glam::Vec3>() * 200. - 100.;
            let scale = rng.random::<f32>() * 2.0 + if i == 1 { 100.0 } else { 1.0 };
            let transform = glam::Mat4::from_translation(translation);
            let transform = transform * glam::Mat4::from_scale(glam::vec3(scale, scale, scale));
            instances.push(Instance {
                transform,
                mesh_id: i % meshes.len() as u32,
                material_id: i % materials.len() as u32,
                _pad: [0; 2],
            });
        }

        let mesh = MeshRenderer::new(
            &device,
            &queue,
            surface.format(),
            &meshes,
            &materials,
            &instances,
            &textures,
        );
        let renderer = Renderer::new(&device, mesh);
        Demo {
            device,
            queue,
            surface,
            renderer,
            projection,
            camera,
            debug_camera,
            camera_controller,
            use_debug_camera: false,
            enable_occlusion_culling: true,

            frame_index: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.surface.resize(width, height);
        if width == 0 || height == 0 {
            return;
        }
        self.projection.aspect = width as f32 / height as f32;
        self.camera.set_projection(self.projection);
        self.debug_camera.set_projection(self.projection);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        // Low-frequency stats: request once per ~120 frames, print when ready.
        self.frame_index += 1;
        if self.frame_index.is_multiple_of(120) {
            self.renderer.mesh_mut().request_stats();
            self.renderer.request_gpu_timing();
        }

        let debug_camera = if self.use_debug_camera {
            Some(self.debug_camera)
        } else {
            None
        };
        let Some(surface_texture) = self.surface.acquire(&self.device) else {
            return;
        };
        self.renderer
            .render(
                &self.device,
                &self.queue,
                FrameInput {
                    frame_index: self.frame_index,
                    surface_texture: &surface_texture.texture,
                    mesh: MeshFrameInput {
                        camera: self.camera,
                        debug_camera,
                        enable_occlusion_culling: self.enable_occlusion_culling,
                    },
                },
            )
            .expect("FrameGraph rendering failed");
        self.queue.present(surface_texture);

        if let Some(stats) = self.renderer.mesh_mut().take_stats(&self.device) {
            println!(
                "Render stats: total={} main_cull_visible={} drawn={} (A: vis={} draw={} | B: vis={} draw={})",
                stats.total_instances,
                stats.visible_after_main_cull,
                stats.drawn_instances,
                stats.list_a_visible,
                stats.list_a_drawn,
                stats.list_b_visible,
                stats.list_b_drawn,
            );
        }
        if let Some(timing) = self.renderer.take_gpu_timing() {
            print_gpu_timing(timing);
        }
    }

    fn mouse_drag(&mut self, dx: f32, dy: f32) {
        self.camera_controller.orbit(dx * 0.01, dy * 0.01);
        self.camera.set_view(self.camera_controller.view_matrix());
    }

    fn mouse_wheel(&mut self, delta_y: f32) {
        self.camera_controller.dolly(delta_y);
        self.camera.set_view(self.camera_controller.view_matrix());
    }

    fn key_input(&mut self, key_event: winit::event::KeyEvent) {
        if key_event.physical_key == winit::keyboard::KeyCode::KeyD
            && key_event.state == winit::event::ElementState::Pressed
        {
            self.use_debug_camera = !self.use_debug_camera;
            println!("Use debug camera: {}", self.use_debug_camera);
        } else if key_event.physical_key == winit::keyboard::KeyCode::KeyO
            && key_event.state == winit::event::ElementState::Pressed
        {
            self.enable_occlusion_culling = !self.enable_occlusion_culling;
            println!(
                "Enable occlusion culling: {}",
                self.enable_occlusion_culling
            );
        }
    }
}

fn main() {
    run::<Demo>(Some(120));
}

fn load_texture_from_assets(name: &str) -> Texture {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir.join("assets").join("textures").join(name);
    let img = image::open(&path).expect("Failed to load asset texture");
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Texture {
        width,
        height,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        pixels: rgba.into_raw(),
    }
}

fn print_gpu_timing(timing: GpuTimingReport) {
    match timing {
        GpuTimingReport::Available {
            frame_index,
            frame_duration,
            nodes,
            ..
        } => {
            println!("GPU timing frame {frame_index}: total {frame_duration:?}");
            for node in nodes {
                println!("  {} ({:?}): {:?}", node.label, node.kind, node.duration);
            }
        }
        GpuTimingReport::Unavailable {
            frame_index,
            reason,
        } => println!("GPU timing frame {frame_index} unavailable: {reason:?}"),
        _ => println!("GPU timing report uses an unknown future format"),
    }
}
