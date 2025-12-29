mod common;
use common::{
    Example,
    orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    run,
};
use rand::Rng;
use std::path::Path;
use std::sync::Arc;
use winit::window::Window;
use zen_proto::{
    camera::{Camera, PerspectiveProjection},
    instance::Instance,
    material, mesh,
    render::{DefaultRenderer, RenderTarget, request_device_and_target},
    texture::Texture,
};

struct Demo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: RenderTarget,
    renderer: DefaultRenderer,
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
        let surface = instance.create_surface(window).unwrap();
        let (device, queue, target) = request_device_and_target(&instance, surface).await;
        let projection = PerspectiveProjection::default();
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

        let mut meshes = vec![];
        meshes.push(mesh::Mesh::create_triangle());
        meshes.push(mesh::Mesh::create_box());
        meshes.push(mesh::Mesh::create_sphere(6));

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

            materials.push(material::Material {
                color: glam::Vec4::new(r + m, g + m, b + m, 1.0),
                texture_id: i % textures_count,
                _pad: [0; 3],
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

        let renderer = DefaultRenderer::new(
            &device, &queue, &target, &meshes, &materials, &instances, &textures,
        );
        Demo {
            device,
            queue,
            target,
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
        self.target.resize(width, height);
        self.projection.aspect = width as f32 / height as f32;
        self.camera.set_projection(self.projection);
        self.debug_camera.set_projection(self.projection);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        // Low-frequency stats: request once per ~120 frames, print when ready.
        self.frame_index += 1;
        if self.frame_index % 120 == 0 {
            self.renderer.request_render_stats();
        }

        let debug_camera = if self.use_debug_camera {
            Some(self.debug_camera)
        } else {
            None
        };
        let target_changed = self.target.apply_pending_resize(&self.device);
        self.renderer.render(
            &self.device,
            &self.queue,
            &self.target,
            self.camera,
            debug_camera,
            self.enable_occlusion_culling,
            target_changed,
        );

        if let Some(stats) = self.renderer.take_render_stats(&self.device) {
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
