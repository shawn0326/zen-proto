mod common;

use common::{
    Example,
    orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    run,
};
use std::path::Path;
use std::sync::Arc;
use winit::window::Window;
use zen_proto::{
    camera::{Camera, PerspectiveProjection},
    render::{DefaultRenderer, RenderTarget, request_device_and_target},
};

use common::gltf_loader::{LoadGltfOptions, load_gltf};

struct Demo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: RenderTarget,
    renderer: DefaultRenderer,
    camera: Camera,
    camera_controller: OrbitCameraController,
    frame_index: u64,
}

impl Example for Demo {
    async fn init(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();
        let (device, queue, target) = request_device_and_target(&instance, surface).await;

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let default_path = manifest_dir
            .join("assets")
            .join("DamagedHelmet")
            .join("glTF")
            .join("DamagedHelmet.gltf");

        let path = std::env::args()
            .nth(1)
            .map(|p| manifest_dir.join(p))
            .unwrap_or(default_path);

        let model = load_gltf(
            &path,
            LoadGltfOptions {
                global_scale: 1.0,
                flip_v: false,
                bake_node_transform: true,
            },
        );

        let (center, radius) = compute_model_bounds(&model.meshes);

        let projection = PerspectiveProjection::default();
        let camera_pos = center + glam::vec3(0.0, 0.0, radius.max(0.01) * 3.0);
        let camera = Camera::new(
            glam::Mat4::look_at_rh(camera_pos, center, glam::vec3(0.0, 1.0, 0.0)).inverse(),
            projection,
        );
        let camera_controller = OrbitCameraController::new(OrbitCameraControllerOptions {
            target: center,
            position: Some(camera_pos),
            ..Default::default()
        });

        let renderer = DefaultRenderer::new(
            &device,
            &queue,
            &target,
            &model.meshes,
            &model.materials,
            &model.instances,
            &model.textures,
        );

        Self {
            device,
            queue,
            target,
            renderer,
            camera,
            camera_controller,
            frame_index: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.target.resize(width, height);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        self.frame_index += 1;
        if self.frame_index % 120 == 0 {
            self.renderer.request_render_stats();
        }

        let target_changed = self.target.apply_pending_resize(&self.device);
        self.renderer.render(
            &self.device,
            &self.queue,
            &self.target,
            self.camera,
            None,
            true,
            target_changed,
        );
    }

    fn mouse_drag(&mut self, dx: f32, dy: f32) {
        self.camera_controller.orbit(dx * 0.01, dy * 0.01);
        self.camera.set_view(self.camera_controller.view_matrix());
    }

    fn mouse_wheel(&mut self, delta_y: f32) {
        self.camera_controller.dolly(delta_y);
        self.camera.set_view(self.camera_controller.view_matrix());
    }
}

fn compute_model_bounds(meshes: &[zen_proto::mesh::Mesh]) -> (glam::Vec3, f32) {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);

    for m in meshes {
        for v in &m.vertices {
            let p = glam::Vec3::new(v.position.x, v.position.y, v.position.z);
            min = min.min(p);
            max = max.max(p);
        }
    }

    if !min.x.is_finite() {
        return (glam::Vec3::ZERO, 1.0);
    }

    let center = (min + max) * 0.5;
    let ext = (max - min) * 0.5;
    let radius = ext.length();
    (center, radius)
}

fn main() {
    run::<Demo>(Some(120));
}
