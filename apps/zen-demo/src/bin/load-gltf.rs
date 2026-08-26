use std::path::Path;
use std::sync::Arc;
use winit::window::Window;
use zen_demo::{
    Example,
    orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    run,
};
use zen_renderer::{
    camera::{Camera, PerspectiveProjection},
    render::{DefaultRenderer, RenderTarget, request_device_and_target},
};

use zen_demo::gltf_loader::{LoadGltfOptions, load_gltf};

struct Demo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: RenderTarget,
    renderer: DefaultRenderer,
    projection: PerspectiveProjection,
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
        let workspace_dir = manifest_dir
            .parent()
            .and_then(Path::parent)
            .expect("zen-demo must be located under apps/ in the workspace");
        let default_path = manifest_dir
            .join("assets")
            .join("DamagedHelmet")
            .join("glTF")
            .join("DamagedHelmet.gltf");

        let path = std::env::args()
            .nth(1)
            .map(|p| resolve_model_path(workspace_dir, manifest_dir, p))
            .unwrap_or(default_path);

        let model = load_gltf(
            &path,
            LoadGltfOptions {
                global_scale: 1.0,
                flip_v: false,
                bake_node_transform: false,
            },
        );

        let (center, radius) = compute_model_bounds(&model.meshes);

        let projection = PerspectiveProjection {
            aspect: target.width() as f32 / target.height() as f32,
            fovy_deg: 45.0,
            near: 0.1,
            far: 1000.0,
        };
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
            projection,
            camera,
            camera_controller,
            frame_index: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.target.resize(width, height);
        self.projection.aspect = width as f32 / height as f32;
        self.camera.set_projection(self.projection);
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
}

fn compute_model_bounds(meshes: &[zen_renderer::mesh::Mesh]) -> (glam::Vec3, f32) {
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

fn resolve_model_path(
    workspace_dir: &Path,
    manifest_dir: &Path,
    path: impl AsRef<Path>,
) -> std::path::PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_owned()
    } else if let Ok(asset_path) = path.strip_prefix("assets") {
        manifest_dir.join("assets").join(asset_path)
    } else {
        workspace_dir.join(path)
    }
}

fn main() {
    run::<Demo>(Some(120));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_paths_from_workspace_root() {
        let workspace_dir = Path::new("workspace");
        let manifest_dir = workspace_dir.join("apps/zen-demo");

        assert_eq!(
            resolve_model_path(workspace_dir, &manifest_dir, "models/model.gltf"),
            workspace_dir.join("models/model.gltf")
        );
    }

    #[test]
    fn maps_legacy_asset_paths_to_demo_assets() {
        let workspace_dir = Path::new("workspace");
        let manifest_dir = workspace_dir.join("apps/zen-demo");

        assert_eq!(
            resolve_model_path(
                workspace_dir,
                &manifest_dir,
                "assets/DamagedHelmet/glTF/DamagedHelmet.gltf",
            ),
            manifest_dir.join("assets/DamagedHelmet/glTF/DamagedHelmet.gltf")
        );
    }

    #[test]
    fn preserves_absolute_paths() {
        let workspace_dir = Path::new("workspace");
        let manifest_dir = workspace_dir.join("apps/zen-demo");
        let absolute_path = std::env::current_dir().unwrap().join("model.gltf");

        assert_eq!(
            resolve_model_path(workspace_dir, &manifest_dir, &absolute_path),
            absolute_path
        );
    }
}
