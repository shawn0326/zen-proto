mod common;
use common::{
    Example,
    orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    run,
};
use rand::Rng;
use std::sync::Arc;
use winit::window::Window;
use zen_proto::{
    camera::{Camera, PerspectiveProjection},
    primitive::Primitive,
    render::{RenderContext, Renderer},
};

struct Demo {
    renderer: Renderer,
    cull_camera: Camera,
    draw_camera: Camera,
    camera_controller: OrbitCameraController,
    render_context: RenderContext,
}

impl Example for Demo {
    async fn init(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();
        let renderer = Renderer::new(&instance, surface).await;
        let projection = PerspectiveProjection::default();
        let cull_camera = Camera::new(
            glam::Mat4::look_at_rh(
                glam::vec3(0.0, 0.0, 10.0),
                glam::vec3(0.0, 0.0, 0.0),
                glam::vec3(0.0, 1.0, 0.0),
            )
            .inverse(),
            projection,
        );
        let draw_camera = Camera::new(
            glam::Mat4::look_at_rh(
                glam::vec3(-50.0, 50.0, 50.0),
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
        let primitive_count = 100_0000u32;
        let mut primitives = Vec::with_capacity(primitive_count as usize);
        let mut rng = rand::rng();
        for _ in 0..primitive_count {
            let translation = rng.random::<glam::Vec3>() * 100. - glam::vec3(50.0, 50.0, 50.0);
            let transform = glam::Mat4::from_translation(translation);
            let sphere = glam::Vec4::new(0.0, 0.0, 0.0, 1.0); // 半径为 1 的单位球体
            primitives.push(Primitive { transform, sphere });
        }
        let render_context = renderer.create_context(&primitives);
        Demo {
            renderer,
            cull_camera,
            draw_camera,
            camera_controller,
            render_context,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        self.renderer
            .render(self.cull_camera, self.draw_camera, &self.render_context);
    }

    fn mouse_drag(&mut self, dx: f32, dy: f32) {
        self.camera_controller.orbit(dx * 0.01, dy * 0.01);
        self.cull_camera
            .set_view(self.camera_controller.view_matrix());
    }

    fn mouse_wheel(&mut self, delta_y: f32) {
        self.camera_controller.dolly(delta_y);
        self.cull_camera
            .set_view(self.camera_controller.view_matrix());
    }
}

fn main() {
    run::<Demo>();
}
