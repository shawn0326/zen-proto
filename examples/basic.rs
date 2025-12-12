mod common;
use common::{
    Example,
    // orbit_camera_controller::{OrbitCameraController, OrbitCameraControllerOptions},
    run,
};
use std::sync::Arc;
use winit::window::Window;
use zen_proto::render::Renderer;

struct Demo {
    renderer: Renderer,
}

impl Example for Demo {
    async fn init(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();
        let renderer = Renderer::new(&instance, surface).await;
        Demo { renderer }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        self.renderer.render();
    }
}

fn main() {
    run::<Demo>();
}
