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
    material, mesh,
    primitive::Primitive,
    render::{RenderContext, Renderer},
};

struct Demo {
    renderer: Renderer,
    camera: Camera,
    debug_camera: Camera,
    camera_controller: OrbitCameraController,
    render_context: RenderContext,
    use_debug_camera: bool,
}

impl Example for Demo {
    async fn init(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();
        let renderer = Renderer::new(&instance, surface).await;
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

        let mut meshes = vec![];
        meshes.push(mesh::create_triangle_mesh());
        meshes.push(mesh::create_box_mesh());

        let mut materials = vec![];
        materials.push(material::Material {
            color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        });
        materials.push(material::Material {
            color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
        });
        materials.push(material::Material {
            color: glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
        });

        let primitive_count = 100_0000u32;
        let mut primitives = Vec::with_capacity(primitive_count as usize);
        let mut rng = rand::rng();
        for i in 0..primitive_count {
            let translation = rng.random::<glam::Vec3>() * 200. - glam::Vec3::ONE * 100.;
            let transform = glam::Mat4::from_translation(translation);
            primitives.push(Primitive {
                transform,
                mesh_id: i % 2,
                material_id: i % 3,
                _pad: [0; 2],
            });
        }

        let render_context = RenderContext::new(&renderer, &meshes, &materials, &primitives);
        Demo {
            renderer,
            camera,
            debug_camera,
            camera_controller,
            render_context,
            use_debug_camera: false,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    fn update(&mut self) {}

    fn render(&mut self) {
        let debug_camera = if self.use_debug_camera {
            Some(self.debug_camera)
        } else {
            None
        };
        self.renderer
            .render(self.camera, debug_camera, &self.render_context);
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
        }
    }
}

fn main() {
    run::<Demo>();
}
