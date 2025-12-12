#[allow(dead_code)]
pub mod frame_rate_tracker;
#[allow(dead_code)]
pub mod orbit_camera_controller;

use frame_rate_tracker::FrameRateTracker;
use pollster::FutureExt;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

pub trait Example {
    async fn init(window: Arc<Window>) -> Self;
    fn resize(&mut self, width: u32, height: u32);
    fn update(&mut self);
    fn render(&mut self);
}

struct App<E: Example> {
    window: Option<Arc<Window>>,
    example: Option<E>,
    fps_tracker: FrameRateTracker,
}

impl<E: Example> Default for App<E> {
    fn default() -> Self {
        Self {
            window: None,
            example: None,
            fps_tracker: FrameRateTracker::default(),
        }
    }
}

impl<E: Example> ApplicationHandler for App<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let example = E::init(window.clone()).block_on();
        self.window = Some(window);
        self.example = Some(example);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(example) = &mut self.example {
                    example.resize(size.width, size.height);
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(example) = &mut self.example {
                    let fps = self.fps_tracker.record_frame();
                    self.window
                        .as_ref()
                        .unwrap()
                        .set_title(&format!("Basic Scene Example - FPS: {:.1}", fps));

                    example.update();
                    example.render();
                }

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

pub fn run<E: Example>() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::<E>::default();
    event_loop.run_app(&mut app).unwrap();
}
