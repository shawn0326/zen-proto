#[allow(dead_code)]
pub mod frame_rate_tracker;
#[allow(dead_code)]
pub mod orbit_camera_controller;

use frame_rate_tracker::FrameRateTracker;
use pollster::FutureExt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

pub trait Example {
    async fn init(window: Arc<Window>) -> Self;
    fn resize(&mut self, width: u32, height: u32);
    fn update(&mut self);
    fn render(&mut self);
    fn mouse_drag(&mut self, _dx: f32, _dy: f32) {}
    fn mouse_wheel(&mut self, _delta_y: f32) {}
    fn key_input(&mut self, _key_event: KeyEvent) {}
}

struct App<E: Example> {
    window: Option<Arc<Window>>,
    example: Option<E>,
    fps_tracker: FrameRateTracker,
    tick_interval: Duration,
    next_tick: Option<Instant>,
    mouse_left_down: bool,
    last_cursor_pos: Option<PhysicalPosition<f64>>,
}

impl<E: Example> Default for App<E> {
    fn default() -> Self {
        Self {
            window: None,
            example: None,
            fps_tracker: FrameRateTracker::default(),
            tick_interval: Duration::from_nanos(8_333_333), // ~120Hz
            next_tick: None,
            mouse_left_down: false,
            last_cursor_pos: None,
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
        self.next_tick = Some(Instant::now());
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(example)) = (self.window.as_ref(), self.example.as_mut()) else {
            return;
        };

        let now = Instant::now();
        let interval = self.tick_interval;
        let mut next = self.next_tick.unwrap_or(now);

        let mut updated = false;
        let mut steps = 0;
        while now >= next && steps < 5 {
            example.update();
            updated = true;
            steps += 1;
            next += interval;
        }
        if steps == 5 && now >= next {
            next = now + interval;
        }

        self.next_tick = Some(next);

        if updated {
            window.request_redraw();
        }

        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
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
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_left_down = state == ElementState::Pressed;
                    if !self.mouse_left_down {
                        self.last_cursor_pos = None;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(example) = &mut self.example {
                    example.key_input(event);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_left_down {
                    if let (Some(example), Some(last)) = (&mut self.example, self.last_cursor_pos) {
                        let dx = (position.x - last.x) as f32;
                        let dy = (position.y - last.y) as f32;
                        if dx != 0.0 || dy != 0.0 {
                            example.mouse_drag(dx, dy);
                        }
                    }
                }
                self.last_cursor_pos = Some(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };

                if let Some(example) = &mut self.example {
                    example.mouse_wheel(dy);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(example) = &mut self.example {
                    let fps = self.fps_tracker.record_frame();
                    self.window
                        .as_ref()
                        .unwrap()
                        .set_title(&format!("Basic Scene Example - FPS: {:.1}", fps));
                    example.render();
                }
            }
            _ => (),
        }
    }
}

pub fn run<E: Example>() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::<E>::default();
    event_loop.run_app(&mut app).unwrap();
}
