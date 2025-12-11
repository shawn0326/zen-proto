use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

pub trait Example {
    fn init(window: &Window) -> Self;
    fn resize(&mut self, width: u32, height: u32);
    fn update(&mut self);
    fn render(&mut self);
}

struct App<E: Example> {
    window: Option<Window>,
    example: Option<E>,
}

impl<E: Example> Default for App<E> {
    fn default() -> Self {
        Self {
            window: None,
            example: None,
        }
    }
}

impl<E: Example> ApplicationHandler for App<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();
        let example = E::init(&window);
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
