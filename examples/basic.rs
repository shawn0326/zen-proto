mod common;
use common::*;

struct Demo {}

impl Example for Demo {
    fn init(_window: &winit::window::Window) -> Self {
        Demo {}
    }

    fn resize(&mut self, _width: u32, _height: u32) {}

    fn update(&mut self) {}

    fn render(&mut self) {}
}

fn main() {
    run::<Demo>();
}
