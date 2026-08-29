#[allow(dead_code)]
pub mod frame_rate_tracker;
#[allow(dead_code)]
pub mod gltf_loader;
#[allow(dead_code)]
pub mod orbit_camera_controller;
pub mod surface_state;

use frame_rate_tracker::FrameRateTracker;
use pollster::FutureExt;
use std::time::{Duration, Instant};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, Modifiers, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};
use zen_renderer::{FrameGraphSnapshotV1, Renderer, frame_graph_snapshot_to_json_pretty};

#[allow(async_fn_in_trait)]
pub trait Example {
    const NAME: &'static str;

    async fn init(window: Arc<Window>) -> Self;
    fn resize(&mut self, width: u32, height: u32);
    fn update(&mut self);
    fn render(&mut self);
    fn mouse_drag(&mut self, _dx: f32, _dy: f32) {}
    fn mouse_wheel(&mut self, _delta_y: f32) {}
    fn key_input(&mut self, _key_event: KeyEvent) {}
    fn frame_graph_snapshot_source(&mut self) -> Option<&mut Renderer> {
        None
    }
}

struct App<E: Example> {
    window: Option<Arc<Window>>,
    example: Option<E>,
    fps_tracker: FrameRateTracker,
    tick_interval: Duration,
    next_tick: Option<Instant>,
    fixed_fps: Option<u8>,
    mouse_left_down: bool,
    last_cursor_pos: Option<PhysicalPosition<f64>>,
    modifiers: Modifiers,
}

impl<E: Example> Default for App<E> {
    fn default() -> Self {
        Self {
            window: None,
            example: None,
            fps_tracker: FrameRateTracker::default(),
            tick_interval: Duration::from_nanos(16_666_667), // ~60Hz
            next_tick: None,
            fixed_fps: None,
            mouse_left_down: false,
            last_cursor_pos: None,
            modifiers: Modifiers::default(),
        }
    }
}

impl<E: Example> App<E> {
    fn with_fixed_fps(fps: u8) -> Self {
        Self {
            tick_interval: Duration::from_secs_f64(1.0 / fps as f64),
            fixed_fps: Some(fps),
            ..Default::default()
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

        if let Some(fps) = self.fixed_fps {
            println!("Using fixed FPS: {}", fps);
            self.tick_interval = Duration::from_secs_f64(1.0 / fps as f64);
        } else {
            let hz = window
                .current_monitor()
                .and_then(|m| m.refresh_rate_millihertz())
                .map(|mhz| mhz as f64 / 1000.0)
                .filter(|hz| *hz > 1.0)
                .unwrap_or(60.0);
            println!("Monitor refresh rate: {:.1} Hz", hz);
            self.tick_interval = Duration::from_secs_f64(1.0 / hz);
        }

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
                    if !handle_snapshot_key(example, &event, &self.modifiers) {
                        example.key_input(event);
                    }
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_left_down
                    && let (Some(example), Some(last)) = (&mut self.example, self.last_cursor_pos)
                {
                    let dx = (position.x - last.x) as f32;
                    let dy = (position.y - last.y) as f32;
                    if dx != 0.0 || dy != 0.0 {
                        example.mouse_drag(dx, dy);
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
                    self.window.as_ref().unwrap().set_title(&format!(
                        "{} - FPS: {:.1}",
                        E::NAME,
                        fps
                    ));
                    example.render();
                    poll_and_write_snapshot(example);
                }
            }
            _ => (),
        }
    }
}

fn handle_snapshot_key<E: Example>(
    example: &mut E,
    event: &KeyEvent,
    modifiers: &Modifiers,
) -> bool {
    if !is_snapshot_shortcut(event.physical_key, modifiers.state()) {
        return false;
    }
    if event.state == ElementState::Pressed && !event.repeat {
        match example.frame_graph_snapshot_source() {
            Some(renderer) => {
                renderer.request_frame_graph_snapshot();
                println!("Snapshot requested for the next eligible successful frame");
            }
            None => println!("{} does not support FrameGraph Snapshot capture", E::NAME),
        }
    }
    true
}

fn is_snapshot_shortcut(physical_key: PhysicalKey, modifiers: ModifiersState) -> bool {
    physical_key == KeyCode::KeyS && modifiers.is_empty()
}

fn poll_and_write_snapshot<E: Example>(example: &mut E) {
    let result = example
        .frame_graph_snapshot_source()
        .and_then(Renderer::take_frame_graph_snapshot);
    let Some(result) = result else {
        return;
    };
    match result {
        Ok(snapshot) => match write_snapshot_to_captures(E::NAME, &snapshot) {
            Ok(path) => println!("FrameGraph Snapshot written to {}", path.display()),
            Err(error) => eprintln!("Failed to write FrameGraph Snapshot: {error}"),
        },
        Err(error) => eprintln!("Failed to create FrameGraph Snapshot: {error}"),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SnapshotFileError {
    #[error("failed to encode Snapshot JSON: {0}")]
    Encode(#[from] zen_renderer::SnapshotJsonError),
    #[error("failed to access capture path {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("demo name does not contain a filename-safe character")]
    InvalidDemoName,
}

/// Encodes and atomically creates a collision-safe file below `./captures`.
pub fn write_snapshot_to_captures(
    demo_name: &str,
    snapshot: &FrameGraphSnapshotV1,
) -> Result<PathBuf, SnapshotFileError> {
    let json = frame_graph_snapshot_to_json_pretty(snapshot)?;
    let directory = std::env::current_dir()
        .map_err(|source| SnapshotFileError::Io {
            path: PathBuf::from("."),
            source,
        })?
        .join("captures");
    write_snapshot_json_to_directory(
        &directory,
        demo_name,
        snapshot.capture.frame_index,
        json.as_bytes(),
    )
}

fn write_snapshot_json_to_directory(
    directory: &Path,
    demo_name: &str,
    frame_index: u64,
    json: &[u8],
) -> Result<PathBuf, SnapshotFileError> {
    fs::create_dir_all(directory).map_err(|source| SnapshotFileError::Io {
        path: directory.to_owned(),
        source,
    })?;
    let demo_name = sanitize_demo_name(demo_name).ok_or(SnapshotFileError::InvalidDemoName)?;
    for collision in 0_u64.. {
        let suffix = if collision == 0 {
            String::new()
        } else {
            format!("-{collision}")
        };
        let path = directory.join(format!(
            "{demo_name}-frame-{frame_index}{suffix}.fgsnapshot.json"
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(json)
                    .and_then(|()| file.flush())
                    .map_err(|source| SnapshotFileError::Io {
                        path: path.clone(),
                        source,
                    })?;
                return path
                    .canonicalize()
                    .map_err(|source| SnapshotFileError::Io { path, source });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(SnapshotFileError::Io { path, source }),
        }
    }
    unreachable!("u64 collision suffix space exhausted")
}

fn sanitize_demo_name(name: &str) -> Option<String> {
    let value: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    value
        .chars()
        .any(|character| character.is_ascii_alphanumeric())
        .then_some(value)
}

pub fn run<E: Example>(fixed_fps: Option<u8>) {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = match fixed_fps {
        Some(fps) => App::<E>::with_fixed_fps(fps),
        None => App::<E>::default(),
    };
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod tests {
    use super::{
        SnapshotFileError, is_snapshot_shortcut, sanitize_demo_name,
        write_snapshot_json_to_directory,
    };
    use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

    #[test]
    fn unmodified_s_is_reserved_by_the_shared_runner() {
        assert!(is_snapshot_shortcut(
            PhysicalKey::Code(KeyCode::KeyS),
            ModifiersState::empty(),
        ));
        assert!(!is_snapshot_shortcut(
            PhysicalKey::Code(KeyCode::KeyS),
            ModifiersState::SHIFT,
        ));
        assert!(!is_snapshot_shortcut(
            PhysicalKey::Code(KeyCode::KeyD),
            ModifiersState::empty(),
        ));
    }

    #[test]
    fn capture_names_are_sanitized_and_collisions_never_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let first = write_snapshot_json_to_directory(
            directory.path(),
            "Load GLTF",
            42,
            br#"{"capture":1}"#,
        )
        .unwrap();
        let second = write_snapshot_json_to_directory(
            directory.path(),
            "Load GLTF",
            42,
            br#"{"capture":2}"#,
        )
        .unwrap();

        assert_eq!(
            first.file_name().unwrap(),
            "load-gltf-frame-42.fgsnapshot.json"
        );
        assert_eq!(
            second.file_name().unwrap(),
            "load-gltf-frame-42-1.fgsnapshot.json"
        );
        assert_eq!(std::fs::read(first).unwrap(), br#"{"capture":1}"#);
        assert_eq!(std::fs::read(second).unwrap(), br#"{"capture":2}"#);
    }

    #[test]
    fn file_system_errors_are_returned_to_the_runner() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("not-a-directory");
        std::fs::write(&file, b"occupied").unwrap();

        assert!(matches!(
            write_snapshot_json_to_directory(&file, "basic", 1, b"{}"),
            Err(SnapshotFileError::Io { .. })
        ));
    }

    #[test]
    fn demo_name_requires_a_safe_character() {
        assert_eq!(
            sanitize_demo_name("Basic Scene"),
            Some("basic-scene".into())
        );
        assert_eq!(sanitize_demo_name("..."), None);
    }
}
