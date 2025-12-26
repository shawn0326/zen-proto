use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    pub total_instances: u32,
    pub visible_after_main_cull: u32,
    pub drawn_instances: u32,

    pub list_a_visible: u32,
    pub list_a_drawn: u32,
    pub list_b_visible: u32,
    pub list_b_drawn: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct CountersSnapshot {
    a_visible: u32,
    a_draw: u32,
    b_visible: u32,
    b_draw: u32,
}

struct InFlightReadback {
    buffer_index: usize,
    enable_occlusion: bool,
    result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
}

struct PendingReadback {
    buffer_index: usize,
    enable_occlusion: bool,
}

pub struct RenderStatsReadback {
    staging: [wgpu::Buffer; 2],
    next_index: usize,
    requested: bool,
    pending: Option<PendingReadback>,
    in_flight: Option<InFlightReadback>,
    ready: Option<(CountersSnapshot, bool)>,
}

impl RenderStatsReadback {
    pub fn new(device: &wgpu::Device) -> Self {
        let make_staging = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            staging: [
                make_staging("render_stats.staging.0"),
                make_staging("render_stats.staging.1"),
            ],
            next_index: 0,
            requested: false,
            pending: None,
            in_flight: None,
            ready: None,
        }
    }

    pub fn request(&mut self) {
        self.requested = true;
    }

    pub fn encode_if_requested(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        enable_occlusion: bool,
        list_a_visible_count: &wgpu::Buffer,
        list_a_draw_count: &wgpu::Buffer,
        list_b_visible_count: &wgpu::Buffer,
        list_b_draw_count: &wgpu::Buffer,
    ) {
        if !self.requested || self.pending.is_some() || self.in_flight.is_some() {
            return;
        }

        self.requested = false;

        let buffer_index = self.next_index;
        self.next_index = (self.next_index + 1) % self.staging.len();

        let dst = &self.staging[buffer_index];
        encoder.copy_buffer_to_buffer(list_a_visible_count, 0, dst, 0, 4);
        encoder.copy_buffer_to_buffer(list_a_draw_count, 0, dst, 4, 4);
        encoder.copy_buffer_to_buffer(list_b_visible_count, 0, dst, 8, 4);
        encoder.copy_buffer_to_buffer(list_b_draw_count, 0, dst, 12, 4);

        // IMPORTANT: do not call `map_async` here.
        // The buffer is a COPY_DST in this command buffer, and mapping it before submit
        // triggers wgpu validation ("buffer is still mapped"). We defer mapping until
        // after `queue.submit`.
        self.pending = Some(PendingReadback {
            buffer_index,
            enable_occlusion,
        });
    }

    pub fn after_submit(&mut self, device: &wgpu::Device) {
        self.begin_map_if_pending();
        if self.in_flight.is_some() {
            self.pump(device);
        }
    }

    fn begin_map_if_pending(&mut self) {
        if self.in_flight.is_some() {
            return;
        }
        let Some(pending) = self.pending.take() else {
            return;
        };

        let result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let result_clone = result.clone();

        let dst = &self.staging[pending.buffer_index];
        let slice = dst.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            *result_clone.lock().unwrap() = Some(r);
        });

        self.in_flight = Some(InFlightReadback {
            buffer_index: pending.buffer_index,
            enable_occlusion: pending.enable_occlusion,
            result,
        });
    }

    pub fn take_ready(
        &mut self,
        device: &wgpu::Device,
        total_instances: u32,
    ) -> Option<RenderStats> {
        self.pump(device);

        let (snapshot, enable_occlusion) = self.ready.take()?;

        let visible_after_main_cull = if enable_occlusion {
            snapshot.a_visible.saturating_add(snapshot.b_visible)
        } else {
            snapshot.a_visible
        };

        let drawn_instances = if enable_occlusion {
            snapshot.a_draw.saturating_add(snapshot.b_draw)
        } else {
            snapshot.a_draw
        };

        Some(RenderStats {
            total_instances,
            visible_after_main_cull,
            drawn_instances,
            list_a_visible: snapshot.a_visible,
            list_a_drawn: snapshot.a_draw,
            list_b_visible: snapshot.b_visible,
            list_b_drawn: snapshot.b_draw,
        })
    }

    pub fn pump(&mut self, device: &wgpu::Device) {
        let Some(in_flight) = &self.in_flight else {
            return;
        };

        let _ = device.poll(wgpu::PollType::Poll);

        let done = {
            let lock = in_flight.result.lock().unwrap();
            lock.is_some()
        };
        if !done {
            return;
        }

        let in_flight = self.in_flight.take().unwrap();
        let result = in_flight.result.lock().unwrap().take().unwrap();

        let dst = &self.staging[in_flight.buffer_index];
        if result.is_ok() {
            let data = dst.slice(..).get_mapped_range();
            let words: [u32; 4] = bytemuck::cast_slice(&data)[0..4].try_into().unwrap();
            drop(data);
            dst.unmap();

            self.ready = Some((
                CountersSnapshot {
                    a_visible: words[0],
                    a_draw: words[1],
                    b_visible: words[2],
                    b_draw: words[3],
                },
                in_flight.enable_occlusion,
            ));
        } else {
            dst.unmap();
        }
    }
}
