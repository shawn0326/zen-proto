use crate::mesh::frame::MeshGraphResources;
use std::sync::{Arc, Mutex};
use zen_frame_graph::{Buffer, Frame, FrameGraphError};

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshRenderStats {
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

pub(crate) struct MeshStatsReadback {
    staging: [wgpu::Buffer; 2],
    next_index: usize,
    requested: bool,
    pending: Option<PendingReadback>,
    in_flight: Option<InFlightReadback>,
    ready: Option<(CountersSnapshot, bool)>,
}

impl MeshStatsReadback {
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

    pub fn planned_buffer_index(&self) -> Option<usize> {
        if !self.requested || self.pending.is_some() || self.in_flight.is_some() {
            return None;
        }
        Some(self.next_index)
    }

    pub fn staging_buffer(&self, buffer_index: usize) -> &wgpu::Buffer {
        &self.staging[buffer_index]
    }

    pub(crate) fn record_copy<'frame>(
        &self,
        frame: &mut Frame<'frame>,
        resources: &MeshGraphResources<'frame>,
    ) -> Result<Buffer<'frame>, FrameGraphError> {
        let destination = resources
            .readback
            .ok_or_else(|| FrameGraphError::Internal {
                message: "stats copy was recorded without a readback buffer".into(),
            })?;
        let mut pass = frame.copy_pass("stats-readback");
        pass.set_side_effect(false);
        for (source, destination_offset) in [
            resources.list_a.visible_count,
            resources.list_a.draw_count,
            resources.list_b.visible_count,
            resources.list_b.draw_count,
        ]
        .into_iter()
        .zip([0, 4, 8, 12])
        {
            pass.copy_buffer_to_buffer(source, 0, destination, destination_offset, 4)?;
        }
        pass.finish()?;
        Ok(destination)
    }

    pub fn commit_submitted(&mut self, buffer_index: usize, enable_occlusion: bool) {
        assert_eq!(
            self.planned_buffer_index(),
            Some(buffer_index),
            "render stats readback plan changed before execution"
        );

        self.requested = false;
        self.next_index = (self.next_index + 1) % self.staging.len();

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
    ) -> Option<MeshRenderStats> {
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

        Some(MeshRenderStats {
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
            let data = match dst.slice(..).get_mapped_range() {
                Ok(data) => data,
                Err(error) => {
                    eprintln!("Failed to read mapped GPU stats: {error}");
                    dst.unmap();
                    return;
                }
            };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planning_does_not_consume_a_request_or_advance_the_ring() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshStatsReadback::new(&device);

        readback.request();
        assert_eq!(readback.planned_buffer_index(), Some(0));
        assert_eq!(readback.planned_buffer_index(), Some(0));
        assert!(readback.requested);
        assert_eq!(readback.next_index, 0);
        assert!(readback.pending.is_none());
    }

    #[test]
    fn successful_execute_commit_is_the_only_state_transition() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshStatsReadback::new(&device);

        readback.request();
        let planned = readback.planned_buffer_index().unwrap();
        readback.commit_submitted(planned, true);

        assert!(!readback.requested);
        assert_eq!(readback.next_index, 1);
        let pending = readback.pending.as_ref().unwrap();
        assert_eq!(pending.buffer_index, 0);
        assert!(pending.enable_occlusion);
        assert_eq!(readback.planned_buffer_index(), None);
    }
}
