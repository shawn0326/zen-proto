use super::config::{MeshletBackend, TASK_MESHLETS_PER_WORKGROUP};
use super::gpu_types::GpuCounters;
use super::stats::{
    MESHLET_STATS_READBACK_DELAY_FRAMES, MeshletGpuFrameTimings, MeshletGpuTimingError,
    MeshletOverflowFlags, MeshletPsoBinStats, MeshletRenderStats,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

const READBACK_RING_SIZE: usize = 3;
const COUNTERS_SIZE: u64 = std::mem::size_of::<GpuCounters>() as u64;

type MapResult = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

enum SlotState {
    Idle,
    Submitted {
        eligible_epoch: u64,
        frame_index: u64,
        sequence: u64,
    },
    Mapping {
        result: MapResult,
        frame_index: u64,
        sequence: u64,
    },
}

/// Asynchronous, three-submission-delayed readback for the meshlet counter block.
///
/// Copying the counter buffer is deliberately owned by the renderer/FrameGraph. A request is only
/// consumed by [`Self::commit_submitted`], so discarding a prepared frame leaves the request intact.
pub(crate) struct MeshletStatsReadback {
    staging: [wgpu::Buffer; READBACK_RING_SIZE],
    slots: [SlotState; READBACK_RING_SIZE],
    next_index: usize,
    requested: bool,
    requested_timing: bool,
    submission_epoch: u64,
    next_sequence: u64,
    total_instances: u32,
    backend: MeshletBackend,
    ready: BTreeMap<u64, MeshletRenderStats>,
    timings: BTreeMap<u64, MeshletGpuFrameTimings>,
    expected_timings: BTreeSet<u64>,
}

impl MeshletStatsReadback {
    pub(crate) fn new(
        device: &wgpu::Device,
        total_instances: u32,
        backend: MeshletBackend,
    ) -> Self {
        let make_staging = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: COUNTERS_SIZE,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            staging: [
                make_staging("meshlet.stats.staging.0"),
                make_staging("meshlet.stats.staging.1"),
                make_staging("meshlet.stats.staging.2"),
            ],
            slots: [SlotState::Idle, SlotState::Idle, SlotState::Idle],
            next_index: 0,
            requested: false,
            requested_timing: false,
            submission_epoch: 0,
            next_sequence: 0,
            total_instances,
            backend,
            ready: BTreeMap::new(),
            timings: BTreeMap::new(),
            expected_timings: BTreeSet::new(),
        }
    }

    pub(crate) fn request(&mut self) {
        self.requested = true;
    }

    pub(crate) fn can_request_immediately(&self) -> bool {
        !self.requested && matches!(&self.slots[self.next_index], SlotState::Idle)
    }

    pub(crate) fn try_request(&mut self) -> bool {
        if !self.can_request_immediately() {
            return false;
        }
        self.requested = true;
        true
    }

    pub(crate) fn try_request_with_gpu_timing(&mut self) -> bool {
        if !self.try_request() {
            return false;
        }
        self.requested_timing = true;
        true
    }

    /// Returns the stable destination chosen for the next counter copy.
    ///
    /// Calling this method is side-effect free. In particular, a subsequently discarded frame does
    /// not consume the request or advance the ring.
    pub(crate) fn planned_buffer_index(&self) -> Option<usize> {
        if !self.requested || !matches!(&self.slots[self.next_index], SlotState::Idle) {
            return None;
        }
        Some(self.next_index)
    }

    pub(crate) fn staging_buffer(&self, index: usize) -> &wgpu::Buffer {
        &self.staging[index]
    }

    /// Marks a planned copy as submitted. The renderer must not call this for a discarded frame.
    pub(crate) fn commit_submitted(&mut self, index: usize, frame_index: u64) {
        assert_eq!(
            self.planned_buffer_index(),
            Some(index),
            "meshlet stats readback plan changed before submission"
        );

        self.requested = false;
        if self.requested_timing {
            assert!(
                !self.expected_timings.contains(&frame_index)
                    && !self
                        .ready
                        .values()
                        .any(|stats| stats.frame_index == frame_index)
                    && !self.slots.iter().enumerate().any(|(slot_index, slot)| {
                        slot_index != index
                            && matches!(
                                slot,
                                SlotState::Submitted {
                                    frame_index: pending,
                                    ..
                                } | SlotState::Mapping {
                                    frame_index: pending,
                                    ..
                                } if *pending == frame_index
                            )
                    }),
                "GPU-timed meshlet stats require a unique in-flight frame_index"
            );
            self.expected_timings.insert(frame_index);
        }
        self.requested_timing = false;
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.slots[index] = SlotState::Submitted {
            eligible_epoch: self
                .submission_epoch
                .saturating_add(u64::from(MESHLET_STATS_READBACK_DELAY_FRAMES)),
            frame_index,
            sequence,
        };
        self.next_index = (index + 1) % READBACK_RING_SIZE;
    }

    /// Advances the submission clock, starts eligible maps, and performs a non-blocking poll.
    pub(crate) fn after_submit(&mut self, device: &wgpu::Device) {
        self.submission_epoch = self.submission_epoch.saturating_add(1);
        self.begin_eligible_maps();
        self.pump(device);
    }

    /// Polls already-started maps and returns the oldest completed snapshot, if one is ready.
    pub(crate) fn take_ready(&mut self, device: &wgpu::Device) -> Option<MeshletRenderStats> {
        self.pump(device);
        let (&sequence, stats) = self.ready.first_key_value()?;
        if self.expected_timings.contains(&stats.frame_index) {
            return None;
        }
        self.ready.remove(&sequence)
    }

    /// Associates an independently completed timestamp report by explicit frame identity.
    pub(crate) fn associate_gpu_timing(
        &mut self,
        report: &zen_frame_graph::GpuTimingReport,
    ) -> Result<(), MeshletGpuTimingError> {
        let timing = match MeshletGpuFrameTimings::from_gpu_timing_report(report) {
            Ok(timing) => timing,
            Err(error @ MeshletGpuTimingError::Unavailable { frame_index, .. }) => {
                self.expected_timings.remove(&frame_index);
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        self.associate_frame_timing(timing)
    }

    fn associate_frame_timing(
        &mut self,
        timing: MeshletGpuFrameTimings,
    ) -> Result<(), MeshletGpuTimingError> {
        if let Some(stats) = self
            .ready
            .values_mut()
            .find(|stats| stats.frame_index == timing.frame_index)
        {
            stats.associate_gpu_timings(timing)?;
            self.expected_timings.remove(&timing.frame_index);
            return Ok(());
        }
        let pending = self.slots.iter().any(|slot| match slot {
            SlotState::Submitted { frame_index, .. } | SlotState::Mapping { frame_index, .. } => {
                *frame_index == timing.frame_index
            }
            SlotState::Idle => false,
        });
        if !pending {
            return Err(MeshletGpuTimingError::NoPendingStats {
                frame_index: timing.frame_index,
            });
        }
        self.timings.insert(timing.frame_index, timing);
        Ok(())
    }

    fn begin_eligible_maps(&mut self) {
        for (slot, staging) in self.slots.iter_mut().zip(&self.staging) {
            let (frame_index, sequence) = match &*slot {
                SlotState::Submitted {
                    eligible_epoch,
                    frame_index,
                    sequence,
                } if *eligible_epoch <= self.submission_epoch => (*frame_index, *sequence),
                SlotState::Idle | SlotState::Submitted { .. } | SlotState::Mapping { .. } => {
                    continue;
                }
            };

            let result: MapResult = Arc::new(Mutex::new(None));
            let callback_result = Arc::clone(&result);
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |map_result| {
                    *callback_result.lock().unwrap() = Some(map_result);
                });
            *slot = SlotState::Mapping {
                result,
                frame_index,
                sequence,
            };
        }
    }

    fn pump(&mut self, device: &wgpu::Device) {
        if !self
            .slots
            .iter()
            .any(|slot| matches!(slot, SlotState::Mapping { .. }))
        {
            return;
        }

        let _ = device.poll(wgpu::PollType::Poll);

        for (slot, buffer) in self.slots.iter_mut().zip(&self.staging) {
            let Some((frame_index, sequence, completed)) = (match &*slot {
                SlotState::Mapping {
                    result,
                    frame_index,
                    sequence,
                } => Some((*frame_index, *sequence, result.lock().unwrap().take())),
                SlotState::Idle | SlotState::Submitted { .. } => None,
            }) else {
                continue;
            };
            let Some(completed) = completed else {
                continue;
            };

            *slot = SlotState::Idle;
            if completed.is_err() {
                self.timings.remove(&frame_index);
                self.expected_timings.remove(&frame_index);
                buffer.unmap();
                continue;
            }

            let mapped = match buffer.slice(..).get_mapped_range() {
                Ok(mapped) => mapped,
                Err(error) => {
                    eprintln!("Failed to read mapped meshlet stats: {error}");
                    self.timings.remove(&frame_index);
                    self.expected_timings.remove(&frame_index);
                    buffer.unmap();
                    continue;
                }
            };
            let counters =
                bytemuck::pod_read_unaligned::<GpuCounters>(&mapped[..COUNTERS_SIZE as usize]);
            drop(mapped);
            buffer.unmap();
            let mut stats = Self::decode(frame_index, self.total_instances, self.backend, counters);
            if let Some(timing) = self.timings.remove(&frame_index) {
                stats
                    .associate_gpu_timings(timing)
                    .expect("timing map is keyed by the same frame identity");
                self.expected_timings.remove(&frame_index);
            }
            self.ready.insert(sequence, stats);
        }
    }

    fn decode(
        frame_index: u64,
        total_instances: u32,
        backend: MeshletBackend,
        counters: GpuCounters,
    ) -> MeshletRenderStats {
        let visible_meshlets_per_bin = MeshletPsoBinStats {
            opaque_backface: counters.visible_count_backface,
            opaque_two_sided: counters.visible_count_two_sided,
        };
        let task_workgroups = if backend == MeshletBackend::TaskMesh {
            counters
                .visible_count_backface
                .div_ceil(TASK_MESHLETS_PER_WORKGROUP)
                .saturating_add(
                    counters
                        .visible_count_two_sided
                        .div_ceil(TASK_MESHLETS_PER_WORKGROUP),
                )
        } else {
            0
        };
        MeshletRenderStats {
            frame_index,
            total_instances,
            // The classify dispatch covers every uploaded instance. The current ABI has no
            // separate invalid-instance counter with which to refine this CPU-known value.
            classified_instances: total_instances,
            visible_instances: counters.instances_visible,
            lod_instances: counters.lod_histogram,
            lod_overflow_instances: counters.lod_overflow_instances,
            candidate_meshlets: counters.candidate_count,
            visible_meshlets: visible_meshlets_per_bin.total(),
            frustum_culled_meshlets: counters.culled_frustum,
            normal_cone_culled_meshlets: counters.culled_cone,
            hiz_culled_meshlets: counters.culled_hiz,
            conservatively_visible_meshlets: counters.conservatively_visible_meshlets,
            task_workgroups,
            visible_meshlets_per_bin,
            // The indexed fallback emits exactly one indirect draw per visible meshlet. Mesh paths
            // retain this logical count so comparisons use the same unit.
            indirect_draws_per_bin: visible_meshlets_per_bin,
            output_vertices: u64::from(counters.output_vertices),
            output_primitives: u64::from(counters.output_primitives),
            overflow: MeshletOverflowFlags::from_bits_retain(counters.overflow),
            gpu_timings: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planning_and_discard_do_not_consume_the_request() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 42, MeshletBackend::IndexedIndirect);

        readback.request();
        assert_eq!(readback.planned_buffer_index(), Some(0));
        assert_eq!(readback.planned_buffer_index(), Some(0));

        // A renderer discard performs no commit. Other submissions must not alter the plan.
        readback.after_submit(&device);
        assert_eq!(readback.planned_buffer_index(), Some(0));
        assert!(readback.requested);
        assert_eq!(readback.next_index, 0);
    }

    #[test]
    fn mapping_does_not_begin_before_three_after_submit_calls() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 1, MeshletBackend::IndexedIndirect);

        readback.request();
        readback.commit_submitted(0, 17);
        assert!(matches!(
            &readback.slots[0],
            SlotState::Submitted {
                eligible_epoch: 3,
                frame_index: 17,
                ..
            }
        ));

        readback.after_submit(&device);
        assert!(matches!(&readback.slots[0], SlotState::Submitted { .. }));
        readback.after_submit(&device);
        assert!(matches!(&readback.slots[0], SlotState::Submitted { .. }));

        readback.after_submit(&device);
        assert!(!matches!(&readback.slots[0], SlotState::Submitted { .. }));
    }

    #[test]
    fn counter_decode_preserves_bins_and_saturates_totals() {
        let counters = GpuCounters {
            candidate_count: 17,
            visible_count_backface: u32::MAX,
            visible_count_two_sided: 1,
            instances_visible: 4,
            culled_frustum: 5,
            culled_cone: 6,
            culled_hiz: 7,
            output_vertices: 8,
            output_primitives: 9,
            overflow: 1 << 31,
            lod_histogram: [1, 2, 1, 0, 0, 0, 0, 0],
            lod_overflow_instances: 11,
            conservatively_visible_meshlets: 13,
        };

        let stats = MeshletStatsReadback::decode(44, 12, MeshletBackend::TaskMesh, counters);
        assert_eq!(stats.frame_index, 44);
        assert_eq!(stats.total_instances, 12);
        assert_eq!(stats.classified_instances, 12);
        assert_eq!(stats.visible_meshlets, u32::MAX);
        assert_eq!(stats.task_workgroups, 134_217_729);
        assert_eq!(stats.visible_meshlets_per_bin.opaque_two_sided, 1);
        assert_eq!(stats.indirect_draws_per_bin, stats.visible_meshlets_per_bin);
        assert_eq!(stats.lod_overflow_instances, 11);
        assert_eq!(stats.conservatively_visible_meshlets, 13);
        assert_eq!(stats.overflow.bits(), 1 << 31);
        assert_eq!(stats.gpu_timings, Default::default());
    }

    #[test]
    fn explicit_timing_association_uses_frame_identity_not_arrival_order() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 1, MeshletBackend::IndexedIndirect);

        assert!(readback.try_request_with_gpu_timing());
        readback.commit_submitted(0, 10);
        assert!(readback.try_request_with_gpu_timing());
        readback.commit_submitted(1, 11);

        let frame_11 = MeshletGpuFrameTimings {
            frame_index: 11,
            frame_total_ns: 22,
            passes: super::super::stats::MeshletGpuPassTimings {
                prefix_scan_ns: Some(7),
                ..Default::default()
            },
        };
        let frame_10 = MeshletGpuFrameTimings {
            frame_index: 10,
            frame_total_ns: 20,
            passes: super::super::stats::MeshletGpuPassTimings {
                prefix_scan_ns: Some(5),
                ..Default::default()
            },
        };
        readback.associate_frame_timing(frame_11).unwrap();
        readback.associate_frame_timing(frame_10).unwrap();
        assert_eq!(readback.timings.get(&10), Some(&frame_10));
        assert_eq!(readback.timings.get(&11), Some(&frame_11));

        let wrong = readback
            .associate_frame_timing(MeshletGpuFrameTimings {
                frame_index: 12,
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(
            wrong,
            MeshletGpuTimingError::NoPendingStats { frame_index: 12 }
        );
    }

    #[test]
    fn discard_has_no_frame_that_can_accept_timing() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 1, MeshletBackend::IndexedIndirect);
        assert!(readback.try_request_with_gpu_timing());
        assert_eq!(readback.planned_buffer_index(), Some(0));

        let error = readback
            .associate_frame_timing(MeshletGpuFrameTimings {
                frame_index: 7,
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(
            error,
            MeshletGpuTimingError::NoPendingStats { frame_index: 7 }
        );
        assert!(readback.timings.is_empty());
    }

    #[test]
    fn unavailable_report_releases_expected_timing_without_cross_frame_data() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 1, MeshletBackend::IndexedIndirect);
        assert!(readback.try_request_with_gpu_timing());
        readback.commit_submitted(0, 8);

        let error = readback
            .associate_gpu_timing(&zen_frame_graph::GpuTimingReport::Unavailable {
                frame_index: 8,
                reason: zen_frame_graph::GpuTimingUnavailableReason::Unsupported,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            MeshletGpuTimingError::Unavailable { frame_index: 8, .. }
        ));
        assert!(!readback.expected_timings.contains(&8));
        assert!(readback.timings.is_empty());
    }

    #[test]
    fn a_full_ring_rejects_timed_requests_without_leaving_a_future_waiter() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 1, MeshletBackend::IndexedIndirect);

        for (slot, frame_index) in [10_u64, 11, 12].into_iter().enumerate() {
            assert!(readback.try_request_with_gpu_timing());
            readback.commit_submitted(slot, frame_index);
        }
        assert!(!readback.can_request_immediately());
        assert!(!readback.try_request_with_gpu_timing());
        assert!(!readback.requested);
        assert!(!readback.requested_timing);

        let error = readback
            .associate_frame_timing(MeshletGpuFrameTimings {
                frame_index: 13,
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(
            error,
            MeshletGpuTimingError::NoPendingStats { frame_index: 13 }
        );
        assert!(!readback.expected_timings.contains(&13));
    }

    #[test]
    fn completed_stats_are_delivered_in_submission_order_not_slot_order() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut readback = MeshletStatsReadback::new(&device, 1, MeshletBackend::IndexedIndirect);
        readback.ready.insert(
            2,
            MeshletStatsReadback::decode(
                30,
                1,
                MeshletBackend::IndexedIndirect,
                GpuCounters::default(),
            ),
        );
        readback.ready.insert(
            1,
            MeshletStatsReadback::decode(
                20,
                1,
                MeshletBackend::IndexedIndirect,
                GpuCounters::default(),
            ),
        );

        assert_eq!(readback.take_ready(&device).unwrap().frame_index, 20);
        assert_eq!(readback.take_ready(&device).unwrap().frame_index, 30);
    }
}
