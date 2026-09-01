use std::ops::{BitOr, BitOrAssign};

use zen_frame_graph::{GpuTimingReport, GpuTimingUnavailableReason};

use super::gpu_types::{
    OVERFLOW_CANDIDATES, OVERFLOW_DISPATCH, OVERFLOW_VISIBLE_BACKFACE, OVERFLOW_VISIBLE_TWO_SIDED,
};

/// Number of individual LOD levels retained in the fixed-size stats snapshot.
pub const MESHLET_STATS_LOD_BUCKETS: usize = 8;
/// Stats are consumed only after this many submitted frames, avoiding a same-frame synchronization.
pub const MESHLET_STATS_READBACK_DELAY_FRAMES: u32 = 3;

/// A fixed GPU arena whose writes must be capacity-checked.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshletCapacityKind {
    Instances,
    CandidateMeshlets,
    VisibleMeshlets,
    IndirectDraws,
    DispatchWorkgroups,
}

impl MeshletCapacityKind {
    #[must_use]
    pub const fn overflow_flag(self) -> MeshletOverflowFlags {
        match self {
            Self::Instances => MeshletOverflowFlags::INSTANCES,
            Self::CandidateMeshlets => MeshletOverflowFlags::CANDIDATE_MESHLETS,
            Self::VisibleMeshlets => MeshletOverflowFlags::VISIBLE_MESHLETS,
            // Indexed draw args and visible work are allocated together, one entry per meshlet.
            Self::IndirectDraws => MeshletOverflowFlags::VISIBLE_MESHLETS,
            Self::DispatchWorkgroups => MeshletOverflowFlags::DISPATCH_WORKGROUPS,
        }
    }
}

/// GPU-written sticky overflow bits for one frame.
///
/// Producers set a bit whenever an atomic reservation is clamped. No pass clears an earlier bit;
/// the counter-clear pass resets the complete word at the beginning of the next frame.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MeshletOverflowFlags(u32);

impl MeshletOverflowFlags {
    pub const NONE: Self = Self(0);
    // Bits 0 through 3 are part of the GpuCounters/WGSL ABI.
    pub const CANDIDATE_MESHLETS: Self = Self(OVERFLOW_CANDIDATES);
    pub const VISIBLE_BACKFACE: Self = Self(OVERFLOW_VISIBLE_BACKFACE);
    pub const VISIBLE_TWO_SIDED: Self = Self(OVERFLOW_VISIBLE_TWO_SIDED);
    pub const VISIBLE_MESHLETS: Self = Self(Self::VISIBLE_BACKFACE.0 | Self::VISIBLE_TWO_SIDED.0);
    pub const DISPATCH_WORKGROUPS: Self = Self(OVERFLOW_DISPATCH);
    /// Reserved for a future GPU-side instance arena; current uploads reject excess instances.
    pub const INSTANCES: Self = Self(1 << 4);
    pub const ALL: Self = Self(
        Self::CANDIDATE_MESHLETS.0
            | Self::VISIBLE_BACKFACE.0
            | Self::VISIBLE_TWO_SIDED.0
            | Self::DISPATCH_WORKGROUPS.0
            | Self::INSTANCES.0,
    );

    #[must_use]
    pub const fn from_bits_retain(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl BitOr for MeshletOverflowFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for MeshletOverflowFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}

/// Counts split by the two static opaque PSO bins.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeshletPsoBinStats {
    pub opaque_backface: u32,
    pub opaque_two_sided: u32,
}

impl MeshletPsoBinStats {
    #[must_use]
    pub const fn total(self) -> u32 {
        self.opaque_backface.saturating_add(self.opaque_two_sided)
    }
}

/// Timestamp-query results for the fixed meshlet frame topology, in nanoseconds.
///
/// A field is `None` when timestamp queries were unavailable or that pass was not recorded for the
/// selected backend/configuration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeshletGpuPassTimings {
    pub clear_frame_counters_ns: Option<u64>,
    pub instance_classify_lod_count_ns: Option<u64>,
    pub prefix_scan_ns: Option<u64>,
    pub candidate_scatter_ns: Option<u64>,
    pub coarse_cull_ns: Option<u64>,
    pub occluder_depth_ns: Option<u64>,
    pub hiz_build_ns: Option<u64>,
    pub clear_coarse_results_ns: Option<u64>,
    pub final_cull_ns: Option<u64>,
    pub indirect_prepare_ns: Option<u64>,
    pub backend_raster_ns: Option<u64>,
    pub stats_copy_ns: Option<u64>,
}

impl MeshletGpuPassTimings {
    #[must_use]
    pub fn measured_total_ns(self) -> u64 {
        [
            self.clear_frame_counters_ns,
            self.instance_classify_lod_count_ns,
            self.prefix_scan_ns,
            self.candidate_scatter_ns,
            self.coarse_cull_ns,
            self.occluder_depth_ns,
            self.hiz_build_ns,
            self.clear_coarse_results_ns,
            self.final_cull_ns,
            self.indirect_prepare_ns,
            self.backend_raster_ns,
            self.stats_copy_ns,
        ]
        .into_iter()
        .flatten()
        .fold(0, u64::saturating_add)
    }

    fn add_label_duration(&mut self, label: &str, duration_ns: u64) {
        let destination = match label {
            "meshlet.clear-frame-counters" => &mut self.clear_frame_counters_ns,
            "meshlet.instance-classify-lod-count" => &mut self.instance_classify_lod_count_ns,
            "meshlet.prefix-scan" => &mut self.prefix_scan_ns,
            "meshlet.candidate-scatter" => &mut self.candidate_scatter_ns,
            "meshlet.coarse-cull" => &mut self.coarse_cull_ns,
            "meshlet.opaque-occluder-depth" => &mut self.occluder_depth_ns,
            "meshlet.clear-coarse-results" => &mut self.clear_coarse_results_ns,
            "meshlet.final-cull" => &mut self.final_cull_ns,
            "meshlet.indirect-prepare" => &mut self.indirect_prepare_ns,
            "meshlet.stats-readback" => &mut self.stats_copy_ns,
            label if label.starts_with("meshlet.hiz-") => &mut self.hiz_build_ns,
            label if label.starts_with("meshlet.backend-raster.") => &mut self.backend_raster_ns,
            _ => return,
        };
        *destination = Some(destination.unwrap_or(0).saturating_add(duration_ns));
    }
}

/// One FrameGraph timing result decoded against the fixed meshlet pass labels.
///
/// The explicit `frame_index` is the association key for delayed counter readback. Timings must
/// never be paired by arrival order because timestamp and counter mappings complete independently.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeshletGpuFrameTimings {
    pub frame_index: u64,
    pub frame_total_ns: u64,
    pub passes: MeshletGpuPassTimings,
}

impl MeshletGpuFrameTimings {
    pub fn from_gpu_timing_report(report: &GpuTimingReport) -> Result<Self, MeshletGpuTimingError> {
        let GpuTimingReport::Available {
            frame_index,
            frame_duration,
            nodes,
            ..
        } = report
        else {
            return match report {
                GpuTimingReport::Unavailable {
                    frame_index,
                    reason,
                } => Err(MeshletGpuTimingError::Unavailable {
                    frame_index: *frame_index,
                    reason: *reason,
                }),
                _ => Err(MeshletGpuTimingError::UnsupportedReport),
            };
        };

        let mut passes = MeshletGpuPassTimings::default();
        for node in nodes {
            passes.add_label_duration(&node.label, duration_ns(node.duration));
        }
        Ok(Self {
            frame_index: *frame_index,
            frame_total_ns: duration_ns(*frame_duration),
            passes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MeshletGpuTimingError {
    #[error("GPU timings for frame {frame_index} are unavailable: {reason:?}")]
    Unavailable {
        frame_index: u64,
        reason: GpuTimingUnavailableReason,
    },
    #[error("unsupported future GPU timing report format")]
    UnsupportedReport,
    #[error(
        "GPU timing frame mismatch: stats are frame {stats_frame_index}, timing is frame {timing_frame_index}"
    )]
    FrameMismatch {
        stats_frame_index: u64,
        timing_frame_index: u64,
    },
    #[error("frame {frame_index} has no pending meshlet counter readback")]
    NoPendingStats { frame_index: u64 },
}

fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Three-frames-delayed counters and timings for one submitted meshlet frame.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeshletRenderStats {
    /// Frame identity supplied in [`super::MeshletRenderInput`].
    pub frame_index: u64,
    pub total_instances: u32,
    pub classified_instances: u32,
    pub visible_instances: u32,
    /// Instance counts for LOD 0 through 7.
    pub lod_instances: [u32; MESHLET_STATS_LOD_BUCKETS],
    /// Instances selecting LOD 8 or above, kept separate to avoid silently losing counts.
    pub lod_overflow_instances: u32,

    pub candidate_meshlets: u32,
    pub visible_meshlets: u32,
    pub frustum_culled_meshlets: u32,
    pub normal_cone_culled_meshlets: u32,
    pub hiz_culled_meshlets: u32,
    /// Near-plane, invalid-bound, and otherwise uncertain tests that were conservatively retained.
    pub conservatively_visible_meshlets: u32,

    /// Logical 32-meshlet task groups for `TaskMesh`; zero for the other concrete backends.
    pub task_workgroups: u32,
    pub visible_meshlets_per_bin: MeshletPsoBinStats,
    pub indirect_draws_per_bin: MeshletPsoBinStats,
    /// Logical vertices in the compute-compacted visible set.
    pub output_vertices: u64,
    /// Logical primitives in the compute-compacted visible set.
    pub output_primitives: u64,

    pub overflow: MeshletOverflowFlags,
    pub gpu_timings: MeshletGpuPassTimings,
}

impl MeshletRenderStats {
    /// Associates one timestamp report with this delayed counter snapshot after checking identity.
    pub fn associate_gpu_timings(
        &mut self,
        timings: MeshletGpuFrameTimings,
    ) -> Result<(), MeshletGpuTimingError> {
        if self.frame_index != timings.frame_index {
            return Err(MeshletGpuTimingError::FrameMismatch {
                stats_frame_index: self.frame_index,
                timing_frame_index: timings.frame_index,
            });
        }
        self.gpu_timings = timings.passes;
        Ok(())
    }

    /// Returns the number of instances represented by the complete reported LOD distribution.
    #[must_use]
    pub fn lod_instance_total(self) -> u32 {
        self.lod_instances
            .into_iter()
            .fold(self.lod_overflow_instances, u32::saturating_add)
    }

    /// A snapshot with no capacity loss. This does not imply that timestamp queries were present.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        self.overflow.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_bits_are_sticky_and_preserve_unknown_gpu_bits() {
        let mut flags = MeshletOverflowFlags::NONE;
        flags |= MeshletCapacityKind::VisibleMeshlets.overflow_flag();
        flags |= MeshletCapacityKind::IndirectDraws.overflow_flag();
        assert!(flags.contains(MeshletOverflowFlags::VISIBLE_MESHLETS));
        assert!(flags.contains(MeshletOverflowFlags::VISIBLE_BACKFACE));
        assert!(flags.contains(MeshletOverflowFlags::VISIBLE_TWO_SIDED));

        let future_bit = 1 << 31;
        let decoded = MeshletOverflowFlags::from_bits_retain(flags.bits() | future_bit);
        assert_eq!(decoded.bits() & future_bit, future_bit);
    }

    #[test]
    fn lod_total_includes_the_overflow_bucket() {
        let stats = MeshletRenderStats {
            lod_instances: [1, 2, 3, 4, 0, 0, 0, 0],
            lod_overflow_instances: 5,
            ..Default::default()
        };
        assert_eq!(stats.lod_instance_total(), 15);
    }

    #[test]
    fn timing_total_ignores_unavailable_passes() {
        let timings = MeshletGpuPassTimings {
            prefix_scan_ns: Some(20),
            backend_raster_ns: Some(30),
            ..Default::default()
        };
        assert_eq!(timings.measured_total_ns(), 50);
    }

    #[test]
    fn timing_labels_aggregate_hiz_mips_and_ignore_other_renderers() {
        let mut timings = MeshletGpuPassTimings::default();
        timings.add_label_duration("meshlet.hiz-depth-to-mip0", 10);
        timings.add_label_duration("meshlet.hiz-mip0-to-mip1", 15);
        timings.add_label_duration("legacy.raster", 75);
        assert_eq!(timings.hiz_build_ns, Some(25));
        assert_eq!(timings.measured_total_ns(), 25);
    }

    #[test]
    fn stats_reject_timing_from_another_frame() {
        let mut stats = MeshletRenderStats {
            frame_index: 4,
            ..Default::default()
        };
        let error = stats
            .associate_gpu_timings(MeshletGpuFrameTimings {
                frame_index: 5,
                frame_total_ns: 9,
                passes: MeshletGpuPassTimings {
                    prefix_scan_ns: Some(3),
                    ..Default::default()
                },
            })
            .unwrap_err();
        assert!(matches!(error, MeshletGpuTimingError::FrameMismatch { .. }));
        assert_eq!(stats.gpu_timings, MeshletGpuPassTimings::default());
    }

    #[test]
    fn pso_bin_total_saturates() {
        let bins = MeshletPsoBinStats {
            opaque_backface: u32::MAX,
            opaque_two_sided: 1,
        };
        assert_eq!(bins.total(), u32::MAX);
    }
}
