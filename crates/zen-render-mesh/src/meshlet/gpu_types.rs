use bytemuck::{Pod, Zeroable};

pub(crate) const FRAME_UNIFORM_SIZE: u64 = 304;
pub(crate) const RASTER_UNIFORM_STRIDE: u64 = 256;
pub(crate) const PSO_BIN_COUNT: u32 = 2;
pub(crate) const PREFIX_SCAN_WORKGROUP_SIZE: u32 = 256;

pub(crate) const fn prefix_scan_block_count(instance_count: u32) -> u32 {
    instance_count.div_ceil(PREFIX_SCAN_WORKGROUP_SIZE)
}

pub(crate) const OVERFLOW_CANDIDATES: u32 = 1 << 0;
pub(crate) const OVERFLOW_VISIBLE_BACKFACE: u32 = 1 << 1;
pub(crate) const OVERFLOW_VISIBLE_TWO_SIDED: u32 = 1 << 2;
pub(crate) const OVERFLOW_TASK_PACKETS: u32 = 1 << 3;
pub(crate) const OVERFLOW_DISPATCH: u32 = 1 << 4;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVertex {
    pub position: [f32; 3],
    pub normal_oct: u32,
    pub uv: [f32; 2],
    pub color: u32,
    pub _pad: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuMeshRecord {
    pub first_lod: u32,
    pub lod_count: u32,
    pub _pad: [u32; 2],
    pub sphere: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuLodRecord {
    pub first_meshlet: u32,
    pub meshlet_count: u32,
    pub geometric_error: f32,
    pub _pad: u32,
    pub sphere: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuMeshletRecord {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub triangle_offset: u32,
    pub triangle_count: u32,
    pub fallback_first_index: u32,
    pub fallback_index_count: u32,
    pub _pad: [u32; 2],
    pub sphere: [f32; 4],
    /// xyz is the unit cone axis; w is meshoptimizer's cone cutoff.
    pub cone: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct CandidateWork {
    pub meshlet_id: u32,
    pub instance_id: u32,
    pub material_id: u32,
    pub pso_bin: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct InstanceClassification {
    pub meshlet_count: u32,
    pub meshlet_offset: u32,
    pub selected_lod: u32,
    pub _pad: u32,
}

pub(crate) type VisibleMeshletWork = CandidateWork;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct TaskPacket {
    pub first_meshlet: u32,
    pub meshlet_count: u32,
    pub instance_id: u32,
    pub material_and_bin: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Non-atomic work counts consumed by the mesh/task stages after indirect preparation.
///
/// Keeping these values separate from [`GpuCounters`] gives wgpu an explicit storage-write to
/// storage-read transition between the compute and mesh/task pipeline stages.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct BackendWorkCounts {
    pub mesh: [u32; 2],
    pub task: [u32; 2],
}

/// CPU mirror of the storage buffer whose first fields are atomic in WGSL.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct GpuCounters {
    pub candidate_count: u32,
    pub packet_count_backface: u32,
    pub packet_count_two_sided: u32,
    pub visible_count_backface: u32,
    pub visible_count_two_sided: u32,
    pub instances_visible: u32,
    pub culled_frustum: u32,
    pub culled_cone: u32,
    pub culled_hiz: u32,
    pub output_vertices: u32,
    pub output_primitives: u32,
    pub overflow: u32,
    pub lod_histogram: [u32; 8],
    pub lod_overflow_instances: u32,
    pub conservatively_visible_meshlets: u32,
    pub raster_claim_backface: u32,
    pub raster_claim_two_sided: u32,
    pub _pad: [u32; 8],
}

impl GpuCounters {
    pub(crate) const VISIBLE_BACKFACE_OFFSET: u64 =
        std::mem::offset_of!(Self, visible_count_backface) as u64;
    pub(crate) const VISIBLE_TWO_SIDED_OFFSET: u64 =
        std::mem::offset_of!(Self, visible_count_two_sided) as u64;
    pub(crate) const CONSERVATIVELY_VISIBLE_OFFSET: u64 =
        std::mem::offset_of!(Self, conservatively_visible_meshlets) as u64;
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FrameUniform {
    pub view_projection: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub camera_position: [f32; 4],
    /// width, height, inverse width, inverse height
    pub viewport: [f32; 4],
    /// lod threshold in pixels, hysteresis ratio, near plane, Hi-Z enabled
    pub parameters: [f32; 4],
    /// instance count, mesh count, candidate capacity, per-bin visible capacity
    pub counts: [u32; 4],
    /// task packet capacity per bin, Hi-Z mip count, max dispatch dimension, perspective flag
    pub limits: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct RasterUniform {
    pub view_projection: [[f32; 4]; 4],
    pub visible_base: u32,
    pub task_packet_base: u32,
    pub render_mode: u32,
    pub pso_bin: u32,
}

const _: () = {
    assert!(std::mem::size_of::<GpuVertex>() == 32);
    assert!(std::mem::size_of::<GpuMeshRecord>() == 32);
    assert!(std::mem::size_of::<GpuLodRecord>() == 32);
    assert!(std::mem::size_of::<GpuMeshletRecord>() == 64);
    assert!(std::mem::size_of::<CandidateWork>() == 16);
    assert!(std::mem::size_of::<InstanceClassification>() == 16);
    assert!(std::mem::size_of::<TaskPacket>() == 16);
    assert!(std::mem::size_of::<DrawIndexedIndirectArgs>() == 20);
    assert!(std::mem::size_of::<DispatchIndirectArgs>() == 12);
    assert!(std::mem::size_of::<BackendWorkCounts>() == 16);
    assert!(std::mem::size_of::<GpuCounters>() == 128);
    assert!(std::mem::size_of::<FrameUniform>() as u64 == FRAME_UNIFORM_SIZE);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_count_offsets_match_the_wgsl_counter_header() {
        assert_eq!(GpuCounters::VISIBLE_BACKFACE_OFFSET, 12);
        assert_eq!(GpuCounters::VISIBLE_TWO_SIDED_OFFSET, 16);
        assert_eq!(std::mem::offset_of!(GpuCounters, overflow), 44);
        assert_eq!(std::mem::offset_of!(GpuCounters, lod_histogram), 48);
        assert_eq!(
            std::mem::offset_of!(GpuCounters, lod_overflow_instances),
            80
        );
        assert_eq!(
            std::mem::offset_of!(GpuCounters, conservatively_visible_meshlets),
            84
        );
        assert_eq!(std::mem::offset_of!(GpuCounters, raster_claim_backface), 88);
        assert_eq!(
            std::mem::offset_of!(GpuCounters, raster_claim_two_sided),
            92
        );
        assert_eq!(std::mem::size_of::<GpuCounters>(), 128);
    }

    #[test]
    fn backend_work_counts_match_the_wgsl_abi() {
        assert_eq!(std::mem::offset_of!(BackendWorkCounts, mesh), 0);
        assert_eq!(std::mem::offset_of!(BackendWorkCounts, task), 8);
        assert_eq!(std::mem::size_of::<BackendWorkCounts>(), 16);
        assert_eq!(std::mem::align_of::<BackendWorkCounts>(), 16);
    }

    #[test]
    fn raster_uniform_keeps_its_dynamic_offset_abi() {
        assert_eq!(std::mem::offset_of!(RasterUniform, view_projection), 0);
        assert_eq!(std::mem::offset_of!(RasterUniform, visible_base), 64);
        assert_eq!(std::mem::offset_of!(RasterUniform, task_packet_base), 68);
        assert_eq!(std::mem::offset_of!(RasterUniform, render_mode), 72);
        assert_eq!(std::mem::offset_of!(RasterUniform, pso_bin), 76);
        assert_eq!(std::mem::size_of::<RasterUniform>(), 80);
        assert_eq!(RASTER_UNIFORM_STRIDE, 256);
    }

    #[test]
    fn all_storage_records_keep_sixteen_byte_alignment() {
        for alignment in [
            std::mem::align_of::<GpuVertex>(),
            std::mem::align_of::<GpuMeshRecord>(),
            std::mem::align_of::<GpuLodRecord>(),
            std::mem::align_of::<GpuMeshletRecord>(),
            std::mem::align_of::<CandidateWork>(),
            std::mem::align_of::<InstanceClassification>(),
            std::mem::align_of::<TaskPacket>(),
        ] {
            assert_eq!(alignment, 16);
        }
    }

    #[test]
    fn prefix_scan_block_count_uses_overflow_safe_ceil_division() {
        assert_eq!(prefix_scan_block_count(0), 0);
        assert_eq!(prefix_scan_block_count(1), 1);
        assert_eq!(prefix_scan_block_count(255), 1);
        assert_eq!(prefix_scan_block_count(256), 1);
        assert_eq!(prefix_scan_block_count(257), 2);
        assert_eq!(prefix_scan_block_count(262_144), 1_024);
        assert_eq!(prefix_scan_block_count(u32::MAX), 16_777_216);
    }
}
