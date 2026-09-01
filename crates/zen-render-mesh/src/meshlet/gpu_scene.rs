use std::mem::size_of;

use wgpu::util::DeviceExt;

use super::{
    config::MeshletCapacityConfig,
    gpu_types::{
        BackendWorkCounts, CandidateWork, DispatchIndirectArgs, DrawIndexedIndirectArgs,
        FRAME_UNIFORM_SIZE, FrameUniform, GpuCounters, GpuLodRecord, GpuMeshRecord,
        GpuMeshletRecord, GpuVertex, InstanceClassification, PSO_BIN_COUNT, RASTER_UNIFORM_STRIDE,
        RasterUniform, VisibleMeshletWork, prefix_scan_block_count,
    },
};
use crate::mesh::{Instance, Material};

pub(crate) struct MeshletGpuSceneUpload {
    pub vertices: Vec<GpuVertex>,
    pub meshes: Vec<GpuMeshRecord>,
    pub lods: Vec<GpuLodRecord>,
    pub meshlets: Vec<GpuMeshletRecord>,
    pub meshlet_vertices: Vec<u32>,
    pub micro_indices: Vec<u32>,
    pub fallback_indices: Vec<u32>,
    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,
}

pub(crate) struct MeshletGpuScene {
    pub vertices: wgpu::Buffer,
    pub meshes: wgpu::Buffer,
    pub lods: wgpu::Buffer,
    pub meshlets: wgpu::Buffer,
    pub meshlet_vertices: wgpu::Buffer,
    pub micro_indices: wgpu::Buffer,
    pub fallback_indices: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub materials: wgpu::Buffer,

    pub classifications: wgpu::Buffer,
    pub scan_blocks: wgpu::Buffer,
    pub lod_history: wgpu::Buffer,
    pub candidates: wgpu::Buffer,
    pub visible: wgpu::Buffer,
    pub draw_args: wgpu::Buffer,
    pub counters: wgpu::Buffer,
    pub backend_work_counts: wgpu::Buffer,
    pub mesh_dispatch: wgpu::Buffer,
    pub task_dispatch: wgpu::Buffer,
    pub candidate_dispatch: wgpu::Buffer,
    pub frame_uniform: wgpu::Buffer,
    pub coarse_frame_uniform: wgpu::Buffer,
    pub raster_uniform: wgpu::Buffer,

    pub instance_count: u32,
    pub mesh_count: u32,
    pub capacities: MeshletCapacityConfig,
}

impl MeshletGpuScene {
    pub(crate) fn new(
        device: &wgpu::Device,
        upload: MeshletGpuSceneUpload,
        capacities: MeshletCapacityConfig,
    ) -> Self {
        assert!(
            upload.instances.len() <= capacities.max_instances as usize,
            "scene has {} instances, exceeding configured capacity {}",
            upload.instances.len(),
            capacities.max_instances,
        );
        let instance_count = upload.instances.len() as u32;
        let mesh_count = upload.meshes.len() as u32;

        Self {
            vertices: storage_init(device, "meshlet.scene.vertices", &upload.vertices),
            meshes: storage_init(device, "meshlet.scene.meshes", &upload.meshes),
            lods: storage_init(device, "meshlet.scene.lods", &upload.lods),
            meshlets: storage_init(device, "meshlet.scene.meshlets", &upload.meshlets),
            meshlet_vertices: storage_init(
                device,
                "meshlet.scene.meshlet-vertices",
                &upload.meshlet_vertices,
            ),
            micro_indices: storage_init(
                device,
                "meshlet.scene.micro-indices",
                &upload.micro_indices,
            ),
            fallback_indices: buffer_init(
                device,
                "meshlet.scene.fallback-indices",
                &upload.fallback_indices,
                wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
            ),
            instances: storage_init(device, "meshlet.scene.instances", &upload.instances),
            materials: storage_init(device, "meshlet.scene.materials", &upload.materials),
            classifications: sized_buffer::<InstanceClassification>(
                device,
                "meshlet.work.classifications",
                capacities.max_instances,
                wgpu::BufferUsages::STORAGE,
            ),
            scan_blocks: sized_buffer::<u32>(
                device,
                "meshlet.work.prefix-scan-blocks",
                prefix_scan_block_count(capacities.max_instances),
                wgpu::BufferUsages::STORAGE,
            ),
            lod_history: sized_buffer::<u32>(
                device,
                "meshlet.work.lod-history",
                capacities.max_instances,
                wgpu::BufferUsages::STORAGE,
            ),
            candidates: sized_buffer::<CandidateWork>(
                device,
                "meshlet.work.candidates",
                capacities.max_candidate_meshlets,
                wgpu::BufferUsages::STORAGE,
            ),
            visible: sized_buffer::<VisibleMeshletWork>(
                device,
                "meshlet.work.visible",
                capacities.max_visible_meshlets,
                wgpu::BufferUsages::STORAGE,
            ),
            draw_args: sized_buffer::<DrawIndexedIndirectArgs>(
                device,
                "meshlet.work.draw-args",
                capacities
                    .max_indirect_draws_per_bin
                    .saturating_mul(PSO_BIN_COUNT),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
            counters: sized_buffer::<GpuCounters>(
                device,
                "meshlet.work.counters",
                1,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            backend_work_counts: sized_buffer::<BackendWorkCounts>(
                device,
                "meshlet.work.backend-work-counts",
                1,
                wgpu::BufferUsages::STORAGE,
            ),
            mesh_dispatch: sized_buffer::<DispatchIndirectArgs>(
                device,
                "meshlet.work.mesh-dispatch",
                2,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
            task_dispatch: sized_buffer::<DispatchIndirectArgs>(
                device,
                "meshlet.work.task-dispatch",
                2,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
            candidate_dispatch: sized_buffer::<DispatchIndirectArgs>(
                device,
                "meshlet.work.candidate-dispatch",
                1,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
            frame_uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("meshlet.frame.uniform"),
                size: FRAME_UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            coarse_frame_uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("meshlet.frame.coarse-uniform"),
                size: FRAME_UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            raster_uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("meshlet.raster.uniform"),
                size: RASTER_UNIFORM_STRIDE * 2,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            instance_count,
            mesh_count,
            capacities,
        }
    }

    pub(crate) fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        frame: &FrameUniform,
        coarse_frame: &FrameUniform,
        raster: &[RasterUniform; 2],
    ) {
        queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(frame));
        queue.write_buffer(
            &self.coarse_frame_uniform,
            0,
            bytemuck::bytes_of(coarse_frame),
        );
        for (index, uniform) in raster.iter().enumerate() {
            queue.write_buffer(
                &self.raster_uniform,
                index as u64 * RASTER_UNIFORM_STRIDE,
                bytemuck::bytes_of(uniform),
            );
        }
    }
}

fn storage_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[T],
) -> wgpu::Buffer {
    buffer_init(device, label, contents, wgpu::BufferUsages::STORAGE)
}

fn buffer_init<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    if contents.is_empty() {
        return device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_of::<T>().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(contents),
        usage,
    })
}

fn sized_buffer<T>(
    device: &wgpu::Device,
    label: &'static str,
    count: u32,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (count as u64)
        .checked_mul(size_of::<T>() as u64)
        .expect("validated meshlet buffer size overflow")
        .max(4);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_work_buffers_include_both_pso_bins() {
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let capacities = MeshletCapacityConfig {
            max_instances: 4,
            max_candidate_meshlets: 8,
            max_visible_meshlets: 8,
            max_indirect_draws_per_bin: 4,
        };
        let scene = MeshletGpuScene::new(
            &device,
            MeshletGpuSceneUpload {
                vertices: Vec::new(),
                meshes: Vec::new(),
                lods: Vec::new(),
                meshlets: Vec::new(),
                meshlet_vertices: Vec::new(),
                micro_indices: Vec::new(),
                fallback_indices: Vec::new(),
                instances: Vec::new(),
                materials: Vec::new(),
            },
            capacities,
        );
        assert_eq!(
            scene.visible.size(),
            8 * size_of::<VisibleMeshletWork>() as u64
        );
        assert_eq!(scene.scan_blocks.size(), size_of::<u32>() as u64);
        assert_eq!(
            scene.draw_args.size(),
            8 * size_of::<DrawIndexedIndirectArgs>() as u64
        );
    }
}
