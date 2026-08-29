//! GPU-driven renderer composition built around a Mesh domain renderer and `zen-frame-graph`.
//!
//! [`Renderer`] owns frame composition and FrameGraph execution. [`mesh::MeshRenderer`] owns
//! Mesh scene resources and records its internal visibility and draw stages into each frame.

pub mod camera;
pub mod device;
pub mod mesh;
mod renderer;

pub use renderer::{FrameInput, Renderer};
#[cfg(feature = "snapshot")]
pub use zen_frame_graph::snapshot::{
    FrameGraphSnapshotV1, SnapshotExportError, SnapshotJsonError,
    to_json as frame_graph_snapshot_to_json, to_json_pretty as frame_graph_snapshot_to_json_pretty,
};
pub use zen_frame_graph::{
    GpuTimingNodeKind, GpuTimingNodeReport, GpuTimingReport, GpuTimingUnavailableReason,
};
