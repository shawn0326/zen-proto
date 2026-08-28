//! GPU-driven renderer composition built around a Mesh domain renderer and `zen-frame-graph`.
//!
//! [`Renderer`] owns frame composition and FrameGraph execution. [`mesh::MeshRenderer`] owns
//! Mesh scene resources and records its internal visibility and draw stages into each frame.

pub mod camera;
pub mod device;
pub mod mesh;
mod renderer;

pub use renderer::{FrameInput, Renderer};
pub use zen_frame_graph::{
    GpuTimingNodeKind, GpuTimingNodeReport, GpuTimingReport, GpuTimingUnavailableReason,
};
