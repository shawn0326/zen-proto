use crate::{CompilationReport, CompilationSummary, FrameGraphError, FullCompilationReport};

/// Stable, owned capture envelope for a full CPU compilation report.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameGraphCaptureV1 {
    pub schema: String,
    pub schema_version: u32,
    pub summary: CompilationSummary,
    pub graph: FullCompilationReport,
}

impl TryFrom<&CompilationReport> for FrameGraphCaptureV1 {
    type Error = FrameGraphError;

    fn try_from(report: &CompilationReport) -> Result<Self, Self::Error> {
        let mut graph = report
            .full
            .clone()
            .ok_or(FrameGraphError::CaptureRequiresFullReport)?;
        graph.debug_groups.clear();
        for node in &mut graph.nodes {
            node.debug_group = None;
            node.recording_order = 0;
        }
        for node in &mut graph.culled_nodes {
            node.debug_group = None;
            node.recording_order = 0;
        }
        for resource in &mut graph.resources {
            resource.debug_group = None;
        }
        Ok(Self {
            schema: "frame-graph-capture-v1".into(),
            schema_version: 1,
            summary: report.summary.clone(),
            graph,
        })
    }
}
