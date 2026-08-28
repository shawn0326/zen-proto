#![cfg(feature = "serde")]

use zen_frame_graph::{
    BufferDesc, BufferRange, CompileOptions, FrameGraph, FrameGraphCaptureV1, WriteContents,
};

#[test]
fn full_report_capture_round_trips() {
    let mut graph = FrameGraph::new();
    let mut frame = graph.begin_frame();
    let buffer = frame.create_buffer(BufferDesc::new("buffer", 4)).unwrap();
    let mut pass = frame.compute_pass("write");
    let _ = pass
        .storage_buffer_write(buffer, BufferRange::whole(), WriteContents::Overwrite)
        .unwrap();
    pass.finish().unwrap();
    frame
        .mark_buffer_root(
            buffer,
            BufferRange::whole(),
            zen_frame_graph::RootReason::Output,
        )
        .unwrap();
    let compiled = frame.compile(CompileOptions::full_report()).unwrap();
    let capture = FrameGraphCaptureV1::try_from(compiled.report().unwrap()).unwrap();
    let json = serde_json::to_string_pretty(&capture).unwrap();
    let decoded: FrameGraphCaptureV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, capture);
    assert_eq!(decoded.schema, "frame-graph-capture-v1");
}

#[test]
fn summary_report_cannot_be_captured() {
    let mut graph = FrameGraph::new();
    let frame = graph.begin_frame();
    let compiled = frame.compile(CompileOptions::summary_report()).unwrap();
    assert!(FrameGraphCaptureV1::try_from(compiled.report().unwrap()).is_err());
}

#[test]
fn fixed_fixture_contains_every_schema_required_field() {
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../schema/frame-graph-capture-v1.schema.json")).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/capture-v1.json")).unwrap();
    assert_required(&schema, &fixture);

    let _: FrameGraphCaptureV1 = serde_json::from_value(fixture).unwrap();
}

#[test]
fn callbacks_and_native_runtime_state_are_not_captured() {
    let mut graph = FrameGraph::new();
    let mut frame = graph.begin_frame();
    frame
        .command_pass("callback-only")
        .finish_command(|_| Ok(()))
        .unwrap();
    let compiled = frame.compile(CompileOptions::full_report()).unwrap();
    let capture = FrameGraphCaptureV1::try_from(compiled.report().unwrap()).unwrap();
    let json = serde_json::to_string(&capture).unwrap();
    assert!(json.contains("callback-only"));
    assert!(!json.contains("native_resources"));
    assert!(!json.contains("executors"));
    assert!(!json.contains("resource_pool"));
    assert!(!json.contains("retained_count"));
}

#[test]
fn debug_group_metadata_does_not_change_capture_v1() {
    let mut graph = FrameGraph::new();
    let mut frame = graph.begin_frame();
    let buffer = frame
        .with_debug_group("Mesh", |frame| {
            let buffer = frame.create_buffer(BufferDesc::new("grouped", 4))?;
            let mut pass = frame.compute_pass("grouped-write");
            let _ =
                pass.storage_buffer_write(buffer, BufferRange::whole(), WriteContents::Overwrite)?;
            pass.finish()?;
            Ok(buffer)
        })
        .unwrap();
    frame
        .mark_buffer_root(
            buffer,
            BufferRange::whole(),
            zen_frame_graph::RootReason::Output,
        )
        .unwrap();
    let compiled = frame.compile(CompileOptions::full_report()).unwrap();
    assert!(
        !compiled
            .report()
            .unwrap()
            .full
            .as_ref()
            .unwrap()
            .debug_groups
            .is_empty()
    );

    let capture = FrameGraphCaptureV1::try_from(compiled.report().unwrap()).unwrap();
    let json = serde_json::to_string(&capture).unwrap();
    assert!(!json.contains("debug_groups"));
    assert!(!json.contains("debug_group"));
    assert!(capture.graph.debug_groups.is_empty());
    assert!(
        capture
            .graph
            .nodes
            .iter()
            .all(|node| node.debug_group.is_none())
    );
    assert!(
        capture
            .graph
            .resources
            .iter()
            .all(|resource| resource.debug_group.is_none())
    );
}

fn assert_required(schema: &serde_json::Value, value: &serde_json::Value) {
    for field in schema["required"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(value.get(field).is_some(), "missing required field {field}");
    }
    for section in ["summary", "graph"] {
        for field in schema["properties"][section]["required"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(
                value[section].get(field).is_some(),
                "missing required field {section}.{field}"
            );
        }
    }
}
