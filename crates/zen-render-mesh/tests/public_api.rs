use zen_render_mesh::{
    Camera, Instance, Material, Mesh, MeshRenderInput, MeshRenderStats, MeshRenderTargets,
    MeshRenderer, MeshletBackend, MeshletBuildConfig, MeshletCapabilities,
    MeshletDeviceRequirements, MeshletRenderInput, MeshletRenderMode, MeshletRenderStats,
    MeshletRenderer, MeshletRendererConfig, MeshletSceneAsset, PerspectiveProjection,
    PreparedMeshFrame, PreparedMeshletFrame, Texture, Vertex,
};

#[test]
fn mesh_domain_public_api_paths_compile() {
    fn accept_mesh_renderer(_: Option<MeshRenderer>) {}
    fn accept_mesh_input(_: Option<MeshRenderInput>) {}
    fn accept_prepared_frame(_: Option<PreparedMeshFrame>) {}
    fn accept_targets<'frame>(_: Option<MeshRenderTargets<'frame>>) {}
    fn accept_stats(_: Option<MeshRenderStats>) {}

    accept_mesh_renderer(None);
    accept_mesh_input(None);
    accept_prepared_frame(None);
    accept_targets(None);
    accept_stats(None);

    let _ = std::mem::size_of::<Camera>();
    let _ = std::mem::size_of::<PerspectiveProjection>();
    let _ = std::mem::size_of::<Mesh>();
    let _ = std::mem::size_of::<Vertex>();
    let _ = std::mem::size_of::<Material>();
    let _ = std::mem::size_of::<Instance>();
    let _ = std::mem::size_of::<Texture>();
    let _ = MeshRenderer::new;
    let _ = MeshRenderer::required_features;
    let _ = MeshRenderer::required_limits;
    let _ = MeshRenderer::prepare_frame;
    let _ = MeshRenderer::record_frame_graph;
    let _ = MeshRenderer::after_submit;
    let _ = MeshRenderer::after_discard;
    let _ = MeshRenderer::request_stats;
    let _ = MeshRenderer::take_stats;
    let _ = MeshRenderTargets::new;
}

#[test]
fn meshlet_domain_public_api_paths_compile() {
    fn accept_renderer(_: Option<MeshletRenderer>) {}
    fn accept_input(_: Option<MeshletRenderInput>) {}
    fn accept_prepared(_: Option<PreparedMeshletFrame>) {}
    fn accept_stats(_: Option<MeshletRenderStats>) {}
    fn accept_requirements(_: Option<MeshletDeviceRequirements>) {}

    accept_renderer(None);
    accept_input(None);
    accept_prepared(None);
    accept_stats(None);
    accept_requirements(None);

    let _ = std::mem::size_of::<MeshletRendererConfig>();
    let _ = std::mem::size_of::<MeshletBuildConfig>();
    let _ = std::mem::size_of::<MeshletCapabilities>();
    let _ = std::mem::size_of::<MeshletSceneAsset>();
    let _ = [
        MeshletBackend::Auto,
        MeshletBackend::IndexedIndirect,
        MeshletBackend::MeshOnly,
        MeshletBackend::TaskMesh,
    ];
    assert_eq!(
        MeshletRenderInput::default().render_mode,
        MeshletRenderMode::Shaded
    );
    let _ = [MeshletRenderMode::Shaded, MeshletRenderMode::MeshletId];
    let _ = MeshletRenderer::new;
    let _ = MeshletRenderer::prepare_frame;
    let _ = MeshletRenderer::record_frame_graph;
    let _ = MeshletRenderer::after_submit;
    let _ = MeshletRenderer::after_discard;
    let _ = MeshletRenderer::request_stats;
    let _ = MeshletRenderer::take_stats;
    let _ = MeshletCapabilities::from_adapter;
    let _ = MeshletCapabilities::resolve_backend;
    let _ = MeshletCapabilities::device_requirements;
    let _ = MeshletSceneAsset::build;
    let _ = MeshletSceneAsset::encode_zenmesh;
    let _ = MeshletSceneAsset::decode_zenmesh;
}
