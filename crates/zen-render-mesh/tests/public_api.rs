use zen_render_mesh::{
    Camera, Instance, Material, Mesh, MeshRenderInput, MeshRenderStats, MeshRenderTargets,
    MeshRenderer, PerspectiveProjection, PreparedMeshFrame, Texture, Vertex,
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
