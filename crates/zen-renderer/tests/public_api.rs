use zen_renderer::{
    FrameInput, Renderer,
    camera::{Camera, PerspectiveProjection},
    device::request_device_and_queue,
    mesh::{
        Instance, Material, Mesh, MeshFrameInput, MeshRenderStats, MeshRenderer, Texture, Vertex,
    },
};

#[test]
fn documented_public_api_paths_compile() {
    fn accept_renderer(_: Option<Renderer>) {}
    fn accept_frame_input(_: Option<FrameInput<'static>>) {}
    fn accept_mesh_renderer(_: Option<MeshRenderer>) {}
    fn accept_mesh_input(_: Option<MeshFrameInput>) {}
    fn accept_stats(_: Option<MeshRenderStats>) {}

    accept_renderer(None);
    accept_frame_input(None);
    accept_mesh_renderer(None);
    accept_mesh_input(None);
    accept_stats(None);

    let _ = std::mem::size_of::<Camera>();
    let _ = std::mem::size_of::<PerspectiveProjection>();
    let _ = std::mem::size_of::<Mesh>();
    let _ = std::mem::size_of::<Vertex>();
    let _ = std::mem::size_of::<Material>();
    let _ = std::mem::size_of::<Instance>();
    let _ = std::mem::size_of::<Texture>();
    let _ = Renderer::render;
    let _ = Renderer::mesh;
    let _ = Renderer::mesh_mut;
    let _ = Renderer::set_gpu_debug_groups_enabled;
    let _ = Renderer::request_gpu_timing;
    let _ = Renderer::take_gpu_timing;
    let _ = MeshRenderer::request_stats;
    let _ = MeshRenderer::take_stats;
    let _ = request_device_and_queue;
}
