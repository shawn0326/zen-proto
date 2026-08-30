use zen_render_mesh::MeshRenderer;

pub async fn request_device_and_queue(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> (wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("no graphics adapter supports the demo surface");

    println!("{:?}", adapter.get_info());
    println!("{:?}", surface.get_capabilities(&adapter).formats);

    let adapter_features = adapter.features();
    let mesh_features = MeshRenderer::required_features();
    if !adapter_features.contains(mesh_features) {
        let missing = mesh_features - adapter_features;
        panic!("Adapter does not support required Mesh renderer features: {missing:?}");
    }

    let optional_timing_features = adapter_features & wgpu::Features::TIMESTAMP_QUERY;
    let required_limits = MeshRenderer::required_limits(&adapter.limits());

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: mesh_features | optional_timing_features,
            required_limits,
            ..Default::default()
        })
        .await
        .expect("failed to create the demo graphics device")
}
