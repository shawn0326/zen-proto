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
        .unwrap();

    println!("{:?}", adapter.get_info());
    println!("{:?}", surface.get_capabilities(&adapter).formats);

    let bindless_features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let enabled_bindless_features = adapter.features() & bindless_features;
    if enabled_bindless_features != bindless_features {
        panic!("Adapter does not support required bindless features");
    }

    let optional_timing_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let required_features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::INDIRECT_FIRST_INSTANCE
        | optional_timing_features
        | enabled_bindless_features;
    let adapter_limits = adapter.limits();
    let required_limits = wgpu::Limits {
        max_binding_array_elements_per_shader_stage: 1024
            .min(adapter_limits.max_binding_array_elements_per_shader_stage),
        ..Default::default()
    };

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits,
            ..Default::default()
        })
        .await
        .unwrap()
}
