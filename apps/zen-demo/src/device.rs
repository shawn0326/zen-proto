use zen_render_mesh::{
    MeshRenderer, TextureSamplingConfig,
    meshlet::{
        MeshletCapabilities, MeshletCapabilityError, MeshletDeviceRequirements,
        MeshletRendererConfig,
    },
};

/// Creates the instance used by the advanced meshlet demo.
///
/// Keeping the backend mask here makes the Vulkan-only contract visible before adapter selection;
/// Direct3D, Metal, GLES, and browser WebGPU adapters cannot accidentally enter this path.
pub fn create_vulkan_instance() -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    wgpu::Instance::new(descriptor)
}

#[derive(Debug, thiserror::Error)]
pub enum MeshletDeviceRequestError {
    #[error("no Vulkan adapter supports the demo surface: {0}")]
    Adapter(String),
    #[error(transparent)]
    Capabilities(#[from] MeshletCapabilityError),
    #[error("failed to create the Vulkan meshlet device: {0}")]
    Device(String),
    #[error("meshlet device configuration failed after adapter selection: {0}")]
    Configuration(String),
}

/// Device request result for a concrete meshlet backend selected once at startup.
pub struct MeshletDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub requirements: MeshletDeviceRequirements,
    pub config: MeshletRendererConfig,
    pub adapter_info: wgpu::AdapterInfo,
    pub sampling: TextureSamplingConfig,
}

/// Selects a Vulkan adapter, resolves `Auto`, and requests exactly the features/limits needed by
/// the selected meshlet backend.
///
/// Mesh/task shader support remains an explicit unsafe opt-in at this application boundary. The
/// indexed-indirect backend uses the normal checked wgpu path and receives a disabled experimental
/// token.
pub async fn request_vulkan_meshlet_device(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
    config: &MeshletRendererConfig,
) -> Result<MeshletDevice, MeshletDeviceRequestError> {
    request_vulkan_meshlet_device_configured(instance, surface, *config, |_, _| Ok(())).await
}

/// Variant that lets the application attach an adapter-identity-bound Auto profile after adapter
/// selection but before capability resolution/device creation.
pub async fn request_vulkan_meshlet_device_configured(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
    mut config: MeshletRendererConfig,
    configure: impl FnOnce(&wgpu::AdapterInfo, &mut MeshletRendererConfig) -> Result<(), String>,
) -> Result<MeshletDevice, MeshletDeviceRequestError> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|error| MeshletDeviceRequestError::Adapter(error.to_string()))?;

    let info = adapter.get_info();
    configure(&info, &mut config).map_err(MeshletDeviceRequestError::Configuration)?;
    let capabilities = MeshletCapabilities::from_adapter(&adapter);
    let sampling = interactive_sampling_config(&adapter);
    let requirements = capabilities.device_requirements(&config)?;
    let surface_capabilities = surface.get_capabilities(&adapter);

    println!("Vulkan adapter: {info:?}");
    println!("Vulkan features: {:?}", capabilities.features());
    println!("Vulkan limits: {:?}", capabilities.limits());
    println!(
        "Vulkan downlevel: {:?}",
        adapter.get_downlevel_capabilities()
    );
    println!("Surface formats: {:?}", surface_capabilities.formats);
    println!(
        "Meshlet backend: {} (bindless textures={}, samplers={})",
        requirements.backend(),
        requirements.bindless_capacity().textures,
        requirements.bindless_capacity().samplers,
    );

    let optional_timing_features = optional_timing_features(capabilities.features());
    // SAFETY: this is the explicit application-level acknowledgement required by wgpu. The
    // requirements object only enables the token for MeshOnly/TaskMesh; IndexedIndirect keeps it
    // disabled. Development runs should keep Vulkan validation layers enabled.
    let experimental_features = unsafe { requirements.experimental_features_token() };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("zen-meshlet-vulkan-device"),
            required_features: requirements.features() | optional_timing_features,
            required_limits: requirements.limits().clone(),
            experimental_features,
            ..Default::default()
        })
        .await
        .map_err(|error| MeshletDeviceRequestError::Device(error.to_string()))?;

    Ok(MeshletDevice {
        device,
        queue,
        requirements,
        config,
        adapter_info: info,
        sampling,
    })
}

pub struct DemoDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub sampling: TextureSamplingConfig,
}

/// Requests the legacy renderer on the same high-performance Vulkan adapter policy used by the
/// meshlet paths, which keeps cross-process benchmark identities comparable.
pub async fn request_vulkan_legacy_device(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> DemoDevice {
    request_device_and_queue_with_info(instance, surface, wgpu::PowerPreference::HighPerformance)
        .await
}

pub async fn request_device_and_queue(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> (wgpu::Device, wgpu::Queue, TextureSamplingConfig) {
    let requested =
        request_device_and_queue_with_info(instance, surface, wgpu::PowerPreference::default())
            .await;
    (requested.device, requested.queue, requested.sampling)
}

pub async fn request_device_and_queue_with_info(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
    power_preference: wgpu::PowerPreference,
) -> DemoDevice {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("no graphics adapter supports the demo surface");

    let adapter_info = adapter.get_info();
    let sampling = interactive_sampling_config(&adapter);
    println!("{adapter_info:?}");
    println!("{:?}", surface.get_capabilities(&adapter).formats);

    let adapter_features = adapter.features();
    let mesh_features = MeshRenderer::required_features();
    if !adapter_features.contains(mesh_features) {
        let missing = mesh_features - adapter_features;
        panic!("Adapter does not support required Mesh renderer features: {missing:?}");
    }

    let optional_timing_features = optional_timing_features(adapter_features);
    let required_limits = MeshRenderer::required_limits(&adapter.limits());

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: mesh_features | optional_timing_features,
            required_limits,
            ..Default::default()
        })
        .await
        .expect("failed to create the demo graphics device");
    DemoDevice {
        device,
        queue,
        adapter_info,
        sampling,
    }
}

fn interactive_sampling_config(adapter: &wgpu::Adapter) -> TextureSamplingConfig {
    let supported = adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::ANISOTROPIC_FILTERING);
    TextureSamplingConfig {
        max_anisotropy: if supported { 16 } else { 1 },
    }
}

fn optional_timing_features(available: wgpu::Features) -> wgpu::Features {
    if !available.contains(wgpu::Features::TIMESTAMP_QUERY) {
        return wgpu::Features::empty();
    }
    available & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
}
