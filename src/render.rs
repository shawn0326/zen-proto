mod dispatch_prepare_pass;
mod draw_pass;
mod draw_prepare_pass;
mod hiz_generate_pass;
mod hiz_texture;
mod main_cull_pass;
mod occlusion_cull_pass;
mod render_stats;
mod render_target;
mod visibility_history;
mod visibility_list;

use crate::camera::Camera;
use crate::instance::Instance;
use crate::material::Material;
use crate::mesh::Mesh;
use crate::resources::Resources;
use crate::texture::Texture;
use dispatch_prepare_pass::DispatchPreparePass;
use draw_pass::DrawPass;
use draw_prepare_pass::DrawPreparePass;
use hiz_generate_pass::HiZGeneratePass;
use hiz_texture::HiZTexture;
use main_cull_pass::MainCullPass;
use occlusion_cull_pass::OcclusionCullPass;
pub use render_stats::RenderStats;
use render_stats::RenderStatsReadback;
pub use render_target::RenderTarget;
use visibility_history::VisibilityHistory;
use visibility_list::VisibilityList;

pub async fn request_device_and_target(
    instance: &wgpu::Instance,
    surface: wgpu::Surface<'static>,
) -> (wgpu::Device, wgpu::Queue, RenderTarget) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    println!("{:?}", adapter.get_info());
    println!("{:?}", surface.get_capabilities(&adapter).formats);

    // Only request bindless features needed for sampled-texture (material) bindless.
    let bindless_features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

    let enabled_bindless_features = adapter.features() & bindless_features;

    if enabled_bindless_features != bindless_features {
        panic!("Adapter does not support required bindless features");
    }

    let required_features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        | wgpu::Features::INDIRECT_FIRST_INSTANCE
        | wgpu::Features::TIMESTAMP_QUERY
        | enabled_bindless_features;

    // Bindless texture arrays require non-zero binding-array limits.
    let adapter_limits = adapter.limits();
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_binding_array_elements_per_shader_stage =
        1024.min(adapter_limits.max_binding_array_elements_per_shader_stage);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: required_features,
            required_limits,
            ..Default::default()
        })
        .await
        .unwrap();

    let target = RenderTarget::new(&device, surface);

    (device, queue, target)
}

pub struct DefaultRenderer {
    resources: Resources,

    list_a: VisibilityList,
    list_b: VisibilityList,

    visibility_history: VisibilityHistory,

    hiz_texture: HiZTexture,

    main_cull_pass: MainCullPass,
    dispatch_prepare_pass: DispatchPreparePass,
    draw_prepare_pass: DrawPreparePass,
    occlusion_cull_pass: OcclusionCullPass,
    draw_pass: DrawPass,
    hiz_generate_pass: HiZGeneratePass,

    stats_readback: RenderStatsReadback,
}

impl DefaultRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &RenderTarget,
        meshes: &[Mesh],
        materials: &[Material],
        instances: &[Instance],
        textures: &[Texture],
    ) -> DefaultRenderer {
        // buffers

        let resources = Resources::new(device, queue, meshes, materials, instances, textures);
        let max_instance_count = resources.instances().instance_count();

        let list_a = VisibilityList::new(device, "list_a", max_instance_count);
        let list_b = VisibilityList::new(device, "list_b", max_instance_count);

        let visibility_history = VisibilityHistory::new(device, max_instance_count);

        let hiz_texture = HiZTexture::new(device, target.width(), target.height());
        let hiz_view = hiz_texture.sampled_full_view();

        // passes

        let main_cull_pass = MainCullPass::new(device);
        main_cull_pass.prepare(device, &resources, &visibility_history, &list_a, &list_b);

        let dispatch_prepare_pass = DispatchPreparePass::new(device);
        dispatch_prepare_pass.prepare(device, &list_a);
        dispatch_prepare_pass.prepare(device, &list_b);

        let draw_prepare_pass = DrawPreparePass::new(device);
        draw_prepare_pass.prepare(device, &resources, &visibility_history, &list_a);
        draw_prepare_pass.prepare(device, &resources, &visibility_history, &list_b);

        let occlusion_cull_pass = OcclusionCullPass::new(device);
        occlusion_cull_pass.prepare(device, &resources, &visibility_history, hiz_view, &list_a);
        occlusion_cull_pass.prepare(device, &resources, &visibility_history, hiz_view, &list_b);

        let draw_pass = DrawPass::new(device, target.format(), &resources);

        let mut hiz_generate_pass = HiZGeneratePass::new(device);
        hiz_generate_pass.prepare(device, target.depth_for_hiz_view(), &hiz_texture);

        // assemble

        Self {
            resources,
            list_a,
            list_b,
            visibility_history,
            hiz_texture,
            main_cull_pass,
            dispatch_prepare_pass,
            draw_prepare_pass,
            occlusion_cull_pass,
            draw_pass,
            hiz_generate_pass,

            stats_readback: RenderStatsReadback::new(device),
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &RenderTarget,
        camera: Camera,
        debug_camera: Option<Camera>,
        enable_occlusion_culling: bool,
        target_changed: bool,
    ) {
        let resources = &self.resources;
        let max_instance_count = resources.instances().instance_count();

        let target_context = target.get_target_context();

        let target_context = match target_context {
            Ok(context) => context,
            Err(wgpu::SurfaceError::Lost) => {
                println!("Surface lost");
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                println!("Out of memory");
                return;
            }
            Err(e) => {
                println!("Failed to acquire next swap chain texture: {:?}", e);
                return;
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        self.list_a.reset(queue);
        self.list_b.reset(queue);

        self.main_cull_pass
            .update(queue, resources, &camera, enable_occlusion_culling);
        self.main_cull_pass.encode(&mut encoder, max_instance_count);

        self.dispatch_prepare_pass
            .encode(&mut encoder, &self.list_a);
        self.draw_prepare_pass.encode(&mut encoder, &self.list_a);

        self.draw_pass.update(&queue, &camera, 0);
        self.draw_pass.encode(
            &mut encoder,
            &target_context,
            &self.resources.meshes().index_buffer(),
            &self.list_a,
            max_instance_count,
            true,
            true,
            0,
        );

        if enable_occlusion_culling {
            if target_changed {
                let hiz_texture = HiZTexture::new(device, target.width(), target.height());
                let hiz_view = hiz_texture.sampled_full_view();

                self.hiz_generate_pass
                    .prepare(device, target.depth_for_hiz_view(), &hiz_texture);

                self.occlusion_cull_pass.clear_cache();
                self.occlusion_cull_pass.prepare(
                    device,
                    &resources,
                    &self.visibility_history,
                    hiz_view,
                    &self.list_a,
                );
                self.occlusion_cull_pass.prepare(
                    device,
                    &resources,
                    &self.visibility_history,
                    hiz_view,
                    &self.list_b,
                );

                self.hiz_texture = hiz_texture;
            }

            self.hiz_generate_pass
                .encode(&mut encoder, &self.hiz_texture);

            self.dispatch_prepare_pass
                .encode(&mut encoder, &self.list_b);

            // Occlusion cull List B: update visibility_history based on Hi-Z.
            // (History for List A will be handled later.)
            self.occlusion_cull_pass
                .update(&queue, &camera, target.width(), target.height());
            self.occlusion_cull_pass.encode(&mut encoder, &self.list_b);

            self.draw_prepare_pass.encode(&mut encoder, &self.list_b);

            self.draw_pass.encode(
                &mut encoder,
                &target_context,
                &self.resources.meshes().index_buffer(),
                &self.list_b,
                max_instance_count,
                false,
                false,
                0,
            );

            self.hiz_generate_pass
                .encode(&mut encoder, &self.hiz_texture);

            // Occlusion cull List B: update visibility_history based on Hi-Z.
            // (History for List A will be handled later.)
            self.occlusion_cull_pass.encode(&mut encoder, &self.list_a);
        }

        if let Some(debug_camera) = debug_camera {
            self.draw_pass.update(&queue, &debug_camera, 1);

            self.draw_pass.encode(
                &mut encoder,
                &target_context,
                &self.resources.meshes().index_buffer(),
                &self.list_a,
                max_instance_count,
                true,
                true,
                1,
            );

            self.draw_pass.encode(
                &mut encoder,
                &target_context,
                &self.resources.meshes().index_buffer(),
                &self.list_b,
                max_instance_count,
                false,
                false,
                1,
            );
        }

        // Optional low-frequency stats readback (copies 4x u32 into a small staging buffer).
        self.stats_readback.encode_if_requested(
            &mut encoder,
            enable_occlusion_culling,
            self.list_a.visible_count_buffer(),
            self.list_a.draw_count_buffer(),
            self.list_b.visible_count_buffer(),
            self.list_b.draw_count_buffer(),
        );

        queue.submit(Some(encoder.finish()));

        // Start mapping after submit, then progress mapping if needed.
        self.stats_readback.after_submit(device);

        target_context.surface_texture.present();
    }

    /// Request a low-overhead GPU->CPU readback of render counters.
    ///
    /// The readback is performed on the *next* `render()` call, and can be retrieved later via
    /// `take_render_stats()`.
    pub fn request_render_stats(&mut self) {
        self.stats_readback.request();
    }

    /// Returns a newly available stats snapshot, if the previously requested readback has finished.
    ///
    /// This is non-blocking; call it every frame (or at your preferred cadence).
    pub fn take_render_stats(&mut self, device: &wgpu::Device) -> Option<RenderStats> {
        self.stats_readback
            .take_ready(device, self.resources.instances().instance_count())
    }
}
