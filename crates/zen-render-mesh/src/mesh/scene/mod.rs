mod geometry;
mod gpu_scene;
mod instance;
mod material;
mod texture;

fn create_non_empty_buffer_init<T: bytemuck::Pod + bytemuck::Zeroable>(
    device: &wgpu::Device,
    label: &str,
    contents: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;

    let dummy = T::zeroed();
    let contents = if contents.is_empty() {
        bytemuck::bytes_of(&dummy)
    } else {
        bytemuck::cast_slice(contents)
    };

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

pub use geometry::{Mesh, Vertex};
pub use instance::Instance;
pub use material::{Material, MaterialTextureBinding};
pub use texture::{
    Texture, TextureAddressMode, TextureMagFilter, TextureMinFilter, TextureResourceError,
    TextureSampler, TextureSamplingConfig,
};

pub(crate) use geometry::{MeshStorage, VertexPacked};
pub(crate) use gpu_scene::MeshGpuScene;
pub(crate) use instance::InstanceStorage;
pub(crate) use material::MaterialStorage;
pub(crate) use texture::{TextureStorage, TextureUploader};
