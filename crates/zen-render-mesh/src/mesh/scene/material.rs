#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialTextureBinding {
    pub texture_id: u32,
    pub sampler_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub albedo_factor: glam::Vec4,
    /// xyz: emissive factor (linear), w: occlusion strength
    pub emissive_ao: glam::Vec4,
    pub albedo: MaterialTextureBinding,
    pub emissive: MaterialTextureBinding,
    pub occlusion: MaterialTextureBinding,
    pub _padding: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<Material>() == 64);
const _: () = assert!(std::mem::align_of::<Material>() == 16);

pub(crate) struct MaterialStorage {
    material_buffer: wgpu::Buffer,
}

impl MaterialStorage {
    pub fn from_materials(device: &wgpu::Device, materials: &[Material]) -> Self {
        let material_buffer = super::create_non_empty_buffer_init(
            device,
            "materials.material_buffer",
            materials,
            wgpu::BufferUsages::STORAGE,
        );

        Self { material_buffer }
    }

    pub(crate) fn material_buffer(&self) -> &wgpu::Buffer {
        &self.material_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::{Material, MaterialTextureBinding};

    #[test]
    fn material_matches_the_64_byte_gpu_abi() {
        assert_eq!(std::mem::size_of::<MaterialTextureBinding>(), 8);
        assert_eq!(std::mem::size_of::<Material>(), 64);
        assert_eq!(std::mem::align_of::<Material>(), 16);
        assert_eq!(std::mem::offset_of!(Material, albedo_factor), 0);
        assert_eq!(std::mem::offset_of!(Material, emissive_ao), 16);
        assert_eq!(std::mem::offset_of!(Material, albedo), 32);
        assert_eq!(std::mem::offset_of!(Material, emissive), 40);
        assert_eq!(std::mem::offset_of!(Material, occlusion), 48);
        assert_eq!(std::mem::offset_of!(Material, _padding), 56);
    }

    #[test]
    fn every_shader_material_matches_the_rust_offsets() {
        for (label, type_name, source) in [
            (
                "legacy",
                "Material",
                include_str!("../../../shaders/mesh/draw.wgsl"),
            ),
            (
                "indexed",
                "MaterialData",
                include_str!("../../../shaders/meshlet/indexed.wgsl"),
            ),
            (
                "mesh",
                "MaterialData",
                include_str!("../../../shaders/meshlet/mesh.wgsl"),
            ),
        ] {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            let ty = module
                .types
                .iter()
                .map(|(_, ty)| ty)
                .find(|ty| ty.name.as_deref() == Some(type_name))
                .unwrap_or_else(|| panic!("{label} shader has no {type_name}"));
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("{label} material is not a struct");
            };
            assert_eq!(*span, 64, "{label} material size");
            let offsets = members
                .iter()
                .map(|member| member.offset)
                .collect::<Vec<_>>();
            assert_eq!(offsets, [0, 16, 32, 40, 48, 56], "{label} offsets");
        }
    }
}
