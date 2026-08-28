use zen_frame_graph::{TextureDesc, TextureViewDesc, UsagePolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HiZPyramidDesc {
    width: u32,
    height: u32,
    mip_level_count: u32,
}

impl HiZPyramidDesc {
    pub(crate) fn new(width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let max_dimension = width.max(height);
        Self {
            width,
            height,
            mip_level_count: u32::BITS - max_dimension.leading_zeros(),
        }
    }

    pub(crate) const fn width(self) -> u32 {
        self.width
    }

    pub(crate) const fn height(self) -> u32 {
        self.height
    }

    pub(crate) const fn mip_level_count(self) -> u32 {
        self.mip_level_count
    }

    pub(crate) fn mip_extent(self, mip: u32) -> wgpu::Extent3d {
        assert!(mip < self.mip_level_count);
        wgpu::Extent3d {
            width: (self.width >> mip).max(1),
            height: (self.height >> mip).max(1),
            depth_or_array_layers: 1,
        }
    }

    pub(crate) fn texture_desc(self) -> TextureDesc {
        TextureDesc {
            label: "hiz-transient".into(),
            size: self.mip_extent(0),
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            view_formats: vec![],
            usage: UsagePolicy::Infer,
        }
    }

    pub(crate) fn mip_view_desc(self, mip: u32) -> TextureViewDesc {
        assert!(mip < self.mip_level_count);
        TextureViewDesc {
            label: format!("hiz-mip-{mip}"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_power_of_two_dimensions_follow_webgpu_mip_extents() {
        let desc = HiZPyramidDesc::new(13, 7);
        assert_eq!(desc.mip_level_count(), 4);
        assert_eq!(
            desc.mip_extent(0),
            wgpu::Extent3d {
                width: 13,
                height: 7,
                depth_or_array_layers: 1,
            }
        );
        assert_eq!(desc.mip_extent(1).width, 6);
        assert_eq!(desc.mip_extent(1).height, 3);
        assert_eq!(desc.mip_extent(3).width, 1);
        assert_eq!(desc.mip_extent(3).height, 1);
    }

    #[test]
    fn one_by_one_and_zero_dimensions_produce_one_mip() {
        for desc in [HiZPyramidDesc::new(1, 1), HiZPyramidDesc::new(0, 0)] {
            assert_eq!(desc.width(), 1);
            assert_eq!(desc.height(), 1);
            assert_eq!(desc.mip_level_count(), 1);
            assert_eq!(desc.mip_extent(0).width, 1);
            assert_eq!(desc.mip_extent(0).height, 1);
        }
    }

    #[test]
    fn logical_descriptors_cover_the_pyramid_without_native_usage() {
        let pyramid = HiZPyramidDesc::new(9, 5);
        let texture = pyramid.texture_desc();
        assert_eq!(texture.format, wgpu::TextureFormat::R32Float);
        assert_eq!(texture.dimension, wgpu::TextureDimension::D2);
        assert_eq!(texture.mip_level_count, 4);
        assert_eq!(texture.sample_count, 1);
        assert_eq!(texture.usage, UsagePolicy::Infer);

        let view = pyramid.mip_view_desc(2);
        assert_eq!(view.base_mip_level, 2);
        assert_eq!(view.mip_level_count, Some(1));
        assert_eq!(view.aspect, wgpu::TextureAspect::All);
    }
}
