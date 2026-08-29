use glam::Mat4;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveProjection {
    pub fovy_deg: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for PerspectiveProjection {
    fn default() -> Self {
        Self {
            fovy_deg: 60.0,
            aspect: 1.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl From<PerspectiveProjection> for Mat4 {
    fn from(projection: PerspectiveProjection) -> Self {
        Mat4::perspective_rh(
            projection.fovy_deg.to_radians(),
            projection.aspect,
            projection.near,
            projection.far,
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrthographicProjection {
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrthographicProjection {
    fn default() -> Self {
        Self {
            left: -1.0,
            right: 1.0,
            bottom: -1.0,
            top: 1.0,
            near: 0.0,
            far: 1.0,
        }
    }
}

impl From<OrthographicProjection> for Mat4 {
    fn from(projection: OrthographicProjection) -> Self {
        Mat4::orthographic_rh(
            projection.left,
            projection.right,
            projection.bottom,
            projection.top,
            projection.near,
            projection.far,
        )
    }
}

impl OrthographicProjection {
    #[inline]
    pub fn from_width_height(width: f32, height: f32, near: f32, far: f32) -> Self {
        let hw = width * 0.5;
        let hh = height * 0.5;
        Self {
            left: -hw,
            right: hw,
            bottom: -hh,
            top: hh,
            near,
            far,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    transform: Mat4,
    projection: Mat4,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            transform: Mat4::IDENTITY,
            projection: PerspectiveProjection::default().into(),
        }
    }
}

impl Camera {
    #[inline]
    pub fn new(transform: Mat4, projection: impl Into<Mat4>) -> Self {
        Self {
            transform,
            projection: projection.into(),
        }
    }

    #[inline]
    pub fn from_projection(projection: impl Into<Mat4>) -> Self {
        Self {
            transform: Mat4::IDENTITY,
            projection: projection.into(),
        }
    }

    #[inline]
    pub fn set_transform(&mut self, transform: Mat4) -> &mut Self {
        self.transform = transform;
        self
    }

    #[inline]
    pub fn transform(&self) -> Mat4 {
        self.transform
    }

    #[inline]
    pub fn set_projection(&mut self, projection: impl Into<Mat4>) -> &mut Self {
        self.projection = projection.into();
        self
    }

    #[inline]
    pub fn projection(&self) -> Mat4 {
        self.projection
    }

    #[inline]
    pub fn set_view(&mut self, view: Mat4) -> &mut Self {
        self.transform = view.inverse();
        self
    }

    #[inline]
    pub fn view(&self) -> Mat4 {
        self.transform.inverse()
    }

    #[inline]
    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view()
    }

    pub fn frustum(&self) -> [glam::Vec4; 6] {
        let view_proj = self.view_projection();
        let m = view_proj.to_cols_array();

        let row = |i| glam::Vec4::new(m[i], m[i + 4], m[i + 8], m[i + 12]);

        let m0 = row(0);
        let m1 = row(1);
        let m2 = row(2);
        let m3 = row(3);

        let normalize_plane = |plane: glam::Vec4| {
            let normal_length = plane.truncate().length();
            if normal_length > 0.0 && normal_length.is_finite() {
                plane / normal_length
            } else {
                // Infinite projections have no finite far plane. A constant-positive plane keeps
                // every sphere on the inside instead of introducing NaNs into culling.
                glam::Vec4::W
            }
        };

        let mut planes = [glam::Vec4::ZERO; 6];
        planes[0] = normalize_plane(m3 + m0); // left: x >= -w
        planes[1] = normalize_plane(m3 - m0); // right: x <= w
        planes[2] = normalize_plane(m3 + m1); // bottom: y >= -w
        planes[3] = normalize_plane(m3 - m1); // top: y <= w
        planes[4] = normalize_plane(m2); // near: z >= 0
        planes[5] = normalize_plane(m3 - m2); // far: z <= w

        planes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3, Vec4};

    fn signed_distance(plane: Vec4, point: Vec3) -> f32 {
        plane.dot(point.extend(1.0))
    }

    fn assert_approx_eq(actual: f32, expected: f32) {
        let error = (actual - expected).abs();
        assert!(
            error <= 2.0e-3,
            "expected {expected}, got {actual} (absolute error {error})"
        );
    }

    #[test]
    fn perspective_frustum_matches_webgpu_clip_depth_and_world_space_distance() {
        let near = 0.25;
        let far = 80.0;
        let projection = PerspectiveProjection {
            fovy_deg: 70.0,
            aspect: 16.0 / 9.0,
            near,
            far,
        };
        let transform = Mat4::from_rotation_translation(
            Quat::from_rotation_y(0.37) * Quat::from_rotation_x(-0.21),
            Vec3::new(12.0, -3.5, 4.8),
        );
        let planes = Camera::new(transform, projection).frustum();

        for plane in planes {
            assert_approx_eq(plane.truncate().length(), 1.0);
        }

        let world_point = |camera_space: Vec3| transform.transform_point3(camera_space);
        let near_plane = planes[4];
        assert_approx_eq(
            signed_distance(near_plane, world_point(Vec3::new(0.0, 0.0, -near))),
            0.0,
        );
        assert_approx_eq(
            signed_distance(near_plane, world_point(Vec3::new(0.0, 0.0, -(near + 2.0)))),
            2.0,
        );
        assert_approx_eq(
            signed_distance(near_plane, world_point(Vec3::new(0.0, 0.0, -(near - 0.1)))),
            -0.1,
        );

        let far_plane = planes[5];
        assert_approx_eq(
            signed_distance(far_plane, world_point(Vec3::new(0.0, 0.0, -far))),
            0.0,
        );
        assert_approx_eq(
            signed_distance(far_plane, world_point(Vec3::new(0.0, 0.0, -(far - 3.0)))),
            3.0,
        );
        assert_approx_eq(
            signed_distance(far_plane, world_point(Vec3::new(0.0, 0.0, -(far + 3.0)))),
            -3.0,
        );
    }

    #[test]
    fn infinite_perspective_uses_an_always_inside_far_plane() {
        let projection = Mat4::perspective_infinite_rh(60.0_f32.to_radians(), 1.0, 0.1);
        let planes = Camera::from_projection(projection).frustum();

        assert_eq!(planes[5], Vec4::W);
        assert!(signed_distance(planes[5], Vec3::new(0.0, 0.0, -1.0e9)) > 0.0);
    }
}
