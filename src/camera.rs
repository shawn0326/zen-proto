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
            near: 1.0,
            far: 1000.0,
        }
    }
}

impl Into<Mat4> for PerspectiveProjection {
    fn into(self) -> Mat4 {
        Mat4::perspective_rh(self.fovy_deg.to_radians(), self.aspect, self.near, self.far)
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

impl Into<Mat4> for OrthographicProjection {
    fn into(self) -> Mat4 {
        Mat4::orthographic_rh(
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
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

        let mut planes = [glam::Vec4::ZERO; 6];
        planes[0] = (m3 + m0).normalize(); // left
        planes[1] = (m3 - m0).normalize(); // right
        planes[2] = (m3 + m1).normalize(); // bottom
        planes[3] = (m3 - m1).normalize(); // top
        planes[4] = (m3 + m2).normalize(); // near
        planes[5] = (m3 - m2).normalize(); // far

        planes
    }
}
