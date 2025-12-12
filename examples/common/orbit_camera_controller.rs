use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct OrbitCameraControllerOptions {
    pub orbit_x: f32,
    pub orbit_y: f32,
    pub max_orbit_x: f32,
    pub min_orbit_x: f32,
    pub max_orbit_y: f32,
    pub min_orbit_y: f32,
    pub constrain_x_orbit: bool,
    pub constrain_y_orbit: bool,

    pub max_distance: f32,
    pub min_distance: f32,
    pub constrain_distance: bool,

    pub target: Vec3,
    pub distance: f32,
    pub position: Option<Vec3>,
}

impl Default for OrbitCameraControllerOptions {
    fn default() -> Self {
        Self {
            orbit_x: 0.0,
            orbit_y: 0.0,
            max_orbit_x: std::f32::consts::FRAC_PI_2,
            min_orbit_x: -std::f32::consts::FRAC_PI_2,
            max_orbit_y: std::f32::consts::PI,
            min_orbit_y: -std::f32::consts::PI,
            constrain_x_orbit: true,
            constrain_y_orbit: false,

            max_distance: 1000.0,
            min_distance: 1.0,
            constrain_distance: true,

            target: Vec3::ZERO,
            distance: 1.0,
            position: None,
        }
    }
}

pub struct OrbitCameraController {
    orbit_x: f32,
    orbit_y: f32,
    max_orbit_x: f32,
    min_orbit_x: f32,
    max_orbit_y: f32,
    min_orbit_y: f32,
    constrain_x_orbit: bool,
    constrain_y_orbit: bool,

    max_distance: f32,
    min_distance: f32,
    constrain_distance: bool,

    distance: f32,
    target: Vec3,

    view_mat: Mat4,
    camera_mat: Mat4,
    dirty: bool,
}

impl Default for OrbitCameraController {
    fn default() -> Self {
        Self::from_options(OrbitCameraControllerOptions::default())
    }
}

impl OrbitCameraController {
    pub fn new(options: OrbitCameraControllerOptions) -> Self {
        Self::from_options(options)
    }

    pub fn from_options(options: OrbitCameraControllerOptions) -> Self {
        let mut controller = Self {
            orbit_x: options.orbit_x,
            orbit_y: options.orbit_y,
            max_orbit_x: options.max_orbit_x,
            min_orbit_x: options.min_orbit_x,
            max_orbit_y: options.max_orbit_y,
            min_orbit_y: options.min_orbit_y,
            constrain_x_orbit: options.constrain_x_orbit,
            constrain_y_orbit: options.constrain_y_orbit,

            max_distance: options.max_distance,
            min_distance: options.min_distance,
            constrain_distance: options.constrain_distance,

            distance: options.distance,
            target: options.target,
            view_mat: Mat4::IDENTITY,
            camera_mat: Mat4::IDENTITY,
            dirty: true,
        };

        controller.apply_constraints();

        if let Some(position) = options.position {
            controller.set_position(position);
        }

        controller
    }

    pub fn orbit(&mut self, x_delta: f32, y_delta: f32) {
        if x_delta == 0.0 && y_delta == 0.0 {
            return;
        }

        self.orbit_y += x_delta;
        self.orbit_x += y_delta;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn orbit_x(&self) -> f32 {
        self.orbit_x
    }

    pub fn orbit_y(&self) -> f32 {
        self.orbit_y
    }

    pub fn set_orbit(&mut self, orbit_y: f32, orbit_x: f32) {
        self.orbit_y = orbit_y;
        self.orbit_x = orbit_x;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn orbit_limits_x(&self) -> (f32, f32) {
        (self.min_orbit_x, self.max_orbit_x)
    }

    pub fn set_orbit_limits_x(&mut self, min_orbit_x: f32, max_orbit_x: f32) {
        self.min_orbit_x = min_orbit_x;
        self.max_orbit_x = max_orbit_x;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn orbit_limits_y(&self) -> (f32, f32) {
        (self.min_orbit_y, self.max_orbit_y)
    }

    pub fn set_orbit_limits_y(&mut self, min_orbit_y: f32, max_orbit_y: f32) {
        self.min_orbit_y = min_orbit_y;
        self.max_orbit_y = max_orbit_y;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn constrain_x_orbit(&self) -> bool {
        self.constrain_x_orbit
    }

    pub fn set_constrain_x_orbit(&mut self, value: bool) {
        self.constrain_x_orbit = value;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn constrain_y_orbit(&self) -> bool {
        self.constrain_y_orbit
    }

    pub fn set_constrain_y_orbit(&mut self, value: bool) {
        self.constrain_y_orbit = value;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn target(&self) -> Vec3 {
        self.target
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
        self.dirty = true;
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn distance_limits(&self) -> (f32, f32) {
        (self.min_distance, self.max_distance)
    }

    pub fn set_distance_limits(&mut self, min_distance: f32, max_distance: f32) {
        self.min_distance = min_distance;
        self.max_distance = max_distance;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn constrain_distance(&self) -> bool {
        self.constrain_distance
    }

    pub fn set_constrain_distance(&mut self, value: bool) {
        self.constrain_distance = value;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn set_position(&mut self, position: Vec3) {
        let offset = position - self.target;
        let len = offset.length();
        if len <= 1.0e-6 {
            self.orbit_x = 0.0;
            self.orbit_y = 0.0;
            self.distance = self.min_distance;
            self.apply_constraints();
            self.dirty = true;
            return;
        }

        let v = offset / len;
        self.orbit_x = v.y.asin();
        self.orbit_y = (-v.x).atan2(v.z);
        self.distance = len;
        self.apply_constraints();
        self.dirty = true;
    }

    pub fn dolly(&mut self, delta: f32) {
        self.set_distance(self.distance + delta);
    }

    pub fn position(&mut self) -> Vec3 {
        self.update_matrices();
        self.camera_mat.transform_point3(Vec3::ZERO)
    }

    pub fn view_matrix(&mut self) -> Mat4 {
        self.update_matrices();
        self.view_mat
    }

    pub fn camera_matrix(&mut self) -> Mat4 {
        self.update_matrices();
        self.camera_mat
    }

    fn apply_constraints(&mut self) {
        if self.min_orbit_x > self.max_orbit_x {
            std::mem::swap(&mut self.min_orbit_x, &mut self.max_orbit_x);
        }
        if self.min_orbit_y > self.max_orbit_y {
            std::mem::swap(&mut self.min_orbit_y, &mut self.max_orbit_y);
        }
        if self.min_distance > self.max_distance {
            std::mem::swap(&mut self.min_distance, &mut self.max_distance);
        }

        if self.constrain_x_orbit {
            self.orbit_x = clamp(self.orbit_x, self.min_orbit_x, self.max_orbit_x);
        } else {
            self.orbit_x = wrap_radians_pi(self.orbit_x);
        }

        if self.constrain_y_orbit {
            self.orbit_y = clamp(self.orbit_y, self.min_orbit_y, self.max_orbit_y);
        } else {
            self.orbit_y = wrap_radians_pi(self.orbit_y);
        }

        if self.constrain_distance {
            self.distance = clamp(self.distance, self.min_distance, self.max_distance);
        }
    }

    fn update_matrices(&mut self) {
        if !self.dirty {
            return;
        }

        let mut camera = Mat4::IDENTITY;
        camera = camera * Mat4::from_translation(self.target);
        camera = camera * Mat4::from_rotation_y(-self.orbit_y);
        camera = camera * Mat4::from_rotation_x(-self.orbit_x);
        camera = camera * Mat4::from_translation(Vec3::new(0.0, 0.0, self.distance));

        self.camera_mat = camera;
        self.view_mat = camera.inverse();
        self.dirty = false;
    }
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

fn wrap_radians_pi(value: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    (value + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    fn assert_vec3_approx(a: Vec3, b: Vec3) {
        let d = (a - b).length();
        assert!(d <= EPS, "vec3 not approx: a={a:?} b={b:?} d={d}");
    }

    fn assert_mat4_approx(a: Mat4, b: Mat4) {
        let aa = a.to_cols_array();
        let bb = b.to_cols_array();
        for i in 0..16 {
            let d = (aa[i] - bb[i]).abs();
            assert!(
                d <= EPS,
                "mat4 not approx at {i}: a={} b={} d={}",
                aa[i],
                bb[i],
                d
            );
        }
    }

    #[test]
    fn set_position_roundtrip_unconstrained() {
        let target = Vec3::new(1.0, 2.0, 3.0);
        let position = Vec3::new(4.0, 6.0, 9.0);

        let options = OrbitCameraControllerOptions {
            target,
            position: Some(position),
            constrain_x_orbit: false,
            constrain_y_orbit: false,
            constrain_distance: false,
            ..Default::default()
        };

        let mut camera = OrbitCameraController::new(options);
        assert_vec3_approx(camera.position(), position);

        let v = camera.view_matrix();
        let c = camera.camera_matrix();
        assert_mat4_approx(v * c, Mat4::IDENTITY);
        assert_mat4_approx(c * v, Mat4::IDENTITY);
    }

    #[test]
    fn orbit_y_rotates_around_target() {
        let options = OrbitCameraControllerOptions {
            target: Vec3::ZERO,
            distance: 1.0,
            constrain_x_orbit: false,
            constrain_y_orbit: false,
            constrain_distance: false,
            ..Default::default()
        };
        let mut camera = OrbitCameraController::new(options);

        camera.set_orbit(std::f32::consts::FRAC_PI_2, 0.0);
        assert_vec3_approx(camera.position(), Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn orbit_x_rotates_around_target() {
        let options = OrbitCameraControllerOptions {
            target: Vec3::ZERO,
            distance: 1.0,
            constrain_x_orbit: false,
            constrain_y_orbit: false,
            constrain_distance: false,
            ..Default::default()
        };
        let mut camera = OrbitCameraController::new(options);

        camera.set_orbit(0.0, std::f32::consts::FRAC_PI_2);
        assert_vec3_approx(camera.position(), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn distance_limits_swap_and_clamp() {
        let options = OrbitCameraControllerOptions {
            min_distance: 10.0,
            max_distance: 1.0,
            constrain_distance: true,
            distance: 100.0,
            ..Default::default()
        };
        let camera = OrbitCameraController::new(options);
        assert_eq!(camera.distance_limits(), (1.0, 10.0));
        assert!((camera.distance() - 10.0).abs() <= EPS);
    }

    #[test]
    fn orbit_wraps_when_unconstrained() {
        let options = OrbitCameraControllerOptions {
            constrain_x_orbit: false,
            constrain_y_orbit: false,
            ..Default::default()
        };
        let mut camera = OrbitCameraController::new(options);

        camera.set_orbit(3.0 * std::f32::consts::PI, 0.0);
        assert!((camera.orbit_y() - (-std::f32::consts::PI)).abs() <= EPS);
    }
}
