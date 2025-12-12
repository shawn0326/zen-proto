use std::{collections::VecDeque, time::Instant};

pub struct FrameRateTracker {
    frame_times: VecDeque<Instant>,
    window_size: usize,
    fps: f64,
}

impl FrameRateTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            frame_times: VecDeque::new(),
            window_size,
            fps: 0.0,
        }
    }

    pub fn record_frame(&mut self) -> f64 {
        let now = Instant::now();
        self.frame_times.push_back(now);

        while self.frame_times.len() > self.window_size {
            self.frame_times.pop_front();
        }

        if self.frame_times.len() > 1 {
            let oldest = self.frame_times.front().unwrap();
            let elapsed = now.duration_since(*oldest).as_secs_f64().max(1e-6);
            self.fps = (self.frame_times.len() - 1) as f64 / elapsed;
        }

        self.fps
    }

    #[inline]
    pub fn fps(&self) -> f64 {
        self.fps
    }
}

impl Default for FrameRateTracker {
    fn default() -> Self {
        Self::new(60)
    }
}
