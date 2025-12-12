use std::{collections::VecDeque, time::Instant};

pub struct FpsCounter {
    frame_times: VecDeque<Instant>,
    sample_size: usize,
}

impl FpsCounter {
    pub fn new(sample_size: usize) -> Self {
        Self {
            frame_times: VecDeque::new(),
            sample_size,
        }
    }

    pub fn tick(&mut self) -> f64 {
        let now = Instant::now();
        self.frame_times.push_back(now);

        while self.frame_times.len() > self.sample_size {
            self.frame_times.pop_front();
        }

        if self.frame_times.len() > 1 {
            let oldest = self.frame_times.front().unwrap();
            let elapsed = now.duration_since(*oldest).as_secs_f64();
            let fps = (self.frame_times.len() - 1) as f64 / elapsed;
            fps
        } else {
            0.0
        }
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new(60)
    }
}
