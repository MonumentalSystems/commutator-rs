use crate::{KeldyshError, Result};

/// A strictly increasing real-time grid with composite-trapezoid weights.
#[derive(Debug, Clone, PartialEq)]
pub struct RealTimeGrid {
    times: Vec<f64>,
    weights: Vec<f64>,
}

impl RealTimeGrid {
    /// Construct a grid and its full-interval composite-trapezoid weights.
    pub fn try_new(times: Vec<f64>) -> Result<Self> {
        if times.len() < 2 {
            return Err(KeldyshError::Empty("real-time grid"));
        }
        if !times.iter().all(|time| time.is_finite()) {
            return Err(KeldyshError::NonFinite("real-time grid"));
        }
        for index in 1..times.len() {
            if times[index] <= times[index - 1] {
                return Err(KeldyshError::UnorderedTimeGrid { index });
            }
        }
        let mut weights = vec![0.0; times.len()];
        weights[0] = 0.5 * (times[1] - times[0]);
        let last = times.len() - 1;
        weights[last] = 0.5 * (times[last] - times[last - 1]);
        for index in 1..last {
            weights[index] = 0.5 * (times[index + 1] - times[index - 1]);
        }
        Ok(Self { times, weights })
    }

    /// Return the time points.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Return the full-interval composite-trapezoid weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Return the number of time points.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Return whether the grid is empty. A valid grid is never empty.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub(crate) fn interval_weight(&self, point: usize, start: usize, end: usize) -> f64 {
        if start >= end || point < start || point > end {
            return 0.0;
        }
        if point == start {
            0.5 * (self.times[start + 1] - self.times[start])
        } else if point == end {
            0.5 * (self.times[end] - self.times[end - 1])
        } else {
            0.5 * (self.times[point + 1] - self.times[point - 1])
        }
    }
}
