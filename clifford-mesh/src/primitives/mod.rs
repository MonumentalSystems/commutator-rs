//! Checked Euclidean primitive tessellators.
//!
//! Coordinates are right-handed and use caller-defined units. Triangle faces
//! are counter-clockwise when viewed from their outward normal. All geometry
//! is `f32`; trigonometric results are not promised to be bitwise identical
//! across platforms.

mod circle;
mod linear;
mod plane;
mod sphere;

pub use circle::{circle_disc, circle_loop, circle_tube};
pub use linear::{arrow, line_segment, open_cylinder};
pub use plane::{plane_grid, plane_normal_indicator, plane_patch};
pub use sphere::{icosphere, uv_sphere};
