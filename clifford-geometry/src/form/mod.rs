//! Geometric forms built on CGA3D primitives.
//!
//! Port of the C++ Versor `form/` directory. Provides higher-level constructs
//! for physics simulation: frames, shapes, kinematic chains, twists,
//! interpolation, and fields.

pub mod chain;
pub mod field;
pub mod frame;
pub mod interp;
pub mod shapes;
pub mod twist;
