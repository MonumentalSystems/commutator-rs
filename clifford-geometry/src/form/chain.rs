//! Kinematic chains: linked rigid body segments using motors.
//!
//! Port of `vsr_chain.h`. A Chain is a sequence of Frames where each joint
//! has a local transformation, and forward kinematics propagates the chain.

use super::frame::Frame;
use crate::cga3d::*;

/// Joint types for kinematic chains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointType {
    /// Rotation around one axis.
    Revolute,
    /// Translation along one axis.
    Prismatic,
    /// Combined rotation + translation along one axis.
    Cylindrical,
    /// Screw motion (rotation + coupled translation).
    Helical,
    /// Two-axis rotation.
    Spherical,
}

/// A kinematic chain: a sequence of joints and links.
///
/// Each joint has a local transformation, and each link describes the
/// relative offset to the next joint. Forward kinematics computes the
/// absolute frame of each joint.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Base frame (root of the chain).
    pub base: Frame,
    /// Joint frames (local socket transformations).
    pub joints: Vec<Frame>,
    /// Link frames (relative offset to next joint).
    pub links: Vec<Frame>,
    /// Computed absolute frames after forward kinematics.
    pub frames: Vec<Frame>,
}

impl Chain {
    /// Create a chain with `n` joints, all starting at unit spacing along Y.
    pub fn new(n: usize) -> Self {
        let mut joints = Vec::with_capacity(n);
        let mut links = Vec::with_capacity(n);
        let frames = vec![Frame::new(); n];
        for _ in 0..n {
            joints.push(Frame::new());
            let mut link = Frame::new();
            link.pos = point(0.0, 1.0, 0.0); // default link length = 1 along Y
            links.push(link);
        }
        let mut chain = Self {
            base: Frame::new(),
            joints,
            links,
            frames,
        };
        chain.fk();
        chain
    }

    /// Number of joints.
    pub fn num(&self) -> usize {
        self.joints.len()
    }

    /// Set joint angle (rotation in local XY plane) for joint `i`.
    pub fn set_joint_angle(&mut self, i: usize, angle: f32) {
        self.joints[i].rot = Gen::rot(&biv(angle, 0.0, 0.0));
    }

    /// Set link length for link `i` (distance along local Y to next joint).
    pub fn set_link_length(&mut self, i: usize, length: f32) {
        self.links[i].pos = point(0.0, length, 0.0);
    }

    /// Forward kinematics: compute absolute frames from base, joints, and links.
    ///
    /// Uses direct point translation and rotation via sandwich products (Bst)
    /// rather than motor composition, because CGA motors in the compact Mot basis
    /// (8 components) lose the e_i4 translation blades.
    pub fn fk(&mut self) {
        if self.joints.is_empty() {
            return;
        }

        // Frame[0] = base position + joint[0] rotation applied to base
        let base_pos = self.base.pos;
        let base_rot = self.base.rot;
        let j0_rot = gp_rot_rot(&base_rot, &self.joints[0].rot);
        self.frames[0] = Frame {
            pos: base_pos,
            rot: j0_rot,
            ..Frame::new()
        };

        for i in 1..self.joints.len() {
            let prev = &self.frames[i - 1];
            // Link offset in local frame: extract Euclidean coords from link position
            let (lx, ly, lz) = Round::location(&self.links[i - 1].pos);
            // Rotate the link offset by the previous frame's rotation
            let local_offset = crate::ega3d::vec3(lx, ly, lz);
            let rotated_offset = crate::ega3d::spin(&prev.rot, &local_offset);
            // Translate previous position by the rotated offset
            let (px, py, pz) = Round::location(&prev.pos);
            let new_pos = point(
                px + rotated_offset[0],
                py + rotated_offset[1],
                pz + rotated_offset[2],
            );
            // Compose rotations
            let new_rot = gp_rot_rot(&prev.rot, &self.joints[i].rot);
            self.frames[i] = Frame {
                pos: new_pos,
                rot: new_rot,
                ..Frame::new()
            };
        }
    }

    /// Get the position of the end effector (last frame).
    pub fn end_effector(&self) -> (f32, f32, f32) {
        if let Some(f) = self.frames.last() {
            f.location()
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Reset all joints to identity.
    pub fn reset_joints(&mut self) {
        for j in self.joints.iter_mut() {
            j.reset();
        }
    }

    /// Reset the entire chain (joints, links, base).
    pub fn reset(&mut self) {
        self.base.reset();
        for j in self.joints.iter_mut() {
            j.reset();
        }
        for l in self.links.iter_mut() {
            *l = Frame::new();
            l.pos = point(0.0, 1.0, 0.0);
        }
        self.fk();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-3;
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_chain_creation() {
        let chain = Chain::new(3);
        assert_eq!(chain.num(), 3);
        assert_eq!(chain.joints.len(), 3);
        assert_eq!(chain.links.len(), 3);
        assert_eq!(chain.frames.len(), 3);
    }

    #[test]
    fn test_chain_fk_straight() {
        // 3-joint chain, all zero angles, links along Y of length 1
        let chain = Chain::new(3);
        // First frame is at the joint[0] = identity at base
        let (x0, y0, z0) = chain.frames[0].location();
        assert!(
            approx(x0, 0.0) && approx(y0, 0.0) && approx(z0, 0.0),
            "frame[0] = ({}, {}, {})",
            x0,
            y0,
            z0
        );

        // Second frame should be at (0, 1, 0) since link[0] is 1 along Y
        let (x1, y1, z1) = chain.frames[1].location();
        assert!(approx(x1, 0.0), "x1={}", x1);
        assert!(approx(y1, 1.0), "y1={}", y1);
        assert!(approx(z1, 0.0), "z1={}", z1);

        // Third frame at (0, 2, 0)
        let (x2, y2, z2) = chain.frames[2].location();
        assert!(approx(x2, 0.0), "x2={}", x2);
        assert!(approx(y2, 2.0), "y2={}", y2);
        assert!(approx(z2, 0.0), "z2={}", z2);
    }

    #[test]
    fn test_chain_end_effector() {
        let chain = Chain::new(4);
        let (x, y, z) = chain.end_effector();
        assert!(approx(x, 0.0));
        assert!(approx(y, 3.0)); // 4 joints, 3 links of length 1
        assert!(approx(z, 0.0));
    }

    #[test]
    fn test_chain_with_angles() {
        let mut chain = Chain::new(2);
        // Rotate first joint by PI/2 in e12 plane (XY plane)
        // This should rotate the Y-link direction by PI/2 in XY, making it point along X
        chain.set_joint_angle(0, PI / 2.0);
        chain.fk();

        // Frame[0] is still at origin (joint rotation doesn't change position of joint itself)
        let (x0, y0, _z0) = chain.frames[0].location();
        assert!(approx(x0, 0.0), "x0={}", x0);
        assert!(approx(y0, 0.0), "y0={}", y0);

        // Frame[1] should be displaced by link[0] in the rotated direction
        let (x1, y1, _z1) = chain.frames[1].location();
        // After PI/2 rotation in e12 plane, Y axis -> -X axis (or X axis depending on convention)
        // The link is along Y, after rotation it should end up along some other axis.
        // Check that it's moved away from (0,1,0)
        let dist = (x1 * x1 + (y1 - 1.0) * (y1 - 1.0)).sqrt();
        assert!(
            dist > 0.5,
            "Joint angle should change end position, got ({}, {})",
            x1,
            y1
        );
    }

    #[test]
    fn test_chain_reset() {
        let mut chain = Chain::new(2);
        chain.set_joint_angle(0, 1.0);
        chain.fk();
        chain.reset();
        let (_, y, _) = chain.frames[1].location();
        assert!(approx(y, 1.0), "After reset, should be straight: y={}", y);
    }
}
