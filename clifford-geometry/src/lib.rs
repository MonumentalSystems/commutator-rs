//! Sparse geometric algebra and geometry primitives.
//!
//! This crate preserves the fixed-size, `f32` Versor layouts and formulas.
//! Use [`adapter`] when crossing into the dense, precision-generic
//! [`clifford_core`] representation; the two crates intentionally differ in
//! commutator normalization, rotor orientation, and some blade layouts.
//!
//! The named geometric types are currently aliases of [`mvec::Multivector`]
//! rather than semantic newtypes. A newtype migration is deferred to a future
//! breaking release so this extraction does not silently change formulas or
//! representation.

#![forbid(unsafe_code)]
// These lints describe deliberate properties of the extracted Versor API:
// fixed-arity constructors mirror blade layouts, indexed loops mirror product
// tables, and several private table builders are retained for source/formula
// fidelity while the public surface is stabilized.
#![allow(dead_code)]
#![allow(clippy::approx_constant)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod adapter;
pub mod basis;
pub mod cga3d;
pub mod cl3;
pub mod cl6;
pub mod ega3d;
pub mod exp;
pub mod form;
pub mod metric;
pub mod mvec;
pub mod pga3d;
pub mod products;
pub mod sta;
