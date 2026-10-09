//! Checked row-major periodic two-dimensional lattice storage.

use crate::error::checked_product;
use crate::{LatticeError, Result};

/// A rectangular two-dimensional lattice with periodic boundaries.
///
/// Sites are stored in row-major order with `index = x * height + y`.
/// Extents must both be at least two so local bond updates never contain
/// self-links.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicLattice2<T> {
    width: usize,
    height: usize,
    sites: Vec<T>,
}

impl<T> PeriodicLattice2<T> {
    /// Constructs a lattice from an explicitly ordered site vector.
    pub fn from_sites(width: usize, height: usize, sites: Vec<T>) -> Result<Self> {
        validate_extents(width, height)?;
        let expected = checked_product("periodic lattice", width, height)?;
        if sites.len() != expected {
            return Err(LatticeError::Shape {
                name: "periodic lattice sites",
                expected,
                actual: sites.len(),
            });
        }
        Ok(Self {
            width,
            height,
            sites,
        })
    }

    /// Returns the horizontal extent.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the vertical extent.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the number of sites.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Returns `true` when the site collection is empty.
    ///
    /// Valid lattices are never empty; this method mirrors slice APIs.
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Returns all sites in row-major order.
    pub fn sites(&self) -> &[T] {
        &self.sites
    }

    /// Returns all sites mutably in row-major order.
    pub fn sites_mut(&mut self) -> &mut [T] {
        &mut self.sites
    }

    /// Consumes the lattice and returns its row-major site vector.
    pub fn into_sites(self) -> Vec<T> {
        self.sites
    }

    /// Returns the checked row-major index for `(x, y)`.
    pub fn index(&self, x: usize, y: usize) -> Result<usize> {
        if x >= self.width || y >= self.height {
            return Err(LatticeError::IndexOutOfBounds);
        }
        Ok(x * self.height + y)
    }

    /// Returns the periodic row-major index for arbitrary coordinates.
    pub fn periodic_index(&self, x: usize, y: usize) -> usize {
        (x % self.width) * self.height + (y % self.height)
    }

    /// Returns the coordinates for a checked row-major index.
    pub fn coordinates(&self, index: usize) -> Result<(usize, usize)> {
        if index >= self.len() {
            return Err(LatticeError::IndexOutOfBounds);
        }
        Ok((index / self.height, index % self.height))
    }

    /// Returns neighbors in `(+x, -x, +y, -y)` order.
    pub fn neighbors(&self, index: usize) -> Result<[usize; 4]> {
        let (x, y) = self.coordinates(index)?;
        let positive_x = (x + 1) % self.width;
        let negative_x = (x + self.width - 1) % self.width;
        let positive_y = (y + 1) % self.height;
        let negative_y = (y + self.height - 1) % self.height;
        Ok([
            positive_x * self.height + y,
            negative_x * self.height + y,
            x * self.height + positive_y,
            x * self.height + negative_y,
        ])
    }

    /// Returns a shared site reference for a checked flat index.
    pub fn get(&self, index: usize) -> Result<&T> {
        self.sites.get(index).ok_or(LatticeError::IndexOutOfBounds)
    }

    /// Replaces a site at a checked flat index.
    pub fn set(&mut self, index: usize, value: T) -> Result<()> {
        let site = self
            .sites
            .get_mut(index)
            .ok_or(LatticeError::IndexOutOfBounds)?;
        *site = value;
        Ok(())
    }
}

impl<T: Clone> PeriodicLattice2<T> {
    /// Constructs a lattice by cloning one initial site.
    pub fn filled(width: usize, height: usize, initial: T) -> Result<Self> {
        validate_extents(width, height)?;
        let count = checked_product("periodic lattice", width, height)?;
        Ok(Self {
            width,
            height,
            sites: vec![initial; count],
        })
    }
}

fn validate_extents(width: usize, height: usize) -> Result<()> {
    if width < 2 {
        return Err(LatticeError::InvalidDomain(
            "periodic lattice width must be at least two",
        ));
    }
    if height < 2 {
        return Err(LatticeError::InvalidDomain(
            "periodic lattice height must be at least two",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_index_and_neighbors_are_periodic() {
        let lattice = PeriodicLattice2::filled(3, 4, 0_u8).unwrap();
        assert_eq!(lattice.index(2, 3).unwrap(), 11);
        assert_eq!(lattice.coordinates(11).unwrap(), (2, 3));
        assert_eq!(lattice.neighbors(0).unwrap(), [4, 8, 1, 3]);
        assert_eq!(lattice.periodic_index(5, 6), lattice.index(2, 2).unwrap());
    }

    #[test]
    fn invalid_shapes_and_indices_are_rejected() {
        assert!(PeriodicLattice2::filled(1, 2, 0_u8).is_err());
        assert!(PeriodicLattice2::from_sites(2, 2, vec![0_u8; 3]).is_err());
        let lattice = PeriodicLattice2::filled(2, 2, 0_u8).unwrap();
        assert!(lattice.index(2, 0).is_err());
        assert!(lattice.neighbors(4).is_err());
    }
}
