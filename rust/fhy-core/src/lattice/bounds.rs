//! The lattice: meets and joins over a partially ordered set.

use std::hash::Hash;

use super::bits::Bits;
use super::error::OrderError;
use super::poset::PartiallyOrderedSet;

/// A partially ordered set asked for greatest lower bounds (meets) and least
/// upper bounds (joins).
///
/// Any partial order can be held. It is a lattice when every pair of its
/// elements has a meet and a join, which [`is_lattice`](Self::is_lattice)
/// checks and [`missing_bounds`](Self::missing_bounds) reports pair by pair.
///
/// The meet of `x` and `y` is the one lower bound (an element at most both)
/// that no other lower bound is above; when there are several such lower
/// bounds, or none, there is no meet. The join is the dual.
#[derive(Debug, Clone)]
pub struct Lattice<T> {
    poset: PartiallyOrderedSet<T>,
}

impl<T> Default for Lattice<T> {
    fn default() -> Self {
        Self {
            poset: PartiallyOrderedSet::default(),
        }
    }
}

/// A pair of elements of a [`Lattice`] without a meet or without a join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MissingBound<'a, T> {
    /// The two elements have no meet.
    Meet(&'a T, &'a T),
    /// The two elements have no join.
    Join(&'a T, &'a T),
}

impl<T: Eq + Hash + Clone> Lattice<T> {
    /// Return an empty lattice.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the partially ordered set of the elements and their order.
    #[must_use]
    pub fn poset(&self) -> &PartiallyOrderedSet<T> {
        &self.poset
    }

    /// Return whether `element` is a member.
    #[must_use]
    pub fn contains(&self, element: &T) -> bool {
        self.poset.contains(element)
    }

    /// Add `element`, as [`PartiallyOrderedSet::add_element`] does.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::AlreadyAMember`] if `element` is a member.
    pub fn add_element(&mut self, element: T) -> Result<(), OrderError<T>> {
        self.poset.add_element(element)
    }

    /// Order `lower` below `upper`, as [`PartiallyOrderedSet::add_order`]
    /// does.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::NotAMember`] or [`OrderError::WouldCycle`].
    pub fn add_order(&mut self, lower: &T, upper: &T) -> Result<(), OrderError<T>> {
        self.poset.add_order(lower, upper)
    }

    /// Return the meet of `x` and `y`, or `None` if they have none.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::NotAMember`] for `x`, then for `y`, if it is not
    /// a member.
    pub fn meet(&self, x: &T, y: &T) -> Result<Option<&T>, OrderError<T>> {
        let x_position = self.poset.position(x)?;
        let y_position = self.poset.position(y)?;
        Ok(self
            .meet_of(x_position, y_position)
            .map(|position| self.poset.element(position)))
    }

    /// Return the join of `x` and `y`, or `None` if they have none.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::NotAMember`] for `x`, then for `y`, if it is not
    /// a member.
    pub fn join(&self, x: &T, y: &T) -> Result<Option<&T>, OrderError<T>> {
        let x_position = self.poset.position(x)?;
        let y_position = self.poset.position(y)?;
        Ok(self
            .join_of(x_position, y_position)
            .map(|position| self.poset.element(position)))
    }

    /// Return whether every pair of elements has a meet and a join.
    #[must_use]
    pub fn is_lattice(&self) -> bool {
        let positions = self.poset.ordered_positions();
        positions.iter().all(|&x| {
            positions
                .iter()
                .all(|&y| self.meet_of(x, y).is_some() && self.join_of(x, y).is_some())
        })
    }

    /// Return, for every ordered pair of elements in the order of
    /// [`PartiallyOrderedSet::iter`], its missing meet and then its missing
    /// join. The list is empty exactly when [`is_lattice`](Self::is_lattice)
    /// holds.
    pub fn missing_bounds(&self) -> impl Iterator<Item = MissingBound<'_, T>> + '_ {
        let positions = self.poset.ordered_positions();
        let mut missing = Vec::new();
        for &x in &positions {
            for &y in &positions {
                let (left, right) = (self.poset.element(x), self.poset.element(y));
                if self.meet_of(x, y).is_none() {
                    missing.push(MissingBound::Meet(left, right));
                }
                if self.join_of(x, y).is_none() {
                    missing.push(MissingBound::Join(left, right));
                }
            }
        }
        missing.into_iter()
    }

    /// Return the position of the meet of the elements at `x` and `y`.
    fn meet_of(&self, x: usize, y: usize) -> Option<usize> {
        let lower_bounds: Vec<usize> = (0..self.poset.len())
            .filter(|&z| self.poset.up_set(z).contains(x) && self.poset.up_set(z).contains(y))
            .collect();
        let mut bounds = Bits::default();
        for &z in &lower_bounds {
            bounds.insert(z);
        }
        single(lower_bounds.iter().copied().filter(|&z| {
            self.poset
                .up_set(z)
                .intersection(&bounds)
                .iter()
                .all(|w| w == z)
        }))
    }

    /// Return the position of the join of the elements at `x` and `y`.
    fn join_of(&self, x: usize, y: usize) -> Option<usize> {
        let upper_bounds = self.poset.up_set(x).intersection(self.poset.up_set(y));
        single(upper_bounds.iter().filter(|&z| {
            upper_bounds
                .iter()
                .all(|w| w == z || !self.poset.up_set(w).contains(z))
        }))
    }
}

/// Return the only item of `items`, or `None` if there are none or several.
fn single(mut items: impl Iterator<Item = usize>) -> Option<usize> {
    let first = items.next()?;
    items.next().is_none().then_some(first)
}
