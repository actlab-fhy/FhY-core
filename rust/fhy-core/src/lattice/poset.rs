//! The partially ordered set.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

use super::bits::Bits;
use super::error::OrderError;

/// A set of elements and a partial order over them.
///
/// The order is the reflexive and transitive closure of the orders added
/// with [`add_order`](Self::add_order): every element is at most itself,
/// and [`add_order`](Self::add_order) refuses an order that would close a
/// cycle. Each element keeps its up-set, the elements it is at most, so
/// [`is_at_most`](Self::is_at_most) is a bit test and adding an order
/// updates the up-sets below it.
///
/// # Examples
///
/// ```
/// use fhy_core::lattice::{OrderError, PartiallyOrderedSet};
///
/// let mut poset = PartiallyOrderedSet::new();
/// for element in [1, 2, 3] {
///     poset.add_element(element)?;
/// }
/// poset.add_order(&1, &2)?;
/// poset.add_order(&2, &3)?;
///
/// assert!(poset.is_at_most(&1, &3)?);
/// assert!(poset.is_at_most(&2, &2)?);
/// assert_eq!(poset.add_order(&3, &1), Err(OrderError::WouldCycle { lower: 3, upper: 1 }));
/// assert_eq!(poset.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
/// # Ok::<(), OrderError<i32>>(())
/// ```
#[derive(Debug, Clone)]
pub struct PartiallyOrderedSet<T> {
    /// The elements, in insertion order.
    elements: Vec<T>,
    /// The position of each element in `elements`.
    positions: HashMap<T, usize>,
    /// Each element's up-set: the positions of the elements it is at most,
    /// its own included.
    up_sets: Vec<Bits>,
    /// The orders added, as the positions of the upper elements of each
    /// lower one, without orders that already held when they were added.
    successors: Vec<Vec<usize>>,
}

impl<T> Default for PartiallyOrderedSet<T> {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            positions: HashMap::new(),
            up_sets: Vec::new(),
            successors: Vec::new(),
        }
    }
}

impl<T: Eq + Hash + Clone> PartiallyOrderedSet<T> {
    /// Return an empty partially ordered set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Return whether the set has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Return whether `element` is a member.
    #[must_use]
    pub fn contains(&self, element: &T) -> bool {
        self.positions.contains_key(element)
    }

    /// Add `element`, ordered with no other element.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::AlreadyAMember`], leaving the set unchanged,
    /// if `element` is a member.
    pub fn add_element(&mut self, element: T) -> Result<(), OrderError<T>> {
        if self.positions.contains_key(&element) {
            return Err(OrderError::AlreadyAMember(element));
        }
        let position = self.elements.len();
        self.positions.insert(element.clone(), position);
        self.elements.push(element);
        self.up_sets.push(Bits::singleton(position));
        self.successors.push(Vec::new());
        Ok(())
    }

    /// Order `lower` below `upper`.
    ///
    /// An order that already holds is accepted and changes nothing.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::NotAMember`] for `lower`, then for `upper`, if
    /// it is not a member, and [`OrderError::WouldCycle`] if `upper` is
    /// already at most `lower`, which includes `lower == upper`. Each
    /// leaves the set unchanged.
    pub fn add_order(&mut self, lower: &T, upper: &T) -> Result<(), OrderError<T>> {
        let lower_position = self.position(lower)?;
        let upper_position = self.position(upper)?;
        if self.up_sets[upper_position].contains(lower_position) {
            return Err(OrderError::WouldCycle {
                lower: lower.clone(),
                upper: upper.clone(),
            });
        }
        if self.up_sets[lower_position].contains(upper_position) {
            return Ok(());
        }
        let added = self.up_sets[upper_position].clone();
        for position in 0..self.up_sets.len() {
            if self.up_sets[position].contains(lower_position) {
                self.up_sets[position].union_with(&added);
            }
        }
        self.successors[lower_position].push(upper_position);
        Ok(())
    }

    /// Return whether `lower` is at most `upper`: equal to it, or ordered
    /// below it directly or through other elements.
    ///
    /// # Errors
    ///
    /// Returns [`OrderError::NotAMember`] for `lower`, then for `upper`, if
    /// it is not a member.
    pub fn is_at_most(&self, lower: &T, upper: &T) -> Result<bool, OrderError<T>> {
        let lower_position = self.position(lower)?;
        let upper_position = self.position(upper)?;
        Ok(self.up_sets[lower_position].contains(upper_position))
    }

    /// Return the elements in a topological order: every element comes
    /// before the elements it is ordered below. Among the elements that can
    /// come next, the one added first does.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + '_ {
        self.topological_positions(|position| position)
            .into_iter()
            .map(|position| &self.elements[position])
    }

    /// Return the elements in a topological order in which, among the
    /// elements that can come next, the one with the least `key` does, and
    /// of equal keys the one added first.
    pub fn iter_by_key<K: Ord>(
        &self,
        mut key: impl FnMut(&T) -> K,
    ) -> impl ExactSizeIterator<Item = &T> + '_ {
        let keys: Vec<K> = self.elements.iter().map(&mut key).collect();
        let ranks = rank_positions(&keys);
        self.topological_positions(|position| ranks[position])
            .into_iter()
            .map(|position| &self.elements[position])
    }

    /// Return the position of `element`, or [`OrderError::NotAMember`].
    pub(super) fn position(&self, element: &T) -> Result<usize, OrderError<T>> {
        self.positions
            .get(element)
            .copied()
            .ok_or_else(|| OrderError::NotAMember(element.clone()))
    }

    /// Return the positions in the topological order that prefers the least
    /// `rank` among the positions ready next, then the least position.
    fn topological_positions(&self, rank: impl Fn(usize) -> usize) -> Vec<usize> {
        let mut in_degrees = vec![0_usize; self.elements.len()];
        for successors in &self.successors {
            for &successor in successors {
                in_degrees[successor] += 1;
            }
        }
        let mut ready: BinaryHeap<Reverse<(usize, usize)>> = in_degrees
            .iter()
            .enumerate()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(position, _)| Reverse((rank(position), position)))
            .collect();
        let mut order = Vec::with_capacity(self.elements.len());
        while let Some(Reverse((_, position))) = ready.pop() {
            order.push(position);
            for &successor in &self.successors[position] {
                in_degrees[successor] -= 1;
                if in_degrees[successor] == 0 {
                    ready.push(Reverse((rank(successor), successor)));
                }
            }
        }
        order
    }

    /// Return the up-set of the element at `position`.
    pub(super) fn up_set(&self, position: usize) -> &Bits {
        &self.up_sets[position]
    }

    /// Return the element at `position`.
    pub(super) fn element(&self, position: usize) -> &T {
        &self.elements[position]
    }

    /// Return the positions in [`iter`](Self::iter)'s order.
    pub(super) fn ordered_positions(&self) -> Vec<usize> {
        self.topological_positions(|position| position)
    }
}

/// Return, for each key, its rank among the sorted keys: equal keys share
/// the rank of the first of them.
fn rank_positions<K: Ord>(keys: &[K]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&left, &right| keys[left].cmp(&keys[right]));
    let mut ranks = vec![0; keys.len()];
    let mut rank = 0;
    for (index, &position) in order.iter().enumerate() {
        if index > 0 && keys[order[index - 1]] != keys[position] {
            rank = index;
        }
        ranks[position] = rank;
    }
    ranks
}
