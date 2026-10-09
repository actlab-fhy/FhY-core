//! A map from identifiers that keeps its keys in insertion order.

use std::collections::HashMap;
use std::fmt;

use crate::identifier::Identifier;

/// A map from identifiers to values that iterates in insertion order.
///
/// It keeps its entries in a vector and an index from each identifier's id
/// to its position, so cloning the index copies it without touching the
/// identifiers. Replacing the value of a key keeps its position; removing a
/// key shifts the entries after it.
#[derive(Clone)]
pub(super) struct OrderedMap<V> {
    entries: Vec<(Identifier, V)>,
    positions: HashMap<u64, usize>,
}

impl<V> OrderedMap<V> {
    /// Return an empty map.
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            positions: HashMap::new(),
        }
    }

    /// Return the number of entries.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return the entries in order.
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = (&Identifier, &V)> + '_ {
        self.entries.iter().map(|(key, value)| (key, value))
    }

    /// Return the values in order, to change them.
    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut V> + '_ {
        self.entries.iter_mut().map(|(_, value)| value)
    }

    /// Return the value of `key`.
    pub(super) fn get(&self, key: &Identifier) -> Option<&V> {
        self.positions
            .get(&key.id())
            .map(|&position| &self.entries[position].1)
    }

    /// Return the stored key equal to `key`, and its value.
    pub(super) fn get_key_value(&self, key: &Identifier) -> Option<(&Identifier, &V)> {
        self.positions.get(&key.id()).map(|&position| {
            let (key, value) = &self.entries[position];
            (key, value)
        })
    }

    /// Return the value of `key`, to change it.
    pub(super) fn get_mut(&mut self, key: &Identifier) -> Option<&mut V> {
        self.positions
            .get(&key.id())
            .map(|&position| &mut self.entries[position].1)
    }

    /// Return whether `key` has a value.
    pub(super) fn contains_key(&self, key: &Identifier) -> bool {
        self.positions.contains_key(&key.id())
    }

    /// Set the value of `key`, keeping its position if it has one and
    /// appending it otherwise, and return the value it replaced.
    pub(super) fn insert(&mut self, key: Identifier, value: V) -> Option<V> {
        if let Some(&position) = self.positions.get(&key.id()) {
            return Some(std::mem::replace(&mut self.entries[position].1, value));
        }
        self.positions.insert(key.id(), self.entries.len());
        self.entries.push((key, value));
        None
    }

    /// Remove `key` and return its value.
    pub(super) fn remove(&mut self, key: &Identifier) -> Option<V> {
        let position = self.positions.remove(&key.id())?;
        let (_, value) = self.entries.remove(position);
        for (later, _) in &self.entries[position..] {
            if let Some(slot) = self.positions.get_mut(&later.id()) {
                *slot -= 1;
            }
        }
        Some(value)
    }

    /// Reorder the entries by identifier id.
    pub(super) fn sort_by_id(&mut self) {
        self.entries.sort_by_key(|(key, _)| key.id());
        for (position, (key, _)) in self.entries.iter().enumerate() {
            if let Some(slot) = self.positions.get_mut(&key.id()) {
                *slot = position;
            }
        }
    }
}

impl<V: fmt::Debug> fmt::Debug for OrderedMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
