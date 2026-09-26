//! A map that keeps its keys in insertion order.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// A map from keys to values that iterates in insertion order.
///
/// It keeps its entries in a vector and an index from each key to its
/// position. Replacing the value of a key keeps its position; removing a key
/// shifts the entries after it.
#[derive(Clone)]
pub(super) struct OrderedMap<K, V> {
    entries: Vec<(K, V)>,
    positions: HashMap<K, usize>,
}

impl<K, V> OrderedMap<K, V> {
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
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + '_ {
        self.entries.iter().map(|(key, value)| (key, value))
    }

    /// Return the values in order, to change them.
    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut V> + '_ {
        self.entries.iter_mut().map(|(_, value)| value)
    }
}

impl<K: Eq + Hash + Clone, V> OrderedMap<K, V> {
    /// Return the value of `key`.
    pub(super) fn get(&self, key: &K) -> Option<&V> {
        self.positions
            .get(key)
            .map(|&position| &self.entries[position].1)
    }

    /// Return the stored key equal to `key`, and its value.
    pub(super) fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        self.positions.get(key).map(|&position| {
            let (key, value) = &self.entries[position];
            (key, value)
        })
    }

    /// Return the value of `key`, to change it.
    pub(super) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.positions
            .get(key)
            .map(|&position| &mut self.entries[position].1)
    }

    /// Return whether `key` has a value.
    pub(super) fn contains_key(&self, key: &K) -> bool {
        self.positions.contains_key(key)
    }

    /// Set the value of `key`, keeping its position if it has one and
    /// appending it otherwise, and return the value it replaced.
    pub(super) fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&position) = self.positions.get(&key) {
            return Some(std::mem::replace(&mut self.entries[position].1, value));
        }
        self.positions.insert(key.clone(), self.entries.len());
        self.entries.push((key, value));
        None
    }

    /// Remove `key` and return its value.
    pub(super) fn remove(&mut self, key: &K) -> Option<V> {
        let position = self.positions.remove(key)?;
        let (_, value) = self.entries.remove(position);
        for (later, _) in &self.entries[position..] {
            if let Some(slot) = self.positions.get_mut(later) {
                *slot -= 1;
            }
        }
        Some(value)
    }

    /// Reorder the entries by the key `rank` gives each; entries of equal
    /// rank keep their order.
    pub(super) fn sort_by_key<R: Ord>(&mut self, mut rank: impl FnMut(&K) -> R) {
        self.entries.sort_by_key(|(key, _)| rank(key));
        for (position, (key, _)) in self.entries.iter().enumerate() {
            if let Some(slot) = self.positions.get_mut(key) {
                *slot = position;
            }
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for OrderedMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(key, value)| (key, value)))
            .finish()
    }
}
