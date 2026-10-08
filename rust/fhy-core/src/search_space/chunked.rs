//! [`Chunked`]: a sequence stored in shared chunks of a fixed size, so a
//! copy that changes a few elements copies only the chunks holding them.
//!
//! A configuration keeps one value per decision; a search step copies the
//! configuration and changes one decision. Kept in chunks, that copy costs
//! a pointer per chunk and the chunks it changes, not every value.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The number of elements of a chunk.
const CHUNK_LENGTH: usize = 64;

/// A sequence of `T` stored in shared chunks of [`CHUNK_LENGTH`] elements,
/// the last one shorter.
///
/// Cloning shares every chunk; [`set`](Self::set) copies the chunk it
/// changes when another sequence shares it. `==` and `Hash` compare the
/// elements in order, as a slice's do.
#[derive(Debug, Clone)]
pub(super) struct Chunked<T> {
    chunks: Vec<Arc<Vec<T>>>,
    length: usize,
}

impl<T: Clone> Chunked<T> {
    /// Return the sequence of `length` copies of `element`.
    pub(super) fn filled(length: usize, element: &T) -> Self {
        let chunks = (0..length)
            .step_by(CHUNK_LENGTH)
            .map(|start| Arc::new(vec![element.clone(); CHUNK_LENGTH.min(length - start)]))
            .collect();
        Self { chunks, length }
    }

    /// Return the number of elements.
    pub(super) fn len(&self) -> usize {
        self.length
    }

    /// Return the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Self::len).
    pub(super) fn get(&self, index: usize) -> &T {
        &self.chunks[index / CHUNK_LENGTH][index % CHUNK_LENGTH]
    }

    /// Put `element` at `index`, copying its chunk first if another
    /// sequence shares it.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Self::len).
    pub(super) fn set(&mut self, index: usize, element: T) {
        Arc::make_mut(&mut self.chunks[index / CHUNK_LENGTH])[index % CHUNK_LENGTH] = element;
    }

    /// Return the elements, in order.
    pub(super) fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }
}

impl<T: Clone> Chunked<Option<T>> {
    /// Return the element at `index`, leaving `None` in its place.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Self::len).
    pub(super) fn take(&mut self, index: usize) -> Option<T> {
        if self.get(index).is_none() {
            return None;
        }
        Arc::make_mut(&mut self.chunks[index / CHUNK_LENGTH])[index % CHUNK_LENGTH].take()
    }
}

impl<T: PartialEq> PartialEq for Chunked<T> {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
            && self
                .chunks
                .iter()
                .zip(&other.chunks)
                .all(|(left, right)| Arc::ptr_eq(left, right) || left == right)
    }
}

impl<T: Eq> Eq for Chunked<T> {}

impl<T: Hash> Hash for Chunked<T> {
    /// Feed the length, then each element in order.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.length.hash(state);
        for chunk in &self.chunks {
            for element in chunk.iter() {
                element.hash(state);
            }
        }
    }
}
