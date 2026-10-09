//! A fixed-width set of element indices, one bit per index.

/// A set of indices below the width it was created with.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct Bits {
    words: Vec<u64>,
}

impl Bits {
    /// Return the set holding only `index`.
    pub(super) fn singleton(index: usize) -> Self {
        let mut bits = Self::default();
        bits.insert(index);
        bits
    }

    /// Add `index`.
    pub(super) fn insert(&mut self, index: usize) {
        let word = index / 64;
        if self.words.len() <= word {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (index % 64);
    }

    /// Return whether the set holds `index`.
    pub(super) fn contains(&self, index: usize) -> bool {
        self.words
            .get(index / 64)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }

    /// Add every index of `other`.
    pub(super) fn union_with(&mut self, other: &Self) {
        if self.words.len() < other.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (word, other_word) in self.words.iter_mut().zip(&other.words) {
            *word |= other_word;
        }
    }

    /// Return the indices both sets hold.
    pub(super) fn intersection(&self, other: &Self) -> Self {
        Self {
            words: self
                .words
                .iter()
                .zip(&other.words)
                .map(|(word, other_word)| word & other_word)
                .collect(),
        }
    }

    /// Return the indices, in increasing order.
    pub(super) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(position, &word)| {
            (0..64).filter_map(move |bit| (word & (1 << bit) != 0).then_some(position * 64 + bit))
        })
    }
}
