//! The hasher of maps and sets keyed by node identities.

use std::hash::{BuildHasherDefault, Hasher};

/// Hashes a [`NodeIdentity`](super::NodeIdentity), an address, by one
/// multiplication folded onto itself, which spreads its bits into both the
/// high and the low bits of the hash.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct IdentityHasher(u64);

/// Builds [`IdentityHasher`]s, for maps and sets keyed by node identities.
pub(crate) type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, value: u64) {
        const MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
        let product = (self.0 ^ value).wrapping_mul(MULTIPLIER);
        self.0 = product ^ (product >> 32);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}
