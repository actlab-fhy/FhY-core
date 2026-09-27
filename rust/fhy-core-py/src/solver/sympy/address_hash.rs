//! The hasher of the walks' memos, keyed by an address: a node identity
//! or a Python object's `id`.

use std::hash::{BuildHasherDefault, Hasher};

/// Hashes an address by one multiplication folded onto itself, which
/// spreads its bits into both the high and the low bits of the hash, as
/// the core's hasher of node identities does.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct AddressHasher(u64);

/// Builds [`AddressHasher`]s, for the memos keyed by addresses.
pub(super) type BuildAddressHasher = BuildHasherDefault<AddressHasher>;

impl Hasher for AddressHasher {
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
