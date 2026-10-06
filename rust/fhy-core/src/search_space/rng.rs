//! [`Rng`]: the random-number generator every random draw of a search
//! takes its numbers from.

#![expect(
    unused_variables,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::num::NonZeroU64;

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

/// A seeded, deterministic generator of 64-bit numbers: SplitMix64.
///
/// Its stream is a contract: for a seed, the numbers it returns, and for
/// each method, which numbers it takes and how it turns them into its
/// result, are the same in every release, on every platform, from Rust and
/// from Python. Changing either is a breaking change.
///
/// - [`next_u64`](Self::next_u64) adds `0x9E37_79B9_7F4A_7C15` to the
///   state, wrapping, and returns the new state mixed: `z ^= z >> 30; z *=
///   0xBF58_476D_1CE4_E5B9; z ^= z >> 27; z *= 0x94D0_49BB_1331_11EB; z ^=
///   z >> 31`, the multiplications wrapping.
/// - [`below`](Self::below) is Lemire's multiply-and-reject method.
/// - [`below_big`](Self::below_big) draws whole limbs and rejects.
/// - [`shuffle`](Self::shuffle) is Fisher-Yates from the last index down.
///
/// Cloning copies the state, so the clone continues the same stream.
/// `serde` writes `{"algorithm": "splitmix64", "state": <u64>}`, from which
/// a generator resumes its stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RngWire", into = "RngWire")]
pub struct Rng {
    state: u64,
}

/// The wire form of an [`Rng`].
#[derive(Serialize, Deserialize)]
#[serde(rename = "Rng", deny_unknown_fields)]
struct RngWire {
    algorithm: String,
    state: u64,
}

impl From<Rng> for RngWire {
    fn from(rng: Rng) -> Self {
        todo!()
    }
}

impl TryFrom<RngWire> for Rng {
    type Error = String;

    /// Refuse an algorithm other than [`Rng::ALGORITHM`], with a message
    /// naming it.
    fn try_from(wire: RngWire) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl Rng {
    /// The generator's name, which its wire form records.
    pub const ALGORITHM: &'static str = "splitmix64";

    /// Return the generator whose state is `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        todo!()
    }

    /// Return the next number of the stream.
    pub fn next_u64(&mut self) -> u64 {
        todo!()
    }

    /// Return a number drawn uniformly from `[0, bound)`.
    ///
    /// Draws `x`, forms the 128-bit product `m = x * bound`, and returns its
    /// high 64 bits unless its low 64 bits are below `(2^64 - bound) mod
    /// bound`, in which case it draws again.
    pub fn below(&mut self, bound: NonZeroU64) -> u64 {
        todo!()
    }

    /// Return a number drawn uniformly from `[0, bound)`.
    ///
    /// With `b` the bit length of `bound` and `n = ceil(b / 64)`, draws `n`
    /// numbers, the first the least significant limb, keeps the low `b - 64
    /// * (n - 1)` bits of the last, and returns the number unless it is at
    /// least `bound`, in which case it draws `n` again.
    ///
    /// # Panics
    ///
    /// Panics with `the bound of below_big must be positive` if `bound` is
    /// zero: every caller passes a cardinality, which is at least one.
    pub fn below_big(&mut self, bound: &BigUint) -> BigUint {
        todo!()
    }

    /// Shuffle `items` uniformly: for each index `i` from the last down to
    /// 1, swap the item at `i` with the one at `below(i + 1)`.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        todo!()
    }

    /// Return a generator of an independent stream, seeded with this one's
    /// next number.
    #[must_use]
    pub fn split(&mut self) -> Self {
        todo!()
    }
}
