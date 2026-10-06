//! [`Rng`]: the random-number generator every random draw of a search
//! takes its numbers from.

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
        Self {
            algorithm: Rng::ALGORITHM.to_owned(),
            state: rng.state,
        }
    }
}

impl TryFrom<RngWire> for Rng {
    type Error = String;

    /// Refuse an algorithm other than [`Rng::ALGORITHM`], with a message
    /// naming it.
    fn try_from(wire: RngWire) -> Result<Self, Self::Error> {
        if wire.algorithm != Rng::ALGORITHM {
            return Err(format!(
                "the generator {:?} is not {:?}",
                wire.algorithm,
                Rng::ALGORITHM
            ));
        }
        Ok(Self { state: wire.state })
    }
}

/// SplitMix64's increment of the state.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// SplitMix64's first mixing multiplier.
const MIX_FIRST: u64 = 0xBF58_476D_1CE4_E5B9;
/// SplitMix64's second mixing multiplier.
const MIX_SECOND: u64 = 0x94D0_49BB_1331_11EB;

/// Return the 32-bit digits of the 64-bit digits `digits`, least
/// significant first.
fn digits_to_u32(digits: &[u64]) -> Vec<u32> {
    digits
        .iter()
        .flat_map(|&digit| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "each half of a 64-bit digit is taken on purpose"
            )]
            [digit as u32, (digit >> 32) as u32]
        })
        .collect()
}

impl Rng {
    /// The generator's name, which its wire form records.
    pub const ALGORITHM: &'static str = "splitmix64";

    /// Return the generator whose state is `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return the next number of the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(MIX_FIRST);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(MIX_SECOND);
        mixed ^ (mixed >> 31)
    }

    /// Return a number drawn uniformly from `[0, bound)`.
    ///
    /// Draws `x`, forms the 128-bit product `m = x * bound`, and returns its
    /// high 64 bits unless its low 64 bits are below `(2^64 - bound) mod
    /// bound`, in which case it draws again.
    pub fn below(&mut self, bound: NonZeroU64) -> u64 {
        let bound = bound.get();
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u128::from(self.next_u64()) * u128::from(bound);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the low half of the 128-bit product is wanted"
            )]
            let low = product as u64;
            if low >= threshold {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the high half of a product of two 64-bit numbers fits 64 bits"
                )]
                return (product >> 64) as u64;
            }
        }
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
        assert!(
            *bound != BigUint::ZERO,
            "the bound of below_big must be positive"
        );
        let bits = bound.bits();
        let limbs = bits.div_ceil(64);
        let top_bits = bits - 64 * (limbs - 1);
        let top_mask = if top_bits == 64 {
            u64::MAX
        } else {
            (1_u64 << top_bits) - 1
        };
        loop {
            let mut digits: Vec<u64> = (0..limbs).map(|_| self.next_u64()).collect();
            if let Some(top) = digits.last_mut() {
                *top &= top_mask;
            }
            let candidate = BigUint::from_slice(&digits_to_u32(&digits));
            if candidate < *bound {
                return candidate;
            }
        }
    }

    /// Shuffle `items` uniformly: for each index `i` from the last down to
    /// 1, swap the item at `i` with the one at `below(i + 1)`.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let count = u64::try_from(index + 1).unwrap_or(u64::MAX);
            let other = self.below(NonZeroU64::MIN.saturating_add(count - 1));
            let other = usize::try_from(other).unwrap_or(index);
            items.swap(index, other);
        }
    }

    /// Return a generator of an independent stream, seeded with this one's
    /// next number.
    #[must_use]
    pub fn split(&mut self) -> Self {
        Self::new(self.next_u64())
    }
}
