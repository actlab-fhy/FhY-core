//! Literal values and the exact decimals they may hold.

mod decimal;

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use num_bigint::{BigInt, Sign};
use serde::{Deserialize, Serialize};

pub use self::decimal::Decimal;
use self::decimal::is_ascii_digit_run;
use super::sort::FunctionSort;

/// The constant held by a literal expression, normalized.
///
/// A literal is a Boolean, an integer of any size, a binary floating-point
/// number (any `f64`, NaN and the infinities included), or an exact
/// [`Decimal`]. [`parse_text`](Self::parse_text) normalizes a numeric text
/// and keeps no spelling: `"05"` is the integer `5`, and `"1.50"` is the
/// decimal `1.5`.
///
/// Equality is a lawful equivalence, and hashing agrees with it. Literals
/// of different variants are unequal, so `1`, `1.0`, the decimal `1` and
/// `true` are pairwise unequal. Integers and decimals compare by value.
/// Floats compare with `==`, except that every NaN equals every other NaN;
/// `-0.0` equals `0.0`.
///
/// `Display` follows Rust conventions: `true` or `false` for a Boolean, the
/// decimal digits of an integer with a leading `-` when negative, a float as
/// `{}` writes an `f64` (`1.5`, `1` for `1.0`, `10000000000000000` for
/// `1e16`, `NaN`, `inf`, `-inf`, `-0`), and a decimal positionally (`1.5`,
/// `100`, `0.5`). So unequal literals may display alike: `1`, `1.0` and the
/// decimal `1` all display as `1`.
///
/// Serializes externally tagged by the lowercase variant name, every number
/// as a string: `{"bool": true}`, `{"int": "-12"}` with the decimal digits
/// of any integer, `{"float": "1.5"}` with the text `{}` writes for the
/// `f64` (`"NaN"`, `"inf"` and `"-inf"` included, so every float literal
/// serializes in every format), and `{"decimal": "1.5"}` with the decimal's
/// `Display` text. Deserializing reads an integer string only in the form
/// `-?(0|[1-9][0-9]*)`, other than `"-0"`, a float string through
/// [`f64::from_str`](std::str::FromStr), and a decimal string through the
/// literal grammar of [`Decimal`]'s `FromStr`.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{BigInt, LiteralValue};
///
/// let integer = LiteralValue::from(5);
/// let padded_text = LiteralValue::parse_text("05")?;
/// assert_eq!(integer, padded_text);
/// assert!(matches!(&padded_text, LiteralValue::Int(value) if *value == BigInt::from(5)));
/// assert_eq!(padded_text.to_string(), "5");
/// assert_eq!(LiteralValue::from(true).to_string(), "true");
/// assert_ne!(LiteralValue::from(1), LiteralValue::from(1.0));
/// # Ok::<(), fhy_core::expr::LiteralTextError>(())
/// ```
#[expect(clippy::exhaustive_enums, reason = "passes match every literal form")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiteralValue {
    /// A Boolean.
    Bool(bool),
    /// An integer of any size.
    Int(
        #[serde(
            serialize_with = "serialize_display_text",
            deserialize_with = "integer_text::deserialize"
        )]
        BigInt,
    ),
    /// A binary floating-point number: any `f64`, NaN and the infinities
    /// included.
    Float(
        #[serde(
            serialize_with = "serialize_display_text",
            deserialize_with = "float_text::deserialize"
        )]
        f64,
    ),
    /// An exact, normalized decimal.
    Decimal(Decimal),
}

impl LiteralValue {
    /// Parse a numeric literal text, normalizing it.
    ///
    /// An integer text, one or more ASCII digits (`"0"`, `"05"`), becomes an
    /// [`Int`](Self::Int) without its leading zeros. A decimal text, ASCII
    /// digits with one decimal point and at least one digit, in the forms
    /// `"1.5"`, `"1."` and `".5"`, becomes a [`Decimal`](Self::Decimal)
    /// without its leading and trailing zeros. Signs, exponents, whitespace,
    /// separators, `inf`, `nan`, and non-ASCII digits are refused.
    ///
    /// # Errors
    ///
    /// Returns [`LiteralTextError`] if `text` is neither an integer text nor
    /// a decimal text.
    pub fn parse_text(text: &str) -> Result<Self, LiteralTextError> {
        if is_ascii_digit_run(text) {
            let value = BigInt::parse_bytes(text.as_bytes(), 10)
                .ok_or_else(|| LiteralTextError { text: text.into() })?;
            return Ok(Self::Int(value));
        }
        text.parse().map(Self::Decimal)
    }
}

impl From<bool> for LiteralValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

/// Implement `From<$integer> for LiteralValue` for each primitive integer
/// type, building the integer literal of the value.
macro_rules! impl_from_integer {
    ($($integer:ty),*) => {
        $(
            impl From<$integer> for LiteralValue {
                fn from(value: $integer) -> Self {
                    Self::Int(BigInt::from(value))
                }
            }
        )*
    };
}

impl_from_integer!(i32, i64, i128, u32, u64, usize);

impl From<BigInt> for LiteralValue {
    fn from(value: BigInt) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for LiteralValue {
    /// Construct a float literal; NaN and the infinities are accepted.
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<Decimal> for LiteralValue {
    fn from(value: Decimal) -> Self {
        Self::Decimal(value)
    }
}

impl PartialEq for LiteralValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Int(left), Self::Int(right)) => left == right,
            (Self::Float(left), Self::Float(right)) => {
                left.partial_cmp(right) == Some(Ordering::Equal)
                    || (left.is_nan() && right.is_nan())
            }
            (Self::Decimal(left), Self::Decimal(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for LiteralValue {}

impl Hash for LiteralValue {
    /// Feed the variant, then the value: nothing more for a NaN, and the
    /// bits of the float with `-0.0` folded into `0.0` for any other float.
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Self::Bool(value) => value.hash(state),
            Self::Int(value) => value.hash(state),
            Self::Float(value) => {
                if !value.is_nan() {
                    (value + 0.0).to_bits().hash(state);
                }
            }
            Self::Decimal(value) => value.hash(state),
        }
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Float(value) => write!(f, "{value}"),
            Self::Decimal(value) => write!(f, "{value}"),
        }
    }
}

impl FunctionSort {
    /// Return whether a literal holding `value` has this sort.
    ///
    /// A Boolean has only the [`Bool`](Self::Bool) sort, and no other
    /// literal has it. An integer has the [`Int`](Self::Int) and
    /// [`Real`](Self::Real) sorts, and the [`Nat`](Self::Nat) sort too when
    /// it is not negative. A float, NaN and the infinities included, and a
    /// decimal have only the [`Real`](Self::Real) sort.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::expr::{FunctionSort, LiteralValue};
    ///
    /// assert!(FunctionSort::Nat.accepts_literal(&LiteralValue::from(3)));
    /// assert!(!FunctionSort::Nat.accepts_literal(&LiteralValue::from(-3)));
    /// assert!(!FunctionSort::Int.accepts_literal(&LiteralValue::from(true)));
    /// ```
    #[must_use]
    pub fn accepts_literal(self, value: &LiteralValue) -> bool {
        match (self, value) {
            (Self::Bool, LiteralValue::Bool(_))
            | (Self::Int | Self::Real, LiteralValue::Int(_))
            | (Self::Real, LiteralValue::Float(_) | LiteralValue::Decimal(_)) => true,
            (Self::Nat, LiteralValue::Int(integer)) => integer.sign() != Sign::Minus,
            (
                Self::Bool,
                LiteralValue::Int(_) | LiteralValue::Float(_) | LiteralValue::Decimal(_),
            )
            | (Self::Nat | Self::Int | Self::Real, LiteralValue::Bool(_))
            | (Self::Nat | Self::Int, LiteralValue::Float(_) | LiteralValue::Decimal(_)) => false,
        }
    }
}

/// A numeric literal text outside the integer and decimal grammar.
///
/// `Display` writes `invalid literal text {text:?}: expected ASCII digits
/// with at most one decimal point`, with the text in Rust string-literal
/// quoting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralTextError {
    text: Box<str>,
}

impl LiteralTextError {
    /// Return the refused text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for LiteralTextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid literal text {:?}: expected ASCII digits with at most one decimal point",
            self.text
        )
    }
}

impl Error for LiteralTextError {}

/// Serialize `value` as the string its `Display` writes: the decimal digits
/// of an integer, or the shortest text that reads back as a float.
fn serialize_display_text<T: fmt::Display, S: serde::Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_str(value)
}

/// The wire form of an integer literal: its decimal digits as a string.
mod integer_text {
    use std::fmt;

    use num_bigint::BigInt;
    use serde::de::{self, Deserializer, Visitor};

    use super::is_ascii_digit_run;

    /// Return whether `text` is the canonical decimal text of an integer:
    /// `-?(0|[1-9][0-9]*)`, other than `-0`.
    fn is_canonical_integer_text(text: &str) -> bool {
        let digits = text.strip_prefix('-').unwrap_or(text);
        is_ascii_digit_run(digits) && (digits == "0" || !digits.starts_with('0')) && text != "-0"
    }

    struct IntegerTextVisitor;

    impl Visitor<'_> for IntegerTextVisitor {
        type Value = BigInt;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("the decimal digits of an integer as a string")
        }

        fn visit_str<E: de::Error>(self, text: &str) -> Result<BigInt, E> {
            let parsed = is_canonical_integer_text(text)
                .then(|| BigInt::parse_bytes(text.as_bytes(), 10))
                .flatten();
            parsed.ok_or_else(|| E::custom(format_args!("invalid integer literal {text:?}")))
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BigInt, D::Error> {
        deserializer.deserialize_str(IntegerTextVisitor)
    }
}

/// The wire form of a float literal: the text `{}` writes for it, which
/// `f64::from_str` reads back to the same value.
mod float_text {
    use std::fmt;
    use std::num::ParseFloatError;

    use serde::de::{self, Deserializer, Visitor};

    struct FloatTextVisitor;

    impl Visitor<'_> for FloatTextVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("the text of a float as a string")
        }

        fn visit_str<E: de::Error>(self, text: &str) -> Result<f64, E> {
            text.parse().map_err(|_refused: ParseFloatError| {
                E::custom(format_args!("invalid float literal {text:?}"))
            })
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        deserializer.deserialize_str(FloatTextVisitor)
    }
}
