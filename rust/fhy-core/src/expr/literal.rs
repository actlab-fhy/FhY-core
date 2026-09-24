//! Literal values and their canonical equivalence keys.

use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};

use num_bigint::{BigInt, Sign};

use crate::python_text::{
    NormalizedDecimal, format_bool, format_float_repr, format_normalized_decimal,
    is_ascii_digit_run, normalize_decimal_text,
};

use super::sort::FunctionSort;

/// The stored form of a literal, as the caller gave it.
///
/// A text also holds the value it denotes, so equality and hashing never
/// parse text.
#[derive(Debug, Clone)]
enum Representation {
    Bool(bool),
    Int(BigInt),
    Float(f64),
    IntegerText {
        text: Box<str>,
        value: BigInt,
    },
    DecimalText {
        text: Box<str>,
        value: NormalizedDecimal,
    },
}

/// The equivalence bucket and canonical value of a literal.
///
/// Two literals are equal exactly when their canonical forms are equal.
#[derive(PartialEq, Eq, Hash)]
enum CanonicalForm<'a> {
    Bool(bool),
    Integer(&'a BigInt),
    /// The bits of a non-NaN float with `-0.0` folded into `0.0`, or `None`
    /// for every NaN.
    BinaryFloat(Option<u64>),
    ExactDecimal(&'a NormalizedDecimal),
}

/// Return `value` with negative zero folded into positive zero.
fn fold_negative_zero(value: f64) -> f64 {
    value + 0.0
}

/// The constant held by a literal expression.
///
/// A literal is a Boolean, an integer of any size, a binary floating-point
/// number, or an exact numeric text. A text keeps the caller's spelling
/// verbatim: an integer text such as `"05"` denotes an integer, and a decimal
/// text such as `"1.50"` denotes an exact decimal.
///
/// Every literal falls in one of four equivalence buckets: Boolean, integer
/// (integers and integer texts), binary float, and exact decimal (decimal
/// texts). Two literals are equal exactly when they share a bucket and a
/// canonical value, which is exactly when their
/// [`canonical_key`](Self::canonical_key)s are equal; equality is therefore
/// a lawful equivalence, and hashing agrees with it. So `5`, `"5"` and
/// `"05"` are equal, every NaN equals every other NaN, `-0.0` equals `0.0`,
/// and `1`, `1.0`, `true` and `"1.0"` are pairwise unequal.
///
/// `Display` writes the value as the caller gave it: `True` or `False` for a
/// Boolean, the decimal digits of an integer, the float text described
/// under [`canonical_key`](Self::canonical_key) but keeping the sign of
/// zero (`1.5`, `1e+16`, `-0.0`, `nan`, `inf`), and a text verbatim (`05`,
/// `1.50`).
///
/// # Examples
///
/// ```
/// use fhy_core::expr::LiteralValue;
///
/// let integer = LiteralValue::from(5);
/// let padded_text = LiteralValue::parse_text("05")?;
/// assert_eq!(integer, padded_text);
/// assert_eq!(padded_text.canonical_key(), "int:5");
/// assert_eq!(padded_text.to_string(), "05");
/// assert_ne!(LiteralValue::from(1), LiteralValue::from(1.0));
/// # Ok::<(), fhy_core::expr::LiteralTextError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LiteralValue {
    representation: Representation,
}

/// A borrowed view of a [`LiteralValue`]'s stored form, for matching.
///
/// Comparing two views with `==` compares stored forms, not literal
/// equivalence: `Int(5)` differs from `IntegerText("05")`, and a NaN
/// `Float` differs from itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiteralKind<'a> {
    /// A Boolean.
    Bool(bool),
    /// An integer.
    Int(&'a BigInt),
    /// A binary floating-point number, possibly NaN or infinite.
    Float(f64),
    /// An integer text of one or more ASCII digits, such as `"05"`.
    IntegerText(&'a str),
    /// A decimal text such as `"1.50"`, `"1."` or `".5"`.
    DecimalText(&'a str),
}

impl LiteralValue {
    /// Construct a Boolean literal.
    #[must_use]
    pub fn from_bool(value: bool) -> Self {
        Self {
            representation: Representation::Bool(value),
        }
    }

    /// Parse a numeric literal text, keeping its spelling verbatim.
    ///
    /// The text is an integer text, one or more ASCII digits (`"0"`,
    /// `"05"`), or a decimal text: ASCII digits with one decimal point and at
    /// least one digit, in the forms `"1.5"`, `"1."` and `".5"`. Signs,
    /// exponents, whitespace, separators, `inf`, `nan`, and non-ASCII digits
    /// are refused.
    ///
    /// # Errors
    ///
    /// Returns [`LiteralTextError`] if `text` is neither an integer text nor
    /// a decimal text.
    pub fn parse_text(text: &str) -> Result<Self, LiteralTextError> {
        let representation = if is_ascii_digit_run(text) {
            let value = BigInt::parse_bytes(text.as_bytes(), 10)
                .ok_or_else(|| LiteralTextError { text: text.into() })?;
            Representation::IntegerText {
                text: text.into(),
                value,
            }
        } else {
            let value = normalize_decimal_text(text)
                .ok_or_else(|| LiteralTextError { text: text.into() })?;
            Representation::DecimalText {
                text: text.into(),
                value,
            }
        };
        Ok(Self { representation })
    }

    /// Return a borrowed view of the stored form.
    #[must_use]
    pub fn kind(&self) -> LiteralKind<'_> {
        match &self.representation {
            Representation::Bool(value) => LiteralKind::Bool(*value),
            Representation::Int(value) => LiteralKind::Int(value),
            Representation::Float(value) => LiteralKind::Float(*value),
            Representation::IntegerText { text, .. } => LiteralKind::IntegerText(text),
            Representation::DecimalText { text, .. } => LiteralKind::DecimalText(text),
        }
    }

    /// Return the canonical key, `"{bucket}:{canonical value}"`, shared by
    /// exactly the literals equal to this one.
    ///
    /// The buckets and canonical values are:
    ///
    /// - `bool`: `True` or `False` (`bool:True`);
    /// - `int`, for integers and integer texts: the integer's decimal digits
    ///   with a leading `-` when negative and no leading zeros (`int:5` for
    ///   `5`, `"5"` and `"05"`);
    /// - `float-binary`, for floats: the shortest digits that read back as
    ///   the same float; with `n` the position of the decimal point relative
    ///   to the first digit (`n = 1` for `1.5`, `n = -3` for `0.0001`), the
    ///   text is positional with a fractional part when `-4 < n <= 16`
    ///   (`1.0`, `0.0001`, `1000000000000000.0`), and otherwise scientific
    ///   with one digit before an optional fractional part, a lowercase `e`,
    ///   and a signed exponent of at least two digits (`1e+16`, `1e-05`,
    ///   `1.5e+300`); `-0.0` is folded into `0.0`, the
    ///   infinities are `inf` and `-inf`, and every NaN is `nan`
    ///   (`float-binary:0.0`, `float-binary:1e+16`, `float-binary:-inf`);
    /// - `float-decimal`, for decimal texts: the exact decimal with leading
    ///   and trailing zeros removed and no rounding at any length, written
    ///   positionally when its exponent is at most zero and its adjusted
    ///   exponent at least `-6`, and otherwise in scientific notation with a
    ///   capital `E` and a signed exponent (`float-decimal:1.5` for `"1.50"`,
    ///   `float-decimal:1E+2` for `"100.0"`, `float-decimal:1E-7`).
    #[must_use]
    pub fn canonical_key(&self) -> String {
        match &self.representation {
            Representation::Bool(value) => format!("bool:{}", format_bool(*value)),
            Representation::Int(value) | Representation::IntegerText { value, .. } => {
                format!("int:{value}")
            }
            Representation::Float(value) => {
                format!(
                    "float-binary:{}",
                    format_float_repr(fold_negative_zero(*value))
                )
            }
            Representation::DecimalText { value, .. } => {
                format!("float-decimal:{}", format_normalized_decimal(value))
            }
        }
    }

    /// Return whether the literal is in the integer bucket: an integer or an
    /// integer text.
    ///
    /// A float or a decimal text is never integer-valued, even when it has no
    /// fractional part, and neither is a Boolean.
    #[must_use]
    pub fn is_integer_valued(&self) -> bool {
        matches!(
            self.representation,
            Representation::Int(_) | Representation::IntegerText { .. }
        )
    }

    /// Return the literal's equivalence bucket and canonical value.
    fn canonical_form(&self) -> CanonicalForm<'_> {
        match &self.representation {
            Representation::Bool(value) => CanonicalForm::Bool(*value),
            Representation::Int(value) | Representation::IntegerText { value, .. } => {
                CanonicalForm::Integer(value)
            }
            Representation::Float(value) => CanonicalForm::BinaryFloat(
                (!value.is_nan()).then(|| fold_negative_zero(*value).to_bits()),
            ),
            Representation::DecimalText { value, .. } => CanonicalForm::ExactDecimal(value),
        }
    }
}

impl From<bool> for LiteralValue {
    fn from(value: bool) -> Self {
        Self::from_bool(value)
    }
}

impl From<i64> for LiteralValue {
    fn from(value: i64) -> Self {
        Self::from(BigInt::from(value))
    }
}

impl From<BigInt> for LiteralValue {
    fn from(value: BigInt) -> Self {
        Self {
            representation: Representation::Int(value),
        }
    }
}

impl From<f64> for LiteralValue {
    /// Construct a float literal; NaN and the infinities are accepted.
    fn from(value: f64) -> Self {
        Self {
            representation: Representation::Float(value),
        }
    }
}

impl PartialEq for LiteralValue {
    fn eq(&self, other: &Self) -> bool {
        self.canonical_form() == other.canonical_form()
    }
}

impl Eq for LiteralValue {}

impl Hash for LiteralValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_form().hash(state);
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.representation {
            Representation::Bool(value) => f.write_str(format_bool(*value)),
            Representation::Int(value) => write!(f, "{value}"),
            Representation::Float(value) => f.write_str(&format_float_repr(*value)),
            Representation::IntegerText { text, .. } | Representation::DecimalText { text, .. } => {
                f.write_str(text)
            }
        }
    }
}

impl FunctionSort {
    /// Return whether a literal holding `value` has this sort.
    ///
    /// A Boolean has only the [`Bool`](Self::Bool) sort, and no other
    /// literal has it. An integer or an integer text has the
    /// [`Int`](Self::Int) and [`Real`](Self::Real) sorts, and the
    /// [`Nat`](Self::Nat) sort too when it is not negative (an integer text
    /// never is). A float, NaN and the infinities included, and a decimal
    /// text have only the [`Real`](Self::Real) sort.
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
        match (self, &value.representation) {
            (Self::Bool, Representation::Bool(_))
            | (
                Self::Int | Self::Real,
                Representation::Int(_) | Representation::IntegerText { .. },
            )
            | (Self::Nat, Representation::IntegerText { .. })
            | (Self::Real, Representation::Float(_) | Representation::DecimalText { .. }) => true,
            (Self::Nat, Representation::Int(integer)) => integer.sign() != Sign::Minus,
            (
                Self::Bool,
                Representation::Int(_)
                | Representation::Float(_)
                | Representation::IntegerText { .. }
                | Representation::DecimalText { .. },
            )
            | (Self::Nat | Self::Int | Self::Real, Representation::Bool(_))
            | (
                Self::Nat | Self::Int,
                Representation::Float(_) | Representation::DecimalText { .. },
            ) => false,
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
