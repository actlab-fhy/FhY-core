//! The core data types, their promotion, and the types literals resolve to.

use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

use crate::expression::LiteralValue;
use crate::expression::operation::impl_name_text;

use super::error::{IntegerFamily, LiteralTypeError, PromotionError};

/// A primitive element type: a weak literal family, a sized integer, float
/// or complex number, or a Boolean.
///
/// `Uint`, `Int` and `Float` are *weak*: they name a family, not a width,
/// and are the types literals take before a context gives them one.
/// [`promote`](Self::promote) joins two types of one family in its
/// promotion order; `Bool` promotes only with itself.
///
/// Displays and parses as its snake-case name (`int32`, `complex64`),
/// which is also its serde form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[repr(u8)]
pub enum CoreDataType {
    /// A non-negative integer literal of no width yet.
    Uint,
    /// A signed integer literal of no width yet.
    Int,
    /// A floating-point literal of no width yet.
    Float,
    /// An 8-bit unsigned integer.
    Uint8,
    /// A 16-bit unsigned integer.
    Uint16,
    /// A 32-bit unsigned integer.
    Uint32,
    /// An 8-bit signed integer.
    Int8,
    /// A 16-bit signed integer.
    Int16,
    /// A 32-bit signed integer.
    Int32,
    /// A 64-bit signed integer.
    Int64,
    /// A 16-bit float.
    Float16,
    /// A 32-bit float.
    Float32,
    /// A 64-bit float.
    Float64,
    /// A complex number of two 16-bit floats.
    Complex32,
    /// A complex number of two 32-bit floats.
    Complex64,
    /// A complex number of two 64-bit floats.
    Complex128,
    /// A one-bit Boolean.
    Bool,
}

/// The number of core data types.
const COUNT: usize = 17;

/// Every core data type, in declaration order.
const ALL: [CoreDataType; COUNT] = [
    CoreDataType::Uint,
    CoreDataType::Int,
    CoreDataType::Float,
    CoreDataType::Uint8,
    CoreDataType::Uint16,
    CoreDataType::Uint32,
    CoreDataType::Int8,
    CoreDataType::Int16,
    CoreDataType::Int32,
    CoreDataType::Int64,
    CoreDataType::Float16,
    CoreDataType::Float32,
    CoreDataType::Float64,
    CoreDataType::Complex32,
    CoreDataType::Complex64,
    CoreDataType::Complex128,
    CoreDataType::Bool,
];

/// The covering pairs of the integer promotion order: the unsigned chain,
/// the signed chain, the weak unsigned type below the weak signed one, and
/// each sized unsigned type below the signed type twice its width.
const INTEGER_ORDER: [(CoreDataType, CoreDataType); 11] = [
    (CoreDataType::Uint, CoreDataType::Uint8),
    (CoreDataType::Uint8, CoreDataType::Uint16),
    (CoreDataType::Uint16, CoreDataType::Uint32),
    (CoreDataType::Int, CoreDataType::Int8),
    (CoreDataType::Int8, CoreDataType::Int16),
    (CoreDataType::Int16, CoreDataType::Int32),
    (CoreDataType::Int32, CoreDataType::Int64),
    (CoreDataType::Uint, CoreDataType::Int),
    (CoreDataType::Uint8, CoreDataType::Int16),
    (CoreDataType::Uint16, CoreDataType::Int32),
    (CoreDataType::Uint32, CoreDataType::Int64),
];

/// The covering pairs of the float and complex promotion order: the real
/// chain from the weak float up, each real float below the complex type
/// twice its width, and the complex chain.
const FLOAT_COMPLEX_ORDER: [(CoreDataType, CoreDataType); 8] = [
    (CoreDataType::Float, CoreDataType::Float16),
    (CoreDataType::Float16, CoreDataType::Float32),
    (CoreDataType::Float32, CoreDataType::Float64),
    (CoreDataType::Float16, CoreDataType::Complex32),
    (CoreDataType::Float32, CoreDataType::Complex64),
    (CoreDataType::Float64, CoreDataType::Complex128),
    (CoreDataType::Complex32, CoreDataType::Complex64),
    (CoreDataType::Complex64, CoreDataType::Complex128),
];

/// Return the bit of `data_type` in a mask of core data types.
const fn bit(data_type: CoreDataType) -> u32 {
    1 << data_type as u32
}

/// Return the mask of the types that appear in `order`.
const fn members(order: &[(CoreDataType, CoreDataType)]) -> u32 {
    let mut mask = 0;
    let mut index = 0;
    while index < order.len() {
        mask |= bit(order[index].0) | bit(order[index].1);
        index += 1;
    }
    mask
}

/// Return each type's up-set in the reflexive and transitive closure of
/// `order`, as a mask.
const fn up_sets(order: &[(CoreDataType, CoreDataType)]) -> [u32; COUNT] {
    let mut up = [0; COUNT];
    let mut index = 0;
    while index < COUNT {
        up[index] = 1 << index;
        index += 1;
    }
    let mut round = 0;
    while round < COUNT {
        let mut pair = 0;
        while pair < order.len() {
            let (lower, upper) = order[pair];
            up[lower as usize] |= up[upper as usize];
            pair += 1;
        }
        round += 1;
    }
    up
}

/// The integer family.
const INTEGERS: u32 = members(&INTEGER_ORDER);
/// The float and complex family.
const FLOATS_AND_COMPLEXES: u32 = members(&FLOAT_COMPLEX_ORDER);
/// The up-sets of the integer promotion order.
const INTEGER_UP_SETS: [u32; COUNT] = up_sets(&INTEGER_ORDER);
/// The up-sets of the float and complex promotion order.
const FLOAT_COMPLEX_UP_SETS: [u32; COUNT] = up_sets(&FLOAT_COMPLEX_ORDER);
/// The unsigned integer types.
const UNSIGNED: u32 = bit(CoreDataType::Uint)
    | bit(CoreDataType::Uint8)
    | bit(CoreDataType::Uint16)
    | bit(CoreDataType::Uint32);
/// The real float types.
const REAL_FLOATS: u32 = bit(CoreDataType::Float)
    | bit(CoreDataType::Float16)
    | bit(CoreDataType::Float32)
    | bit(CoreDataType::Float64);

/// Return the least type of the intersection of the up-sets of `left` and
/// `right` in the order whose up-sets are `up`.
fn join(up: &[u32; COUNT], left: CoreDataType, right: CoreDataType) -> Option<CoreDataType> {
    let common = up[left as usize] & up[right as usize];
    ALL.into_iter().find(|&candidate| {
        common & bit(candidate) != 0 && up[candidate as usize] & common == common
    })
}

impl CoreDataType {
    /// Return every core data type, in declaration order.
    #[must_use]
    pub fn all() -> impl ExactSizeIterator<Item = Self> {
        ALL.into_iter()
    }

    /// Return the snake-case name, such as `int32`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uint => "uint",
            Self::Int => "int",
            Self::Float => "float",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Complex32 => "complex32",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
            Self::Bool => "bool",
        }
    }

    /// Return the width in bits, or `None` for a weak type. A Boolean is one
    /// bit wide, and a complex type is as wide as its two parts together.
    #[must_use]
    pub const fn bit_width(self) -> Option<u32> {
        match self {
            Self::Uint | Self::Int | Self::Float => None,
            Self::Bool => Some(1),
            Self::Uint8 | Self::Int8 => Some(8),
            Self::Uint16 | Self::Int16 | Self::Float16 => Some(16),
            Self::Uint32 | Self::Int32 | Self::Float32 | Self::Complex32 => Some(32),
            Self::Int64 | Self::Float64 | Self::Complex64 => Some(64),
            Self::Complex128 => Some(128),
        }
    }

    /// Return whether the type is weak: `Uint`, `Int` or `Float`.
    #[must_use]
    pub const fn is_weak(self) -> bool {
        matches!(self, Self::Uint | Self::Int | Self::Float)
    }

    /// Return whether the type is an integer, weak or sized, of either
    /// sign.
    #[must_use]
    pub const fn is_integral(self) -> bool {
        INTEGERS & bit(self) != 0
    }

    /// Return whether the type is an unsigned integer, weak or sized.
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        UNSIGNED & bit(self) != 0
    }

    /// Return whether the type is a signed integer, weak or sized.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        self.is_integral() && !self.is_unsigned()
    }

    /// Return whether the type is a real float, weak or sized.
    #[must_use]
    pub const fn is_real_float(self) -> bool {
        REAL_FLOATS & bit(self) != 0
    }

    /// Return whether the type is a complex number.
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64 | Self::Complex128)
    }

    /// Return whether the type is a real float or a complex number.
    #[must_use]
    pub const fn is_float_like(self) -> bool {
        FLOATS_AND_COMPLEXES & bit(self) != 0
    }

    /// Return the type both `self` and `other` promote to: their join in
    /// the integer order, or in the float and complex order.
    ///
    /// # Errors
    ///
    /// Returns [`PromotionError::Boolean`] when exactly one of the two is
    /// `Bool`, and [`PromotionError::AcrossFamilies`] when the two belong to
    /// different families.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::types::CoreDataType;
    ///
    /// assert_eq!(CoreDataType::Uint16.promote(CoreDataType::Int8)?, CoreDataType::Int32);
    /// assert_eq!(CoreDataType::Float16.promote(CoreDataType::Complex64)?, CoreDataType::Complex64);
    /// assert!(CoreDataType::Int8.promote(CoreDataType::Float32).is_err());
    /// # Ok::<(), fhy_core::types::PromotionError>(())
    /// ```
    pub fn promote(self, other: Self) -> Result<Self, PromotionError> {
        if self == Self::Bool && other == Self::Bool {
            return Ok(Self::Bool);
        }
        if self == Self::Bool || other == Self::Bool {
            return Err(PromotionError::Boolean(self, other));
        }
        let joined = if self.is_integral() && other.is_integral() {
            join(&INTEGER_UP_SETS, self, other)
        } else if self.is_float_like() && other.is_float_like() {
            join(&FLOAT_COMPLEX_UP_SETS, self, other)
        } else {
            None
        };
        joined.ok_or(PromotionError::AcrossFamilies(self, other))
    }

    /// Return the concrete type the literal `literal` takes in the context
    /// `context`.
    ///
    /// - A Boolean resolves to `Bool`, only in the `Bool` context.
    /// - A float resolves to a float or complex context, the weak `Float`
    ///   to `Float64`.
    /// - A non-negative integer in an unsigned context resolves to its
    ///   narrowest unsigned type, promoted with the context (`Uint` counts as
    ///   `Uint8`); in a signed context, to its narrowest signed type,
    ///   promoted likewise (`Int` counts as `Int8`); in a float or complex
    ///   context, to the context, the weak `Float` to `Float64`.
    ///
    /// # Errors
    ///
    /// Returns a [`LiteralTypeError`] for a literal the context cannot hold:
    /// of the wrong kind, out of every width of the context's family, or a
    /// decimal, which has no core data type yet.
    pub fn resolve_literal(
        literal: &LiteralValue,
        context: Self,
    ) -> Result<Self, LiteralTypeError> {
        match literal {
            LiteralValue::Bool(value) => {
                if context == Self::Bool {
                    Ok(Self::Bool)
                } else {
                    Err(LiteralTypeError::BooleanIncompatible {
                        value: *value,
                        context,
                    })
                }
            }
            _ if context == Self::Bool => Err(LiteralTypeError::NonBooleanInBooleanContext {
                literal: literal.clone(),
            }),
            LiteralValue::Float(_) => {
                if context.is_float_like() {
                    Ok(concrete_float_like(context))
                } else {
                    Err(LiteralTypeError::Incompatible {
                        literal: literal.clone(),
                        context,
                    })
                }
            }
            LiteralValue::Int(value) => resolve_integer(value, literal, context),
            LiteralValue::Decimal(_) => Err(LiteralTypeError::UnsupportedDecimal),
        }
    }
}

/// Return the weak `Float` as `Float64`, and any other type as itself.
fn concrete_float_like(context: CoreDataType) -> CoreDataType {
    if context == CoreDataType::Float {
        CoreDataType::Float64
    } else {
        context
    }
}

/// Resolve the integer literal `value` in `context`.
fn resolve_integer(
    value: &BigInt,
    literal: &LiteralValue,
    context: CoreDataType,
) -> Result<CoreDataType, LiteralTypeError> {
    if *value >= BigInt::ZERO && context.is_unsigned() {
        let narrowest = narrowest_integer_type(value, IntegerFamily::Unsigned)?;
        let context = if context == CoreDataType::Uint {
            CoreDataType::Uint8
        } else {
            context
        };
        return Ok(narrowest
            .promote(context)
            .unwrap_or_else(|_one_family| unreachable!("two integer types of one sign promote")));
    }
    if context.is_signed() {
        let narrowest = narrowest_integer_type(value, IntegerFamily::Signed)?;
        let context = if context == CoreDataType::Int {
            CoreDataType::Int8
        } else {
            context
        };
        return Ok(narrowest
            .promote(context)
            .unwrap_or_else(|_one_family| unreachable!("two integer types of one sign promote")));
    }
    if context.is_float_like() {
        return Ok(concrete_float_like(context));
    }
    Err(LiteralTypeError::Incompatible {
        literal: literal.clone(),
        context,
    })
}

/// Return the narrowest sized type of `family` that holds `value`.
fn narrowest_integer_type(
    value: &BigInt,
    family: IntegerFamily,
) -> Result<CoreDataType, LiteralTypeError> {
    let candidates: &[(CoreDataType, u32)] = match family {
        IntegerFamily::Unsigned => &[
            (CoreDataType::Uint8, 8),
            (CoreDataType::Uint16, 16),
            (CoreDataType::Uint32, 32),
        ],
        IntegerFamily::Signed => &[
            (CoreDataType::Int8, 8),
            (CoreDataType::Int16, 16),
            (CoreDataType::Int32, 32),
            (CoreDataType::Int64, 64),
        ],
    };
    for &(candidate, width) in candidates {
        let (lower, upper) = match family {
            IntegerFamily::Unsigned => (BigInt::ZERO, (BigInt::from(1) << width) - 1),
            IntegerFamily::Signed => (
                -(BigInt::from(1) << (width - 1)),
                (BigInt::from(1) << (width - 1)) - 1,
            ),
        };
        if lower <= *value && *value <= upper {
            return Ok(candidate);
        }
    }
    Err(LiteralTypeError::OutOfRange {
        value: value.clone(),
        family,
    })
}

impl_name_text!(CoreDataType, "core data type");
