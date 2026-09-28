//! Errors of promotion, literal resolution, building types, and binding,
//! substitution and unification.

use std::error::Error;
use std::fmt;

use crate::expression::{
    BigInt, Expression, FormatOptions, IdentifierStyle, LiteralValue, PiecewiseError,
};
use crate::foreign::BoxError;
use crate::identifier::Identifier;

use super::core_data_type::CoreDataType;
use super::data_type::{DataType, TemplateDataType};
use super::ty::Type;

/// Two core data types with no common promotion.
///
/// Displays as `unsupported primitive data type promotion: int8, float32`,
/// or `... involving boolean: bool, int8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PromotionError {
    /// Exactly one of the two is `Bool`, which promotes only with itself.
    Boolean(CoreDataType, CoreDataType),
    /// The two belong to different promotion families.
    AcrossFamilies(CoreDataType, CoreDataType),
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean(left, right) => write!(
                f,
                "unsupported primitive data type promotion involving boolean: {left}, {right}"
            ),
            Self::AcrossFamilies(left, right) => {
                write!(
                    f,
                    "unsupported primitive data type promotion: {left}, {right}"
                )
            }
        }
    }
}

impl Error for PromotionError {}

/// The two integer families a literal's narrowest type is sought in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegerFamily {
    /// `Uint8` to `Uint32`.
    Unsigned,
    /// `Int8` to `Int64`.
    Signed,
}

/// A literal a context cannot give a core data type.
///
/// Displays one lowercase line, such as `literal 300 does not fit in a
/// supported uint type` or `boolean literal true is incompatible with
/// int32`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LiteralTypeError {
    /// A Boolean in a context other than `Bool`.
    BooleanIncompatible {
        /// The Boolean.
        value: bool,
        /// The context.
        context: CoreDataType,
    },
    /// A number in the `Bool` context.
    NonBooleanInBooleanContext {
        /// The number.
        literal: LiteralValue,
    },
    /// A number in a context of another family.
    Incompatible {
        /// The number.
        literal: LiteralValue,
        /// The context.
        context: CoreDataType,
    },
    /// An integer outside every width of the family its context asks for.
    OutOfRange {
        /// The integer.
        value: BigInt,
        /// The family.
        family: IntegerFamily,
    },
    /// A decimal literal, which has no core data type yet.
    UnsupportedDecimal,
}

impl LiteralTypeError {
    /// Return whether the literal's kind has no core data type at all, as
    /// opposed to a literal the context refuses.
    #[must_use]
    pub const fn is_unsupported(&self) -> bool {
        matches!(self, Self::UnsupportedDecimal)
    }
}

impl fmt::Display for LiteralTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BooleanIncompatible { value, context } => {
                write!(f, "boolean literal {value} is incompatible with {context}")
            }
            Self::NonBooleanInBooleanContext { literal } => write!(
                f,
                "non-boolean literal {} is incompatible with the bool context",
                Expression::from(literal.clone())
            ),
            Self::Incompatible { literal, context } => {
                let kind = if matches!(literal, LiteralValue::Float(_)) {
                    "float literal"
                } else {
                    "literal"
                };
                write!(
                    f,
                    "{kind} {} is incompatible with {context}",
                    Expression::from(literal.clone())
                )
            }
            Self::OutOfRange { value, family } => {
                let family = match family {
                    IntegerFamily::Unsigned => "uint",
                    IntegerFamily::Signed => "int",
                };
                write!(
                    f,
                    "literal {value} does not fit in a supported {family} type"
                )
            }
            Self::UnsupportedDecimal => f.write_str("decimal literals are not yet supported"),
        }
    }
}

impl Error for LiteralTypeError {}

/// A width constraint of a [`TemplateDataType`] that holds a width of zero,
/// or no width at all, which no data type could bind.
///
/// Displays as `template data type widths must be positive, but got 0`, or
/// `template data type widths must not be empty`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TemplateWidthError {
    is_empty_list: bool,
}

impl TemplateWidthError {
    /// Return the error of a width of zero.
    pub(crate) const fn zero_width() -> Self {
        Self {
            is_empty_list: false,
        }
    }

    /// Return the error of an empty width list.
    pub(crate) const fn empty_list() -> Self {
        Self {
            is_empty_list: true,
        }
    }

    /// Return whether the width list was empty, rather than holding a zero.
    #[must_use]
    pub const fn is_empty_list(&self) -> bool {
        self.is_empty_list
    }
}

impl fmt::Display for TemplateWidthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty_list {
            f.write_str("template data type widths must not be empty")
        } else {
            f.write_str("template data type widths must be positive, but got 0")
        }
    }
}

impl Error for TemplateWidthError {}

/// Which operation a [`UnificationError`] arose in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypeOperation {
    /// Binding a template pattern against a concrete value.
    Bind,
    /// Unifying two values with placeholders on either side.
    Unify,
}

/// A pattern and a value that cannot be bound or unified.
///
/// Displays one lowercase line. Types print as their
/// [`Display`](fmt::Display) form, template data types and identifiers as
/// `name::id`, and expressions with identifier ids, such as `occurs check
/// failed: identifier N::7 appears in (N::7 + 1) after substitution through
/// existing bindings ((N::7 + 1))`.
#[derive(Debug)]
#[non_exhaustive]
pub enum UnificationError {
    /// Two types without a rule of their own are not structurally
    /// equivalent.
    TypeMismatch {
        /// The operation.
        operation: TypeOperation,
        /// The pattern, or the expected type.
        expected: Type,
        /// The actual type.
        actual: Type,
    },
    /// Two data types are not structurally equivalent.
    DataTypeMismatch {
        /// The operation.
        operation: TypeOperation,
        /// The pattern, or the expected data type.
        expected: DataType,
        /// The actual data type.
        actual: DataType,
    },
    /// A rule for one kind of type met a value of another kind.
    KindMismatch {
        /// The operation.
        operation: TypeOperation,
        /// The kind of the pattern, or of the expected value.
        expected: String,
        /// The kind of the actual value.
        actual: String,
    },
    /// Two primitive data types differ.
    CoreDataTypeMismatch {
        /// The pattern's core data type.
        expected: CoreDataType,
        /// The actual core data type.
        actual: CoreDataType,
    },
    /// Two numerical types have shapes of different ranks.
    RankMismatch {
        /// The operation.
        operation: TypeOperation,
        /// The rank of the pattern, or of the expected type.
        expected: usize,
        /// The rank of the actual type.
        actual: usize,
    },
    /// A shape dimension of the pattern that is no placeholder differs from
    /// the actual one.
    DimensionMismatch {
        /// The pattern's dimension.
        expected: Expression,
        /// The actual dimension.
        actual: Expression,
    },
    /// A shape variable is bound to one expression and met another.
    ConflictingExpressionBinding {
        /// The shape variable.
        identifier: Identifier,
        /// Its binding.
        bound: Expression,
        /// The expression it met.
        actual: Expression,
    },
    /// A template data type is bound to one data type and met another.
    ConflictingDataTypeBinding {
        /// The template's identifier.
        identifier: Identifier,
        /// Its binding.
        bound: DataType,
        /// The data type it met.
        actual: DataType,
    },
    /// A full-type placeholder is bound to one type and met another.
    ConflictingTypeBinding {
        /// The placeholder's identifier.
        identifier: Identifier,
        /// Its binding.
        bound: Type,
        /// The type it met.
        actual: Type,
    },
    /// Two different template data types met.
    DistinctTemplates {
        /// The operation.
        operation: TypeOperation,
        /// The pattern's, or the expected, template.
        expected: TemplateDataType,
        /// The actual template.
        actual: TemplateDataType,
    },
    /// A template with a width constraint met a data type that is not
    /// primitive.
    WidthOnNonPrimitive {
        /// The template.
        template: TemplateDataType,
        /// The data type it met.
        actual: DataType,
    },
    /// A template with a width constraint met a primitive type of another
    /// width, or of none.
    WidthMismatch {
        /// The template.
        template: TemplateDataType,
        /// The core data type it met.
        actual: CoreDataType,
    },
    /// A placeholder would be bound to an expression that holds it.
    OccursCheck {
        /// The placeholder.
        identifier: Identifier,
        /// The expression it would be bound to.
        expression: Expression,
        /// That expression with the existing bindings substituted.
        substituted: Expression,
    },
    /// Two expressions that are no placeholders differ.
    ExpressionMismatch {
        /// The left expression.
        left: Expression,
        /// The right expression.
        right: Expression,
    },
    /// A wildcard dimension in the actual type of a binding.
    WildcardInActual,
    /// A wildcard dimension in a unification.
    WildcardInUnification,
    /// A type or data type defined outside this crate failed.
    Extension(BoxError),
    /// Substituting the existing shape bindings into an expression was
    /// refused, such as a binding that puts a non-Boolean literal in a
    /// piecewise condition.
    Substitution(PiecewiseError),
}

/// Write `expression` with its identifiers' ids.
fn write_expression(f: &mut fmt::Formatter<'_>, expression: &Expression) -> fmt::Result {
    let options = FormatOptions::default().with_identifier_style(IdentifierStyle::NameHintWithId);
    write!(f, "{}", expression.display(options))
}

/// Write `data_type`, a template as `name::id`.
pub(super) fn write_data_type(f: &mut fmt::Formatter<'_>, data_type: &DataType) -> fmt::Result {
    match data_type {
        DataType::Template(template) => write!(f, "{:?}", template.identifier()),
        other => write!(f, "{other}"),
    }
}

impl fmt::Display for UnificationError {
    #[expect(
        clippy::too_many_lines,
        reason = "one arm per variant, each writing its one line"
    )]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch {
                operation: TypeOperation::Bind,
                expected,
                actual,
            } => write!(
                f,
                "cannot bind {expected} against {actual}: structural mismatch"
            ),
            Self::TypeMismatch {
                operation: TypeOperation::Unify,
                expected,
                actual,
            } => write!(
                f,
                "cannot unify {expected} with {actual}: structural mismatch"
            ),
            Self::DataTypeMismatch {
                operation,
                expected,
                actual,
            } => {
                match operation {
                    TypeOperation::Bind => f.write_str("cannot bind data type ")?,
                    TypeOperation::Unify => {
                        f.write_str("data type mismatch during unification: ")?;
                    }
                }
                write_data_type(f, expected)?;
                f.write_str(match operation {
                    TypeOperation::Bind => " against ",
                    TypeOperation::Unify => " vs ",
                })?;
                write_data_type(f, actual)?;
                if *operation == TypeOperation::Bind {
                    f.write_str(": structural mismatch")?;
                }
                Ok(())
            }
            Self::KindMismatch {
                operation: TypeOperation::Bind,
                expected,
                actual,
            } => write!(f, "cannot bind {expected} pattern against {actual}"),
            Self::KindMismatch {
                operation: TypeOperation::Unify,
                expected,
                actual,
            } => write!(f, "cannot unify {expected} with {actual}"),
            Self::CoreDataTypeMismatch { expected, actual } => {
                write!(f, "core data type mismatch: {expected} vs {actual}")
            }
            Self::RankMismatch {
                operation: TypeOperation::Bind,
                expected,
                actual,
            } => write!(
                f,
                "shape rank mismatch: pattern has {expected} dimensions, actual has {actual}"
            ),
            Self::RankMismatch {
                operation: TypeOperation::Unify,
                expected,
                actual,
            } => write!(
                f,
                "shape rank mismatch during unification: {expected} vs {actual}"
            ),
            Self::DimensionMismatch { expected, actual } => {
                f.write_str("shape dimension mismatch: ")?;
                write_expression(f, expected)?;
                f.write_str(" vs ")?;
                write_expression(f, actual)
            }
            Self::ConflictingExpressionBinding {
                identifier,
                bound,
                actual,
            } => {
                write!(f, "conflicting binding for shape variable {identifier:?}: ")?;
                write_expression(f, bound)?;
                f.write_str(" vs ")?;
                write_expression(f, actual)
            }
            Self::ConflictingDataTypeBinding {
                identifier,
                bound,
                actual,
            } => {
                write!(f, "conflicting data-type binding for {identifier:?}: ")?;
                write_data_type(f, bound)?;
                f.write_str(" vs ")?;
                write_data_type(f, actual)
            }
            Self::ConflictingTypeBinding {
                identifier,
                bound,
                actual,
            } => write!(
                f,
                "conflicting full-type binding for {identifier:?}: {bound} vs {actual}"
            ),
            Self::DistinctTemplates {
                operation,
                expected,
                actual,
            } => {
                let verb = match operation {
                    TypeOperation::Bind => "bind",
                    TypeOperation::Unify => "unify",
                };
                write!(
                    f,
                    "cannot {verb} distinct template data types: {:?} vs {:?}",
                    expected.identifier(),
                    actual.identifier()
                )
            }
            Self::WidthOnNonPrimitive { template, actual } => {
                write!(
                    f,
                    "cannot satisfy width constraint {} on {:?} with non-primitive actual ",
                    WidthList(template),
                    template.identifier()
                )?;
                write_data_type(f, actual)
            }
            Self::WidthMismatch { template, actual } => {
                write!(
                    f,
                    "width mismatch for template {:?}: actual {actual} ",
                    template.identifier()
                )?;
                match actual.bit_width() {
                    Some(width) => write!(f, "has width {width}")?,
                    None => f.write_str("has no width")?,
                }
                write!(f, ", not in {}", WidthList(template))
            }
            Self::OccursCheck {
                identifier,
                expression,
                substituted,
            } => {
                write!(
                    f,
                    "occurs check failed: identifier {identifier:?} appears in "
                )?;
                write_expression(f, expression)?;
                f.write_str(" after substitution through existing bindings (")?;
                write_expression(f, substituted)?;
                f.write_str(")")
            }
            Self::ExpressionMismatch { left, right } => {
                f.write_str("cannot unify expressions ")?;
                write_expression(f, left)?;
                f.write_str(" and ")?;
                write_expression(f, right)
            }
            Self::WildcardInActual => {
                f.write_str("wildcard `...` cannot appear in `actual` during template binding")
            }
            Self::WildcardInUnification => {
                f.write_str("wildcard `...` is not supported during unification")
            }
            Self::Extension(_) => f.write_str("a type defined outside the core failed"),
            Self::Substitution(_) => {
                f.write_str("substituting the existing shape bindings was refused")
            }
        }
    }
}

impl Error for UnificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Extension(source) => Some(source.as_ref()),
            Self::Substitution(source) => Some(source),
            _ => None,
        }
    }
}

/// The width list of a template, as `[8, 16]`.
struct WidthList<'a>(&'a TemplateDataType);

impl fmt::Display for WidthList<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (index, width) in self.0.widths().unwrap_or_default().iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{width}")?;
        }
        f.write_str("]")
    }
}
