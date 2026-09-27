//! Errors of type checking and of the body checks.

use std::error::Error;
use std::fmt;

use crate::expression::{Bounded, Expression, FormatOptions, FunctionSort, IdentifierStyle};
use crate::foreign::BoxError;

use super::super::core_data_type::CoreDataType;
use super::super::ty::Type;
use super::body::FunctionLabel;

/// The kind of rule a [`TypeRule`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TypeRuleKind {
    /// An identifier has no type and is no native constant.
    UnboundIdentifier,
    /// A type was supplied for a native constant's identifier.
    SuppliedConstantType,
    /// An identifier qualified `output` was read.
    OutputRead,
    /// A sub-expression resolves to a type that is neither a scalar
    /// numerical type of a primitive data type nor an index type.
    NotAValueType,
    /// An index type's stride is the literal `0`.
    ZeroStride,
    /// A literal was checked against an index type.
    LiteralAgainstIndex,
    /// A literal that the context cannot hold.
    Literal,
    /// An operation met a Boolean operand it is not defined for, or a
    /// Boolean position met another type.
    Boolean,
    /// An operation met an index type it is not defined for.
    Index,
    /// Two primitive data types without a common promotion.
    Promotion,
    /// A synthesized type does not fit the expected one.
    ExpectedType,
    /// A piecewise case condition that is not Boolean, or a branch that is
    /// no scalar numerical type.
    Piecewise,
    /// A call of a constant, a call with the wrong number of arguments, or
    /// an argument outside its parameter's sort.
    Call,
    /// A call of a function no target resolves.
    UnknownCall,
    /// A construct the checker has no rule for yet: a decimal literal, or a
    /// tensor operand.
    Unsupported,
}

/// A type rule an expression breaks, with the reason in words.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TypeRule {
    kind: TypeRuleKind,
    reason: String,
}

impl TypeRule {
    pub(super) fn new(kind: TypeRuleKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }

    /// Return the kind of rule.
    #[must_use]
    pub fn kind(&self) -> TypeRuleKind {
        self.kind
    }

    /// Return the reason, one lowercase line.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Return whether the rule is a construct the checker does not support
    /// yet, as opposed to a type error.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        self.kind == TypeRuleKind::Unsupported
    }
}

/// The failure of a call target lookup.
#[derive(Debug)]
#[non_exhaustive]
pub enum CallTargetError {
    /// No target is known under the name.
    Unknown {
        /// The name the call refers to.
        name: String,
        /// Why, in words.
        message: String,
        /// The lookup's own error, when it has one.
        source: Option<BoxError>,
    },
    /// The lookup failed.
    Callback(BoxError),
}

impl fmt::Display for CallTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { message, .. } => f.write_str(message),
            Self::Callback(_) => f.write_str("the call target lookup failed"),
        }
    }
}

impl Error for CallTargetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Unknown { source, .. } => source.as_deref().map(|source| source as _),
            Self::Callback(source) => Some(source.as_ref()),
        }
    }
}

/// The failure of type checking an expression.
///
/// A broken rule displays framed by the expression checked and, when
/// different, the sub-expression where it failed: ``type error while
/// inferring the type of `(x::7 + 1)` at sub-expression `x::7`: <reason>``,
/// with identifier ids.
#[derive(Debug)]
#[non_exhaustive]
pub enum TypeCheckError {
    /// The expression breaks a type rule.
    Rule {
        /// The expression checked.
        root: Expression,
        /// The sub-expression where the rule failed; the root itself when it
        /// failed there.
        at: Expression,
        /// The rule.
        rule: TypeRule,
    },
    /// A call's target is unknown and the checker defers such calls: the
    /// lookup's own error, unframed.
    UnknownCall(CallTargetError),
    /// An identifier or call target lookup failed.
    Callback(BoxError),
}

/// Write `expression` with its identifiers' ids, up to 64 node
/// occurrences and then `…`, so a message about a DAG stays short (R2-010).
pub(super) fn format_expression(expression: &Expression) -> String {
    Bounded::message(expression)
        .with_options(
            FormatOptions::default().with_identifier_style(IdentifierStyle::NameHintWithId),
        )
        .to_string()
}

impl fmt::Display for TypeCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rule { root, at, rule } => {
                write!(
                    f,
                    "type error while inferring the type of `{}`",
                    format_expression(root)
                )?;
                if !Expression::ptr_eq(root, at) {
                    write!(f, " at sub-expression `{}`", format_expression(at))?;
                }
                write!(f, ": {}", rule.reason)
            }
            Self::UnknownCall(error) => write!(f, "{error}"),
            Self::Callback(_) => f.write_str("a type-checking lookup failed"),
        }
    }
}

impl Error for TypeCheckError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rule { .. } => None,
            Self::UnknownCall(error) => error.source(),
            Self::Callback(source) => Some(source.as_ref()),
        }
    }
}

/// A function signature whose parts disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignatureError {
    /// The parameters and their sorts differ in number.
    LengthMismatch {
        /// The function.
        function: FunctionLabel,
        /// The number of parameters.
        parameters: usize,
        /// The number of parameter sorts.
        parameter_sorts: usize,
    },
}

impl fmt::Display for SignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                function,
                parameters,
                parameter_sorts,
            } => write!(
                f,
                "function '{function}' has {parameters} parameter(s) but {parameter_sorts} \
                 parameter sort(s)"
            ),
        }
    }
}

impl Error for SignatureError {}

/// A function body that does not satisfy its declared signature.
///
/// Displays one lowercase line naming the function, such as `function
/// 'scale' body synthesized type float64 is not compatible with the
/// declared result sort int`.
#[derive(Debug)]
#[non_exhaustive]
pub enum BodyCheckError {
    /// The body calls a function no target resolves.
    UnknownCall {
        /// The function whose body is checked.
        function: FunctionLabel,
        /// The lookup's error.
        error: CallTargetError,
    },
    /// The body uses a construct the checker does not support yet.
    Unsupported {
        /// The function whose body is checked.
        function: FunctionLabel,
        /// The checker's error.
        error: TypeCheckError,
    },
    /// The body breaks a type rule.
    IllTyped {
        /// The function whose body is checked.
        function: FunctionLabel,
        /// The checker's error.
        error: TypeCheckError,
    },
    /// The body's type is no scalar numerical type of a primitive data
    /// type.
    NotScalar {
        /// The function whose body is checked.
        function: FunctionLabel,
        /// The body's type.
        body_type: Type,
    },
    /// The body's core data type is outside the declared result sort.
    IncompatibleResult {
        /// The function whose body is checked.
        function: FunctionLabel,
        /// The body's core data type.
        body: CoreDataType,
        /// The declared result sort.
        sort: FunctionSort,
    },
    /// A call target lookup failed.
    Callback(BoxError),
    /// The function's signature is malformed, so its body was not checked.
    Signature(SignatureError),
}

impl fmt::Display for BodyCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCall { function, error } => write!(
                f,
                "function '{function}' body calls a function that is not registered: {error}; \
                 every call target must resolve by the time the body is held to its declared \
                 result sort"
            ),
            Self::Unsupported { function, error } => write!(
                f,
                "function '{function}' body uses a construct the body type checker does not \
                 support: {error}"
            ),
            Self::IllTyped { function, error } => {
                write!(
                    f,
                    "function '{function}' body failed to type-check: {error}"
                )
            }
            Self::NotScalar {
                function,
                body_type,
            } => write!(
                f,
                "function '{function}' body must synthesize a scalar numerical type, but got \
                 {body_type}"
            ),
            Self::IncompatibleResult {
                function,
                body,
                sort,
            } => write!(
                f,
                "function '{function}' body synthesized type {body} is not compatible with the \
                 declared result sort {sort}"
            ),
            Self::Callback(_) => f.write_str("a call target lookup failed"),
            Self::Signature(error) => write!(f, "{error}"),
        }
    }
}

impl Error for BodyCheckError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Callback(source) => Some(source.as_ref()),
            Self::Signature(error) => error.source(),
            // The inner errors are fields, and the text already holds them.
            _ => None,
        }
    }
}
