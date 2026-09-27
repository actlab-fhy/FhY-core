//! The errors of the SymPy backend.

use std::error::Error;
use std::fmt;

use pyo3::PyErr;

use fhy_core::expression::{Callee, FunctionName, NonBooleanLogicalOperandError};
use fhy_core::identifier::Identifier;

/// Why the SymPy backend cannot run.
#[derive(Debug)]
pub(crate) enum SympyUnavailableError {
    /// Importing SymPy failed; the error is Python's.
    MissingSympy(PyErr),
    /// SymPy imported, but loading what the backend reads from it, or its
    /// prelude, failed, as with an incompatible SymPy version.
    Incompatible(PyErr),
}

impl fmt::Display for SympyUnavailableError {
    /// Write one lowercase line naming what is missing.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSympy(_) => f.write_str("the sympy package cannot be imported"),
            Self::Incompatible(_) => {
                f.write_str("the sympy package does not provide what the backend needs")
            }
        }
    }
}

impl Error for SympyUnavailableError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MissingSympy(error) | Self::Incompatible(error) => Some(error),
        }
    }
}

/// The phase of the backend's work an error arose in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SympyPhase {
    /// Lowering an expression to SymPy.
    Lowering,
    /// Simplifying a SymPy object.
    Simplification,
    /// Substituting into a SymPy object.
    Substitution,
    /// Lifting a SymPy object to an expression.
    Lifting,
}

impl SympyPhase {
    /// Return the phase's name: `"lowering"`, `"simplification"`,
    /// `"substitution"` or `"lifting"`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Lowering => "lowering",
            Self::Simplification => "simplification",
            Self::Substitution => "substitution",
            Self::Lifting => "lifting",
        }
    }
}

impl fmt::Display for SympyPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What went wrong in a [`SympyError`].
#[derive(Debug)]
pub(crate) enum SympyErrorKind {
    /// The backend cannot run.
    Unavailable(SympyUnavailableError),
    /// A Boolean position of the expression to lower provably holds a
    /// number.
    IllTyped(NonBooleanLogicalOperandError),
    /// A call of a function defined by an expression, which inlining
    /// expands first.
    CallNeedsInlining(Callee),
    /// A call of a registered native constant, which is not callable.
    ConstantCalled(FunctionName),
    /// A call of a registered native function, which SymPy has no
    /// function for.
    NoSympyLowering(FunctionName),
    /// A call of a name neither the catalogue nor the context's registry
    /// knows.
    UnknownFunction(FunctionName),
    /// A reference to a user constant whose value the context does not
    /// know, since it holds no registry.
    ConstantValueUnknown(Identifier),
    /// A built-in constant SymPy has no value for; the catalogue is
    /// `#[non_exhaustive]`, and every constant it holds today has one.
    UnsupportedConstant(Identifier),
    /// A substitution binds native constants the object refers to.
    BoundNativeConstant(Vec<Identifier>),
    /// SymPy's complex infinity, which no expression denotes.
    ComplexInfinity,
    /// A piecewise with no final `True` branch, which no expression
    /// denotes; the text is the SymPy node's `repr`.
    PartialPiecewise(String),
    /// A value that is not a SymPy object; the text is its type.
    UnsupportedNode(String),
    /// A SymPy expression of a kind the lifting does not know; the text is
    /// its type.
    UnsupportedExpression(String),
    /// A SymPy Boolean of a kind the lifting does not know; the text is its
    /// type.
    UnsupportedBoolean(String),
    /// A SymPy relational of a kind the lifting does not know; the text is
    /// its type.
    UnsupportedRelational(String),
    /// A SymPy node with another number of arguments than its kind has;
    /// the text names the kind and the count it needs.
    Arity(String),
    /// A SymPy `Implies`, which the lifting does not support; the text is
    /// its `repr`.
    Implies(String),
    /// A symbol whose name is not `<name hint>_<id>`; the text is the name.
    UnreadableSymbol(String),
    /// An exception Python raised.
    Python(PyErr),
}

/// A failure of the SymPy backend: what went wrong, and in which phase.
#[derive(Debug)]
pub(crate) struct SympyError {
    phase: SympyPhase,
    kind: SympyErrorKind,
}

impl SympyError {
    /// Return the error of `kind` in `phase`.
    pub(crate) fn new(phase: SympyPhase, kind: SympyErrorKind) -> Self {
        Self { phase, kind }
    }

    /// Return the phase the error arose in.
    pub(crate) fn phase(&self) -> SympyPhase {
        self.phase
    }

    /// Return what went wrong.
    pub(crate) fn kind(&self) -> &SympyErrorKind {
        &self.kind
    }

    /// Return what went wrong, consuming the error.
    pub(crate) fn into_kind(self) -> SympyErrorKind {
        self.kind
    }
}

impl fmt::Display for SympyError {
    /// Write one lowercase line naming the node or name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            SympyErrorKind::Unavailable(_) => f.write_str("the sympy backend cannot run"),
            SympyErrorKind::IllTyped(_) => {
                f.write_str("the expression cannot be lowered to sympy: it is ill-typed")
            }
            SympyErrorKind::CallNeedsInlining(callee) => write!(
                f,
                "cannot lower a call of the expression-bodied function {:?} to sympy; \
                 call inline_functions first to expand {:?}",
                callee.name(),
                callee.name()
            ),
            SympyErrorKind::ConstantCalled(name) => write!(
                f,
                "call to {:?} is to a registered native constant, which is not callable",
                name.as_str()
            ),
            SympyErrorKind::NoSympyLowering(name) => write!(
                f,
                "native function {:?} has no sympy lowering",
                name.as_str()
            ),
            SympyErrorKind::UnknownFunction(name) => write!(
                f,
                "cannot lower call to unknown function {:?} to sympy",
                name.as_str()
            ),
            SympyErrorKind::ConstantValueUnknown(identifier) => write!(
                f,
                "the value of the native constant {identifier:?} is unknown without a registry"
            ),
            SympyErrorKind::UnsupportedConstant(identifier) => {
                write!(f, "the built-in constant {identifier:?} has no sympy value")
            }
            SympyErrorKind::BoundNativeConstant(identifiers) => {
                f.write_str(
                    "cannot bind the native constants the expression refers to, \
                     whose values are fixed by the registry: ",
                )?;
                write_identifiers(f, identifiers)
            }
            SympyErrorKind::ComplexInfinity => f.write_str(
                "cannot lift zoo to an expression: sympy folds a quotient by zero to its \
                 directionless complex infinity, and no expression denotes that value",
            ),
            SympyErrorKind::PartialPiecewise(node) => write!(
                f,
                "cannot represent the partial sympy.Piecewise {node} as an expression: \
                 its final branch's condition is not sympy.true, so it has no value where \
                 every condition fails"
            ),
            SympyErrorKind::UnsupportedNode(kind) => write!(f, "unsupported node type: {kind}"),
            SympyErrorKind::UnsupportedExpression(kind) => {
                write!(f, "unsupported expression type: {kind}")
            }
            SympyErrorKind::UnsupportedBoolean(kind) => {
                write!(f, "unsupported boolean expression type: {kind}")
            }
            SympyErrorKind::UnsupportedRelational(kind) => {
                write!(f, "unsupported relational type: {kind}")
            }
            SympyErrorKind::Arity(expected) => write!(f, "expected {expected}"),
            SympyErrorKind::Implies(node) => write!(f, "implies is not supported: {node}"),
            SympyErrorKind::UnreadableSymbol(name) => write!(
                f,
                "cannot read an identifier from the symbol {name:?}: the lowering names \
                 identifiers <name_hint>_<id>"
            ),
            SympyErrorKind::Python(_) => {
                write!(f, "python raised an exception during {}", self.phase)
            }
        }
    }
}

/// Write `identifiers` as `x::7, y::8`.
fn write_identifiers(f: &mut fmt::Formatter<'_>, identifiers: &[Identifier]) -> fmt::Result {
    for (index, identifier) in identifiers.iter().enumerate() {
        if index > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{identifier:?}")?;
    }
    Ok(())
}

impl Error for SympyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            SympyErrorKind::Unavailable(error) => Some(error),
            SympyErrorKind::IllTyped(error) => Some(error),
            SympyErrorKind::Python(error) => Some(error),
            _ => None,
        }
    }
}
