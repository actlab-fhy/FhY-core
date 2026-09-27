//! A simplifier that runs SymPy through an embedded or host Python
//! interpreter, behind the `sympy` feature.

mod boolean;
mod error;
mod lift;
mod load;
mod lower;
mod simplify;
mod substitute;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::hash::BuildHasher;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString};

use crate::expression::{BooleanScreen, Expression};
use crate::identifier::Identifier;

use super::backend::{Simplifier, SimplifyContext};
use super::screen::is_native_constant;
use crate::foreign::BoxError;

pub use error::{SympyError, SympyErrorKind, SympyPhase, SympyUnavailableError};

use load::Handles;

/// A [`Simplifier`] that lowers an expression to SymPy, simplifies it with
/// `sympy.simplify`, and lifts the result back.
///
/// It needs a Python interpreter with the `sympy` package at run time, and
/// never starts one on its own: inside a Python process, such as the
/// `fhy_core` extension module, it attaches to the running interpreter,
/// and a Rust program starts one first, with
/// [`with_embedded_python`](Self::with_embedded_python) or
/// `pyo3::Python::initialize`. Without an interpreter, or without SymPy,
/// every operation fails with [`SympyUnavailableError`], and
/// [`load`](Self::load) tells whether simplification can run.
///
/// Nothing is imported until the first operation, which imports SymPy and
/// loads the backend's prelude, a small Python module of the SymPy classes
/// and hooks only Python code can define, published once per interpreter
/// as `_fhy_core_sympy`. A failed load is retried by the next operation.
///
/// Each operation attaches to the interpreter once for its whole run, so
/// operations on several threads run one at a time, as all SymPy work does.
///
/// # Semantics
///
/// - **Lowering.** A Boolean literal becomes `true` or `false`, an integer
///   an `Integer`, a float the `Float` of its value, and a decimal the exact
///   `Rational` it denotes. An identifier becomes the symbol
///   `<name_hint>_<id>`, a built-in constant `pi`, `E`, `oo` or `nan`, and a
///   user constant of the context's registry the number of its value. The
///   operations are SymPy's, `//` is the `floor` of the quotient, and a
///   piecewise is a piecewise with a final `True` branch. The 19 native
///   built-ins lower to SymPy's functions; any other call is refused.
/// - **Simplification** is best-effort: where `sympy.simplify` raises
///   `PrecisionExhausted`, drops a piecewise's final branch, or cannot
///   compare a piecewise that must be split per branch, the expression is
///   kept as it stands.
/// - **Lifting** inverts the lowering. SymPy's n-ary sums and products fold
///   to the right, a rational becomes the decimal literal of its value when
///   a binary float equals it and the exact quotient otherwise, and `oo`,
///   `-oo` and `nan` become the built-in constants. Complex infinity and a
///   piecewise without a final `True` branch are refused.
///
/// # Examples
///
/// ```no_run
/// use std::collections::HashMap;
///
/// use fhy_core::expression::Expression;
/// use fhy_core::expression::registry::FunctionRegistry;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::solver::{SimplifyContext, Solver, SympySimplifier};
///
/// let simplifier = SympySimplifier::with_embedded_python();
/// simplifier.load()?;
/// let x = Identifier::new("x");
/// let registry = FunctionRegistry::new();
/// let solver = Solver::new().with_simplifier(simplifier);
///
/// let decided = solver.simplify(
///     &Expression::from(x.clone()).greater(0),
///     &HashMap::from([(x, Expression::literal(3))]),
///     &SimplifyContext::from_registry(&registry),
/// )?;
///
/// assert_eq!(decided, Expression::literal(true));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SympySimplifier {
    handles: PyOnceLock<Arc<Handles>>,
}

impl SympySimplifier {
    /// Return the backend, loading nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            handles: PyOnceLock::new(),
        }
    }

    /// Start an embedded Python interpreter if none is running, and return
    /// the backend.
    ///
    /// The interpreter is the one this crate was linked against (found
    /// through `PYO3_PYTHON`, or `python3` on `PATH`, at build time), and
    /// runs without signal handlers. It finds SymPy on its default path or
    /// through `PYTHONPATH`. Inside a running interpreter this starts
    /// nothing. The interpreter is never finalized.
    #[must_use]
    pub fn with_embedded_python() -> Self {
        Python::initialize();
        Self::new()
    }

    /// Import SymPy and load the backend's prelude now, if no operation
    /// has.
    ///
    /// # Errors
    ///
    /// Returns [`SympyUnavailableError`] when no interpreter is running or
    /// SymPy cannot be loaded.
    pub fn load(&self) -> Result<(), SympyUnavailableError> {
        Python::try_attach(|py| self.handles(py).map(drop))
            .unwrap_or(Err(SympyUnavailableError::NoInterpreter))
    }

    /// Return the loaded handles, loading them first if needed.
    fn handles(&self, py: Python<'_>) -> Result<&Arc<Handles>, SympyUnavailableError> {
        self.handles
            .get_or_try_init(py, || Handles::load(py).map(Arc::new))
    }

    /// Return the handles in `phase`, or the error that SymPy is
    /// unavailable.
    fn handles_in(&self, py: Python<'_>, phase: SympyPhase) -> Result<&Arc<Handles>, SympyError> {
        self.handles(py)
            .map_err(|error| SympyError::new(phase, SympyErrorKind::Unavailable(error)))
    }

    /// Return the SymPy object of `expression`.
    ///
    /// # Errors
    ///
    /// Returns a [`SympyError`] in [`SympyPhase::Lowering`]: `IllTyped` if a
    /// Boolean position of `expression` provably holds a number, as
    /// [`BooleanScreen::check_logical_operands`] judges it with the
    /// context's sorts; a call refusal for a call SymPy has no function
    /// for; `ConstantValueUnknown` for a user constant of a context
    /// without a registry; `PartialPiecewise` for a Boolean piecewise
    /// without an otherwise branch; and any exception SymPy raises.
    pub fn lower<'py>(
        &self,
        py: Python<'py>,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Bound<'py, PyAny>, SympyError> {
        let phase = SympyPhase::Lowering;
        let handles = self.handles_in(py, phase)?;
        BooleanScreen::new()
            .with_sorts(context.sorts())
            .check_logical_operands(expression)
            .map_err(|error| SympyError::new(phase, SympyErrorKind::IllTyped(error)))?;
        lower::Lowerer::new(handles, context)
            .lower(py, expression)
            .map_err(|kind| SympyError::new(phase, kind))
    }

    /// Return the expression the SymPy object `object` denotes.
    ///
    /// # Errors
    ///
    /// Returns a [`SympyError`] in [`SympyPhase::Lifting`] for an object no
    /// expression denotes, or of a kind the lifting does not know.
    pub fn lift(&self, object: &Bound<'_, PyAny>) -> Result<Expression, SympyError> {
        let phase = SympyPhase::Lifting;
        let handles = self.handles_in(object.py(), phase)?;
        lift::Lifter::new(handles)
            .lift(object)
            .map_err(|kind| SympyError::new(phase, kind))
    }

    /// Return the best-effort simplification of the SymPy object `object`,
    /// or `object` itself where SymPy gives up.
    ///
    /// # Errors
    ///
    /// Returns a [`SympyError`] in [`SympyPhase::Simplification`] for any
    /// other failure, such as an exception `sympy.simplify` raises.
    pub fn simplify_object<'py>(
        &self,
        object: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, SympyError> {
        let phase = SympyPhase::Simplification;
        let handles = self.handles_in(object.py(), phase)?;
        simplify::try_simplify(handles, object)
            .map(|simplified| simplified.unwrap_or_else(|| object.clone()))
            .map_err(|kind| SympyError::new(phase, kind))
    }

    /// Return the SymPy object `object` with `environment` substituted: the
    /// symbol of each identifier replaced by the lowering of its value,
    /// simultaneously, so a value is never itself substituted.
    ///
    /// # Errors
    ///
    /// Returns a [`SympyError`]: in [`SympyPhase::Substitution`],
    /// `BoundNativeConstant` if `environment` binds a native constant, by
    /// the context's sorts, whose symbol `object` holds free, and any
    /// exception a rebuilt node raises; and in [`SympyPhase::Lowering`] the
    /// errors of [`lower`](Self::lower) for a value.
    pub fn substitute<'py, S: BuildHasher>(
        &self,
        object: &Bound<'py, PyAny>,
        environment: &HashMap<Identifier, Expression, S>,
        context: &SimplifyContext<'_>,
    ) -> Result<Bound<'py, PyAny>, SympyError> {
        let py = object.py();
        let phase = SympyPhase::Substitution;
        let handles = self.handles_in(py, phase)?;
        let python = |error: PyErr| SympyError::new(phase, SympyErrorKind::Python(error));
        if object.is_instance_of::<pyo3::types::PyBool>() {
            return Ok(object.clone());
        }
        let referenced: Vec<String> = object
            .getattr("free_symbols")
            .and_then(|symbols| {
                symbols
                    .try_iter()?
                    .map(|symbol| symbol?.getattr("name")?.extract::<String>())
                    .collect()
            })
            .map_err(python)?;
        let mut bound_constants: Vec<Identifier> = environment
            .keys()
            .filter(|identifier| {
                referenced.contains(&lower::symbol_name(identifier))
                    && is_native_constant(identifier, context.sorts())
            })
            .cloned()
            .collect();
        if !bound_constants.is_empty() {
            bound_constants.sort_by_key(Identifier::id);
            return Err(SympyError::new(
                phase,
                SympyErrorKind::BoundNativeConstant(bound_constants),
            ));
        }
        let replacements = PyDict::new(py);
        for (identifier, value) in environment {
            let symbol = handles
                .symbol
                .bind(py)
                .call1((PyString::new(py, &lower::symbol_name(identifier)),))
                .map_err(python)?;
            let lowered = self.lower(py, value, context)?;
            replacements.set_item(symbol, lowered).map_err(python)?;
        }
        substitute::substitute_symbols(handles, object, replacements.as_any())
            .map_err(|kind| SympyError::new(phase, kind))
    }

    /// Return the SymPy object `object` with `replacements`, a mapping from
    /// SymPy symbols to SymPy objects, applied, simultaneously, keeping
    /// every Boolean position Boolean.
    ///
    /// # Errors
    ///
    /// Returns a [`SympyError`] in [`SympyPhase::Substitution`] for an
    /// exception a rebuilt node raises, such as SymPy refusing a
    /// comparison with complex infinity.
    pub fn substitute_symbols<'py>(
        &self,
        object: &Bound<'py, PyAny>,
        replacements: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, SympyError> {
        let phase = SympyPhase::Substitution;
        let handles = self.handles_in(object.py(), phase)?;
        substitute::substitute_symbols(handles, object, replacements)
            .map_err(|kind| SympyError::new(phase, kind))
    }

    /// Lower, simplify and lift `expression`, attached to the interpreter.
    fn simplify_attached(
        &self,
        py: Python<'_>,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Expression, SympyError> {
        let lowered = self.lower(py, expression, context)?;
        let simplified = self.simplify_object(&lowered)?;
        self.lift(&simplified)
    }
}

impl Default for SympySimplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SympySimplifier {
    /// Write the type name only: the handles are Python objects.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SympySimplifier").finish_non_exhaustive()
    }
}

impl Simplifier for SympySimplifier {
    /// Return `sympy`.
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("sympy")
    }

    /// Lower `expression`, simplify it best-effort, and lift the result,
    /// attached to the interpreter once.
    ///
    /// Fails with a [`SympyError`].
    fn simplify(
        &self,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Expression, BoxError> {
        Python::try_attach(|py| self.simplify_attached(py, expression, context))
            .unwrap_or_else(|| {
                Err(SympyError::new(
                    SympyPhase::Lowering,
                    SympyErrorKind::Unavailable(SympyUnavailableError::NoInterpreter),
                ))
            })
            .map_err(|error| Box::new(error) as BoxError)
    }
}
