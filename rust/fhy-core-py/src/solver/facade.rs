//! `fhy_core._rs.Solver`: the core's facade over the backends a Python
//! caller gives it (P2; D-S8-13), with the methods the module functions of
//! `fhy_core.symbolic.solver` call.
//!
//! Every question reads the sorts of user constants and named functions
//! from a snapshot of the function registry (S7), and runs with the
//! interpreter detached, so a native backend lets other threads run; a
//! Python backend attaches again for its one call.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyMapping;

use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    Answer, CheckLimits, QueryContext, QueryKind, Question, SimplifyContext, SimplifyLimits,
    SolveError, Solver, UnknownReason,
};

use crate::expression::{PyExpression, materialize_substituted, registry_snapshot};
use fhy_core::tree::NodeHandle;

use crate::identifier::{read_identifier_id, restore_identifier};

use super::backends::{build_simplifier, build_smt_solver, run_simplification, type_name};
use super::error::{solve_error_to_py, undecidable_error, warn_hazard, warn_unknown};
use super::values::{read_query_kind, read_symbol_types};

/// The fixed `UndecidableError.reason` of a question the hazard screen
/// refused.
const HAZARD_SCREEN_REASON: &str = "hazard_screen";

/// Return `expression` as a Rust expression, or raise the `TypeError`
/// naming `owner` and `field`.
fn read_expression(value: &Bound<'_, PyAny>, owner: &str, field: &str) -> PyResult<Expression> {
    value
        .cast::<PyExpression>()
        .map(|expression| expression.get().expression().clone())
        .map_err(|_not_an_expression| {
            PyTypeError::new_err(format!(
                "{owner} {field} must be an Expression, got {}.",
                type_name(value)
            ))
        })
}

/// Check `timeout_milliseconds` with `validate_timeout_milliseconds` of
/// `fhy_core.symbolic.solver`, and return the limits it gives.
///
/// # Errors
///
/// Raises `ValueError` unless it is `None` or a positive integer, below
/// `2**64` milliseconds.
pub(crate) fn read_limits(timeout_milliseconds: &Bound<'_, PyAny>) -> PyResult<CheckLimits> {
    static VALIDATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = timeout_milliseconds.py();
    VALIDATE
        .import(
            py,
            "fhy_core.symbolic.solver",
            "validate_timeout_milliseconds",
        )?
        .call1((timeout_milliseconds,))?;
    if timeout_milliseconds.is_none() {
        return Ok(CheckLimits::new());
    }
    let milliseconds: u64 = timeout_milliseconds.extract().map_err(|_too_large| {
        PyValueError::new_err(format!(
            "timeout_milliseconds must be None or a positive integer, but got {}.",
            timeout_milliseconds
                .repr()
                .map_or_else(|_| "?".to_owned(), |text| text.to_string())
        ))
    })?;
    Ok(CheckLimits::new().with_timeout(Duration::from_millis(milliseconds)))
}

/// Return the identifiers of the iterable `considered`.
///
/// # Errors
///
/// Raises `TypeError` for an item that is not an `Identifier`.
fn read_considered(considered: &Bound<'_, PyAny>) -> PyResult<HashSet<Identifier>> {
    let mut read = HashSet::new();
    for item in considered.try_iter()? {
        read.insert(restore_identifier(
            &item?,
            "holds_for_all_free_assignments",
            "considered identifier",
        )?);
    }
    Ok(read)
}

/// The owned parts of a logical question, which a detached call can hold.
enum OwnedQuestion {
    Satisfiability(Expression),
    Implication(Expression, Expression),
    UniversalValidity(HashSet<Identifier>, Expression),
}

impl OwnedQuestion {
    fn kind(&self) -> QueryKind {
        match self {
            Self::Satisfiability(_) => QueryKind::Satisfiability,
            Self::Implication(..) => QueryKind::Implication,
            Self::UniversalValidity(..) => QueryKind::UniversalValidity,
        }
    }

    fn question(&self) -> Question<'_> {
        match self {
            Self::Satisfiability(expression) => Question::Satisfiability(expression),
            Self::Implication(antecedent, consequent) => Question::Implication {
                antecedent,
                consequent,
            },
            Self::UniversalValidity(considered, expression) => Question::UniversalValidity {
                considered,
                expression,
            },
        }
    }
}

/// Answers questions about expressions with the backends it holds: an
/// `SmtSolver` for satisfiability, implication and universal validity, and
/// a `Simplifier` for simplification.
///
/// The methods are the functions of `fhy_core.symbolic.solver`, without
/// their `backend`; each checks and screens its question in Rust and calls
/// its backend once. A solver keeps no state between questions.
#[pyclass(frozen, module = "fhy_core._rs", name = "Solver")]
pub(crate) struct PySolver {
    solver: Solver,
    smt_solver: Option<Py<PyAny>>,
    simplifier: Option<Py<PyAny>>,
}

impl PySolver {
    /// Return the core solver.
    pub(crate) fn core(&self) -> &Solver {
        &self.solver
    }

    /// Answer `question` for the entry point `context`, logging a refusal or
    /// a backend's `unknown`.
    fn ask(
        &self,
        py: Python<'_>,
        context: &str,
        question: &OwnedQuestion,
        symbol_types: &Bound<'_, PyAny>,
        timeout_milliseconds: &Bound<'_, PyAny>,
    ) -> PyResult<Answer> {
        if !self.solver.can_answer(question.kind()) {
            return Err(solve_error_to_py(
                py,
                SolveError::NoCapableBackend(question.kind()),
            ));
        }
        let limits = read_limits(timeout_milliseconds)?;
        let symbol_types = read_symbol_types(Some(symbol_types))?;
        let registry = registry_snapshot();
        let solver = &self.solver;
        let answer = py
            .detach(|| {
                let context = QueryContext::new(&symbol_types)
                    .with_sorts(registry.registry())
                    .with_limits(limits);
                solver.ask(&question.question(), &context)
            })
            .map_err(|error| solve_error_to_py(py, error))?;
        match &answer {
            Answer::Unknown(UnknownReason::Refused(hazard)) => {
                warn_hazard(py, context, hazard, &symbol_types)?;
            }
            Answer::Unknown(UnknownReason::GaveUp { reason }) => {
                warn_unknown(py, context, &self.backend_name(), reason)?;
            }
            Answer::Yes | Answer::No | Answer::Unknown(_) => {}
        }
        Ok(answer)
    }

    /// Return the name of the SMT backend.
    pub(crate) fn backend_name(&self) -> String {
        self.solver
            .smt_solver()
            .map_or_else(String::new, |backend| backend.name().into_owned())
    }

    /// Return the decided answer of the strict entry point `context`, or
    /// raise `UndecidableError` naming why it is undecided.
    fn decide(
        &self,
        py: Python<'_>,
        context: &str,
        subject: &str,
        answer: Answer,
    ) -> PyResult<bool> {
        match answer {
            Answer::Yes => Ok(true),
            Answer::No => Ok(false),
            Answer::Unknown(UnknownReason::GaveUp { reason }) => Err(undecidable_error(
                py,
                format!(
                    "{context}: the backend {} returned `unknown` ({reason}); the {subject} is \
                     undecidable with the current solver configuration.",
                    self.backend_name()
                ),
                &reason,
            )),
            Answer::Unknown(_) => Err(undecidable_error(
                py,
                format!(
                    "{context}: the expression was refused by the solver seam's hazard screen \
                     before the solver was consulted; see the WARNING logged just above for the \
                     offending node. The {subject} is undecidable with the current solver \
                     configuration."
                ),
                HAZARD_SCREEN_REASON,
            )),
        }
    }
}

#[pymethods]
impl PySolver {
    /// Create the solver answering the logical questions with
    /// `smt_solver`, an `SmtSolver`, and simplification with `simplifier`,
    /// a `Simplifier`; either may be `None`.
    ///
    /// Raises `TypeError` for a backend of another type.
    #[new]
    #[pyo3(signature = (smt_solver = None, simplifier = None))]
    fn new(
        smt_solver: Option<&Bound<'_, PyAny>>,
        simplifier: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let smt_solver = smt_solver.filter(|backend| !backend.is_none());
        let simplifier = simplifier.filter(|backend| !backend.is_none());
        let mut solver = Solver::new();
        if let Some(backend) = smt_solver {
            solver = solver.with_shared_smt_solver(build_smt_solver(backend)?);
        }
        if let Some(backend) = simplifier {
            solver = solver.with_shared_simplifier(build_simplifier(backend)?);
        }
        Ok(Self {
            solver,
            smt_solver: smt_solver.map(|backend| backend.clone().unbind()),
            simplifier: simplifier.map(|backend| backend.clone().unbind()),
        })
    }

    /// The backend answering the logical questions, or `None`.
    #[getter]
    fn smt_solver(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.smt_solver
            .as_ref()
            .map(|backend| backend.clone_ref(py))
    }

    /// The backend answering simplification, or `None`.
    #[getter]
    fn simplifier(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.simplifier
            .as_ref()
            .map(|backend| backend.clone_ref(py))
    }

    /// Return whether the solver holds a backend answering `kind`, a
    /// `SolverQueryKind`.
    ///
    /// Raises `ValueError` for another value.
    fn can_answer(&self, kind: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.solver.can_answer(read_query_kind(kind)?))
    }

    /// Simplify `expression`, after substituting `environment`, a mapping
    /// of identifiers to expressions, into it.
    ///
    /// The simplifier receives the substituted expression, which is
    /// `expression` itself when `environment` binds none of its
    /// identifiers, and the object it returns is returned. It reads
    /// `timeout_milliseconds` from its `context`, and honors it if it can.
    ///
    /// Raises `SolverCapabilityError` without a simplifier, `TypeError` for
    /// a value of `environment` that is not an `Expression`, `ValueError`
    /// for a bad `timeout_milliseconds`, `NonBooleanLogicalOperandError`
    /// for a number in a Boolean position, counting an identifier bound to
    /// one, `NativeConstantBindingError` when `environment` binds a native
    /// constant `expression` refers to, and whatever the simplifier raises.
    #[pyo3(signature = (expression, environment = None, *, timeout_milliseconds = None))]
    fn simplify_expression<'py>(
        &self,
        expression: &Bound<'py, PyAny>,
        environment: Option<&Bound<'py, PyAny>>,
        timeout_milliseconds: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = expression.py();
        if !self.solver.can_answer(QueryKind::Simplification) {
            return Err(solve_error_to_py(
                py,
                SolveError::NoCapableBackend(QueryKind::Simplification),
            ));
        }
        let input = expression
            .cast::<PyExpression>()
            .map_err(|_not_an_expression| {
                PyTypeError::new_err(format!(
                    "simplify_expression expression must be an Expression, got {}.",
                    type_name(expression)
                ))
            })?
            .clone();
        let limits = match timeout_milliseconds {
            Some(timeout_milliseconds) => read_limits(timeout_milliseconds)?.timeout(),
            None => None,
        }
        .map_or_else(SimplifyLimits::new, |timeout| {
            SimplifyLimits::new().with_timeout(timeout)
        });
        let mut bindings: HashMap<Identifier, Expression> = HashMap::new();
        let mut known: Vec<(Expression, Py<PyAny>)> = Vec::new();
        if let Some(environment) = environment.filter(|environment| !environment.is_none()) {
            for item in environment.cast::<PyMapping>()?.items()?.iter() {
                let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
                if read_identifier_id(&key)?.is_none() {
                    continue;
                }
                let bound = value.cast::<PyExpression>().map_err(|_not_an_expression| {
                    PyTypeError::new_err(format!(
                        "environment values must be Expressions, got {} for {}.",
                        type_name(&value),
                        key.repr()
                            .map_or_else(|_| "?".to_owned(), |text| text.to_string())
                    ))
                })?;
                let handle = bound.get().expression().clone();
                known.push((handle.clone(), value.clone().unbind()));
                bindings.insert(restore_identifier(&key, "environment", "key")?, handle);
            }
        }
        let rust_input = input.get().expression().clone();
        let registry = registry_snapshot();
        let solver = &self.solver;
        let known_objects: HashMap<_, _> = known
            .iter()
            .map(|(handle, object)| (handle.identity(), object.bind(py).clone()))
            .collect();
        let (result, returned) = run_simplification(input.clone().unbind(), known, || {
            py.detach(|| {
                solver.simplify(
                    &rust_input,
                    &bindings,
                    &SimplifyContext::from_registry(registry.registry()).with_limits(limits),
                )
            })
        });
        let result = result.map_err(|error| solve_error_to_py(py, error))?;
        if let Some(returned) = returned {
            let returned = returned.into_bound(py);
            if let Ok(object) = returned.cast::<PyExpression>() {
                if Expression::ptr_eq(object.get().expression(), &result) {
                    return Ok(returned);
                }
            }
        }
        materialize_substituted(&input, &result, known_objects)
    }

    /// Return whether some assignment of `expression`'s identifiers
    /// satisfies it: `True`, `False`, or `None` when the hazard screen
    /// refuses it or the backend answers `unknown`.
    ///
    /// Raises `SolverCapabilityError` without an SMT backend, `ValueError`
    /// for a bad `timeout_milliseconds`, `KeyError` for identifiers
    /// `symbol_types` misses, and `NonBooleanLogicalOperandError` for an
    /// expression that cannot be a predicate.
    #[pyo3(signature = (expression, symbol_types, *, timeout_milliseconds = None))]
    fn check_expression_satisfiability(
        &self,
        expression: &Bound<'_, PyAny>,
        symbol_types: &Bound<'_, PyAny>,
        timeout_milliseconds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let py = expression.py();
        let question = OwnedQuestion::Satisfiability(read_expression(
            expression,
            "check_expression_satisfiability",
            "expression",
        )?);
        let none = py.None().into_bound(py);
        self.ask(
            py,
            "check_expression_satisfiability",
            &question,
            symbol_types,
            timeout_milliseconds.unwrap_or(&none),
        )
        .map(|answer| answer.decided())
    }

    /// Return whether `antecedent` implies `consequent`: `True`, `False`,
    /// or `None` when the hazard screen refuses either or the backend
    /// answers `unknown`.
    ///
    /// Raises as `check_expression_satisfiability` does, for either side.
    #[pyo3(signature = (antecedent, consequent, symbol_types, *, timeout_milliseconds = None))]
    fn does_expression_imply(
        &self,
        antecedent: &Bound<'_, PyAny>,
        consequent: &Bound<'_, PyAny>,
        symbol_types: &Bound<'_, PyAny>,
        timeout_milliseconds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let py = antecedent.py();
        let question = OwnedQuestion::Implication(
            read_expression(antecedent, "does_expression_imply", "antecedent")?,
            read_expression(consequent, "does_expression_imply", "consequent")?,
        );
        let none = py.None().into_bound(py);
        self.ask(
            py,
            "does_expression_imply",
            &question,
            symbol_types,
            timeout_milliseconds.unwrap_or(&none),
        )
        .map(|answer| answer.decided())
    }

    /// Return whether, for every assignment of `expression`'s identifiers
    /// outside `considered_identifiers`, some assignment of the considered
    /// ones satisfies it: `True`, `False`, or `None`.
    ///
    /// Raises as `check_expression_satisfiability` does; a considered
    /// identifier needs a symbol type too.
    #[pyo3(signature = (considered_identifiers, expression, symbol_types, *, timeout_milliseconds = None))]
    fn holds_for_all_free_assignments(
        &self,
        considered_identifiers: &Bound<'_, PyAny>,
        expression: &Bound<'_, PyAny>,
        symbol_types: &Bound<'_, PyAny>,
        timeout_milliseconds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let py = expression.py();
        let question = OwnedQuestion::UniversalValidity(
            read_considered(considered_identifiers)?,
            read_expression(expression, "holds_for_all_free_assignments", "expression")?,
        );
        let none = py.None().into_bound(py);
        self.ask(
            py,
            "holds_for_all_free_assignments",
            &question,
            symbol_types,
            timeout_milliseconds.unwrap_or(&none),
        )
        .map(|answer| answer.decided())
    }

    /// Return `holds_for_all_free_assignments`'s decided answer, raising
    /// `UndecidableError` instead of returning `None`; its `reason` is the
    /// backend's, or `"hazard_screen"`.
    #[pyo3(signature = (considered_identifiers, expression, symbol_types, *, timeout_milliseconds = None))]
    fn assert_holds_for_all_free_assignments(
        &self,
        considered_identifiers: &Bound<'_, PyAny>,
        expression: &Bound<'_, PyAny>,
        symbol_types: &Bound<'_, PyAny>,
        timeout_milliseconds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = expression.py();
        let context = "assert_holds_for_all_free_assignments";
        let question = OwnedQuestion::UniversalValidity(
            read_considered(considered_identifiers)?,
            read_expression(expression, context, "expression")?,
        );
        let none = py.None().into_bound(py);
        let answer = self.ask(
            py,
            context,
            &question,
            symbol_types,
            timeout_milliseconds.unwrap_or(&none),
        )?;
        self.decide(py, context, "property", answer)
    }

    /// Return `does_expression_imply`'s decided answer, raising
    /// `UndecidableError` instead of returning `None`; its `reason` is the
    /// backend's, or `"hazard_screen"`.
    #[pyo3(signature = (antecedent, consequent, symbol_types, *, timeout_milliseconds = None))]
    fn assert_expression_implies(
        &self,
        antecedent: &Bound<'_, PyAny>,
        consequent: &Bound<'_, PyAny>,
        symbol_types: &Bound<'_, PyAny>,
        timeout_milliseconds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = antecedent.py();
        let context = "assert_expression_implies";
        let question = OwnedQuestion::Implication(
            read_expression(antecedent, context, "antecedent")?,
            read_expression(consequent, context, "consequent")?,
        );
        let none = py.None().into_bound(py);
        let answer = self.ask(
            py,
            context,
            &question,
            symbol_types,
            timeout_milliseconds.unwrap_or(&none),
        )?;
        self.decide(py, context, "implication", answer)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let render = |backend: &Option<Py<PyAny>>| -> PyResult<String> {
            match backend {
                Some(backend) => Ok(backend.bind(py).repr()?.to_string()),
                None => Ok("None".to_owned()),
            }
        };
        Ok(format!(
            "Solver(smt_solver={}, simplifier={})",
            render(&self.smt_solver)?,
            render(&self.simplifier)?
        ))
    }
}
