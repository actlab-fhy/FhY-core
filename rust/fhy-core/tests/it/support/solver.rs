//! Fake solver backends that record what they are asked, for the solver
//! tests.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use fhy_core::expression::{Expression, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    BackendError, CheckLimits, SatResult, Simplifier, SimplifyContext, SmtScript, SmtSolver, Solver,
};

/// An error a fake backend reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FakeBackendError(pub(crate) String);

impl fmt::Display for FakeBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FakeBackendError {}

/// An [`SmtSolver`] answering every check the same way and recording each
/// script's text and limits.
#[derive(Debug)]
pub(crate) struct RecordingSmtSolver {
    answer: Result<SatResult, FakeBackendError>,
    checks: Mutex<Vec<(String, CheckLimits)>>,
}

impl RecordingSmtSolver {
    /// Return the backend answering every check `answer`.
    pub(crate) fn answering(answer: SatResult) -> Arc<Self> {
        Arc::new(Self {
            answer: Ok(answer),
            checks: Mutex::new(Vec::new()),
        })
    }

    /// Return the backend failing every check with `message`.
    pub(crate) fn failing(message: &str) -> Arc<Self> {
        Arc::new(Self {
            answer: Err(FakeBackendError(message.to_owned())),
            checks: Mutex::new(Vec::new()),
        })
    }

    /// Return the text and limits of every check so far, in order.
    pub(crate) fn checks(&self) -> Vec<(String, CheckLimits)> {
        self.checks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Return the text of the one check so far.
    ///
    /// # Panics
    ///
    /// Panics unless exactly one check was made.
    pub(crate) fn only_script(&self) -> String {
        let checks = self.checks();
        assert_eq!(checks.len(), 1, "expected one check, got {checks:?}");
        checks[0].0.clone()
    }

    /// Return a solver holding this backend.
    pub(crate) fn solver(self: &Arc<Self>) -> Solver {
        Solver::new().with_shared_smt_solver(Arc::clone(self) as Arc<dyn SmtSolver>)
    }
}

impl SmtSolver for RecordingSmtSolver {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("recording")
    }

    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BackendError> {
        self.checks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((script.to_string(), *limits));
        self.answer
            .clone()
            .map_err(|error| Box::new(error) as BackendError)
    }
}

/// A [`Simplifier`] recording each input and returning a fixed result, or
/// the input itself.
#[derive(Debug)]
pub(crate) struct RecordingSimplifier {
    result: Option<Result<Expression, FakeBackendError>>,
    inputs: Mutex<Vec<Expression>>,
}

impl RecordingSimplifier {
    /// Return the simplifier returning each input itself.
    pub(crate) fn identity() -> Arc<Self> {
        Arc::new(Self {
            result: None,
            inputs: Mutex::new(Vec::new()),
        })
    }

    /// Return the simplifier returning `result` for every input.
    pub(crate) fn returning(result: Expression) -> Arc<Self> {
        Arc::new(Self {
            result: Some(Ok(result)),
            inputs: Mutex::new(Vec::new()),
        })
    }

    /// Return the simplifier failing with `message`.
    pub(crate) fn failing(message: &str) -> Arc<Self> {
        Arc::new(Self {
            result: Some(Err(FakeBackendError(message.to_owned()))),
            inputs: Mutex::new(Vec::new()),
        })
    }

    /// Return every input so far, in order.
    pub(crate) fn inputs(&self) -> Vec<Expression> {
        self.inputs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Return a solver holding this simplifier.
    pub(crate) fn solver(self: &Arc<Self>) -> Solver {
        Solver::new().with_shared_simplifier(Arc::clone(self) as Arc<dyn Simplifier>)
    }
}

impl Simplifier for RecordingSimplifier {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("recording")
    }

    fn simplify(
        &self,
        expression: &Expression,
        _context: &SimplifyContext<'_>,
    ) -> Result<Expression, BackendError> {
        self.inputs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(expression.clone());
        match &self.result {
            None => Ok(expression.clone()),
            Some(result) => result
                .clone()
                .map_err(|error| Box::new(error) as BackendError),
        }
    }
}

/// Return the symbol types declaring each identifier of `entries` its sort.
pub(crate) fn build_symbol_types(
    entries: &[(&Identifier, SymbolType)],
) -> HashMap<Identifier, SymbolType> {
    entries
        .iter()
        .map(|(identifier, sort)| ((*identifier).clone(), *sort))
        .collect()
}

/// Return the quoted SMT-LIB2 symbol of `identifier`: `|<name>_<id>|`.
pub(crate) fn quoted_symbol(identifier: &Identifier) -> String {
    format!("|{}_{}|", identifier.name_hint(), identifier.id())
}
