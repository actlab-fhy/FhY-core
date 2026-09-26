//! The backend traits a [`Solver`](super::Solver) delegates to, and the
//! values they exchange with it.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::time::Duration;

use crate::expression::registry::FunctionRegistry;
use crate::expression::{Expression, NoRegisteredSorts, SortLookup};

use super::smt::SmtScript;

/// The error a backend reports: any error, boxed.
///
/// The [`Solver`](super::Solver) wraps it in
/// [`SolveError::Backend`](super::SolveError::Backend) with the backend's
/// name, and returns it as that error's source.
pub type BackendError = Box<dyn Error + Send + Sync + 'static>;

/// A backend that decides SMT-LIB2 scripts: whether their assertions can
/// hold together.
///
/// A [`Solver`](super::Solver) holding one answers satisfiability,
/// implication and universal-validity questions. It encodes each question
/// as one script and calls [`check`](Self::check) once, after the screens
/// have passed the question's expressions.
///
/// The crate ships [`SmtLib2Process`](super::SmtLib2Process), which drives
/// any SMT-LIB2 executable, and, with the `z3` feature, `Z3Solver`.
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
///
/// use fhy_core::solver::{BackendError, CheckLimits, SatResult, SmtScript, SmtSolver};
///
/// /// A backend that finds every script satisfiable.
/// #[derive(Debug)]
/// struct Optimist;
///
/// impl SmtSolver for Optimist {
///     fn name(&self) -> Cow<'_, str> {
///         Cow::Borrowed("optimist")
///     }
///
///     fn check(&self, _script: &SmtScript, _limits: &CheckLimits) -> Result<SatResult, BackendError> {
///         Ok(SatResult::Sat)
///     }
/// }
/// ```
pub trait SmtSolver: Send + Sync + fmt::Debug {
    /// Return the backend's name, which errors and logs name it by.
    fn name(&self) -> Cow<'_, str>;

    /// Decide whether the assertions of `script` can hold together.
    ///
    /// A backend enforces [`CheckLimits::timeout`] its own way, and
    /// answers [`SatResult::Unknown`] with the reason `"timeout"` when it
    /// runs out of time.
    ///
    /// # Errors
    ///
    /// Returns any failure of the backend itself: one that is not an
    /// answer, such as a solver that cannot be run or that reports an
    /// error.
    fn check(&self, script: &SmtScript, limits: &CheckLimits) -> Result<SatResult, BackendError>;
}

/// A backend that simplifies expressions.
///
/// A [`Solver`](super::Solver) holding one answers simplification
/// questions. It screens the expression, applies the environment with
/// [`Expression::substitute`], and hands the result to
/// [`simplify`](Self::simplify), so a simplifier sees an expression with
/// nothing left to substitute.
pub trait Simplifier: Send + Sync + fmt::Debug {
    /// Return the backend's name, which errors and logs name it by.
    fn name(&self) -> Cow<'_, str>;

    /// Return a simpler expression equal to `expression`.
    ///
    /// Simplification is best-effort: returning `expression` itself means
    /// that the backend found nothing simpler. With every identifier
    /// substituted, simplifying is evaluating, and the result is a literal
    /// whenever the backend can decide the value.
    ///
    /// # Errors
    ///
    /// Returns any failure of the backend itself, such as a value it
    /// cannot represent.
    fn simplify(
        &self,
        expression: &Expression,
        context: &SimplifyContext<'_>,
    ) -> Result<Expression, BackendError>;
}

/// What a [`Simplifier`] is told about a simplification besides the
/// expression: the sorts of the native constants and named functions the
/// expression may refer to, and, when it has one, the registry holding
/// them.
///
/// A backend that lowers an expression to another system reads a user
/// constant's value, and what kind of entry a called name is, from the
/// [`registry`](Self::registry). A context built from sorts alone has no
/// registry, and such a backend refuses what it would need one for.
///
/// It is a struct rather than more parameters, so that it can carry more
/// without changing the trait.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
/// use fhy_core::expression::{FunctionName, FunctionSort, NoRegisteredSorts};
/// use fhy_core::solver::SimplifyContext;
///
/// let mut registry = FunctionRegistry::new();
/// let answer = registry.register_constant(NativeConstant::try_new(
///     FunctionName::try_new("answer")?,
///     FunctionSort::Int,
///     42,
/// )?)?;
///
/// let context = SimplifyContext::from_registry(&registry);
/// assert_eq!(context.sorts().native_constant_sort(&answer), Some(FunctionSort::Int));
/// assert!(context.registry().is_some());
/// assert!(SimplifyContext::new(&NoRegisteredSorts).registry().is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy)]
pub struct SimplifyContext<'a> {
    sorts: &'a dyn SortLookup,
    registry: Option<&'a FunctionRegistry>,
}

impl<'a> SimplifyContext<'a> {
    /// Return the context reading native constants and named functions'
    /// result sorts from `sorts`, with no registry.
    #[must_use]
    pub fn new(sorts: &'a dyn SortLookup) -> Self {
        Self {
            sorts,
            registry: None,
        }
    }

    /// Return the context reading sorts, constants' values and entries from
    /// `registry`.
    #[must_use]
    pub fn from_registry(registry: &'a FunctionRegistry) -> Self {
        Self {
            sorts: registry,
            registry: Some(registry),
        }
    }

    /// Return the sorts of native constants and named functions.
    #[must_use]
    pub fn sorts(&self) -> &'a dyn SortLookup {
        self.sorts
    }

    /// Return the registry the context was built from, or `None` for a
    /// context built from sorts alone.
    #[must_use]
    pub fn registry(&self) -> Option<&'a FunctionRegistry> {
        self.registry
    }
}

impl Default for SimplifyContext<'_> {
    /// Return the context that knows no native constant and no named
    /// function.
    fn default() -> Self {
        Self::new(&NoRegisteredSorts)
    }
}

impl fmt::Debug for SimplifyContext<'_> {
    /// Write the type name only: the sort lookup need not implement
    /// `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimplifyContext").finish_non_exhaustive()
    }
}

/// The answer of an SMT-LIB2 `check-sat`.
///
/// # Examples
///
/// ```
/// use fhy_core::solver::SatResult;
///
/// let gave_up = SatResult::Unknown { reason: "timeout".to_owned() };
///
/// assert_eq!(gave_up.to_string(), "unknown (timeout)");
/// assert_eq!(SatResult::Sat.to_string(), "sat");
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "check-sat has exactly these three answers"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SatResult {
    /// The assertions can hold together.
    Sat,
    /// The assertions cannot hold together.
    Unsat,
    /// The backend could not decide.
    Unknown {
        /// Why, in the backend's words, such as `"timeout"`.
        reason: String,
    },
}

impl fmt::Display for SatResult {
    /// Write `sat`, `unsat`, or `unknown` followed by the reason in
    /// parentheses.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sat => f.write_str("sat"),
            Self::Unsat => f.write_str("unsat"),
            Self::Unknown { reason } => write!(f, "unknown ({reason})"),
        }
    }
}

/// The bounds a backend's check runs under.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use fhy_core::solver::CheckLimits;
///
/// let limits = CheckLimits::new().with_timeout(Duration::from_millis(500));
///
/// assert_eq!(limits.timeout(), Some(Duration::from_millis(500)));
/// assert_eq!(CheckLimits::new().timeout(), None);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct CheckLimits {
    timeout: Option<Duration>,
}

impl CheckLimits {
    /// Return the limits that bound nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return these limits with the check bounded to `timeout`.
    #[must_use]
    pub fn with_timeout(self, timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
        }
    }

    /// Return how long a check may run, or `None` when it is unbounded.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}
