//! Questions about expressions, answered by pluggable backends: whether an
//! expression is satisfiable, whether one implies another, whether one
//! holds for every assignment of its free identifiers, and what an
//! expression simplifies to.
//!
//! A [`Solver`] holds the backends: an [`SmtSolver`] for the three logical
//! questions, and a [`Simplifier`] for simplification. It checks and
//! screens each question in Rust, encodes a logical question as one
//! SMT-LIB2 [`SmtScript`], and calls its backend once.
//!
//! # The order of checks
//!
//! [`Solver::ask`] checks, in this order:
//!
//! 1. that the solver holds an [`SmtSolver`]
//!    ([`SolveError::NoCapableBackend`]);
//! 2. that every identifier the question mentions, the considered ones of a
//!    universal-validity question included, has a symbol type, except a
//!    native constant's ([`SolveError::MissingSymbolTypes`]);
//! 3. that each expression, in order, can be a predicate, as
//!    [`BooleanScreen::check_predicate`](crate::expression::BooleanScreen::check_predicate)
//!    judges it ([`SolveError::IllTyped`]);
//! 4. that each expression, in order and on its own, passes the hazard
//!    screen of [`Hazard::find`]; the first hazard found answers the
//!    question [`Answer::Unknown`] with [`UnknownReason::Refused`].
//!
//! A question that passes is lowered and handed to the backend. Only a
//! call has no lowering once the screens have passed
//! ([`SolveError::Lowering`]).
//!
//! # The encodings
//!
//! | Question | Script | [`Answer::Yes`] when |
//! |---|---|---|
//! | satisfiability of `e` | `(assert e)` | `sat` |
//! | `a` implies `c` | `(assert (and a (not c)))` | `unsat` |
//! | universal validity of `e`, with the considered identifiers `C` of `e` and the free ones `F` | `(assert (forall (C) (not e)))` with `F` declared, when both are non-empty; `(assert (not e))` when `C` is empty; `(assert e)` when `F` is empty | `unsat`, `unsat`, `sat` |
//!
//! A quantifier appears only where the question alternates them, so a
//! quantifier-free backend answers every other question. The answer that
//! is not [`Answer::Yes`] is [`Answer::No`], and `unknown` is
//! [`Answer::Unknown`] with [`UnknownReason::GaveUp`].
//!
//! # Examples
//!
//! ```
//! use std::borrow::Cow;
//! use std::collections::HashMap;
//!
//! use fhy_core::expression::{Expression, SymbolType};
//! use fhy_core::identifier::Identifier;
//! use fhy_core::solver::{
//!     Answer, BackendError, CheckLimits, QueryContext, Question, SatResult, SmtScript,
//!     SmtSolver, Solver,
//! };
//!
//! /// A backend that finds every script satisfiable.
//! #[derive(Debug)]
//! struct Optimist;
//!
//! impl SmtSolver for Optimist {
//!     fn name(&self) -> Cow<'_, str> {
//!         Cow::Borrowed("optimist")
//!     }
//!
//!     fn check(&self, _script: &SmtScript, _limits: &CheckLimits) -> Result<SatResult, BackendError> {
//!         Ok(SatResult::Sat)
//!     }
//! }
//!
//! let x = Identifier::new("x");
//! let positive = Expression::from(x.clone()).greater(0);
//! let symbol_types = HashMap::from([(x, SymbolType::Int)]);
//! let solver = Solver::new().with_smt_solver(Optimist);
//!
//! let answer = solver.ask(
//!     &Question::Satisfiability(&positive),
//!     &QueryContext::new(&symbol_types),
//! )?;
//!
//! assert_eq!(answer, Answer::Yes);
//! # Ok::<(), fhy_core::solver::SolveError>(())
//! ```

mod backend;
mod error;
mod process;
mod screen;
mod smt;
#[cfg(feature = "z3")]
mod z3;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::BuildHasher;
use std::sync::Arc;

use crate::expression::{
    BooleanScreen, Expression, NoRegisteredSorts, SortLookup, SymbolType, SymbolTypes,
};
use crate::identifier::Identifier;

pub use backend::{BackendError, CheckLimits, SatResult, Simplifier, SimplifyContext, SmtSolver};
pub use error::{LoweringError, SolveError};
pub use process::{ProcessError, SmtLib2Process};
pub use screen::Hazard;
pub use smt::{Declaration, Logic, SmtScript};
#[cfg(feature = "z3")]
pub use z3::{Z3Solver, Z3TermError};

use screen::is_native_constant;
use smt::{Assertion, Lowerer, Operator};

/// A kind of question a [`Solver`] answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueryKind {
    /// What an expression simplifies to; a [`Simplifier`] answers it.
    Simplification,
    /// Whether some assignment satisfies an expression; an [`SmtSolver`]
    /// answers it.
    Satisfiability,
    /// Whether every assignment satisfying one expression satisfies
    /// another; an [`SmtSolver`] answers it.
    Implication,
    /// Whether every assignment of an expression's free identifiers has a
    /// witness among its considered ones; an [`SmtSolver`] answers it.
    UniversalValidity,
}

impl QueryKind {
    /// Return the snake-case name of the kind, such as
    /// `"universal_validity"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Simplification => "simplification",
            Self::Satisfiability => "satisfiability",
            Self::Implication => "implication",
            Self::UniversalValidity => "universal_validity",
        }
    }

    /// Return the kind in words, such as `"universal validity"`.
    fn description(self) -> &'static str {
        match self {
            Self::UniversalValidity => "universal validity",
            other => other.as_str(),
        }
    }
}

impl fmt::Display for QueryKind {
    /// Write the kind in words, such as `universal validity`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

/// A logical question about expressions.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Question<'a> {
    /// Whether some assignment of the expression's identifiers satisfies
    /// it.
    Satisfiability(&'a Expression),
    /// Whether every assignment satisfying `antecedent` satisfies
    /// `consequent`.
    Implication {
        /// The premise.
        antecedent: &'a Expression,
        /// The conclusion.
        consequent: &'a Expression,
    },
    /// Whether, for every assignment of the expression's identifiers that
    /// are not `considered`, some assignment of the considered ones
    /// satisfies it: `forall F. exists C. e`.
    UniversalValidity {
        /// The existentially quantified identifiers. One the expression
        /// does not mention needs a symbol type but quantifies nothing.
        considered: &'a HashSet<Identifier>,
        /// The expression.
        expression: &'a Expression,
    },
}

impl Question<'_> {
    /// Return the kind of the question.
    #[must_use]
    pub fn kind(&self) -> QueryKind {
        match self {
            Self::Satisfiability(_) => QueryKind::Satisfiability,
            Self::Implication { .. } => QueryKind::Implication,
            Self::UniversalValidity { .. } => QueryKind::UniversalValidity,
        }
    }
}

/// What a [`Solver`] is told about a logical question besides its
/// expressions: the symbol types of their identifiers, the sorts of native
/// constants and named functions, and the limits of the backend's check.
#[derive(Clone, Copy)]
pub struct QueryContext<'a> {
    symbol_types: &'a dyn SymbolTypes,
    sorts: &'a dyn SortLookup,
    limits: CheckLimits,
}

impl<'a> QueryContext<'a> {
    /// Return the context reading the value kinds of identifiers from
    /// `symbol_types`, knowing no native constant or named function, and
    /// with unbounded checks.
    #[must_use]
    pub fn new(symbol_types: &'a dyn SymbolTypes) -> Self {
        Self {
            symbol_types,
            sorts: &NoRegisteredSorts,
            limits: CheckLimits::new(),
        }
    }

    /// Return the context reading native constants and named functions'
    /// result sorts from `sorts`.
    #[must_use]
    pub fn with_sorts(self, sorts: &'a dyn SortLookup) -> Self {
        Self { sorts, ..self }
    }

    /// Return the context bounding the backend's check by `limits`.
    #[must_use]
    pub fn with_limits(self, limits: CheckLimits) -> Self {
        Self { limits, ..self }
    }

    /// Return the limits of the backend's check.
    #[must_use]
    pub fn limits(&self) -> CheckLimits {
        self.limits
    }
}

impl fmt::Debug for QueryContext<'_> {
    /// Write the limits only: the lookups need not implement `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryContext")
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

/// The answer to a logical question.
#[expect(
    clippy::exhaustive_enums,
    reason = "a question has exactly these three answers"
)]
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// The question holds.
    Yes,
    /// The question does not hold.
    No,
    /// The question was not decided.
    Unknown(UnknownReason),
}

impl Answer {
    /// Return `Some(true)` for [`Yes`](Self::Yes), `Some(false)` for
    /// [`No`](Self::No), and `None` when the question was not decided.
    #[must_use]
    pub fn decided(&self) -> Option<bool> {
        match self {
            Self::Yes => Some(true),
            Self::No => Some(false),
            Self::Unknown(_) => None,
        }
    }
}

/// Why a question was not decided.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UnknownReason {
    /// The hazard screen refused an expression of the question, so no
    /// backend was asked. A longer timeout cannot change this.
    Refused(Hazard),
    /// The backend answered `unknown`.
    GaveUp {
        /// Why, in the backend's words, such as `"timeout"`.
        reason: String,
    },
}

/// Answers questions about expressions with the backends it holds.
///
/// A new solver holds no backend. Cloning one shares its backends, which
/// are held behind an [`Arc`]. It keeps no state between questions.
#[derive(Debug, Clone, Default)]
pub struct Solver {
    smt_solver: Option<Arc<dyn SmtSolver>>,
    simplifier: Option<Arc<dyn Simplifier>>,
}

impl Solver {
    /// Return the solver holding no backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return this solver with `backend` answering the logical questions.
    #[must_use]
    pub fn with_smt_solver(self, backend: impl SmtSolver + 'static) -> Self {
        self.with_shared_smt_solver(Arc::new(backend))
    }

    /// Return this solver with the shared `backend` answering the logical
    /// questions.
    #[must_use]
    pub fn with_shared_smt_solver(self, backend: Arc<dyn SmtSolver>) -> Self {
        Self {
            smt_solver: Some(backend),
            ..self
        }
    }

    /// Return this solver with `backend` answering simplification
    /// questions.
    #[must_use]
    pub fn with_simplifier(self, backend: impl Simplifier + 'static) -> Self {
        self.with_shared_simplifier(Arc::new(backend))
    }

    /// Return this solver with the shared `backend` answering
    /// simplification questions.
    #[must_use]
    pub fn with_shared_simplifier(self, backend: Arc<dyn Simplifier>) -> Self {
        Self {
            simplifier: Some(backend),
            ..self
        }
    }

    /// Return the backend answering the logical questions, if any.
    #[must_use]
    pub fn smt_solver(&self) -> Option<&Arc<dyn SmtSolver>> {
        self.smt_solver.as_ref()
    }

    /// Return the backend answering simplification questions, if any.
    #[must_use]
    pub fn simplifier(&self) -> Option<&Arc<dyn Simplifier>> {
        self.simplifier.as_ref()
    }

    /// Return whether this solver holds a backend answering `kind`.
    #[must_use]
    pub fn can_answer(&self, kind: QueryKind) -> bool {
        match kind {
            QueryKind::Simplification => self.simplifier.is_some(),
            QueryKind::Satisfiability | QueryKind::Implication | QueryKind::UniversalValidity => {
                self.smt_solver.is_some()
            }
        }
    }

    /// Answer `question`, with what `context` tells about its identifiers.
    ///
    /// The checks and encodings are the module's.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::NoCapableBackend`] without an [`SmtSolver`],
    /// [`SolveError::MissingSymbolTypes`] and [`SolveError::IllTyped`] as
    /// the module's order of checks describes, [`SolveError::Lowering`]
    /// for a call, and [`SolveError::Backend`] when the backend fails.
    pub fn ask(
        &self,
        question: &Question<'_>,
        context: &QueryContext<'_>,
    ) -> Result<Answer, SolveError> {
        let backend = self
            .smt_solver
            .as_ref()
            .ok_or(SolveError::NoCapableBackend(question.kind()))?;
        let expressions = question.expressions();
        let mut mentioned: HashSet<Identifier> = HashSet::new();
        for expression in &expressions {
            mentioned.extend(expression.free_identifiers());
        }
        if let Question::UniversalValidity { considered, .. } = question {
            mentioned.extend(considered.iter().cloned());
        }
        let mut missing: Vec<Identifier> = mentioned
            .into_iter()
            .filter(|identifier| {
                context.symbol_types.symbol_type(identifier).is_none()
                    && !is_native_constant(identifier, context.sorts)
            })
            .collect();
        if !missing.is_empty() {
            missing.sort_by_key(Identifier::id);
            return Err(SolveError::MissingSymbolTypes(missing));
        }
        let screen = BooleanScreen::new()
            .with_sorts(context.sorts)
            .with_symbol_types(context.symbol_types);
        for expression in &expressions {
            screen
                .check_predicate(expression)
                .map_err(SolveError::IllTyped)?;
        }
        for expression in &expressions {
            if let Some(hazard) = Hazard::find(expression, context.symbol_types, context.sorts) {
                return Ok(Answer::Unknown(UnknownReason::Refused(hazard)));
            }
        }
        let (script, is_yes_when_sat) =
            encode(question, context.symbol_types).map_err(SolveError::Lowering)?;
        let result =
            backend
                .check(&script, &context.limits)
                .map_err(|source| SolveError::Backend {
                    backend: backend.name().into_owned(),
                    source,
                })?;
        Ok(match result {
            SatResult::Sat if is_yes_when_sat => Answer::Yes,
            SatResult::Unsat if !is_yes_when_sat => Answer::Yes,
            SatResult::Sat | SatResult::Unsat => Answer::No,
            SatResult::Unknown { reason } => Answer::Unknown(UnknownReason::GaveUp { reason }),
        })
    }

    /// Simplify `expression` after substituting `environment` into it.
    ///
    /// The solver checks, in this order, that it holds a [`Simplifier`];
    /// that no Boolean position of the expression holds a number, counting
    /// an identifier bound to one, as
    /// [`BooleanScreen::check_logical_operands`](crate::expression::BooleanScreen::check_logical_operands)
    /// judges it with `environment`; and that `environment` binds no native
    /// constant the expression refers to. It then substitutes
    /// `environment` with [`Expression::substitute`] and hands the result
    /// to the simplifier, which receives `expression` itself when
    /// `environment` binds none of its identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::NoCapableBackend`],
    /// [`SolveError::IllTyped`] and [`SolveError::BoundNativeConstant`] for
    /// those checks, [`SolveError::Substitution`] if the substitution puts
    /// a literal other than a Boolean in a case condition, and
    /// [`SolveError::Backend`] when the simplifier fails.
    pub fn simplify<S: BuildHasher>(
        &self,
        expression: &Expression,
        environment: &HashMap<Identifier, Expression, S>,
        sorts: &dyn SortLookup,
    ) -> Result<Expression, SolveError> {
        let simplifier = self
            .simplifier
            .as_ref()
            .ok_or(SolveError::NoCapableBackend(QueryKind::Simplification))?;
        BooleanScreen::new()
            .with_sorts(sorts)
            .with_environment(environment)
            .check_logical_operands(expression)
            .map_err(SolveError::IllTyped)?;
        let referenced = expression.free_identifiers();
        let mut bound_constants: Vec<Identifier> = environment
            .keys()
            .filter(|identifier| {
                referenced.contains(*identifier) && is_native_constant(identifier, sorts)
            })
            .cloned()
            .collect();
        if !bound_constants.is_empty() {
            bound_constants.sort_by_key(Identifier::id);
            return Err(SolveError::BoundNativeConstant(bound_constants));
        }
        let substituted = expression
            .substitute(environment)
            .map_err(SolveError::Substitution)?;
        simplifier
            .simplify(&substituted, &SimplifyContext::new(sorts))
            .map_err(|source| SolveError::Backend {
                backend: simplifier.name().into_owned(),
                source,
            })
    }
}

/// Return the script of `question`, and whether `sat` answers it
/// [`Answer::Yes`], by the encodings of the module.
fn encode(
    question: &Question<'_>,
    symbol_types: &dyn SymbolTypes,
) -> Result<(SmtScript, bool), LoweringError> {
    let mut lowerer = Lowerer::new(symbol_types);
    let (assertion, is_yes_when_sat) = match question {
        Question::Satisfiability(expression) => (
            Assertion {
                quantified: Vec::new(),
                body: lowerer.lower(expression)?,
            },
            true,
        ),
        Question::Implication {
            antecedent,
            consequent,
        } => {
            let antecedent = lowerer.lower(antecedent)?;
            let consequent = lowerer.lower(consequent)?;
            let refuted = lowerer.apply(Operator::Not, vec![consequent], SymbolType::Bool);
            let body = lowerer.apply(Operator::And, vec![antecedent, refuted], SymbolType::Bool);
            (
                Assertion {
                    quantified: Vec::new(),
                    body,
                },
                false,
            )
        }
        Question::UniversalValidity {
            considered,
            expression,
        } => {
            let body = lowerer.lower(expression)?;
            let free = expression.free_identifiers();
            let quantified: Vec<usize> = free
                .iter()
                .filter(|identifier| considered.contains(*identifier))
                .filter_map(|identifier| lowerer.symbol_of(identifier))
                .collect();
            let is_every_identifier_considered = quantified.len() == free.len();
            if quantified.is_empty() || !is_every_identifier_considered {
                let negated = lowerer.apply(Operator::Not, vec![body], SymbolType::Bool);
                (
                    Assertion {
                        quantified,
                        body: negated,
                    },
                    false,
                )
            } else {
                (
                    Assertion {
                        quantified: Vec::new(),
                        body,
                    },
                    true,
                )
            }
        }
    };
    Ok((lowerer.finish(vec![assertion], None), is_yes_when_sat))
}

impl<'a> Question<'a> {
    /// Return the expressions of the question, in the order they are
    /// checked and screened.
    fn expressions(&self) -> Vec<&'a Expression> {
        match *self {
            Self::Satisfiability(expression) | Self::UniversalValidity { expression, .. } => {
                vec![expression]
            }
            Self::Implication {
                antecedent,
                consequent,
            } => vec![antecedent, consequent],
        }
    }
}
