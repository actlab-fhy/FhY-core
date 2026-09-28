//! What a param question is asked with: the solver, the function registry,
//! and an observer of why an answer is undecided.

use std::collections::HashMap;
use std::fmt;

use crate::constraint::{
    Bindings, Constraint, ConstraintContext, ConstraintError, ConstraintEvent, ConstraintObserver,
    ConstraintSystem, Member, Outcome,
};
use crate::expression::SymbolType;
use crate::expression::registry::FunctionRegistry;
use crate::identifier::Identifier;
use crate::solver::{SolveError, Solver};

/// Why screening dropped or narrowed a constraint, reported in
/// [`ParamEvent::Screened`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ScreenReason<'a> {
    /// An equation's scope reaches beyond the variable.
    DependentScope,
    /// A set constraint constrains another variable.
    ForeignVariable,
    /// An in-set constraint has a member that does not lift to an
    /// expression.
    UnliftableMember(&'a ConstraintError),
    /// None of a not-in-set constraint's members lifts to an expression.
    NoLiftableMember,
    /// A not-in-set constraint was narrowed to the members that lift.
    Narrowed {
        /// The members kept.
        liftable: &'a [Member],
        /// The members dropped.
        excluded: &'a [Member],
    },
}

/// Why a param question is undecided or weakened, reported to a
/// [`ParamObserver`] when the reason arises.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ParamEvent<'a> {
    /// A constraint reported `event` while it was evaluated under
    /// `bindings`.
    Member {
        /// The constraint.
        constraint: &'a Constraint,
        /// The bindings it was evaluated under.
        bindings: &'a Bindings,
        /// What it reported.
        event: &'a ConstraintEvent<'a>,
    },
    /// A constraint of a conjunction evaluated under bindings answered
    /// [`Outcome::Undecided`].
    UndecidedMember {
        /// The constraint.
        constraint: &'a Constraint,
    },
    /// Evaluating a constraint failed with an error the observer judged
    /// undecidable, so the constraint counts as undecided.
    BridgeFailed {
        /// The constraint.
        constraint: &'a Constraint,
        /// The bindings it was evaluated under.
        bindings: &'a Bindings,
        /// The error.
        error: &'a ConstraintError,
    },
    /// A system reported `event` while the solver was asked about it.
    Question {
        /// The system asked about.
        system: &'a ConstraintSystem,
        /// The symbol types of the question.
        symbol_types: &'a HashMap<Identifier, SymbolType>,
        /// What it reported.
        event: &'a ConstraintEvent<'a>,
    },
    /// Screening dropped or narrowed a constraint for `variable`.
    Screened {
        /// The constraint.
        constraint: &'a Constraint,
        /// The variable the system is screened for.
        variable: &'a Identifier,
        /// Why.
        reason: ScreenReason<'a>,
    },
    /// No in-set candidate of `variable` was decided feasible, and the
    /// equations left `candidates` undecided.
    EnumerationUndecided {
        /// The variable.
        variable: &'a Identifier,
        /// The undecided candidates, in order.
        candidates: &'a [Member],
    },
    /// The candidates of `own` could not be decided against `other` on
    /// both sides.
    SubsetEnumerationUndecided {
        /// The candidate subset's variable.
        own: &'a Identifier,
        /// The candidate superset's variable.
        other: &'a Identifier,
        /// The undecided candidates, in order.
        candidates: &'a [Member],
    },
    /// The solver found `variable` satisfiable, but only on a screened
    /// system that lost constraints, so the answer is undecided.
    SatisfiedOnInexactSystem {
        /// The variable.
        variable: &'a Identifier,
    },
    /// The solver found `variable` unsatisfiable, but over the REAL sort
    /// with a not-in-set float member the sort conflates with other kinds,
    /// so the answer is undecided.
    ViolatedUnderKindConflation {
        /// The variable.
        variable: &'a Identifier,
    },
    /// The solver could not decide whether `variable` is satisfiable.
    SatisfiabilityUndecided {
        /// The variable.
        variable: &'a Identifier,
    },
    /// The solver could not decide whether `own` implies `other`.
    ImplicationUndecided {
        /// The antecedent's variable.
        own: &'a Identifier,
        /// The consequent's variable.
        other: &'a Identifier,
    },
    /// The solver's `outcome` to whether `own` implies `other` rests on a
    /// weakened side, so the answer is undecided.
    ImplicationDowngraded {
        /// The solver's answer.
        outcome: Outcome,
        /// The antecedent's variable.
        own: &'a Identifier,
        /// The consequent's variable.
        other: &'a Identifier,
    },
    /// `variable` provably admits a value outside the `permitted` values
    /// the other side admits.
    WitnessOutside {
        /// The variable.
        variable: &'a Identifier,
        /// How many values the other side permits.
        permitted: usize,
    },
}

/// Receives the [`ParamEvent`]s that explain undecided or weakened answers,
/// and judges which evaluation failures are undecided answers.
pub trait ParamObserver: Sync {
    /// Receive `event`.
    fn notify(&self, event: &ParamEvent<'_>);

    /// Return whether evaluating a constraint that failed with `error`
    /// counts as undecided rather than failing the question.
    ///
    /// The default answers true for a backend's failure, such as a
    /// simplifier that cannot lower an expression.
    fn is_undecidable(&self, error: &ConstraintError) -> bool {
        matches!(error, ConstraintError::Solve(SolveError::Backend { .. }))
    }
}

/// A [`ParamObserver`] that ignores every event and judges failures by the
/// default rule.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoParamObserver;

impl ParamObserver for NoParamObserver {
    fn notify(&self, _event: &ParamEvent<'_>) {}
}

/// What a param question is asked with: the [`ConstraintContext`] its
/// constraints are evaluated and asked about with, holding the [`Solver`]
/// whose simplifier evaluates equations and whose SMT backend answers
/// questions and the [`FunctionRegistry`] that knows the native constants,
/// and the [`ParamObserver`].
///
/// It lends its constraint context to the constraint procedures a question
/// calls, each time with an observer that forwards the constraint's events
/// to the param observer.
#[derive(Clone, Copy)]
pub struct ParamContext<'a> {
    constraint: ConstraintContext<'a>,
    observer: &'a dyn ParamObserver,
}

impl<'a> ParamContext<'a> {
    /// Return the context asking `solver`, with no registry and no
    /// observer.
    #[must_use]
    pub fn new(solver: &'a Solver) -> Self {
        Self {
            constraint: ConstraintContext::new(solver),
            observer: &NoParamObserver,
        }
    }

    /// Return the context reading native constants from `registry`.
    #[must_use]
    pub fn with_registry(self, registry: &'a FunctionRegistry) -> Self {
        Self {
            constraint: self.constraint.with_registry(registry),
            ..self
        }
    }

    /// Return the context reporting to `observer`.
    #[must_use]
    pub fn with_observer(self, observer: &'a dyn ParamObserver) -> Self {
        Self { observer, ..self }
    }

    /// Return the solver.
    #[must_use]
    pub const fn solver(&self) -> &'a Solver {
        self.constraint.solver()
    }

    /// Return the registry, if any.
    #[must_use]
    pub const fn registry(&self) -> Option<&'a FunctionRegistry> {
        self.constraint.registry()
    }

    /// Return the constraint context of the solver and the registry, which
    /// reports no event; a param question evaluates its constraints with it,
    /// forwarding their events to the param observer.
    #[must_use]
    pub const fn constraint_context(&self) -> &ConstraintContext<'a> {
        &self.constraint
    }

    /// Return the constraint context reporting to `observer`.
    pub(super) fn constraint_context_with<'b>(
        &self,
        observer: &'b dyn ConstraintObserver,
    ) -> ConstraintContext<'b>
    where
        'a: 'b,
    {
        self.constraint.with_observer(observer)
    }

    /// Return whether `identifier` is a native constant's canonical
    /// identifier.
    #[must_use]
    pub fn is_native_constant(&self, identifier: &Identifier) -> bool {
        self.constraint.is_native_constant(identifier)
    }

    /// Report `event` to the observer.
    pub(super) fn notify(&self, event: &ParamEvent<'_>) {
        self.observer.notify(event);
    }

    /// Return whether `error` counts as undecided, as the observer judges.
    pub(super) fn is_undecidable(&self, error: &ConstraintError) -> bool {
        self.observer.is_undecidable(error)
    }
}

impl fmt::Debug for ParamContext<'_> {
    /// Write the solver only: the observer need not implement `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamContext")
            .field("solver", &self.solver())
            .finish_non_exhaustive()
    }
}

/// The constraint observer of one constraint evaluated under bindings: it
/// reports each event as a [`ParamEvent::Member`].
pub(super) struct MemberForwarder<'a> {
    pub(super) context: &'a ParamContext<'a>,
    pub(super) constraint: &'a Constraint,
    pub(super) bindings: &'a Bindings,
}

impl ConstraintObserver for MemberForwarder<'_> {
    fn notify(&self, event: &ConstraintEvent<'_>) {
        self.context.notify(&ParamEvent::Member {
            constraint: self.constraint,
            bindings: self.bindings,
            event,
        });
    }
}

/// The constraint observer of one question about a system: it reports each
/// event as a [`ParamEvent::Question`].
pub(super) struct QuestionForwarder<'a> {
    pub(super) context: &'a ParamContext<'a>,
    pub(super) system: &'a ConstraintSystem,
    pub(super) symbol_types: &'a HashMap<Identifier, SymbolType>,
}

impl ConstraintObserver for QuestionForwarder<'_> {
    fn notify(&self, event: &ConstraintEvent<'_>) {
        self.context.notify(&ParamEvent::Question {
            system: self.system,
            symbol_types: self.symbol_types,
            event,
        });
    }
}
