//! What a constraint is told when it is evaluated or asked about: the
//! solver, the function registry, and an observer of why an outcome is
//! undecided.

use std::fmt;

use crate::expression::builtins::BuiltinConstant;
use crate::expression::registry::FunctionRegistry;
use crate::expression::{Expression, NoRegisteredSorts, SortLookup};
use crate::identifier::Identifier;
use crate::solver::{Hazard, QueryKind, SimplifyContext, Solver};

/// Why an evaluation or a question is undecided, reported to an
/// [`Observer`] when the reason arises.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Event<'a> {
    /// A set constraint's variable is unbound.
    Unbound {
        /// The variable.
        variable: &'a Identifier,
    },
    /// A set constraint's variable is bound to an expression that is not a
    /// literal, so its membership cannot be decided.
    SymbolicBinding {
        /// The variable.
        variable: &'a Identifier,
        /// The expression it is bound to.
        binding: &'a Expression,
    },
    /// The bindings bind native constants the constraint refers to, which
    /// name values rather than variables, so they cannot take part in the
    /// decision.
    BoundNativeConstants {
        /// The native constants' identifiers, ordered by id.
        identifiers: &'a [Identifier],
    },
    /// A member of a [`ConstraintSystem`](super::ConstraintSystem) reported
    /// `event` while the system evaluated it.
    InMember {
        /// The member's position in the system's canonical order.
        index: usize,
        /// What the member reported.
        event: &'a Event<'a>,
    },
    /// A member of a system answered [`Outcome::Undecided`](super::Outcome)
    /// while the system evaluated it.
    UndecidedMember {
        /// The member's position in the system's canonical order.
        index: usize,
    },
    /// The solver's hazard screen refused a question a system asked, so no
    /// backend was asked.
    Refused {
        /// The kind of question.
        kind: QueryKind,
        /// What the screen refused.
        hazard: &'a Hazard,
    },
    /// The solver's backend answered `unknown` to a question a system
    /// asked.
    GaveUp {
        /// The kind of question.
        kind: QueryKind,
        /// Why, in the backend's words.
        reason: &'a str,
    },
    /// An equation simplified to an expression that is not a literal.
    Residual {
        /// The simplified expression.
        residual: &'a Expression,
        /// Whether it still has free identifiers, as when a binding is
        /// missing, rather than the simplifier failing to decide a ground
        /// expression.
        has_free_identifiers: bool,
    },
}

/// Receives the [`Event`]s that explain undecided outcomes.
///
/// A constraint reports each event once, when its reason arises, whether
/// or not the outcome it explains is the final one.
pub trait Observer: Sync {
    /// Receive `event`.
    fn notify(&self, event: &Event<'_>);
}

/// An [`Observer`] that ignores every event.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct NoObserver;

impl Observer for NoObserver {
    fn notify(&self, _event: &Event<'_>) {}
}

/// What a constraint is evaluated with: the [`Solver`] whose simplifier
/// decides an equation, the [`FunctionRegistry`] that knows the native
/// constants and named functions, and the [`Observer`] of undecided
/// outcomes.
///
/// A context without a registry knows the built-in constants only.
#[derive(Clone, Copy)]
pub struct ConstraintContext<'a> {
    solver: &'a Solver,
    registry: Option<&'a FunctionRegistry>,
    observer: &'a dyn Observer,
}

impl<'a> ConstraintContext<'a> {
    /// Return the context asking `solver`, with no registry and no
    /// observer.
    #[must_use]
    pub fn new(solver: &'a Solver) -> Self {
        Self {
            solver,
            registry: None,
            observer: &NoObserver,
        }
    }

    /// Return the context reading native constants and named functions
    /// from `registry`.
    #[must_use]
    pub fn with_registry(self, registry: &'a FunctionRegistry) -> Self {
        Self {
            registry: Some(registry),
            ..self
        }
    }

    /// Return the context reporting undecided outcomes to `observer`.
    #[must_use]
    pub fn with_observer(self, observer: &'a dyn Observer) -> Self {
        Self { observer, ..self }
    }

    /// Return the solver.
    #[must_use]
    pub fn solver(&self) -> &'a Solver {
        self.solver
    }

    /// Return the registry, if any.
    #[must_use]
    pub fn registry(&self) -> Option<&'a FunctionRegistry> {
        self.registry
    }

    /// Return the sorts of native constants and named functions.
    #[must_use]
    pub fn sorts(&self) -> &'a dyn SortLookup {
        match self.registry {
            Some(registry) => registry,
            None => &NoRegisteredSorts,
        }
    }

    /// Return the context a simplification is asked with.
    pub(super) fn simplify_context(&self) -> SimplifyContext<'a> {
        match self.registry {
            Some(registry) => SimplifyContext::from_registry(registry),
            None => SimplifyContext::new(&NoRegisteredSorts),
        }
    }

    /// Return whether `identifier` is a native constant's canonical
    /// identifier: a built-in constant's, or a registered one's.
    #[must_use]
    pub fn is_native_constant(&self, identifier: &Identifier) -> bool {
        BuiltinConstant::of_identifier(identifier).is_some()
            || self.sorts().native_constant_sort(identifier).is_some()
    }

    /// Report `event` to the observer.
    pub(super) fn notify(&self, event: &Event<'_>) {
        self.observer.notify(event);
    }
}

impl fmt::Debug for ConstraintContext<'_> {
    /// Write the solver only: the observer need not implement `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintContext")
            .field("solver", &self.solver)
            .finish_non_exhaustive()
    }
}
