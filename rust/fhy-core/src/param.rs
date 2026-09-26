//! Params: variables ranging over a value domain, narrowed by constraints.
//!
//! - A [`ParamDomain`] is the value space of a param: the integers
//!   ([`IntegerDomain`], and [`IntervalIntegerDomain`] for interval
//!   arithmetic), the reals ([`RealDomain`]), a finite ordered set
//!   ([`OrdinalDomain`]), a finite unordered set ([`CategoricalDomain`]),
//!   the permutations of a fixed set ([`PermutationDomain`]), or a domain
//!   defined elsewhere ([`CustomDomain`]).
//! - A domain decides which values it admits and which constraints it
//!   allows, and answers the questions about a set of constraints over one
//!   variable, a [`Side`]: whether some value satisfies them, and whether
//!   one side's set is a subset of another's. Finite domains enumerate
//!   their values; numeric domains enumerate the candidates of an in-set
//!   constraint, or ask the [`Solver`](crate::solver::Solver) about a
//!   screened system of the constraints it can be asked about.
//! - Values are the constraint module's [`Value`](crate::constraint::Value)s,
//!   and a finite domain's values are [`Member`](crate::constraint::Member)s,
//!   so matching is type-strict: a Boolean, an integer and a float never
//!   match.
//! - Why an answer is undecided or weakened is reported to the
//!   [`ParamContext`]'s [`ParamObserver`] as a [`ParamEvent`].
//!
//! # Examples
//!
//! ```
//! use fhy_core::constraint::{Outcome, Value};
//! use fhy_core::identifier::Identifier;
//! use fhy_core::param::{ParamContext, ParamDomain, OrdinalDomain, Side};
//! use fhy_core::solver::Solver;
//!
//! let domain = ParamDomain::from(OrdinalDomain::new(vec![
//!     Value::Int(3.into()),
//!     Value::Int(1.into()),
//!     Value::Int(2.into()),
//! ])?);
//! let solver = Solver::new();
//! let context = ParamContext::new(&solver);
//! let x = Identifier::new("x");
//!
//! assert_eq!(domain.has_feasible_value(Side::new(&[], &x), &context)?, Outcome::Satisfied);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod algebra;
mod context;
mod custom;
mod decide;
mod domain;
mod error;
mod screen;
mod value;

pub use context::{NoParamObserver, ParamContext, ParamEvent, ParamObserver, ScreenReason};
pub use custom::CustomDomain;
pub use decide::{
    Evaluation, are_all_constraints_satisfied, compute_constraint_implication_subset,
    evaluate_constraints,
};
pub use domain::{
    CategoricalDomain, DomainKind, IntegerDomain, IntervalIntegerDomain, IntervalProfile,
    OrdinalDomain, ParamDomain, PermutationDomain, RealDomain, Side, is_bound_expression,
};
pub use error::{ParamError, SetOperation};
