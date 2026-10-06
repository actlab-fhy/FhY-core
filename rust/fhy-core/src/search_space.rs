//! Search spaces: the decisions a compiler search makes, the points it
//! visits, and when two spaces or two points are the same up to renaming.
//!
//! - A [`Variable`] is a decision over the values of a
//!   [`Param`](crate::param::Param). An [`Alternative`] is one option of a
//!   [`Choice`], with the variables and sub-choices that exist only when it
//!   is chosen and the identifiers it binds. Both are traits that other
//!   crates implement, and a [`Part<dyn Variable>`](crate::foreign::Part)
//!   or a [`Part<dyn Alternative>`](crate::foreign::Part) holds any
//!   implementation; [`PlainVariable`] and [`PlainAlternative`] are this
//!   module's own.
//! - A [`Space`] holds top-level variables and choices, [`Condition`]s on
//!   when a decision is active, and [`Forbidden`] combinations of values.
//!   Its **decisions** are its variables and choices at every depth, in
//!   **canonical order**: the top-level variables, then the top-level
//!   choices, each choice followed by each of its alternatives' variables
//!   and then their sub-choices. Every name in a space is distinct: the
//!   space's own, every decision's, every alternative's and every
//!   identifier an alternative binds.
//! - A [`Configuration`] is a point of a space: a value for some of its
//!   active decisions, a choice's value being the
//!   [identifier value](crate::constraint::Value::Identifier) of the chosen
//!   alternative's name. It is checked against its space when it is built,
//!   so every configuration is valid; [`Configuration::is_complete`] says
//!   whether it assigns every active decision. Its
//!   [`ConfigurationKey`] identifies it within its space, the same for
//!   configurations of spaces that differ only in their names.
//!
//! # Activity
//!
//! A decision is [active](Activity::Active) in a configuration when its
//! parent is (a top-level decision has none; a decision under an
//! alternative needs its choice active and that alternative chosen) and
//! its [`Condition`] holds. A condition reads the values of the decisions
//! it names: one that is inactive makes it false, and one that is active
//! but unassigned leaves the target [pending](Activity::Pending), which a
//! configuration may not assign either. A [`Forbidden`] clause applies once
//! every decision it names is active and assigned, and the configuration
//! must then violate it.
//!
//! A condition or a forbidden clause names a choice only in a set
//! constraint, `choice in {a, b}`, whose members are alternatives' names;
//! an equation names variables only. [`Space::new`] refuses an equation
//! that names a choice.
//!
//! # Equivalence
//!
//! Every name a space holds is a binder of the space. Two spaces are alpha
//! equivalent when they have the same shape and, with their names paired
//! in canonical order, every part corresponds: kinds equal, params with the
//! same domains and constraints, each implementation's own data by its
//! hook, and the conditions and forbidden clauses. An identifier inside a
//! domain or a set constraint's members is a reference, and corresponds
//! to an identifier on the other side only as
//! [`AlphaRenaming::is_corresponding`](crate::term::AlphaRenaming::is_corresponding)
//! says, so a free name never matches a bound one. A choice, an
//! alternative or a variable compared on its own binds its own names the
//! same way. Structural equivalence is the same comparison with every name
//! compared by `==`, so it implies alpha equivalence.
//!
//! # Implementing [`Variable`] and [`Alternative`]
//!
//! An implementation keeps this contract, which
//! `testing::check_variable_conformance` and
//! `testing::check_alternative_conformance`, behind the `testing` feature,
//! check where they can:
//!
//! 1. Every getter answers the same for the value's whole life.
//! 2. `kind()` is unique to the implementing type, stable across releases
//!    and processes, and the type id its `to_foreign` writes.
//! 3. An alternative's `bound_identifiers` are distinct, the same on every
//!    call, and as many for two values that should correspond.
//! 4. The `is_extension_*` hooks are equivalence relations; the structural
//!    hook implies the alpha hook under the empty renaming; they read the
//!    renaming only through `is_corresponding`, compare type-strictly,
//!    compare only the implementation's own data, never the name, param,
//!    variables, sub-choices or notes, which the core compares, and report
//!    a failure as `Err`, never by panicking.
//! 5. Every identifier the implementation's own data binds is one of its
//!    `bound_identifiers`; any other identifier in it is a reference.
//! 6. `to_foreign` gives a part that the implementation's resolver turns
//!    back into an equivalent value.
//! 7. A variable's `search_domain`, when it offers one, holds exactly the
//!    values its param's domain admits, before the param's constraints, in
//!    an order that is the same for the value's life.
//!
//! # Searching
//!
//! A search runs as a sequence of **steps**, each a decision offered to a
//! [`SearchOracle`] with the [`StepDomain`] it may be answered from and
//! answered with a [`Coordinate`], a position in that domain. A **static**
//! step asks a decision of a [`Space`] and takes its domain from it; a
//! **dynamic** step asks about something the space does not declare and
//! carries its own domain. A [`Recorder`] drives a run: it asks each step,
//! checks the answer, records it in a [`Trace`], and, over a space, grows
//! the run's [`Configuration`].
//!
//! [`RandomOracle`] draws, [`ReplayOracle`] answers a recorded trace back
//! and refuses a run that leaves its path, and [`ExhaustiveOracle`] takes
//! every path of a stream once over successive runs. Over a space alone,
//! [`Space::sample`], [`Space::sample_uniform`], [`Space::replay`],
//! [`Space::enumerate`], [`Space::cardinality`] and [`Space::mutate`] do
//! the rest. Every random draw takes its numbers from an [`Rng`] the caller
//! passes, whose stream is the same in every release and on every
//! platform.
//!
//! # Examples
//!
//! A choice between tiling and not, with a tile size that exists only
//! while tiling is chosen:
//!
//! ```
//! use fhy_core::constraint::Value;
//! use fhy_core::foreign::Part;
//! use fhy_core::identifier::Identifier;
//! use fhy_core::param::{CategoricalDomain, Param, ParamContext, ParamDomain};
//! use fhy_core::search_space::{
//!     Activity, Choice, Configuration, PlainAlternative, PlainVariable, Space,
//! };
//! use fhy_core::solver::Solver;
//!
//! let solver = Solver::new();
//! let context = ParamContext::new(&solver);
//! let [space, layout, tiled, flat, tile] =
//!     ["space", "layout", "tiled", "flat", "tile"].map(Identifier::new);
//! let sizes = Param::new(
//!     ParamDomain::from(CategoricalDomain::new(vec![Value::Int(4.into()), Value::Int(8.into())])?),
//!     Identifier::new("size"),
//!     [],
//!     &context,
//! )?;
//! let choice = Choice::new(
//!     layout.clone(),
//!     vec![
//!         Part::new(PlainAlternative::new(
//!             tiled.clone(),
//!             vec![Part::new(PlainVariable::new(tile.clone(), sizes))],
//!             vec![],
//!         )?),
//!         Part::new(PlainAlternative::new(flat.clone(), vec![], vec![])?),
//!     ],
//! )?;
//! let space = Space::new(space, vec![], vec![choice], vec![], vec![])?;
//!
//! let chose_flat = Configuration::new(&space, [(layout.clone(), Value::Identifier(flat))], &context)?;
//! assert_eq!(chose_flat.activity(&tile), Some(Activity::Inactive));
//! assert!(chose_flat.is_complete());
//!
//! let chose_tiled = Configuration::new(&space, [(layout, Value::Identifier(tiled))], &context)?;
//! assert_eq!(chose_tiled.activity(&tile), Some(Activity::Active));
//! assert!(!chose_tiled.is_complete());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Serialization
//!
//! [`wire`] gives the shapes and builds them with a resolver of the parts
//! other crates define. A space, a choice and a configuration serialize
//! with their parts tagged: `{"plain": {..}}` for this module's own
//! implementations and `{"foreign": {"type_id", "data"}}` for another's.

mod alternative;
mod choice;
mod configuration;
mod domain;
mod equivalence;
mod error;
mod exploration;
mod oracle;
mod recorder;
mod rng;
mod space;
#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub mod testing;
mod trace;
mod variable;
pub mod wire;

pub use alternative::{Alternative, PlainAlternative};
pub use choice::Choice;
pub use configuration::{Activity, Configuration, ConfigurationKey};
pub use domain::{
    ChoiceDomain, Coordinate, DecisionKind, DomainSignature, OrderDomain, StepDomain,
    StridedDomain, StridedRun,
};
pub use error::{
    ConfigurationError, ConfigurationErrors, EmptyKind, EquivalenceError, ReplayError, SpaceError,
    StepDomainError, TraceError,
};
pub use exploration::{Cardinality, Enumeration};
pub use oracle::{ExhaustiveOracle, PendingStep, RandomOracle, ReplayOracle, SearchOracle};
pub use recorder::{Recorded, Recorder};
pub use rng::Rng;
pub use space::{Condition, Decision, Forbidden, Space};
pub use trace::{Trace, TraceStep};
pub use variable::{PlainVariable, Variable};
