//! The wire forms of domains, params and assignments, which hold opaque
//! values, custom constraints and custom domains as [`Foreign`] parts until
//! a resolver builds them.
//!
//! A [`ParamDomain`] serializes as `{"integer": {"non_negative",
//! "zero_included"}}`, `{"interval_integer": {"prefer_inclusive",
//! "non_negative", "zero_included"}}`, `{"real": {}}`, `{"ordinal":
//! {"sorted_values"}}` (in ordinal order), `{"categorical": {"categories"}}`
//! (in canonical order), `{"permutation": {"ordered_members"}}` (as given)
//! or `{"custom": <foreign part>}`, each value in the constraint module's
//! form. A [`Param`] serializes as `{"domain", "variable",
//! "constraint_system"}` and a [`ParamAssignment`] as `{"param", "value"}`.
//! Decoding builds each domain through its constructor, so a payload
//! outside a domain's invariants fails with its constructor's error; a
//! param is built through [`Param::new`], which checks its constraints
//! against its domain, and an assignment without a check, as unpickling
//! does. [`ParamDomainData`], [`ParamData`] and [`ParamAssignmentData`] read
//! the same shapes and resolve the foreign parts in `build`; the types' own
//! `Deserialize` builds with [`NoForeign`], which refuses them, and a
//! context of a solver without backends and no registry.
//!
//! # Examples
//!
//! ```
//! use fhy_core::param::{IntegerDomain, ParamDomain, Sign, ZeroInclusion};
//!
//! let domain = ParamDomain::from(IntegerDomain::new(Sign::NonNegative, ZeroInclusion::Excluded));
//! let text = serde_json::to_string(&domain)?;
//! assert_eq!(text, r#"{"integer":{"non_negative":true,"zero_included":false}}"#);
//! assert!(serde_json::from_str::<ParamDomain>(&text)?.is_structurally_equivalent(&domain));
//! # Ok::<(), serde_json::Error>(())
//! ```

use serde::de::{self, Deserializer};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};

use crate::constraint::wire::{ConstraintResolver, ConstraintSystemData, ValueData};
use crate::constraint::{ConstraintError, Member, OpaqueValue, Value};
use crate::foreign::{BuildError, Foreign, ForeignError, NoForeign, Part, Resolve};
use crate::identifier::Identifier;
use crate::solver::{SolveError, Solver};

use super::assignment::ParamAssignment;
use super::context::{ParamContext, ParamEvent, ParamObserver};
use super::custom::CustomDomain;
use super::domain::{
    CategoricalDomain, IntegerDomain, IntervalIntegerDomain, OrdinalDomain, ParamDomain,
    PermutationDomain, RealDomain, Sign, ZeroInclusion,
};
use super::interval::Inclusivity;
use super::parameter::Param;

/// The resolvers a param's foreign parts need.
pub trait ParamResolver: ConstraintResolver + Resolve<Part<dyn CustomDomain>> {}

impl<R: ConstraintResolver + Resolve<Part<dyn CustomDomain>> + ?Sized> ParamResolver for R {}

/// The wire form of a [`ParamDomain`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParamDomainData(DomainRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "ParamDomain", rename_all = "snake_case")]
enum DomainRepr {
    Integer(IntegerRepr),
    IntervalInteger(IntervalIntegerRepr),
    Real(RealRepr),
    Ordinal(OrdinalRepr),
    Categorical(CategoricalRepr),
    Permutation(PermutationRepr),
    Custom(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "IntegerDomain", deny_unknown_fields)]
struct IntegerRepr {
    non_negative: bool,
    zero_included: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "IntervalIntegerDomain", deny_unknown_fields)]
struct IntervalIntegerRepr {
    prefer_inclusive: bool,
    non_negative: bool,
    zero_included: bool,
}

/// The empty fields of the real domain's encoding, so that every domain
/// encodes as a map.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "RealDomain", deny_unknown_fields)]
struct RealRepr {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "OrdinalDomain", deny_unknown_fields)]
struct OrdinalRepr {
    sorted_values: Vec<ValueData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "CategoricalDomain", deny_unknown_fields)]
struct CategoricalRepr {
    categories: Vec<ValueData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "PermutationDomain", deny_unknown_fields)]
struct PermutationRepr {
    ordered_members: Vec<ValueData>,
}

fn of_values(members: &[Member]) -> Result<Vec<ValueData>, ForeignError> {
    members
        .iter()
        .map(|member| ValueData::of_value(&member_value(member)))
        .collect()
}

/// Return the value a member was read from.
fn member_value(member: &Member) -> Value {
    use crate::constraint::MemberKind;
    match member.kind() {
        MemberKind::Bool(value) => Value::Bool(value),
        MemberKind::Int(value) => Value::Int(value.clone()),
        MemberKind::Float(value) => Value::Float(value),
        MemberKind::Str(value) => Value::Str(value.to_owned()),
        MemberKind::Identifier(_) => todo!(),
        MemberKind::Tuple(members) => Value::Tuple(members.iter().map(member_value).collect()),
        MemberKind::FrozenSet(members) => {
            Value::FrozenSet(members.iter().map(member_value).collect())
        }
        MemberKind::Opaque(value) => Value::Opaque(value.clone()),
    }
}

fn build_values<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
    values: Vec<ValueData>,
    resolver: &R,
) -> Result<Vec<Value>, BuildError> {
    values
        .into_iter()
        .map(|value| value.build(resolver))
        .collect()
}

impl ParamDomainData {
    /// Return the wire form of `domain`, asking each foreign part for its
    /// form.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(domain: &ParamDomain) -> Result<Self, ForeignError> {
        Ok(Self(match domain {
            ParamDomain::Integer(domain) => DomainRepr::Integer(IntegerRepr {
                non_negative: domain.is_non_negative(),
                zero_included: domain.is_zero_included(),
            }),
            ParamDomain::IntervalInteger(domain) => {
                DomainRepr::IntervalInteger(IntervalIntegerRepr {
                    prefer_inclusive: domain.is_inclusive_preferred(),
                    non_negative: domain.is_non_negative(),
                    zero_included: domain.is_zero_included(),
                })
            }
            ParamDomain::Real(_) => DomainRepr::Real(RealRepr {}),
            ParamDomain::Ordinal(domain) => DomainRepr::Ordinal(OrdinalRepr {
                sorted_values: of_values(domain.values())?,
            }),
            ParamDomain::Categorical(domain) => DomainRepr::Categorical(CategoricalRepr {
                categories: of_values(domain.values())?,
            }),
            ParamDomain::Permutation(domain) => DomainRepr::Permutation(PermutationRepr {
                ordered_members: of_values(domain.values())?,
            }),
            ParamDomain::Custom(domain) => DomainRepr::Custom(domain.get().to_foreign()?),
        }))
    }

    /// Return the foreign part of a custom domain, or `None` for a built-in
    /// one.
    #[must_use]
    pub const fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            DomainRepr::Custom(foreign) => Some(foreign),
            DomainRepr::Integer(_)
            | DomainRepr::IntervalInteger(_)
            | DomainRepr::Real(_)
            | DomainRepr::Ordinal(_)
            | DomainRepr::Categorical(_)
            | DomainRepr::Permutation(_) => None,
        }
    }

    /// Return the domain, its foreign parts resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses, and
    /// [`BuildError::Invalid`] with the [`ParamError`](super::ParamError) of
    /// values a finite domain's constructor refuses.
    pub fn build<R: ParamResolver + ?Sized>(self, resolver: &R) -> Result<ParamDomain, BuildError> {
        Ok(match self.0 {
            DomainRepr::Integer(domain) => ParamDomain::from(IntegerDomain::new(
                Sign::non_negative_if(domain.non_negative),
                ZeroInclusion::included_if(domain.zero_included),
            )),
            DomainRepr::IntervalInteger(domain) => ParamDomain::from(IntervalIntegerDomain::new(
                Inclusivity::inclusive_if(domain.prefer_inclusive),
                Sign::non_negative_if(domain.non_negative),
                ZeroInclusion::included_if(domain.zero_included),
            )),
            DomainRepr::Real(RealRepr {}) => ParamDomain::from(RealDomain),
            DomainRepr::Ordinal(domain) => ParamDomain::from(
                OrdinalDomain::new(build_values(domain.sorted_values, resolver)?)
                    .map_err(BuildError::invalid)?,
            ),
            DomainRepr::Categorical(domain) => ParamDomain::from(
                CategoricalDomain::new(build_values(domain.categories, resolver)?)
                    .map_err(BuildError::invalid)?,
            ),
            DomainRepr::Permutation(domain) => ParamDomain::from(
                PermutationDomain::new(build_values(domain.ordered_members, resolver)?)
                    .map_err(BuildError::invalid)?,
            ),
            DomainRepr::Custom(foreign) => ParamDomain::Custom(resolver.resolve(&foreign)?),
        })
    }
}

/// Serializes the shape of the [module documentation](self).
impl Serialize for ParamDomain {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ParamDomainData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing a
/// foreign part.
impl<'de> Deserialize<'de> for ParamDomain {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ParamDomainData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// The wire form of a [`Param`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Param", deny_unknown_fields)]
pub struct ParamData {
    domain: ParamDomainData,
    variable: Identifier,
    constraint_system: ConstraintSystemData,
}

impl ParamData {
    /// Return the wire form of `param`.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(param: &Param) -> Result<Self, ForeignError> {
        Ok(Self {
            domain: ParamDomainData::of(param.domain())?,
            variable: param.variable().clone(),
            constraint_system: ConstraintSystemData::of(param.constraint_system())?,
        })
    }

    /// Return the domain's wire form, the variable and the system's wire
    /// form, for a caller that builds the param itself.
    #[must_use]
    pub fn into_parts(self) -> (ParamDomainData, Identifier, ConstraintSystemData) {
        (self.domain, self.variable, self.constraint_system)
    }

    /// Return the param, its foreign parts resolved by `resolver`, built
    /// through [`Param::new`] under `context`.
    ///
    /// # Errors
    ///
    /// Returns what the parts' `build` returns, and
    /// [`BuildError::Invalid`] with the [`ParamError`](super::ParamError)
    /// [`Param::new`] returns.
    pub fn build<R: ParamResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<Param, BuildError> {
        let domain = self.domain.build(resolver)?;
        let system = self.constraint_system.build(resolver)?;
        Param::new(
            domain,
            self.variable,
            system.constraints().to_vec(),
            context,
        )
        .map_err(BuildError::invalid)
    }
}

/// Serializes as `{"domain", "variable", "constraint_system"}`.
impl Serialize for Param {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ParamData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"domain", "variable", "constraint_system"}`, refusing a
/// foreign part, under a context of a solver without backends.
impl<'de> Deserialize<'de> for Param {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new();
        ParamData::deserialize(deserializer)?
            .build(&NoForeign, &ParamContext::new(&solver))
            .map_err(de::Error::custom)
    }
}

/// The wire form of a [`ParamAssignment`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "ParamAssignment", deny_unknown_fields)]
pub struct ParamAssignmentData {
    param: ParamData,
    value: ValueData,
}

impl ParamAssignmentData {
    /// Return the wire form of `assignment`.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(assignment: &ParamAssignment) -> Result<Self, ForeignError> {
        Ok(Self {
            param: ParamData::of(assignment.param())?,
            value: ValueData::of_value(assignment.value())?,
        })
    }

    /// Return the param's wire form and the value's, for a caller that
    /// builds the assignment itself.
    #[must_use]
    pub fn into_parts(self) -> (ParamData, ValueData) {
        (self.param, self.value)
    }

    /// Return the assignment, its foreign parts resolved by `resolver`, its
    /// param built under `context`, and its value checked as
    /// [`ParamAssignment::restore`] checks it: refused when inadmissible or
    /// violating a constraint, and accepted when a constraint is left
    /// undecided.
    ///
    /// # Errors
    ///
    /// Returns what [`ParamData::build`] and [`ValueData::build`] return,
    /// and [`BuildError::Invalid`] with the
    /// [`AssignmentError`](super::AssignmentError) `restore` returns.
    pub fn build<R: ParamResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<ParamAssignment, BuildError> {
        let param = self.param.build(resolver, context)?;
        let value = self.value.build(resolver)?;
        ParamAssignment::restore(param, value, context).map_err(BuildError::invalid)
    }
}

/// Serializes as `{"param", "value"}`.
impl Serialize for ParamAssignment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ParamAssignmentData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// The observer an assignment's `Deserialize` checks its value under: a
/// constraint the solver without backends cannot evaluate, an equation,
/// counts as undecided, which [`ParamAssignment::restore`] accepts.
struct WithoutBackends;

impl ParamObserver for WithoutBackends {
    fn notify(&self, _event: &ParamEvent<'_>) {}

    fn is_undecidable(&self, error: &ConstraintError) -> bool {
        matches!(
            error,
            ConstraintError::Solve(SolveError::Backend { .. } | SolveError::NoCapableBackend(_))
        )
    }
}

/// Deserializes `{"param", "value"}`, refusing a foreign part and a value
/// [`ParamAssignment::restore`] refuses, under a context of a solver
/// without backends: an inadmissible value, or one a set constraint's
/// membership refuses, fails to decode, and an equation, which only a
/// simplifier evaluates, is left undecided.
impl<'de> Deserialize<'de> for ParamAssignment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new();
        ParamAssignmentData::deserialize(deserializer)?
            .build(
                &NoForeign,
                &ParamContext::new(&solver).with_observer(&WithoutBackends),
            )
            .map_err(de::Error::custom)
    }
}
