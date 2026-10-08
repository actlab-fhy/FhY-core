//! The wire forms of variables, alternatives, choices, spaces and
//! configurations, which hold the parts other crates define as
//! [`Foreign`] parts until a resolver builds them.
//!
//! A variable or an alternative inside a container serializes tagged:
//! `{"plain": {..}}` for this module's [`PlainVariable`] or
//! [`PlainAlternative`], and `{"foreign": {"type_id", "data"}}`, the part's
//! own [`to_foreign`](crate::foreign::ForeignPart::to_foreign), for any
//! other implementation. The shapes are:
//!
//! | Type | Shape |
//! |---|---|
//! | [`PlainVariable`] | `{"identifier", "param", "notes"}` |
//! | [`PlainAlternative`] | `{"identifier", "variables", "choices", "notes"}` |
//! | [`Choice`] | `{"identifier", "alternatives", "notes"}` |
//! | [`Space`] | `{"identifier", "variables", "choices", "conditions": [{"target", "when"}, ..], "forbidden": [{"when"}, ..], "notes"}` |
//! | [`Configuration`] | `{"space", "entries": [{"name", "value"}, ..]}` |
//! | [`ConfigurationKey`] | `{"entries": [..]}`, one per decision in canonical order: `{"inactive": {}}`, `{"unassigned": {}}`, `{"alternative": {"index"}}` or `{"value": ..}`, a value being `{"leaf": <value>}`, `{"bound": {"position"}}`, `{"tuple": [..]}` or `{"frozen_set": [..]}` |
//! | [`TraceKey`] | `{"steps": [{"kind", "decision", "domain", "coordinate"}, ..]}` |
//! | [`Measurement`] | `{"key", "status", "values": [{"objective": {"name", "direction"}, "value"}, ..], "notes"}`, the key `{"configuration": <configuration key>}` or `{"trace": <trace key>}`, the status `"ok"`, `{"infeasible": {"reason"}}`, `{"failed": {"reason"}}` or `"timeout"` |
//!
//! Params, constraint systems and values are in their own modules' forms,
//! conditions are written one per target in canonical order of the
//! targets, and entries in canonical order of their decisions.
//!
//! Decoding builds each value through its constructor, so a payload
//! outside a constructor's invariants fails with its error, and a
//! configuration is checked as
//! [`Configuration::new`] checks one, except that a variable's value is
//! checked as [`ParamAssignment::restore`](crate::param::ParamAssignment::restore)
//! checks it. [`VariableData`], [`AlternativeData`], [`ChoiceData`],
//! [`SpaceData`] and [`ConfigurationData`] read the same shapes and resolve
//! the foreign parts in `build`; the types' own `Deserialize` builds with
//! [`NoForeign`], which refuses them, under a
//! context of a solver without backends, and a configuration's under a
//! solver holding the
//! [`GroundSimplifier`], so its
//! conditions' equations evaluate.
//!
//! Choices nest through their alternatives, and a key's values through
//! their tuples and sets, and serde recurses once per level of either, so
//! decoding refuses, in every format, a choice whose choices nest more
//! than [`MAX_CHOICE_DEPTH`] levels, with "choice nesting exceeds 16
//! levels", and a key value inside more than [`MAX_VALUE_DEPTH`] nested
//! tuples or sets, with "value nesting exceeds 128 levels", before
//! reading their insides.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::sync::Arc;

use serde::de::{self, DeserializeSeed, Deserializer, VariantAccess};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};

use crate::constraint::wire::{ConstraintSystemData, MAX_VALUE_DEPTH, ValueData};
use crate::constraint::{CustomConstraint, OpaqueValue};
use crate::diagnostic::Note;
use crate::foreign::{BuildError, Foreign, ForeignError, NoForeign, Part, Resolve};
use crate::identifier::Identifier;
use crate::param::wire::{ParamData, ParamResolver};
use crate::param::{CustomDomain, ParamContext};
use crate::solver::{GroundSimplifier, Solver};

use super::alternative::{Alternative, PlainAlternative};
use super::choice::{Choice, MAX_CHOICE_DEPTH};
use super::configuration::{Configuration, ConfigurationKey, KeyEntry, KeyValue};
use super::error::{MeasurementError, RegistryError};
use super::measurement::{Measurement, MeasurementKey, MeasurementStatus, Objective};
use super::space::{Condition, Forbidden, Space};
use super::trace::TraceKey;
use super::variable::{PlainVariable, Variable};

/// The resolvers a search space's foreign parts need: a param's, and the
/// variables' and alternatives' other crates define.
pub trait SearchSpaceResolver:
    ParamResolver + Resolve<Part<dyn Variable>> + Resolve<Part<dyn Alternative>>
{
}

impl<R: ParamResolver + Resolve<Part<dyn Variable>> + Resolve<Part<dyn Alternative>> + ?Sized>
    SearchSpaceResolver for R
{
}

/// Builds a variable of one kind from its foreign part, with the resolver
/// of the parts inside it, such as its param's, and the context its param
/// is built under.
pub type VariableResolverFn = fn(
    &Foreign,
    &dyn SearchSpaceResolver,
    &ParamContext<'_>,
) -> Result<Part<dyn Variable>, ForeignError>;

/// Builds an alternative of one kind from its foreign part, with the
/// resolver of the parts inside it, such as its variables', and the
/// context their params are built under.
pub type AlternativeResolverFn = fn(
    &Foreign,
    &dyn SearchSpaceResolver,
    &ParamContext<'_>,
) -> Result<Part<dyn Alternative>, ForeignError>;

/// Builds an opaque value of one type from its foreign part.
pub type OpaqueValueResolverFn = fn(&Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError>;

/// Builds a custom constraint of one type from its foreign part.
pub type CustomConstraintResolverFn =
    fn(&Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError>;

/// Builds a custom domain of one type from its foreign part.
pub type CustomDomainResolverFn = fn(&Foreign) -> Result<Part<dyn CustomDomain>, ForeignError>;

/// The functions that build the foreign parts of a search space, by type
/// id, per family: variables, alternatives, opaque values, custom
/// constraints and custom domains.
///
/// Each crate that defines parts exports a function that registers them,
/// and a program composes the crates it links by registering each crate's
/// parts into one registry, or by [merging](Self::merge) their registries.
/// [`resolver`](Self::resolver) lends the registry, with the context
/// params are built under, as a [`SearchSpaceResolver`] for the wire
/// forms' `build`: a part whose type id the registry holds is built by its
/// function, which is handed that resolver for the parts inside it, and any
/// other part is refused as [`NoForeign`] refuses it.
///
/// Cloning copies the tables; the registry holds no other state.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::param::ParamContext;
/// use fhy_core::search_space::Space;
/// use fhy_core::search_space::wire::{ResolverRegistry, SpaceData};
/// use fhy_core::solver::Solver;
///
/// let space = Space::new(Identifier::new("s"), vec![], vec![], vec![], vec![])?;
/// let text = serde_json::to_string(&SpaceData::of(&space)?)?;
///
/// let registry = ResolverRegistry::new();
/// let solver = Solver::new();
/// let context = ParamContext::new(&solver);
/// let data: SpaceData = serde_json::from_str(&text)?;
/// let read = data.build(&registry.resolver(&context), &context)?;
/// assert_eq!(read, space);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct ResolverRegistry {
    variables: HashMap<Arc<str>, VariableResolverFn>,
    alternatives: HashMap<Arc<str>, AlternativeResolverFn>,
    opaque_values: HashMap<Arc<str>, OpaqueValueResolverFn>,
    custom_constraints: HashMap<Arc<str>, CustomConstraintResolverFn>,
    custom_domains: HashMap<Arc<str>, CustomDomainResolverFn>,
}

/// Register `resolve` under `type_id` in `table`.
///
/// # Errors
///
/// Returns [`RegistryError::RepeatedTypeId`] for a type id the table
/// holds.
fn register<F>(
    table: &mut HashMap<Arc<str>, F>,
    type_id: &str,
    resolve: F,
) -> Result<(), RegistryError> {
    match table.entry(Arc::from(type_id)) {
        Entry::Occupied(_) => Err(RegistryError::RepeatedTypeId {
            type_id: type_id.to_owned(),
        }),
        Entry::Vacant(slot) => {
            slot.insert(resolve);
            Ok(())
        }
    }
}

/// Move the functions of `other` into `table`.
///
/// # Errors
///
/// Returns [`RegistryError::RepeatedTypeId`] for the least type id both
/// hold; `table` is then unspecified.
fn absorb<F>(
    table: &mut HashMap<Arc<str>, F>,
    other: HashMap<Arc<str>, F>,
) -> Result<(), RegistryError> {
    let mut entries: Vec<(Arc<str>, F)> = other.into_iter().collect();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
    entries
        .into_iter()
        .try_for_each(|(type_id, resolve)| register(table, &type_id, resolve))
}

impl ResolverRegistry {
    /// Return the registry holding no function.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return this registry with `resolve` building the variables of kind
    /// `type_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::ReservedTypeId`] for
    /// [`PlainVariable::KIND`] and [`RegistryError::RepeatedTypeId`] for a
    /// kind already registered.
    pub fn with_variable_kind(
        mut self,
        type_id: &str,
        resolve: VariableResolverFn,
    ) -> Result<Self, RegistryError> {
        if type_id == PlainVariable::KIND {
            return Err(RegistryError::ReservedTypeId {
                type_id: type_id.to_owned(),
            });
        }
        register(&mut self.variables, type_id, resolve)?;
        Ok(self)
    }

    /// Return this registry with `resolve` building the alternatives of
    /// kind `type_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::ReservedTypeId`] for
    /// [`PlainAlternative::KIND`] and [`RegistryError::RepeatedTypeId`] for
    /// a kind already registered.
    pub fn with_alternative_kind(
        mut self,
        type_id: &str,
        resolve: AlternativeResolverFn,
    ) -> Result<Self, RegistryError> {
        if type_id == PlainAlternative::KIND {
            return Err(RegistryError::ReservedTypeId {
                type_id: type_id.to_owned(),
            });
        }
        register(&mut self.alternatives, type_id, resolve)?;
        Ok(self)
    }

    /// Return this registry with `resolve` building the opaque values of
    /// type `type_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::RepeatedTypeId`] for a type already
    /// registered.
    pub fn with_opaque_value(
        mut self,
        type_id: &str,
        resolve: OpaqueValueResolverFn,
    ) -> Result<Self, RegistryError> {
        register(&mut self.opaque_values, type_id, resolve)?;
        Ok(self)
    }

    /// Return this registry with `resolve` building the custom constraints
    /// of type `type_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::RepeatedTypeId`] for a type already
    /// registered.
    pub fn with_custom_constraint(
        mut self,
        type_id: &str,
        resolve: CustomConstraintResolverFn,
    ) -> Result<Self, RegistryError> {
        register(&mut self.custom_constraints, type_id, resolve)?;
        Ok(self)
    }

    /// Return this registry with `resolve` building the custom domains of
    /// type `type_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::RepeatedTypeId`] for a type already
    /// registered.
    pub fn with_custom_domain(
        mut self,
        type_id: &str,
        resolve: CustomDomainResolverFn,
    ) -> Result<Self, RegistryError> {
        register(&mut self.custom_domains, type_id, resolve)?;
        Ok(self)
    }

    /// Return the registry holding the functions of this one and of
    /// `other`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::RepeatedTypeId`] for the first type id,
    /// in a family, that both hold, in the order of the families above
    /// and then of the ids.
    pub fn merge(mut self, other: Self) -> Result<Self, RegistryError> {
        absorb(&mut self.variables, other.variables)?;
        absorb(&mut self.alternatives, other.alternatives)?;
        absorb(&mut self.opaque_values, other.opaque_values)?;
        absorb(&mut self.custom_constraints, other.custom_constraints)?;
        absorb(&mut self.custom_domains, other.custom_domains)?;
        Ok(self)
    }

    /// Return the resolver building the parts this registry holds, their
    /// params under `context`.
    #[must_use]
    pub const fn resolver<'a, 'c>(
        &'a self,
        context: &'a ParamContext<'c>,
    ) -> RegistryResolver<'a, 'c> {
        RegistryResolver {
            registry: self,
            context,
        }
    }
}

/// A [`ResolverRegistry`] lent with the context params are built under: a
/// [`SearchSpaceResolver`], from [`ResolverRegistry::resolver`].
#[derive(Debug, Clone, Copy)]
pub struct RegistryResolver<'a, 'c> {
    registry: &'a ResolverRegistry,
    context: &'a ParamContext<'c>,
}

/// Return the error for the part `foreign`, whose type id the registry
/// does not hold, as [`NoForeign`] refuses it.
fn unresolved(foreign: &Foreign) -> ForeignError {
    ForeignError::Unresolved {
        type_id: foreign.type_id().to_owned(),
    }
}

impl Resolve<Part<dyn Variable>> for RegistryResolver<'_, '_> {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
        let build = self
            .registry
            .variables
            .get(foreign.type_id())
            .ok_or_else(|| unresolved(foreign))?;
        build(foreign, self, self.context)
    }
}

impl Resolve<Part<dyn Alternative>> for RegistryResolver<'_, '_> {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
        let build = self
            .registry
            .alternatives
            .get(foreign.type_id())
            .ok_or_else(|| unresolved(foreign))?;
        build(foreign, self, self.context)
    }
}

impl Resolve<Part<dyn OpaqueValue>> for RegistryResolver<'_, '_> {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
        let build = self
            .registry
            .opaque_values
            .get(foreign.type_id())
            .ok_or_else(|| unresolved(foreign))?;
        build(foreign)
    }
}

impl Resolve<Part<dyn CustomConstraint>> for RegistryResolver<'_, '_> {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
        let build = self
            .registry
            .custom_constraints
            .get(foreign.type_id())
            .ok_or_else(|| unresolved(foreign))?;
        build(foreign)
    }
}

impl Resolve<Part<dyn CustomDomain>> for RegistryResolver<'_, '_> {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
        let build = self
            .registry
            .custom_domains
            .get(foreign.type_id())
            .ok_or_else(|| unresolved(foreign))?;
        build(foreign)
    }
}

/// The wire form of a variable, a [`PlainVariable`] or another
/// implementation's [`Foreign`] part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VariableData(VariableRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Variable", rename_all = "snake_case")]
enum VariableRepr {
    Plain(PlainVariableRepr),
    Foreign(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "PlainVariable", deny_unknown_fields)]
struct PlainVariableRepr {
    identifier: Identifier,
    param: ParamData,
    notes: Vec<Note>,
}

impl PlainVariableRepr {
    /// Return the fields of `variable`.
    fn of(variable: &PlainVariable) -> Result<Self, ForeignError> {
        Ok(Self {
            identifier: variable.name().clone(),
            param: ParamData::of(variable.param())?,
            notes: variable.notes().to_vec(),
        })
    }

    /// Return the plain variable of these fields.
    fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<PlainVariable, BuildError> {
        let param = self.param.build(resolver, context)?;
        Ok(PlainVariable::new(self.identifier, param).with_notes(self.notes))
    }
}

impl VariableData {
    /// Return the wire form of `variable`: a plain variable's fields, or
    /// another implementation's foreign part.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(variable: &Part<dyn Variable>) -> Result<Self, ForeignError> {
        let part = variable.get();
        Ok(Self(match part.as_any().downcast_ref::<PlainVariable>() {
            Some(plain) => VariableRepr::Plain(PlainVariableRepr::of(plain)?),
            None => VariableRepr::Foreign(part.to_foreign()?),
        }))
    }

    /// Return the foreign part of another implementation's variable, or
    /// `None` for a plain one.
    #[must_use]
    pub const fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            VariableRepr::Foreign(foreign) => Some(foreign),
            VariableRepr::Plain(_) => None,
        }
    }

    /// Return the variable, its foreign parts resolved by `resolver` and
    /// its param built under `context`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses, and
    /// [`BuildError::Invalid`] with the error of a constructor that
    /// refuses the data.
    pub fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<Part<dyn Variable>, BuildError> {
        Ok(match self.0 {
            VariableRepr::Plain(repr) => Part::new(repr.build(resolver, context)?),
            VariableRepr::Foreign(foreign) => resolver.resolve(&foreign)?,
        })
    }
}

/// The wire form of an alternative, a [`PlainAlternative`] or another
/// implementation's [`Foreign`] part.
///
/// Decoding refuses sub-choices nested more than [`MAX_CHOICE_DEPTH`]
/// levels.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct AlternativeData(AlternativeRepr);

/// Decodes through [`AlternativeSeed`], so its sub-choices' nesting is
/// bounded.
#[derive(Debug, Clone, Serialize)]
#[serde(rename = "Alternative", rename_all = "snake_case")]
enum AlternativeRepr {
    Plain(PlainAlternativeRepr),
    Foreign(Foreign),
}

/// Decodes through [`PlainAlternativeSeed`], so its sub-choices' nesting
/// is bounded.
#[derive(Debug, Clone, Serialize)]
#[serde(rename = "PlainAlternative")]
struct PlainAlternativeRepr {
    identifier: Identifier,
    variables: Vec<VariableData>,
    choices: Vec<ChoiceData>,
    notes: Vec<Note>,
}

impl PlainAlternativeRepr {
    /// Return the fields of `alternative`.
    fn of(alternative: &PlainAlternative) -> Result<Self, ForeignError> {
        Ok(Self {
            identifier: alternative.name().clone(),
            variables: alternative
                .variables()
                .iter()
                .map(VariableData::of)
                .collect::<Result<_, _>>()?,
            choices: alternative
                .choices()
                .iter()
                .map(ChoiceData::of)
                .collect::<Result<_, _>>()?,
            notes: alternative.notes().to_vec(),
        })
    }

    /// Return the plain alternative of these fields.
    fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<PlainAlternative, BuildError> {
        let variables = self
            .variables
            .into_iter()
            .map(|variable| variable.build(resolver, context))
            .collect::<Result<Vec<_>, _>>()?;
        let choices = self
            .choices
            .into_iter()
            .map(|choice| choice.build(resolver, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PlainAlternative::new(self.identifier, variables, choices)
            .map_err(BuildError::invalid)?
            .with_notes(self.notes))
    }
}

impl AlternativeData {
    /// Return the wire form of `alternative`: a plain alternative's fields,
    /// or another implementation's foreign part.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(alternative: &Part<dyn Alternative>) -> Result<Self, ForeignError> {
        let part = alternative.get();
        Ok(Self(
            match part.as_any().downcast_ref::<PlainAlternative>() {
                Some(plain) => AlternativeRepr::Plain(PlainAlternativeRepr::of(plain)?),
                None => AlternativeRepr::Foreign(part.to_foreign()?),
            },
        ))
    }

    /// Return the foreign part of another implementation's alternative, or
    /// `None` for a plain one.
    #[must_use]
    pub const fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            AlternativeRepr::Foreign(foreign) => Some(foreign),
            AlternativeRepr::Plain(_) => None,
        }
    }

    /// Return the alternative, its foreign parts resolved by `resolver` and
    /// its params built under `context`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses, and
    /// [`BuildError::Invalid`] with the error of a constructor that
    /// refuses the data.
    pub fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<Part<dyn Alternative>, BuildError> {
        Ok(match self.0 {
            AlternativeRepr::Plain(repr) => Part::new(repr.build(resolver, context)?),
            AlternativeRepr::Foreign(foreign) => resolver.resolve(&foreign)?,
        })
    }
}

/// The wire form of a [`Choice`], its foreign parts unresolved.
///
/// Decoding refuses a choice whose choices nest more than
/// [`MAX_CHOICE_DEPTH`] levels, itself included.
#[derive(Debug, Clone, Serialize)]
#[serde(rename = "Choice")]
pub struct ChoiceData {
    identifier: Identifier,
    alternatives: Vec<AlternativeData>,
    notes: Vec<Note>,
}

impl ChoiceData {
    /// Return the wire form of `choice`.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(choice: &Choice) -> Result<Self, ForeignError> {
        Ok(Self {
            identifier: choice.name().clone(),
            alternatives: choice
                .alternatives()
                .iter()
                .map(AlternativeData::of)
                .collect::<Result<_, _>>()?,
            notes: choice.notes().to_vec(),
        })
    }

    /// Return the choice, its foreign parts resolved by `resolver` and its
    /// params built under `context`, built through [`Choice::new`].
    ///
    /// # Errors
    ///
    /// Returns what the parts' `build` returns, and [`BuildError::Invalid`]
    /// with the [`SpaceError`](super::SpaceError) [`Choice::new`] returns.
    pub fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<Choice, BuildError> {
        let alternatives = self
            .alternatives
            .into_iter()
            .map(|alternative| alternative.build(resolver, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Choice::new(self.identifier, alternatives)
            .map_err(BuildError::invalid)?
            .with_notes(self.notes))
    }
}

/// The wire form of a [`Space`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Space", deny_unknown_fields)]
pub struct SpaceData {
    identifier: Identifier,
    variables: Vec<VariableData>,
    choices: Vec<ChoiceData>,
    conditions: Vec<ConditionRepr>,
    forbidden: Vec<ForbiddenRepr>,
    notes: Vec<Note>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Condition", deny_unknown_fields)]
struct ConditionRepr {
    target: Identifier,
    when: ConstraintSystemData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Forbidden", deny_unknown_fields)]
struct ForbiddenRepr {
    when: ConstraintSystemData,
}

impl SpaceData {
    /// Return the wire form of `space`.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(space: &Space) -> Result<Self, ForeignError> {
        Ok(Self {
            identifier: space.name().clone(),
            variables: space
                .variables()
                .iter()
                .map(VariableData::of)
                .collect::<Result<_, _>>()?,
            choices: space
                .choices()
                .iter()
                .map(ChoiceData::of)
                .collect::<Result<_, _>>()?,
            conditions: space
                .conditions()
                .iter()
                .map(|condition| {
                    Ok(ConditionRepr {
                        target: condition.target().clone(),
                        when: ConstraintSystemData::of(condition.when())?,
                    })
                })
                .collect::<Result<_, ForeignError>>()?,
            forbidden: space
                .forbidden()
                .iter()
                .map(|clause| {
                    Ok(ForbiddenRepr {
                        when: ConstraintSystemData::of(clause.when())?,
                    })
                })
                .collect::<Result<_, ForeignError>>()?,
            notes: space.notes().to_vec(),
        })
    }

    /// Return the space, its foreign parts resolved by `resolver` and its
    /// params built under `context`, built through [`Space::new`].
    ///
    /// # Errors
    ///
    /// Returns what the parts' `build` returns, and [`BuildError::Invalid`]
    /// with the [`SpaceError`](super::SpaceError) a constructor returns.
    pub fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<Space, BuildError> {
        let variables = self
            .variables
            .into_iter()
            .map(|variable| variable.build(resolver, context))
            .collect::<Result<Vec<_>, _>>()?;
        let choices = self
            .choices
            .into_iter()
            .map(|choice| choice.build(resolver, context))
            .collect::<Result<Vec<_>, _>>()?;
        let conditions = self
            .conditions
            .into_iter()
            .map(|condition| {
                Ok(Condition::new(
                    condition.target,
                    condition.when.build(resolver)?,
                ))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let forbidden = self
            .forbidden
            .into_iter()
            .map(|clause| Ok(Forbidden::new(clause.when.build(resolver)?)))
            .collect::<Result<Vec<_>, BuildError>>()?;
        Ok(
            Space::new(self.identifier, variables, choices, conditions, forbidden)
                .map_err(BuildError::invalid)?
                .with_notes(self.notes),
        )
    }
}

/// The wire form of a [`Configuration`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Configuration", deny_unknown_fields)]
pub struct ConfigurationData {
    space: SpaceData,
    entries: Vec<EntryRepr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Entry", deny_unknown_fields)]
struct EntryRepr {
    name: Identifier,
    value: ValueData,
}

impl ConfigurationData {
    /// Return the wire form of `configuration`.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(configuration: &Configuration) -> Result<Self, ForeignError> {
        Ok(Self {
            space: SpaceData::of(configuration.space())?,
            entries: configuration
                .entries()
                .map(|(name, value)| {
                    Ok(EntryRepr {
                        name: name.clone(),
                        value: ValueData::of_value(value)?,
                    })
                })
                .collect::<Result<_, ForeignError>>()?,
        })
    }

    /// Return the configuration, its foreign parts resolved by `resolver`,
    /// its space built under `context`, and its entries checked under
    /// `context` as [`Configuration::new`] checks them, except that a
    /// variable's value is checked as
    /// [`ParamAssignment::restore`](crate::param::ParamAssignment::restore)
    /// checks it.
    ///
    /// # Errors
    ///
    /// Returns what [`SpaceData::build`] and the values' `build` return,
    /// and [`BuildError::Invalid`] with the
    /// [`ConfigurationErrors`](super::ConfigurationErrors) the check
    /// returns.
    pub fn build<R: SearchSpaceResolver + ?Sized>(
        self,
        resolver: &R,
        context: &ParamContext<'_>,
    ) -> Result<Configuration, BuildError> {
        let space = self.space.build(resolver, context)?;
        let entries = self
            .entries
            .into_iter()
            .map(|entry| Ok((entry.name, entry.value.build(resolver)?)))
            .collect::<Result<Vec<_>, BuildError>>()?;
        Configuration::restore(&space, entries, context).map_err(BuildError::invalid)
    }
}

/// The names of [`AlternativeRepr`]'s variants, in declaration order.
const ALTERNATIVE_VARIANTS: &[&str] = &["plain", "foreign"];

/// The variant tags of [`AlternativeRepr`].
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AlternativeTag {
    Plain,
    Foreign,
}

/// The names of [`ChoiceData`]'s fields, in declaration order.
const CHOICE_FIELDS: &[&str] = &["identifier", "alternatives", "notes"];

/// The fields of [`ChoiceData`].
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ChoiceField {
    Identifier,
    Alternatives,
    Notes,
}

/// The names of [`PlainAlternativeRepr`]'s fields, in declaration order.
const PLAIN_ALTERNATIVE_FIELDS: &[&str] = &["identifier", "variables", "choices", "notes"];

/// The fields of [`PlainAlternativeRepr`].
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum PlainAlternativeField {
    Identifier,
    Variables,
    Choices,
    Notes,
}

/// Decodes a sequence, each element through the seed it holds.
#[derive(Clone, Copy)]
struct SequenceSeed<S>(S);

impl<'de, S: DeserializeSeed<'de> + Copy> DeserializeSeed<'de> for SequenceSeed<S> {
    type Value = Vec<S::Value>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, S: DeserializeSeed<'de> + Copy> de::Visitor<'de> for SequenceSeed<S> {
    type Value = Vec<S::Value>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        // A size hint comes from the input, so it only bounds the first
        // allocation.
        let mut elements = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
        while let Some(element) = seq.next_element_seed(self.0)? {
            elements.push(element);
        }
        Ok(elements)
    }
}

/// Decodes a [`ChoiceData`] at level `depth` of choices, counting from 1,
/// refusing one past [`MAX_CHOICE_DEPTH`].
#[derive(Clone, Copy)]
struct ChoiceSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for ChoiceSeed {
    type Value = ChoiceData;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<ChoiceData, D::Error> {
        if self.depth > MAX_CHOICE_DEPTH {
            return Err(de::Error::custom(format_args!(
                "choice nesting exceeds {MAX_CHOICE_DEPTH} levels"
            )));
        }
        deserializer.deserialize_struct("Choice", CHOICE_FIELDS, self)
    }
}

impl<'de> de::Visitor<'de> for ChoiceSeed {
    type Value = ChoiceData;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("struct Choice")
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<ChoiceData, A::Error> {
        let expected = &"struct Choice with 3 elements";
        let identifier = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, expected))?;
        let alternatives = seq
            .next_element_seed(SequenceSeed(AlternativeSeed { depth: self.depth }))?
            .ok_or_else(|| de::Error::invalid_length(1, expected))?;
        let notes = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(2, expected))?;
        Ok(ChoiceData {
            identifier,
            alternatives,
            notes,
        })
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<ChoiceData, A::Error> {
        let (mut identifier, mut alternatives, mut notes) = (None, None, None);
        while let Some(field) = map.next_key()? {
            match field {
                ChoiceField::Identifier => {
                    if identifier.is_some() {
                        return Err(de::Error::duplicate_field("identifier"));
                    }
                    identifier = Some(map.next_value()?);
                }
                ChoiceField::Alternatives => {
                    if alternatives.is_some() {
                        return Err(de::Error::duplicate_field("alternatives"));
                    }
                    alternatives =
                        Some(map.next_value_seed(SequenceSeed(AlternativeSeed {
                            depth: self.depth,
                        }))?);
                }
                ChoiceField::Notes => {
                    if notes.is_some() {
                        return Err(de::Error::duplicate_field("notes"));
                    }
                    notes = Some(map.next_value()?);
                }
            }
        }
        Ok(ChoiceData {
            identifier: identifier.ok_or_else(|| de::Error::missing_field("identifier"))?,
            alternatives: alternatives.ok_or_else(|| de::Error::missing_field("alternatives"))?,
            notes: notes.ok_or_else(|| de::Error::missing_field("notes"))?,
        })
    }
}

/// Decodes `{"identifier", "alternatives", "notes"}`, refusing choices
/// nested more than [`MAX_CHOICE_DEPTH`] levels.
impl<'de> Deserialize<'de> for ChoiceData {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ChoiceSeed { depth: 1 }.deserialize(deserializer)
    }
}

/// Decodes an [`AlternativeData`] of a choice at level `depth` of
/// choices, 0 for an alternative on its own.
#[derive(Clone, Copy)]
struct AlternativeSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for AlternativeSeed {
    type Value = AlternativeData;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<AlternativeData, D::Error> {
        deserializer.deserialize_enum("Alternative", ALTERNATIVE_VARIANTS, self)
    }
}

impl<'de> de::Visitor<'de> for AlternativeSeed {
    type Value = AlternativeData;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("enum Alternative")
    }

    fn visit_enum<A: de::EnumAccess<'de>>(self, data: A) -> Result<AlternativeData, A::Error> {
        let (tag, variant) = data.variant::<AlternativeTag>()?;
        Ok(AlternativeData(match tag {
            AlternativeTag::Plain => AlternativeRepr::Plain(
                variant.newtype_variant_seed(PlainAlternativeSeed { depth: self.depth })?,
            ),
            AlternativeTag::Foreign => AlternativeRepr::Foreign(variant.newtype_variant()?),
        }))
    }
}

/// Decodes the tagged alternative, refusing sub-choices nested more than
/// [`MAX_CHOICE_DEPTH`] levels.
impl<'de> Deserialize<'de> for AlternativeData {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        AlternativeSeed { depth: 0 }.deserialize(deserializer)
    }
}

/// Decodes a [`PlainAlternativeRepr`] of a choice at level `depth` of
/// choices, 0 for an alternative on its own: its sub-choices are at the
/// next level.
#[derive(Clone, Copy)]
struct PlainAlternativeSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for PlainAlternativeSeed {
    type Value = PlainAlternativeRepr;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<PlainAlternativeRepr, D::Error> {
        deserializer.deserialize_struct("PlainAlternative", PLAIN_ALTERNATIVE_FIELDS, self)
    }
}

impl<'de> de::Visitor<'de> for PlainAlternativeSeed {
    type Value = PlainAlternativeRepr;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("struct PlainAlternative")
    }

    fn visit_seq<A: de::SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> Result<PlainAlternativeRepr, A::Error> {
        let expected = &"struct PlainAlternative with 4 elements";
        let identifier = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(0, expected))?;
        let variables = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(1, expected))?;
        let choices = seq
            .next_element_seed(SequenceSeed(ChoiceSeed {
                depth: self.depth + 1,
            }))?
            .ok_or_else(|| de::Error::invalid_length(2, expected))?;
        let notes = seq
            .next_element()?
            .ok_or_else(|| de::Error::invalid_length(3, expected))?;
        Ok(PlainAlternativeRepr {
            identifier,
            variables,
            choices,
            notes,
        })
    }

    fn visit_map<A: de::MapAccess<'de>>(
        self,
        mut map: A,
    ) -> Result<PlainAlternativeRepr, A::Error> {
        let (mut identifier, mut variables, mut choices, mut notes) = (None, None, None, None);
        while let Some(field) = map.next_key()? {
            match field {
                PlainAlternativeField::Identifier => {
                    if identifier.is_some() {
                        return Err(de::Error::duplicate_field("identifier"));
                    }
                    identifier = Some(map.next_value()?);
                }
                PlainAlternativeField::Variables => {
                    if variables.is_some() {
                        return Err(de::Error::duplicate_field("variables"));
                    }
                    variables = Some(map.next_value()?);
                }
                PlainAlternativeField::Choices => {
                    if choices.is_some() {
                        return Err(de::Error::duplicate_field("choices"));
                    }
                    choices = Some(map.next_value_seed(SequenceSeed(ChoiceSeed {
                        depth: self.depth + 1,
                    }))?);
                }
                PlainAlternativeField::Notes => {
                    if notes.is_some() {
                        return Err(de::Error::duplicate_field("notes"));
                    }
                    notes = Some(map.next_value()?);
                }
            }
        }
        Ok(PlainAlternativeRepr {
            identifier: identifier.ok_or_else(|| de::Error::missing_field("identifier"))?,
            variables: variables.ok_or_else(|| de::Error::missing_field("variables"))?,
            choices: choices.ok_or_else(|| de::Error::missing_field("choices"))?,
            notes: notes.ok_or_else(|| de::Error::missing_field("notes"))?,
        })
    }
}

/// Decodes `{"identifier", "variables", "choices", "notes"}`, refusing
/// sub-choices nested more than [`MAX_CHOICE_DEPTH`] levels.
impl<'de> Deserialize<'de> for PlainAlternativeRepr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        PlainAlternativeSeed { depth: 0 }.deserialize(deserializer)
    }
}

/// Serializes as `{"identifier", "param", "notes"}`.
impl Serialize for PlainVariable {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        PlainVariableRepr::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"identifier", "param", "notes"}`, refusing a foreign
/// part, under a context of a solver without backends.
impl<'de> Deserialize<'de> for PlainVariable {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new();
        PlainVariableRepr::deserialize(deserializer)?
            .build(&NoForeign, &ParamContext::new(&solver))
            .map_err(de::Error::custom)
    }
}

/// Serializes as `{"identifier", "variables", "choices", "notes"}`.
impl Serialize for PlainAlternative {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        PlainAlternativeRepr::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"identifier", "variables", "choices", "notes"}`,
/// refusing a foreign part, under a context of a solver without backends.
impl<'de> Deserialize<'de> for PlainAlternative {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new();
        PlainAlternativeRepr::deserialize(deserializer)?
            .build(&NoForeign, &ParamContext::new(&solver))
            .map_err(de::Error::custom)
    }
}

/// Serializes as `{"identifier", "alternatives", "notes"}`.
impl Serialize for Choice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ChoiceData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"identifier", "alternatives", "notes"}`, refusing a
/// foreign part, under a context of a solver without backends.
impl<'de> Deserialize<'de> for Choice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new();
        ChoiceData::deserialize(deserializer)?
            .build(&NoForeign, &ParamContext::new(&solver))
            .map_err(de::Error::custom)
    }
}

/// Serializes as `{"identifier", "variables", "choices", "conditions",
/// "forbidden", "notes"}`.
impl Serialize for Space {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SpaceData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the space's shape, refusing a foreign part, under a
/// context of a solver without backends.
impl<'de> Deserialize<'de> for Space {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new();
        SpaceData::deserialize(deserializer)?
            .build(&NoForeign, &ParamContext::new(&solver))
            .map_err(de::Error::custom)
    }
}

/// Serializes as `{"space", "entries"}`.
impl Serialize for Configuration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ConfigurationData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"space", "entries"}`, refusing a foreign part, under a
/// context of a solver holding the
/// [`GroundSimplifier`].
impl<'de> Deserialize<'de> for Configuration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let solver = Solver::new().with_simplifier(GroundSimplifier::new());
        ConfigurationData::deserialize(deserializer)?
            .build(&NoForeign, &ParamContext::new(&solver))
            .map_err(de::Error::custom)
    }
}

/// The wire form of a [`ConfigurationKey`], its opaque values unresolved.
///
/// A key is self-contained: every identifier its space binds is written as
/// its position among the space's names, and any other value as the value
/// itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "ConfigurationKey", deny_unknown_fields)]
pub struct ConfigurationKeyData {
    entries: Vec<KeyEntryRepr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "KeyEntry", rename_all = "snake_case")]
enum KeyEntryRepr {
    Inactive {},
    Unassigned {},
    Alternative { index: usize },
    Value(KeyValueRepr),
}

/// Decodes through [`KeyValueSeed`], so its nesting is bounded.
#[derive(Debug, Clone, Serialize)]
#[serde(rename = "KeyValue", rename_all = "snake_case")]
enum KeyValueRepr {
    Leaf(ValueData),
    Bound { position: usize },
    Tuple(Vec<KeyValueRepr>),
    FrozenSet(Vec<KeyValueRepr>),
}

/// The names of [`KeyValueRepr`]'s variants, in declaration order.
const KEY_VALUE_VARIANTS: &[&str] = &["leaf", "bound", "tuple", "frozen_set"];

/// The variant tags of [`KeyValueRepr`].
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum KeyValueTag {
    Leaf,
    Bound,
    Tuple,
    FrozenSet,
}

/// The fields of [`KeyValueRepr::Bound`], decoded as the variant's one
/// field: a struct variant and a newtype variant holding the struct read
/// alike in every format.
#[derive(Deserialize)]
#[serde(rename = "Bound")]
struct BoundRepr {
    position: usize,
}

/// Decodes a [`KeyValueRepr`] inside `depth` tuples or sets, refusing one
/// inside more than [`MAX_VALUE_DEPTH`].
#[derive(Clone, Copy)]
struct KeyValueSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for KeyValueSeed {
    type Value = KeyValueRepr;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<KeyValueRepr, D::Error> {
        if self.depth > MAX_VALUE_DEPTH {
            return Err(de::Error::custom(format_args!(
                "value nesting exceeds {MAX_VALUE_DEPTH} levels"
            )));
        }
        deserializer.deserialize_enum("KeyValue", KEY_VALUE_VARIANTS, self)
    }
}

impl<'de> de::Visitor<'de> for KeyValueSeed {
    type Value = KeyValueRepr;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("enum KeyValue")
    }

    fn visit_enum<A: de::EnumAccess<'de>>(self, data: A) -> Result<KeyValueRepr, A::Error> {
        let (tag, variant) = data.variant::<KeyValueTag>()?;
        let elements = SequenceSeed(Self {
            depth: self.depth + 1,
        });
        Ok(match tag {
            KeyValueTag::Leaf => KeyValueRepr::Leaf(variant.newtype_variant()?),
            KeyValueTag::Bound => KeyValueRepr::Bound {
                position: variant.newtype_variant::<BoundRepr>()?.position,
            },
            KeyValueTag::Tuple => KeyValueRepr::Tuple(variant.newtype_variant_seed(elements)?),
            KeyValueTag::FrozenSet => {
                KeyValueRepr::FrozenSet(variant.newtype_variant_seed(elements)?)
            }
        })
    }
}

/// Decodes the key value's shape, refusing one nested deeper than
/// [`MAX_VALUE_DEPTH`].
impl<'de> Deserialize<'de> for KeyValueRepr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        KeyValueSeed { depth: 0 }.deserialize(deserializer)
    }
}

impl ConfigurationKeyData {
    /// Return the wire form of `key`.
    ///
    /// # Errors
    ///
    /// Returns the error of an opaque value that cannot give its foreign
    /// form.
    pub fn of(key: &ConfigurationKey) -> Result<Self, ForeignError> {
        Ok(Self {
            entries: key
                .entries()
                .iter()
                .map(KeyEntryRepr::of)
                .collect::<Result<_, _>>()?,
        })
    }

    /// Return the key, its opaque values resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for an opaque value `resolver`
    /// refuses.
    pub fn build<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<ConfigurationKey, BuildError> {
        let entries = self
            .entries
            .into_iter()
            .map(|entry| entry.build(resolver))
            .collect::<Result<_, _>>()?;
        Ok(ConfigurationKey::from_entries(entries))
    }
}

impl KeyEntryRepr {
    /// Return the wire form of `entry`.
    fn of(entry: &KeyEntry) -> Result<Self, ForeignError> {
        Ok(match entry {
            KeyEntry::Inactive => Self::Inactive {},
            KeyEntry::Unassigned => Self::Unassigned {},
            KeyEntry::Alternative(index) => Self::Alternative { index: *index },
            KeyEntry::Value(value) => Self::Value(KeyValueRepr::of(value)?),
        })
    }

    /// Return the entry, its opaque values resolved by `resolver`.
    fn build<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<KeyEntry, BuildError> {
        Ok(match self {
            Self::Inactive {} => KeyEntry::Inactive,
            Self::Unassigned {} => KeyEntry::Unassigned,
            Self::Alternative { index } => KeyEntry::Alternative(index),
            Self::Value(value) => KeyEntry::Value(value.build(resolver)?),
        })
    }
}

impl KeyValueRepr {
    /// Return the wire form of `value`.
    fn of(value: &KeyValue) -> Result<Self, ForeignError> {
        Ok(match value {
            KeyValue::Leaf(value) => Self::Leaf(ValueData::of_value(value)?),
            KeyValue::Bound(position) => Self::Bound {
                position: *position,
            },
            KeyValue::Tuple(values) => {
                Self::Tuple(values.iter().map(Self::of).collect::<Result<_, _>>()?)
            }
            KeyValue::FrozenSet(values) => {
                Self::FrozenSet(values.iter().map(Self::of).collect::<Result<_, _>>()?)
            }
        })
    }

    /// Return the value, its opaque values resolved by `resolver`.
    fn build<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<KeyValue, BuildError> {
        Ok(match self {
            Self::Leaf(value) => KeyValue::Leaf(value.build(resolver)?),
            Self::Bound { position } => KeyValue::Bound(position),
            Self::Tuple(values) => KeyValue::Tuple(
                values
                    .into_iter()
                    .map(|value| value.build(resolver))
                    .collect::<Result<_, _>>()?,
            ),
            Self::FrozenSet(values) => KeyValue::FrozenSet(
                values
                    .into_iter()
                    .map(|value| value.build(resolver))
                    .collect::<Result<_, _>>()?,
            ),
        })
    }
}

/// Serializes as `{"entries"}`.
impl Serialize for ConfigurationKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ConfigurationKeyData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"entries"}`, refusing an opaque value.
impl<'de> Deserialize<'de> for ConfigurationKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ConfigurationKeyData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// The wire form of a [`Measurement`], its key's opaque values unresolved:
/// `{"key", "status", "values": [{"objective", "value"}, ..], "notes"}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Measurement", deny_unknown_fields)]
pub struct MeasurementData {
    key: MeasurementKeyRepr,
    status: MeasurementStatus,
    values: Vec<MeasuredValueRepr>,
    notes: Vec<Note>,
}

/// The wire form of a [`MeasurementKey`], a configuration key's opaque
/// values unresolved: `{"configuration": ..}` or `{"trace": ..}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "MeasurementKey", rename_all = "snake_case")]
enum MeasurementKeyRepr {
    Configuration(ConfigurationKeyData),
    Trace(TraceKey),
}

impl MeasurementKeyRepr {
    /// Return the wire form of `key`.
    ///
    /// # Errors
    ///
    /// Returns the error of an opaque value of a configuration key that
    /// cannot give its foreign form.
    fn of(key: &MeasurementKey) -> Result<Self, ForeignError> {
        Ok(match key {
            MeasurementKey::Configuration(key) => {
                Self::Configuration(ConfigurationKeyData::of(key)?)
            }
            MeasurementKey::Trace(key) => Self::Trace(key.clone()),
        })
    }

    /// Return the key, a configuration key's opaque values resolved by
    /// `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for an opaque value `resolver`
    /// refuses.
    fn build<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<MeasurementKey, BuildError> {
        Ok(match self {
            Self::Configuration(key) => MeasurementKey::Configuration(key.build(resolver)?),
            Self::Trace(key) => MeasurementKey::Trace(key),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "MeasuredValue", deny_unknown_fields)]
struct MeasuredValueRepr {
    objective: Objective,
    value: f64,
}

impl MeasurementData {
    /// Return the wire form of `measurement`.
    ///
    /// # Errors
    ///
    /// Returns the error of an opaque value of the key that cannot give
    /// its foreign form.
    pub fn of(measurement: &Measurement) -> Result<Self, ForeignError> {
        Ok(Self {
            key: MeasurementKeyRepr::of(measurement.key())?,
            status: measurement.status().clone(),
            values: measurement
                .values()
                .iter()
                .map(|(objective, value)| MeasuredValueRepr {
                    objective: objective.clone(),
                    value: *value,
                })
                .collect(),
            notes: measurement.notes().to_vec(),
        })
    }

    /// Return the measurement, its key's opaque values resolved by
    /// `resolver`, checked as its constructor checks it.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for an opaque value `resolver`
    /// refuses, and [`BuildError::Invalid`] with the
    /// [`MeasurementError`] of values a constructor refuses, or
    /// [`UnexpectedValues`](MeasurementError::UnexpectedValues) for values
    /// held by a measurement that did not succeed.
    pub fn build<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<Measurement, BuildError> {
        let key = self.key.build(resolver)?;
        let values: Vec<(Objective, f64)> = self
            .values
            .into_iter()
            .map(|entry| (entry.objective, entry.value))
            .collect();
        let measurement = match self.status {
            MeasurementStatus::Ok => Measurement::ok(key, values).map_err(BuildError::invalid)?,
            _ if !values.is_empty() => {
                return Err(BuildError::invalid(MeasurementError::UnexpectedValues));
            }
            MeasurementStatus::Infeasible { reason } => Measurement::infeasible(key, reason),
            MeasurementStatus::Failed { reason } => Measurement::failed(key, reason),
            MeasurementStatus::Timeout => Measurement::timeout(key),
        };
        Ok(measurement.with_notes(self.notes))
    }
}

/// Serializes `{"key", "status", "values", "notes"}`.
impl Serialize for Measurement {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MeasurementData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"key", "status", "values", "notes"}`, refusing an opaque
/// value in the key.
impl<'de> Deserialize<'de> for Measurement {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        MeasurementData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}
