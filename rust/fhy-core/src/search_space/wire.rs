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

use serde::de::{self, Deserializer};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};

use crate::constraint::OpaqueValue;
use crate::constraint::wire::{ConstraintSystemData, ValueData};
use crate::diagnostic::Note;
use crate::foreign::{BuildError, Foreign, ForeignError, NoForeign, Part, Resolve};
use crate::identifier::Identifier;
use crate::param::ParamContext;
use crate::param::wire::{ParamData, ParamResolver};
use crate::solver::{GroundSimplifier, Solver};

use super::alternative::{Alternative, PlainAlternative};
use super::choice::Choice;
use super::configuration::{Configuration, ConfigurationKey, KeyEntry, KeyValue};
use super::space::{Condition, Forbidden, Space};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AlternativeData(AlternativeRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Alternative", rename_all = "snake_case")]
enum AlternativeRepr {
    Plain(PlainAlternativeRepr),
    Foreign(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "PlainAlternative", deny_unknown_fields)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Choice", deny_unknown_fields)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "KeyValue", rename_all = "snake_case")]
enum KeyValueRepr {
    Leaf(ValueData),
    Bound { position: usize },
    Tuple(Vec<KeyValueRepr>),
    FrozenSet(Vec<KeyValueRepr>),
}

impl ConfigurationKeyData {
    /// Return the wire form of `key`.
    ///
    /// # Errors
    ///
    /// Returns the error of an opaque value that cannot give its foreign
    /// form.
    #[expect(
        clippy::todo,
        reason = "interface stub; the body is todo!() until implementation"
    )]
    pub fn of(key: &ConfigurationKey) -> Result<Self, ForeignError> {
        let _: (&[KeyEntry], Option<&KeyValue>) = (key.entries(), None);
        todo!()
    }

    /// Return the key, its opaque values resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for an opaque value `resolver`
    /// refuses.
    #[expect(
        unused_variables,
        clippy::todo,
        reason = "interface stub; the body is todo!() until implementation"
    )]
    pub fn build<R: Resolve<Part<dyn OpaqueValue>> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<ConfigurationKey, BuildError> {
        let _ = ConfigurationKey::from_entries;
        todo!()
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
