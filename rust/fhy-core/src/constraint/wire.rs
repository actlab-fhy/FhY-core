//! The wire forms of values, constraints and systems, which hold opaque
//! values and custom constraints as [`Foreign`] parts until a resolver
//! builds them.
//!
//! A [`Value`] or [`Member`] serializes as `{"bool": b}`, `{"int": "12"}`
//! (the decimal digits), `{"float": "1.5"}` (the text `{}` writes, `"NaN"`
//! and the infinities included), `{"decimal": "1.5"}`, `{"str": "a"}`,
//! `{"tuple": [..]}`, `{"frozen_set": [..]}` or `{"opaque": <foreign
//! part>}`; a member's sets are written in canonical order. A
//! [`Constraint`] serializes as `{"equation": {"expression"}}`, `{"in_set":
//! {"variable", "values"}}`, `{"not_in_set": {"variable", "values"}}` or
//! `{"custom": <foreign part>}`, with its members in canonical order, and a
//! [`ConstraintSystem`] as `{"constraints": [..]}`, in its canonical order.
//! Decoding accepts members and constraints in any order. [`ValueData`],
//! [`ConstraintData`] and [`ConstraintSystemData`] read the same shapes and
//! resolve the foreign parts in `build`; the types' own `Deserialize`
//! builds with [`NoForeign`], which refuses them.
//!
//! Values nest, and serde recurses once per level of a tuple or a set;
//! JSON limits the depth, and postcard does not.
//!
//! # Examples
//!
//! ```
//! use fhy_core::constraint::{Constraint, Member, MemberSet, Polarity, SetConstraint, Value};
//! use fhy_core::identifier::Identifier;
//!
//! let x = Identifier::try_restore(61_200, "x")?;
//! let members = MemberSet::new([Value::Str("a".into()), Value::Int(2.into())]
//!     .into_iter()
//!     .map(Member::try_from_value)
//!     .collect::<Result<Vec<_>, _>>()?);
//! let constraint = Constraint::Set(SetConstraint::new(x, members, Polarity::In));
//!
//! let text = serde_json::to_string(&constraint)?;
//! assert_eq!(
//!     text,
//!     r#"{"in_set":{"variable":{"id":61200,"name_hint":"x"},"values":[{"int":"2"},{"str":"a"}]}}"#
//! );
//! let decoded: Constraint = serde_json::from_str(&text)?;
//! assert!(decoded.is_structurally_equivalent(&constraint));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::sync::Arc;

use serde::de::{self, Deserializer};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};

use crate::expression::{BigInt, Decimal, float_text, integer_text, serialize_display_text};
use crate::foreign::{BuildError, Foreign, ForeignError, NoForeign, Resolve};
use crate::identifier::Identifier;

use super::Constraint;
use super::custom::CustomConstraint;
use super::equation::EquationConstraint;
use super::set::{Polarity, SetConstraint};
use super::system::ConstraintSystem;
use super::value::{Member, MemberKind, MemberSet, Opaque, Value};

/// The wire form of a [`Value`] or a [`Member`], its opaque parts
/// unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ValueData(ValueRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Value", rename_all = "snake_case")]
enum ValueRepr {
    Bool(bool),
    Int(
        #[serde(
            serialize_with = "serialize_display_text",
            deserialize_with = "integer_text::deserialize"
        )]
        BigInt,
    ),
    Float(
        #[serde(
            serialize_with = "serialize_display_text",
            deserialize_with = "float_text::deserialize"
        )]
        f64,
    ),
    Decimal(Decimal),
    Str(String),
    Tuple(Vec<ValueRepr>),
    FrozenSet(Vec<ValueRepr>),
    Opaque(Foreign),
}

impl ValueRepr {
    fn of_value(value: &Value) -> Result<Self, ForeignError> {
        Ok(match value {
            Value::Bool(value) => Self::Bool(*value),
            Value::Int(value) => Self::Int(value.clone()),
            Value::Float(value) => Self::Float(*value),
            Value::Decimal(value) => Self::Decimal(value.clone()),
            Value::Str(value) => Self::Str(value.clone()),
            Value::Tuple(values) => Self::Tuple(
                values
                    .iter()
                    .map(Self::of_value)
                    .collect::<Result<_, _>>()?,
            ),
            Value::FrozenSet(values) => Self::FrozenSet(
                values
                    .iter()
                    .map(Self::of_value)
                    .collect::<Result<_, _>>()?,
            ),
            Value::Opaque(value) => Self::Opaque(value.get().to_foreign()?),
        })
    }

    fn of_member(member: &Member) -> Result<Self, ForeignError> {
        Ok(match member.kind() {
            MemberKind::Bool(value) => Self::Bool(value),
            MemberKind::Int(value) => Self::Int(value.clone()),
            MemberKind::Float(value) => Self::Float(value),
            MemberKind::Str(value) => Self::Str(value.to_owned()),
            MemberKind::Tuple(members) => Self::Tuple(
                members
                    .iter()
                    .map(Self::of_member)
                    .collect::<Result<_, _>>()?,
            ),
            MemberKind::FrozenSet(members) => Self::FrozenSet(of_members(members)?),
            MemberKind::Opaque(value) => Self::Opaque(value.get().to_foreign()?),
        })
    }

    fn build<R: Resolve<Opaque> + ?Sized>(self, resolver: &R) -> Result<Value, BuildError> {
        Ok(match self {
            Self::Bool(value) => Value::Bool(value),
            Self::Int(value) => Value::Int(value),
            Self::Float(value) => Value::Float(value),
            Self::Decimal(value) => Value::Decimal(value),
            Self::Str(value) => Value::Str(value),
            Self::Tuple(values) => Value::Tuple(build_values(values, resolver)?),
            Self::FrozenSet(values) => Value::FrozenSet(build_values(values, resolver)?),
            Self::Opaque(foreign) => Value::Opaque(resolver.resolve(&foreign)?),
        })
    }
}

fn build_values<R: Resolve<Opaque> + ?Sized>(
    values: Vec<ValueRepr>,
    resolver: &R,
) -> Result<Vec<Value>, BuildError> {
    values
        .into_iter()
        .map(|value| value.build(resolver))
        .collect()
}

fn of_members(members: &MemberSet) -> Result<Vec<ValueRepr>, ForeignError> {
    members.iter().map(ValueRepr::of_member).collect()
}

fn build_member_set<R: Resolve<Opaque> + ?Sized>(
    values: Vec<ValueRepr>,
    resolver: &R,
) -> Result<MemberSet, BuildError> {
    let members = values
        .into_iter()
        .map(|value| Member::try_from_value(value.build(resolver)?).map_err(BuildError::invalid))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MemberSet::new(members))
}

impl ValueData {
    /// Return the wire form of `value`, asking each opaque value for its
    /// foreign part.
    ///
    /// # Errors
    ///
    /// Returns the error of an opaque value that cannot give its part.
    pub fn of_value(value: &Value) -> Result<Self, ForeignError> {
        ValueRepr::of_value(value).map(Self)
    }

    /// Return the value, its opaque parts resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses.
    pub fn build<R: Resolve<Opaque> + ?Sized>(self, resolver: &R) -> Result<Value, BuildError> {
        self.0.build(resolver)
    }

    /// Return the member, its opaque parts resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses, and
    /// [`BuildError::Invalid`] with the [`MemberError`](super::MemberError)
    /// of a value that is no member.
    pub fn build_member<R: Resolve<Opaque> + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<Member, BuildError> {
        Member::try_from_value(self.0.build(resolver)?).map_err(BuildError::invalid)
    }
}

/// Serializes the shape of the [module documentation](self).
impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ValueRepr::of_value(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing an
/// opaque part.
impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ValueData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// Serializes the shape of the [module documentation](self).
impl Serialize for Member {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ValueRepr::of_member(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing an
/// opaque part and a value that is no member.
impl<'de> Deserialize<'de> for Member {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ValueData::deserialize(deserializer)?
            .build_member(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// Serializes as `{"expression": <expression>}`.
impl Serialize for EquationConstraint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        EquationRef {
            expression: self.expression(),
        }
        .serialize(serializer)
    }
}

/// Deserializes `{"expression": <expression>}`.
impl<'de> Deserialize<'de> for EquationConstraint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(
            EquationWire::deserialize(deserializer)?.expression,
        ))
    }
}

#[derive(Serialize)]
#[serde(rename = "EquationConstraint")]
struct EquationRef<'a> {
    expression: &'a crate::expression::Expression,
}

#[derive(Deserialize)]
#[serde(rename = "EquationConstraint", deny_unknown_fields)]
struct EquationWire {
    expression: crate::expression::Expression,
}

/// The resolvers a constraint's foreign parts need.
pub trait ConstraintResolver: Resolve<Opaque> + Resolve<Arc<dyn CustomConstraint>> {}

impl<R: Resolve<Opaque> + Resolve<Arc<dyn CustomConstraint>> + ?Sized> ConstraintResolver for R {}

/// The wire form of a [`Constraint`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConstraintData(ConstraintRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Constraint", rename_all = "snake_case")]
enum ConstraintRepr {
    Equation(EquationConstraint),
    InSet(SetRepr),
    NotInSet(SetRepr),
    Custom(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "SetConstraint", deny_unknown_fields)]
struct SetRepr {
    variable: Identifier,
    values: Vec<ValueRepr>,
}

impl ConstraintData {
    /// Return the wire form of `constraint`, asking each opaque member and
    /// custom constraint for its foreign part.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(constraint: &Constraint) -> Result<Self, ForeignError> {
        Ok(Self(match constraint {
            Constraint::Equation(equation) => ConstraintRepr::Equation(equation.clone()),
            Constraint::Set(set) => {
                let repr = SetRepr {
                    variable: set.variable().clone(),
                    values: of_members(set.members())?,
                };
                match set.polarity() {
                    Polarity::In => ConstraintRepr::InSet(repr),
                    Polarity::NotIn => ConstraintRepr::NotInSet(repr),
                }
            }
            Constraint::Custom(custom) => ConstraintRepr::Custom(custom.to_foreign()?),
        }))
    }

    /// Return the foreign part of a custom constraint, or `None` for a
    /// built-in one.
    #[must_use]
    pub fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            ConstraintRepr::Custom(foreign) => Some(foreign),
            ConstraintRepr::Equation(_)
            | ConstraintRepr::InSet(_)
            | ConstraintRepr::NotInSet(_) => None,
        }
    }

    /// Return the constraint, its foreign parts resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses, and
    /// [`BuildError::Invalid`] for a set member that is no member.
    pub fn build<R: ConstraintResolver + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<Constraint, BuildError> {
        Ok(match self.0 {
            ConstraintRepr::Equation(equation) => Constraint::Equation(equation),
            ConstraintRepr::InSet(set) => Constraint::Set(SetConstraint::new(
                set.variable,
                build_member_set(set.values, resolver)?,
                Polarity::In,
            )),
            ConstraintRepr::NotInSet(set) => Constraint::Set(SetConstraint::new(
                set.variable,
                build_member_set(set.values, resolver)?,
                Polarity::NotIn,
            )),
            ConstraintRepr::Custom(foreign) => Constraint::Custom(resolver.resolve(&foreign)?),
        })
    }
}

/// Serializes the shape of the [module documentation](self).
impl Serialize for Constraint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ConstraintData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing a
/// foreign part.
impl<'de> Deserialize<'de> for Constraint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ConstraintData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// The wire form of a [`ConstraintSystem`], its foreign parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "ConstraintSystem", deny_unknown_fields)]
pub struct ConstraintSystemData {
    constraints: Vec<ConstraintData>,
}

impl ConstraintSystemData {
    /// Return the wire form of `system`, in its canonical order.
    ///
    /// # Errors
    ///
    /// Returns the error of a part that cannot give its foreign form.
    pub fn of(system: &ConstraintSystem) -> Result<Self, ForeignError> {
        Ok(Self {
            constraints: system
                .constraints()
                .iter()
                .map(ConstraintData::of)
                .collect::<Result<_, _>>()?,
        })
    }

    /// Return the constraints' wire forms, in order.
    #[must_use]
    pub fn into_constraints(self) -> Vec<ConstraintData> {
        self.constraints
    }

    /// Return the system, its foreign parts resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns what [`ConstraintData::build`] returns.
    pub fn build<R: ConstraintResolver + ?Sized>(
        self,
        resolver: &R,
    ) -> Result<ConstraintSystem, BuildError> {
        let constraints = self
            .constraints
            .into_iter()
            .map(|constraint| constraint.build(resolver))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ConstraintSystem::new(constraints))
    }
}

/// Serializes as `{"constraints": [..]}`, in canonical order.
impl Serialize for ConstraintSystem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ConstraintSystemData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"constraints": [..]}`, in any order, refusing a foreign
/// part.
impl<'de> Deserialize<'de> for ConstraintSystem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ConstraintSystemData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}
