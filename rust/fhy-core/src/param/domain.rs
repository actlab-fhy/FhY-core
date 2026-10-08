//! The value domains of params: what values a param ranges over, which
//! constraints it may carry, and the questions each kind answers.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

use crate::constraint::wire::MAX_VALUE_DEPTH;
use crate::constraint::{
    Constraint, EquationConstraint, Member, MemberError, MemberSet, Outcome, Value,
};
use crate::error::impl_from_name;
use crate::expression::{
    BigInt, BinaryOperation, Expression, ExpressionKind, LiteralValue, SymbolType,
};
use crate::foreign::Part;
use crate::identifier::Identifier;

use super::context::ParamContext;
use super::custom::CustomDomain;
use super::error::{DomainError, ParamBuildError, ParamError};
use super::interval::Inclusivity;
use super::value::{compare_ordinal, sort_tolerantly};
use super::{algebra, decide};

/// The kind of a [`ParamDomain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DomainKind {
    /// [`IntegerDomain`].
    Integer,
    /// [`IntervalIntegerDomain`].
    IntervalInteger,
    /// [`RealDomain`].
    Real,
    /// [`OrdinalDomain`].
    Ordinal,
    /// [`CategoricalDomain`].
    Categorical,
    /// [`PermutationDomain`].
    Permutation,
    /// A [`CustomDomain`].
    Custom,
}

impl DomainKind {
    /// Return the kind's name: `integer`, `interval integer`, `real`,
    /// `ordinal`, `categorical`, `permutation` or `custom`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Integer => "integer",
            Self::IntervalInteger => "interval integer",
            Self::Real => "real",
            Self::Ordinal => "ordinal",
            Self::Categorical => "categorical",
            Self::Permutation => "permutation",
            Self::Custom => "custom",
        }
    }

    /// Return the kind's name after its indefinite article, and `domain`.
    pub(super) fn article(self) -> String {
        let article = match self {
            Self::Integer | Self::IntervalInteger | Self::Ordinal => "an",
            Self::Real | Self::Categorical | Self::Permutation | Self::Custom => "a",
        };
        format!("{article} {} domain", self.name())
    }
}

impl fmt::Display for DomainKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl_from_name!(
    DomainKind,
    name,
    "domain kind",
    [
        Integer,
        IntervalInteger,
        Real,
        Ordinal,
        Categorical,
        Permutation,
        Custom
    ]
);

/// What interval arithmetic and the natural-number bound gates read from a
/// domain.
///
/// A param's interval lives in its bound constraints; the domain
/// contributes only these facts. It is built with [`new`](Self::new) and
/// [`with_only_bounds`](Self::with_only_bounds), and read through its
/// getters, so it can gain facts.
///
/// # Examples
///
/// ```
/// use fhy_core::param::{Inclusivity, IntervalProfile, Sign, ZeroInclusion};
///
/// let profile = IntervalProfile::new(Sign::NonNegative, ZeroInclusion::Excluded, Inclusivity::Inclusive)
///     .with_only_bounds();
///
/// assert!(profile.is_bounds_only());
/// assert!(profile.is_non_negative());
/// assert!(!profile.is_zero_included());
/// assert!(profile.is_inclusive_preferred());
/// ```
#[expect(
    clippy::struct_excessive_bools,
    reason = "four independent facts about a domain, behind getters"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct IntervalProfile {
    admits_only_bounds: bool,
    non_negative: bool,
    zero_included: bool,
    prefer_inclusive: bool,
}

impl IntervalProfile {
    /// Return the profile of a domain of `sign`, with zero as `zero` says,
    /// whose derived bounds render as `preferred` says, admitting any
    /// constraint.
    ///
    /// `zero` means nothing for [`Sign::Any`]; it is kept as given, and the
    /// gates read it only for a non-negative domain.
    #[must_use]
    pub const fn new(sign: Sign, zero: ZeroInclusion, preferred: Inclusivity) -> Self {
        Self {
            admits_only_bounds: false,
            non_negative: sign.is_non_negative(),
            zero_included: zero.is_included(),
            prefer_inclusive: preferred.is_inclusive(),
        }
    }

    /// Return this profile of a domain that admits only bound constraints,
    /// so a param over it is an interval operand as it stands.
    #[must_use]
    pub const fn with_only_bounds(self) -> Self {
        Self {
            admits_only_bounds: true,
            ..self
        }
    }

    /// Return whether the domain admits only bound constraints.
    #[must_use]
    pub const fn is_bounds_only(&self) -> bool {
        self.admits_only_bounds
    }

    /// Return whether the domain admits only non-negative values.
    #[must_use]
    pub const fn is_non_negative(&self) -> bool {
        self.non_negative
    }

    /// Return whether the domain admits zero, given it is non-negative.
    #[must_use]
    pub const fn is_zero_included(&self) -> bool {
        self.zero_included
    }

    /// Return whether bounds that arithmetic derives render in inclusive
    /// form.
    #[must_use]
    pub const fn is_inclusive_preferred(&self) -> bool {
        self.prefer_inclusive
    }
}

/// The constraints of one side of a question, and the variable they
/// constrain.
#[derive(Debug, Clone, Copy)]
pub struct Side<'a> {
    constraints: &'a [Constraint],
    variable: &'a Identifier,
}

impl<'a> Side<'a> {
    /// Return the side of `constraints` over `variable`.
    #[must_use]
    pub const fn new(constraints: &'a [Constraint], variable: &'a Identifier) -> Self {
        Self {
            constraints,
            variable,
        }
    }

    /// Return the constraints.
    #[must_use]
    pub const fn constraints(&self) -> &'a [Constraint] {
        self.constraints
    }

    /// Return the variable.
    #[must_use]
    pub const fn variable(&self) -> &'a Identifier {
        self.variable
    }
}

/// Whether an integer domain admits every integer or only the
/// non-negative ones.
#[expect(
    clippy::exhaustive_enums,
    reason = "the two signs an integer domain is restricted to, which callers match"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Every integer.
    Any,
    /// The non-negative integers, the natural numbers.
    NonNegative,
}

impl Sign {
    /// Return [`NonNegative`](Self::NonNegative) if `is_non_negative`, and
    /// [`Any`](Self::Any) otherwise.
    #[must_use]
    pub const fn non_negative_if(is_non_negative: bool) -> Self {
        if is_non_negative {
            Self::NonNegative
        } else {
            Self::Any
        }
    }

    /// Return whether the sign is [`NonNegative`](Self::NonNegative).
    #[must_use]
    pub const fn is_non_negative(self) -> bool {
        matches!(self, Self::NonNegative)
    }
}

/// Whether a non-negative integer domain admits zero.
#[expect(
    clippy::exhaustive_enums,
    reason = "zero is in or out, which callers match"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroInclusion {
    /// Zero is admissible.
    Included,
    /// Zero is not admissible.
    Excluded,
}

impl ZeroInclusion {
    /// Return [`Included`](Self::Included) if `is_included`, and
    /// [`Excluded`](Self::Excluded) otherwise.
    #[must_use]
    pub const fn included_if(is_included: bool) -> Self {
        if is_included {
            Self::Included
        } else {
            Self::Excluded
        }
    }

    /// Return whether zero is [`Included`](Self::Included).
    #[must_use]
    pub const fn is_included(self) -> bool {
        matches!(self, Self::Included)
    }
}

/// Integers, optionally restricted to the natural numbers.
///
/// Every integer is admissible; a non-negative domain implies the bound
/// `x >= 0`, or `x > 0` without zero. Any constraint is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntegerDomain {
    non_negative: bool,
    zero_included: bool,
}

impl IntegerDomain {
    /// Return the integers of `sign`, with zero as `zero` says.
    ///
    /// `zero` means nothing for [`Sign::Any`], so zero is then included.
    #[must_use]
    pub const fn new(sign: Sign, zero: ZeroInclusion) -> Self {
        let non_negative = sign.is_non_negative();
        Self {
            non_negative,
            zero_included: zero.is_included() || !non_negative,
        }
    }

    /// Return whether the domain admits only non-negative values.
    #[must_use]
    pub const fn is_non_negative(&self) -> bool {
        self.non_negative
    }

    /// Return whether the domain admits zero, given it is non-negative.
    #[must_use]
    pub const fn is_zero_included(&self) -> bool {
        self.zero_included
    }
}

/// Integers whose params carry their interval as bound constraints, which
/// interval arithmetic reads.
///
/// Every integer is admissible; only bound equations are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntervalIntegerDomain {
    prefer_inclusive: bool,
    non_negative: bool,
    zero_included: bool,
}

impl IntervalIntegerDomain {
    /// Return the interval integers, rendering derived bounds as
    /// `preferred` says, restricted as [`IntegerDomain::new`] is.
    #[must_use]
    pub const fn new(preferred: Inclusivity, sign: Sign, zero: ZeroInclusion) -> Self {
        let non_negative = sign.is_non_negative();
        Self {
            prefer_inclusive: preferred.is_inclusive(),
            non_negative,
            zero_included: zero.is_included() || !non_negative,
        }
    }

    /// Return whether derived bounds render inclusively.
    #[must_use]
    pub const fn is_inclusive_preferred(&self) -> bool {
        self.prefer_inclusive
    }

    /// Return whether the domain admits only non-negative values.
    #[must_use]
    pub const fn is_non_negative(&self) -> bool {
        self.non_negative
    }

    /// Return whether the domain admits zero, given it is non-negative.
    #[must_use]
    pub const fn is_zero_included(&self) -> bool {
        self.zero_included
    }
}

/// The reals: finite floats, and strings in the literal grammar, which
/// denote exact decimals.
#[expect(
    clippy::exhaustive_structs,
    reason = "a stateless unit type that callers name as a value"
)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RealDomain;

/// A finite, totally ordered set of values.
///
/// The values are kept ascending: numbers numerically and exactly across
/// Booleans, integers and floats, strings by code point, and opaque values
/// by their producer's order; values the order does not tell apart by
/// kind, `bool`, `float`, `int`, and then as given. Cloning one shares it.
#[derive(Debug, Clone)]
pub struct OrdinalDomain(Arc<FiniteValues>);

/// A finite, unordered set of categories, kept in the members' canonical
/// order. A category is a leaf value or a tuple or frozen set of
/// categories, such as a tile shape `(8, 8)`. Cloning one shares it.
#[derive(Debug, Clone)]
pub struct CategoricalDomain(Arc<FiniteValues>);

/// The permutations of a fixed, ordered set of members, kept as given.
/// Cloning one shares it.
#[derive(Debug, Clone)]
pub struct PermutationDomain(Arc<FiniteValues>);

/// The values of a finite domain in its order, and a set for lookup.
#[derive(Debug)]
struct FiniteValues {
    values: Vec<Member>,
    lookup: MemberSet,
}

/// What a value of a finite domain may be, beyond the Boolean, integer,
/// string, identifier and member-shaped opaque leaves every domain admits.
#[derive(Clone, Copy)]
struct Admits {
    /// Whether a float is a leaf.
    float: bool,
    /// Whether a tuple or frozen set of admitted values is.
    composite: bool,
}

/// Return whether `value` is admitted, with `remaining` more tuples or
/// frozen sets allowed around its innermost value.
fn is_admitted(value: &Value, admits: Admits, remaining: usize) -> bool {
    match value {
        Value::Bool(_) | Value::Int(_) | Value::Str(_) | Value::Identifier(_) => true,
        Value::Float(_) => admits.float,
        Value::Opaque(opaque) => opaque.get().is_member_shaped(),
        Value::Decimal(_) => false,
        Value::Tuple(elements) | Value::FrozenSet(elements) => {
            admits.composite
                && remaining > 0
                && elements
                    .iter()
                    .all(|element| is_admitted(element, admits, remaining - 1))
        }
    }
}

/// Return the members of the `values` of a finite domain of `kind`,
/// checked in order: non-empty, each value admitted by `admits`, then no
/// NaN. A composite value nests at most [`MAX_VALUE_DEPTH`] deep.
fn read_finite_members(
    kind: DomainKind,
    values: Vec<Value>,
    admits: Admits,
) -> Result<Vec<Member>, DomainError> {
    if values.is_empty() {
        return Err(DomainError::EmptyValues(kind));
    }
    for (index, value) in values.iter().enumerate() {
        if !is_admitted(value, admits, MAX_VALUE_DEPTH) {
            return Err(DomainError::NotALeafValue { kind, index });
        }
    }
    if values
        .iter()
        .any(|value| matches!(value, Value::Float(number) if number.is_nan()))
    {
        return Err(DomainError::NanValue(kind));
    }
    values
        .into_iter()
        .map(|value| {
            Member::try_from(value).map_err(|error| match error {
                MemberError::OrderingKey { source, .. } => DomainError::Custom(source),
                _ => DomainError::NanValue(kind),
            })
        })
        .collect()
}

/// Return the finite values `members` of `kind`, refusing equal members.
fn build_finite_values(
    kind: DomainKind,
    members: Vec<Member>,
) -> Result<FiniteValues, DomainError> {
    let lookup = MemberSet::new(members.iter().cloned());
    if lookup.len() != members.len() {
        return Err(DomainError::DuplicateValues(kind));
    }
    Ok(FiniteValues {
        values: members,
        lookup,
    })
}

impl OrdinalDomain {
    /// Return the ordinal domain of `values`.
    ///
    /// # Errors
    ///
    /// In order: [`DomainError::EmptyValues`] for no value;
    /// [`DomainError::NotALeafValue`] for a value that is no Boolean,
    /// integer, float, string, identifier or member-shaped opaque value;
    /// [`DomainError::NanValue`] for a NaN; [`DomainError::IncomparableValues`]
    /// for two values that do not order, such as two identifiers; and
    /// [`DomainError::DuplicateValues`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, DomainError> {
        let members = read_finite_members(
            DomainKind::Ordinal,
            values,
            Admits {
                float: true,
                composite: false,
            },
        )?;
        let sorted = sort_tolerantly(members, |left, right| {
            compare_ordinal(left, right)
                .map_err(DomainError::Custom)?
                .ok_or(DomainError::IncomparableValues)
        })?;
        build_finite_values(DomainKind::Ordinal, sorted).map(|values| Self(Arc::new(values)))
    }

    /// Return the values, ascending.
    #[must_use]
    pub fn values(&self) -> &[Member] {
        &self.0.values
    }

    /// Return whether the domain holds a member equal to `value`.
    #[must_use]
    pub fn contains_value(&self, value: &Value) -> bool {
        self.0.lookup.contains_value(value)
    }
}

impl CategoricalDomain {
    /// Return the categorical domain of `values`.
    ///
    /// # Errors
    ///
    /// In order: [`DomainError::EmptyValues`] for no value;
    /// [`DomainError::NotALeafValue`] for a value that is no Boolean,
    /// integer, string, identifier or member-shaped opaque value, or tuple
    /// or frozen set of such values at any depth (a float or a decimal
    /// refused, inside a tuple too); and
    /// [`DomainError::DuplicateValues`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, DomainError> {
        let members = read_finite_members(
            DomainKind::Categorical,
            values,
            Admits {
                float: false,
                composite: true,
            },
        )?;
        let count = members.len();
        let lookup = MemberSet::new(members);
        if lookup.len() != count {
            return Err(DomainError::DuplicateValues(DomainKind::Categorical));
        }
        Ok(Self(Arc::new(FiniteValues {
            values: lookup.iter().cloned().collect(),
            lookup,
        })))
    }

    /// Return the categories, in the members' canonical order.
    #[must_use]
    pub fn values(&self) -> &[Member] {
        &self.0.values
    }

    /// Return whether the domain holds a member equal to `value`.
    #[must_use]
    pub fn contains_value(&self, value: &Value) -> bool {
        self.0.lookup.contains_value(value)
    }
}

impl PermutationDomain {
    /// Return the permutations of `values`, in the order given.
    ///
    /// # Errors
    ///
    /// In order: [`DomainError::EmptyValues`] for no value;
    /// [`DomainError::NotALeafValue`] for a value that is no Boolean,
    /// integer, float, string, identifier or member-shaped opaque value;
    /// [`DomainError::NanValue`] for a NaN; and
    /// [`DomainError::DuplicateValues`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, DomainError> {
        let members = read_finite_members(
            DomainKind::Permutation,
            values,
            Admits {
                float: true,
                composite: false,
            },
        )?;
        build_finite_values(DomainKind::Permutation, members).map(|values| Self(Arc::new(values)))
    }

    /// Return the members, in the order given.
    #[must_use]
    pub fn values(&self) -> &[Member] {
        &self.0.values
    }

    /// Return whether the domain holds a member equal to `value`.
    #[must_use]
    pub fn contains_value(&self, value: &Value) -> bool {
        self.0.lookup.contains_value(value)
    }

    /// Return whether `value` is a permutation of the members: a tuple
    /// holding each member once.
    #[must_use]
    pub fn is_permutation(&self, value: &Value) -> bool {
        let Value::Tuple(elements) = value else {
            return false;
        };
        if elements.len() != self.0.values.len()
            || !elements.iter().all(|element| self.contains_value(element))
        {
            return false;
        }
        let members: Result<Vec<Member>, _> =
            elements.iter().cloned().map(Member::try_from).collect();
        members.is_ok_and(|members| MemberSet::new(members).len() == elements.len())
    }
}

/// The value domain of a param: one of the six kinds, or a custom one.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ParamDomain {
    /// The integers, optionally the natural numbers.
    Integer(IntegerDomain),
    /// The integers of interval arithmetic.
    IntervalInteger(IntervalIntegerDomain),
    /// The reals.
    Real(RealDomain),
    /// A finite, totally ordered set of values.
    Ordinal(OrdinalDomain),
    /// A finite, unordered set of categories.
    Categorical(CategoricalDomain),
    /// The permutations of a fixed set of members.
    Permutation(PermutationDomain),
    /// A domain of a kind defined elsewhere.
    Custom(Part<dyn CustomDomain>),
}

impl From<IntegerDomain> for ParamDomain {
    fn from(domain: IntegerDomain) -> Self {
        Self::Integer(domain)
    }
}

impl From<IntervalIntegerDomain> for ParamDomain {
    fn from(domain: IntervalIntegerDomain) -> Self {
        Self::IntervalInteger(domain)
    }
}

impl From<RealDomain> for ParamDomain {
    fn from(domain: RealDomain) -> Self {
        Self::Real(domain)
    }
}

impl From<OrdinalDomain> for ParamDomain {
    fn from(domain: OrdinalDomain) -> Self {
        Self::Ordinal(domain)
    }
}

impl From<CategoricalDomain> for ParamDomain {
    fn from(domain: CategoricalDomain) -> Self {
        Self::Categorical(domain)
    }
}

impl From<PermutationDomain> for ParamDomain {
    fn from(domain: PermutationDomain) -> Self {
        Self::Permutation(domain)
    }
}

/// Return the implied bound of a non-negative integer domain on
/// `variable`: `x >= 0`, or `x > 0` without zero.
fn non_negative_bound(
    variable: &Identifier,
    non_negative: bool,
    zero_included: bool,
) -> Vec<Constraint> {
    if !non_negative {
        return Vec::new();
    }
    let reference = Expression::from(variable);
    let zero = Expression::literal(LiteralValue::Int(0.into()));
    let bound = if zero_included {
        reference.greater_equal(zero)
    } else {
        reference.greater(zero)
    };
    vec![Constraint::from(EquationConstraint::new(bound))]
}

/// The lower bound of a numeric domain's sign restriction, weakest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SignBound {
    /// No bound.
    Unbounded,
    /// `x >= 0`.
    NonNegative,
    /// `x > 0`.
    Positive,
}

/// Return whether `constraint` is an equation, as the constraints of a
/// finite domain are not.
const fn is_set_constraint(constraint: &Constraint) -> bool {
    matches!(constraint, Constraint::Set(_))
}

impl ParamDomain {
    /// Return the domain's kind.
    #[must_use]
    pub const fn kind(&self) -> DomainKind {
        match self {
            Self::Integer(_) => DomainKind::Integer,
            Self::IntervalInteger(_) => DomainKind::IntervalInteger,
            Self::Real(_) => DomainKind::Real,
            Self::Ordinal(_) => DomainKind::Ordinal,
            Self::Categorical(_) => DomainKind::Categorical,
            Self::Permutation(_) => DomainKind::Permutation,
            Self::Custom(_) => DomainKind::Custom,
        }
    }

    /// Return the sort the solver reasons about the values in: `Int` for
    /// the integer kinds, `Real` for the reals, `None` for the finite
    /// kinds.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn symbol_type(&self) -> Result<Option<SymbolType>, ParamError> {
        Ok(match self {
            Self::Integer(_) | Self::IntervalInteger(_) => Some(SymbolType::Int),
            Self::Real(_) => Some(SymbolType::Real),
            Self::Ordinal(_) | Self::Categorical(_) | Self::Permutation(_) => None,
            Self::Custom(domain) => domain.get().symbol_type().map_err(ParamError::Custom)?,
        })
    }

    /// Return whether `value` lies in the domain's value set: an integer
    /// for the integer kinds, within their sign restriction (at least `0`
    /// for a non-negative domain, above `0` without zero); a finite float
    /// or a string in the literal grammar for the reals; a member, compared
    /// type-strictly, for the ordinal and categorical kinds; a tuple holding
    /// each member once for the permutation kind. A custom domain answers
    /// for its own restriction.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn is_value_admissible(&self, value: &Value) -> Result<bool, ParamError> {
        Ok(match self {
            Self::Integer(_) | Self::IntervalInteger(_) => match value {
                Value::Int(integer) => match self.sign_bound() {
                    SignBound::Unbounded => true,
                    SignBound::NonNegative => *integer >= BigInt::ZERO,
                    SignBound::Positive => *integer > BigInt::ZERO,
                },
                _ => false,
            },
            Self::Real(_) => match value {
                Value::Float(number) => number.is_finite(),
                Value::Str(text) => LiteralValue::parse_text(text).is_ok(),
                _ => false,
            },
            Self::Ordinal(domain) => domain.contains_value(value),
            Self::Categorical(domain) => domain.contains_value(value),
            Self::Permutation(domain) => domain.is_permutation(value),
            Self::Custom(domain) => domain
                .get()
                .is_value_admissible(value)
                .map_err(ParamError::Custom)?,
        })
    }

    /// Refuse `constraint` on `variable` if the domain forbids it: an
    /// interval domain allows only bound equations, and a finite domain only
    /// set constraints.
    ///
    /// # Errors
    ///
    /// Returns [`ParamBuildError::ForbiddenConstraintKind`] for a constraint of
    /// a forbidden kind, [`ParamBuildError::NotABound`] for an interval domain's
    /// equation that is no bound, and [`ParamBuildError::Custom`] for a custom
    /// domain's refusal.
    pub fn validate_constraint(
        &self,
        constraint: &Constraint,
        variable: &Identifier,
    ) -> Result<(), ParamBuildError> {
        match self {
            Self::Integer(_) | Self::Real(_) => Ok(()),
            Self::IntervalInteger(_) => match constraint {
                Constraint::Equation(equation) if is_bound_expression(equation.expression()) => {
                    Ok(())
                }
                Constraint::Equation(_) => Err(ParamBuildError::NotABound),
                _ => Err(ParamBuildError::ForbiddenConstraintKind(
                    DomainKind::IntervalInteger,
                )),
            },
            Self::Ordinal(_) | Self::Categorical(_) | Self::Permutation(_) => {
                if is_set_constraint(constraint) {
                    Ok(())
                } else {
                    Err(ParamBuildError::ForbiddenConstraintKind(self.kind()))
                }
            }
            Self::Custom(domain) => domain
                .get()
                .validate_constraint(constraint, variable)
                .map_err(ParamBuildError::Custom),
        }
    }

    /// Return the constraints the domain imposes on `variable`: the sign
    /// bound of a non-negative integer domain, and none otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`ParamBuildError::Custom`] for a custom domain that fails.
    pub fn implied_constraints(
        &self,
        variable: &Identifier,
    ) -> Result<Vec<Constraint>, ParamBuildError> {
        Ok(match self {
            Self::Integer(domain) => {
                non_negative_bound(variable, domain.non_negative, domain.zero_included)
            }
            Self::IntervalInteger(domain) => {
                non_negative_bound(variable, domain.non_negative, domain.zero_included)
            }
            Self::Real(_) | Self::Ordinal(_) | Self::Categorical(_) | Self::Permutation(_) => {
                Vec::new()
            }
            Self::Custom(domain) => domain
                .get()
                .implied_constraints(variable)
                .map_err(ParamBuildError::Custom)?,
        })
    }

    /// Return what interval arithmetic reads from the domain: a profile for
    /// the integer kinds, which admits only bounds for the interval kind,
    /// and `None` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn interval_profile(&self) -> Result<Option<IntervalProfile>, ParamError> {
        Ok(match self {
            Self::Integer(domain) => Some(IntervalProfile {
                admits_only_bounds: false,
                non_negative: domain.non_negative,
                zero_included: domain.zero_included,
                prefer_inclusive: true,
            }),
            Self::IntervalInteger(domain) => Some(IntervalProfile {
                admits_only_bounds: true,
                non_negative: domain.non_negative,
                zero_included: domain.zero_included,
                prefer_inclusive: domain.prefer_inclusive,
            }),
            Self::Real(_) | Self::Ordinal(_) | Self::Categorical(_) | Self::Permutation(_) => None,
            Self::Custom(domain) => domain
                .get()
                .interval_profile()
                .map_err(ParamError::Custom)?,
        })
    }

    /// Return whether the domain's value set is a subset of `other`'s: a
    /// numeric domain's of a domain of its sort whose restriction is no
    /// stronger (the naturals lie in the integers, not the other way
    /// round), a finite domain's of one of its kind holding each of its
    /// values (a permutation domain's of one of as many members).
    ///
    /// Against a custom domain of its sort, a numeric domain's set is a
    /// subset when each of the custom domain's implied constraints is one
    /// of its own; any other restriction is not proven, and answers `false`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn is_value_set_subset(
        &self,
        other: &Self,
        context: &ParamContext<'_>,
    ) -> Result<bool, ParamError> {
        Ok(match self {
            Self::Integer(_) | Self::IntervalInteger(_) | Self::Real(_) => {
                let own = self.symbol_type()?;
                if own.is_none() || other.symbol_type()? != own {
                    return Ok(false);
                }
                match other {
                    Self::Integer(_) | Self::IntervalInteger(_) | Self::Real(_) => {
                        self.sign_bound() >= other.sign_bound()
                    }
                    _ => {
                        let variable = Identifier::new("value");
                        let own_implied = self
                            .implied_constraints(&variable)
                            .map_err(decide::restriction_error)?;
                        other
                            .implied_constraints(&variable)
                            .map_err(decide::restriction_error)?
                            .iter()
                            .all(|implied| {
                                own_implied
                                    .iter()
                                    .any(|own| own.is_structurally_equivalent(implied))
                            })
                    }
                }
            }
            Self::Ordinal(domain) => match other {
                Self::Ordinal(other) => {
                    are_all_contained(domain.values(), |value| other.contains_value(value))
                }
                _ => false,
            },
            Self::Categorical(domain) => match other {
                Self::Categorical(other) => {
                    are_all_contained(domain.values(), |value| other.contains_value(value))
                }
                _ => false,
            },
            Self::Permutation(domain) => match other {
                Self::Permutation(other) => {
                    domain.values().len() == other.values().len()
                        && are_all_contained(domain.values(), |value| other.contains_value(value))
                }
                _ => false,
            },
            Self::Custom(domain) => domain
                .get()
                .is_value_set_subset(other, context)
                .map_err(ParamError::Custom)?,
        })
    }

    /// Decide whether `own`'s constrained set, over this domain, is a
    /// subset of `other`'s, over `other_domain`.
    ///
    /// Each side holds its domain's [`implied_constraints`](Self::implied_constraints)
    /// as well as its own, as a param's side does, so the domains'
    /// restrictions count.
    ///
    /// Domains of different value spaces decide [`Outcome::Violated`]. A
    /// numeric domain asks
    /// [`compute_constraint_implication_subset`](super::compute_constraint_implication_subset);
    /// a finite domain enumerates its values (a permutation domain its
    /// permutations) and always decides.
    ///
    /// # Errors
    ///
    /// Returns a constraint's error, and [`ParamError::Custom`] for a
    /// custom domain that fails.
    pub fn feasibility_subset(
        &self,
        own: Side<'_>,
        other_domain: &Self,
        other: Side<'_>,
        context: &ParamContext<'_>,
    ) -> Result<Outcome, ParamError> {
        decide::feasibility_subset(self, own, other_domain, other, context)
    }

    /// Decide whether some admissible value satisfies `side`'s constraints
    /// and the domain's [`implied_constraints`](Self::implied_constraints).
    ///
    /// A numeric domain enumerates the candidates of an in-set constraint,
    /// or asks the solver about the screened system; a finite domain
    /// enumerates its values and always decides.
    ///
    /// # Errors
    ///
    /// Returns a constraint's error, and [`ParamError::Custom`] for a
    /// custom domain that fails.
    pub fn has_feasible_value(
        &self,
        side: Side<'_>,
        context: &ParamContext<'_>,
    ) -> Result<Outcome, ParamError> {
        decide::has_feasible_value(self, side, context)
    }

    /// Return the domain and constraints of the union of the two value
    /// sets over `variable`, or `None` for a kind that represents no union:
    /// only the ordinal and categorical kinds do, baking both sides'
    /// effective values into a new domain with no constraint. Each side
    /// holds its domain's implied constraints as well as its own.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::KindMismatch`] for domains of different kinds,
    /// [`ParamError::EmptyUnion`] for an empty union, the new domain's
    /// errors, a constraint's error, and [`ParamError::Custom`].
    pub fn union(
        &self,
        own: Side<'_>,
        other_domain: &Self,
        other: Side<'_>,
        variable: &Identifier,
        context: &ParamContext<'_>,
    ) -> Result<Option<(Self, Vec<Constraint>)>, ParamError> {
        algebra::union(self, own, other_domain, other, variable, context)
    }

    /// Return the domain and constraints of the intersection of the two
    /// value sets over `variable`.
    ///
    /// A finite domain bakes the effective values both sides admit into a
    /// new domain with no constraint; a permutation domain keeps its
    /// members; a numeric domain merges the restrictions. The latter two
    /// carry both sides' constraints rescoped to `variable`, each side
    /// holding its domain's implied constraints as well as its own.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::KindMismatch`] for domains of different kinds,
    /// [`ParamError::EmptyIntersection`] and
    /// [`ParamError::DifferentPermutationMembers`] for an empty finite
    /// intersection, the rescoping's errors, a constraint's error, and
    /// [`ParamError::Custom`].
    pub fn intersection(
        &self,
        own: Side<'_>,
        other_domain: &Self,
        other: Side<'_>,
        variable: &Identifier,
        context: &ParamContext<'_>,
    ) -> Result<(Self, Vec<Constraint>), ParamError> {
        algebra::intersection(self, own, other_domain, other, variable, context)
    }

    /// Return the lower bound a built-in numeric domain's sign restriction
    /// imposes; any other domain is unbounded here.
    const fn sign_bound(&self) -> SignBound {
        let (non_negative, zero_included) = match self {
            Self::Integer(domain) => (domain.non_negative, domain.zero_included),
            Self::IntervalInteger(domain) => (domain.non_negative, domain.zero_included),
            _ => return SignBound::Unbounded,
        };
        match (non_negative, zero_included) {
            (false, _) => SignBound::Unbounded,
            (true, true) => SignBound::NonNegative,
            (true, false) => SignBound::Positive,
        }
    }

    /// Return whether `other` is a structurally identical domain: of the
    /// same kind, with the same restrictions or values (a categorical
    /// domain's in any order), compared type-strictly.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => left == right,
            (Self::IntervalInteger(left), Self::IntervalInteger(right)) => left == right,
            (Self::Real(_), Self::Real(_)) => true,
            (Self::Ordinal(left), Self::Ordinal(right)) => left.values() == right.values(),
            (Self::Categorical(left), Self::Categorical(right)) => left.0.lookup == right.0.lookup,
            (Self::Permutation(left), Self::Permutation(right)) => left.values() == right.values(),
            (Self::Custom(left), Self::Custom(right)) => left == right,
            _ => false,
        }
    }
}

impl PartialEq for ParamDomain {
    /// Compare as [`is_structurally_equivalent`](Self::is_structurally_equivalent)
    /// does: of one kind, built-in kinds by their restrictions or values,
    /// and custom domains through their [`eq_part`](CustomDomain::eq_part).
    fn eq(&self, other: &Self) -> bool {
        self.is_structurally_equivalent(other)
    }
}

impl Eq for ParamDomain {}

impl Hash for ParamDomain {
    /// Feed the kind, then the restrictions or values: a categorical
    /// domain's as a set, and a custom domain through its
    /// [`hash_part`](CustomDomain::hash_part).
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Self::Integer(domain) => domain.hash(state),
            Self::IntervalInteger(domain) => domain.hash(state),
            Self::Real(domain) => domain.hash(state),
            Self::Ordinal(domain) => domain.values().hash(state),
            Self::Categorical(domain) => domain.0.lookup.hash(state),
            Self::Permutation(domain) => domain.values().hash(state),
            Self::Custom(domain) => domain.hash(state),
        }
    }
}

impl fmt::Display for ParamDomain {
    /// Write the kind and its restrictions or values: `integer`,
    /// `non-negative integer` and `positive integer`, the same for
    /// `interval integer` with its preferred bounds, `real`, `ordinal (1,
    /// 2)` in order, `categorical {1, 2}`, `permutation of (1, 2)`, and a
    /// custom domain as its type name in angle brackets.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = |non_negative: bool, zero_included: bool| match (non_negative, zero_included) {
            (false, _) => "",
            (true, true) => "non-negative ",
            (true, false) => "positive ",
        };
        let values = |f: &mut fmt::Formatter<'_>, values: &[Member]| -> fmt::Result {
            f.write_str("(")?;
            for (position, value) in values.iter().enumerate() {
                if position > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{value}")?;
            }
            f.write_str(if values.len() == 1 { ",)" } else { ")" })
        };
        match self {
            Self::Integer(domain) => write!(
                f,
                "{}integer",
                sign(domain.is_non_negative(), domain.is_zero_included())
            ),
            Self::IntervalInteger(domain) => write!(
                f,
                "{}interval integer ({} bounds)",
                sign(domain.is_non_negative(), domain.is_zero_included()),
                if domain.is_inclusive_preferred() {
                    "inclusive"
                } else {
                    "exclusive"
                }
            ),
            Self::Real(_) => f.write_str("real"),
            Self::Ordinal(domain) => {
                f.write_str("ordinal ")?;
                values(f, domain.values())
            }
            Self::Categorical(domain) => write!(f, "categorical {}", domain.0.lookup),
            Self::Permutation(domain) => {
                f.write_str("permutation of ")?;
                values(f, domain.values())
            }
            Self::Custom(domain) => write!(f, "<{}>", domain.get().type_name()),
        }
    }
}

/// Return whether `contains` holds for the value of each of `members`.
fn are_all_contained(members: &[Member], contains: impl Fn(&Value) -> bool) -> bool {
    members
        .iter()
        .all(|member| contains(&super::value::member_value(member)))
}

/// Return whether `expression` is an integer bound of the form `x <cmp> k`
/// or `k <cmp> x`: a comparison `>=`, `>`, `<=` or `<` of an identifier and
/// an integer literal.
#[must_use]
pub fn is_bound_expression(expression: &Expression) -> bool {
    let ExpressionKind::Binary(binary) = expression.kind() else {
        return false;
    };
    if !matches!(
        binary.operation(),
        BinaryOperation::GreaterEqual
            | BinaryOperation::Greater
            | BinaryOperation::LessEqual
            | BinaryOperation::Less
    ) {
        return false;
    }
    let (left, right) = (binary.left().kind(), binary.right().kind());
    let is_identifier = |kind: &ExpressionKind| matches!(kind, ExpressionKind::Identifier(_));
    let integer_literal =
        |kind: &ExpressionKind| matches!(kind, ExpressionKind::Literal(LiteralValue::Int(_)));
    (is_identifier(left) && integer_literal(right))
        || (integer_literal(left) && is_identifier(right))
}
