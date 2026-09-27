//! The value domains of params: what values a param ranges over, which
//! constraints it may carry, and the questions each kind answers.

use std::fmt;
use std::sync::Arc;

use crate::constraint::{Constraint, EquationConstraint, Member, MemberSet, Outcome, Value};
use crate::error::impl_from_name;
use crate::expression::{BinaryOperation, Expression, ExpressionKind, LiteralValue, SymbolType};
use crate::identifier::Identifier;

use super::context::ParamContext;
use super::custom::CustomDomain;
use super::error::ParamError;
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
    pub fn name(self) -> &'static str {
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
/// contributes only these facts.
#[expect(
    clippy::exhaustive_structs,
    reason = "the four facts interval arithmetic reads, which callers build and read directly"
)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "four independent facts about a domain"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntervalProfile {
    /// Whether the domain admits only bound constraints, so a param over it
    /// is an interval operand as it stands.
    pub admits_only_bounds: bool,
    /// Whether the domain admits only non-negative values.
    pub non_negative: bool,
    /// Whether the domain admits zero, given it is non-negative.
    pub zero_included: bool,
    /// Whether bounds that arithmetic derives render in inclusive form.
    pub prefer_inclusive: bool,
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
    pub fn new(constraints: &'a [Constraint], variable: &'a Identifier) -> Self {
        Self {
            constraints,
            variable,
        }
    }

    /// Return the constraints.
    #[must_use]
    pub fn constraints(&self) -> &'a [Constraint] {
        self.constraints
    }

    /// Return the variable.
    #[must_use]
    pub fn variable(&self) -> &'a Identifier {
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
    pub fn new(sign: Sign, zero: ZeroInclusion) -> Self {
        let non_negative = sign.is_non_negative();
        Self {
            non_negative,
            zero_included: zero.is_included() || !non_negative,
        }
    }

    /// Return whether the domain admits only non-negative values.
    #[must_use]
    pub fn is_non_negative(&self) -> bool {
        self.non_negative
    }

    /// Return whether the domain admits zero, given it is non-negative.
    #[must_use]
    pub fn is_zero_included(&self) -> bool {
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
    pub fn new(preferred: Inclusivity, sign: Sign, zero: ZeroInclusion) -> Self {
        let non_negative = sign.is_non_negative();
        Self {
            prefer_inclusive: preferred.is_inclusive(),
            non_negative,
            zero_included: zero.is_included() || !non_negative,
        }
    }

    /// Return whether derived bounds render inclusively.
    #[must_use]
    pub fn is_inclusive_preferred(&self) -> bool {
        self.prefer_inclusive
    }

    /// Return whether the domain admits only non-negative values.
    #[must_use]
    pub fn is_non_negative(&self) -> bool {
        self.non_negative
    }

    /// Return whether the domain admits zero, given it is non-negative.
    #[must_use]
    pub fn is_zero_included(&self) -> bool {
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
/// order. Cloning one shares it.
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

/// Return the members of the leaf `values` of a finite domain of `kind`,
/// checked in order: non-empty, each value a leaf (a float only when
/// `allows_float`), then no NaN.
fn read_leaf_values(
    kind: DomainKind,
    values: Vec<Value>,
    allows_float: bool,
) -> Result<Vec<Member>, ParamError> {
    if values.is_empty() {
        return Err(ParamError::EmptyValues(kind));
    }
    for (index, value) in values.iter().enumerate() {
        let is_leaf = match value {
            Value::Bool(_) | Value::Int(_) | Value::Str(_) => true,
            Value::Float(_) => allows_float,
            Value::Opaque(opaque) => opaque.get().is_member_shaped(),
            Value::Decimal(_) | Value::Tuple(_) | Value::FrozenSet(_) => false,
        };
        if !is_leaf {
            return Err(ParamError::NotALeafValue { kind, index });
        }
    }
    if values
        .iter()
        .any(|value| matches!(value, Value::Float(number) if number.is_nan()))
    {
        return Err(ParamError::NanValue(kind));
    }
    values
        .into_iter()
        .map(|value| Member::try_from(value).map_err(|_nan| ParamError::NanValue(kind)))
        .collect()
}

/// Return the finite values `members` of `kind`, refusing equal members.
fn build_finite_values(kind: DomainKind, members: Vec<Member>) -> Result<FiniteValues, ParamError> {
    let lookup = MemberSet::new(members.iter().cloned());
    if lookup.len() != members.len() {
        return Err(ParamError::DuplicateValues(kind));
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
    /// In order: [`ParamError::EmptyValues`] for no value;
    /// [`ParamError::NotALeafValue`] for a value that is no Boolean,
    /// integer, float, string or member-shaped opaque value;
    /// [`ParamError::NanValue`] for a NaN; [`ParamError::IncomparableValues`]
    /// for two values that do not order; and
    /// [`ParamError::DuplicateValues`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, ParamError> {
        let members = read_leaf_values(DomainKind::Ordinal, values, true)?;
        let sorted = sort_tolerantly(members, compare_ordinal)
            .map_err(|()| ParamError::IncomparableValues)?;
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
    /// In order: [`ParamError::EmptyValues`] for no value;
    /// [`ParamError::NotALeafValue`] for a value that is no Boolean,
    /// integer, string or member-shaped opaque value, a float included; and
    /// [`ParamError::DuplicateValues`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, ParamError> {
        let members = read_leaf_values(DomainKind::Categorical, values, false)?;
        let count = members.len();
        let lookup = MemberSet::new(members);
        if lookup.len() != count {
            return Err(ParamError::DuplicateValues(DomainKind::Categorical));
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
    /// In order: [`ParamError::EmptyValues`] for no value;
    /// [`ParamError::NotALeafValue`] for a value that is no Boolean,
    /// integer, float, string or member-shaped opaque value;
    /// [`ParamError::NanValue`] for a NaN; and
    /// [`ParamError::DuplicateValues`] for two equal values.
    pub fn new(values: Vec<Value>) -> Result<Self, ParamError> {
        let members = read_leaf_values(DomainKind::Permutation, values, true)?;
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
    Custom(Arc<dyn CustomDomain>),
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

/// Return whether `constraint` is an equation, as the constraints of a
/// finite domain are not.
fn is_set_constraint(constraint: &Constraint) -> bool {
    matches!(constraint, Constraint::Set(_))
}

impl ParamDomain {
    /// Return the domain's kind.
    #[must_use]
    pub fn kind(&self) -> DomainKind {
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
            Self::Custom(domain) => domain.symbol_type().map_err(ParamError::Custom)?,
        })
    }

    /// Return whether `value` lies in the domain's value set: an integer
    /// for the integer kinds; a finite float or a string in the literal
    /// grammar for the reals; a member, compared type-strictly, for the
    /// ordinal and categorical kinds; a tuple holding each member once for
    /// the permutation kind.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn is_value_admissible(&self, value: &Value) -> Result<bool, ParamError> {
        Ok(match self {
            Self::Integer(_) | Self::IntervalInteger(_) => matches!(value, Value::Int(_)),
            Self::Real(_) => match value {
                Value::Float(number) => number.is_finite(),
                Value::Str(text) => LiteralValue::parse_text(text).is_ok(),
                _ => false,
            },
            Self::Ordinal(domain) => domain.contains_value(value),
            Self::Categorical(domain) => domain.contains_value(value),
            Self::Permutation(domain) => domain.is_permutation(value),
            Self::Custom(domain) => domain
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
    /// Returns [`ParamError::ForbiddenConstraintKind`] for a constraint of
    /// a forbidden kind, [`ParamError::NotABound`] for an interval domain's
    /// equation that is no bound, and [`ParamError::Custom`] for a custom
    /// domain's refusal.
    pub fn validate_constraint(
        &self,
        constraint: &Constraint,
        variable: &Identifier,
    ) -> Result<(), ParamError> {
        match self {
            Self::Integer(_) | Self::Real(_) => Ok(()),
            Self::IntervalInteger(_) => match constraint {
                Constraint::Equation(equation) if is_bound_expression(equation.expression()) => {
                    Ok(())
                }
                Constraint::Equation(_) => Err(ParamError::NotABound),
                _ => Err(ParamError::ForbiddenConstraintKind(
                    DomainKind::IntervalInteger,
                )),
            },
            Self::Ordinal(_) | Self::Categorical(_) | Self::Permutation(_) => {
                if is_set_constraint(constraint) {
                    Ok(())
                } else {
                    Err(ParamError::ForbiddenConstraintKind(self.kind()))
                }
            }
            Self::Custom(domain) => domain
                .validate_constraint(constraint, variable)
                .map_err(ParamError::Custom),
        }
    }

    /// Return the constraints the domain imposes on `variable`: the sign
    /// bound of a non-negative integer domain, and none otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn implied_constraints(
        &self,
        variable: &Identifier,
    ) -> Result<Vec<Constraint>, ParamError> {
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
                .implied_constraints(variable)
                .map_err(ParamError::Custom)?,
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
            Self::Custom(domain) => domain.interval_profile().map_err(ParamError::Custom)?,
        })
    }

    /// Return whether the domain's value set is a subset of `other`'s: a
    /// numeric domain's of any domain of its sort, a finite domain's of one
    /// of its kind holding each of its values (a permutation domain's of
    /// one of as many members).
    ///
    /// # Errors
    ///
    /// Returns [`ParamError::Custom`] for a custom domain that fails.
    pub fn is_value_set_subset(&self, other: &Self) -> Result<bool, ParamError> {
        Ok(match self {
            Self::Integer(_) | Self::IntervalInteger(_) | Self::Real(_) => {
                let own = self.symbol_type()?;
                own.is_some() && other.symbol_type()? == own
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
                .is_value_set_subset(other)
                .map_err(ParamError::Custom)?,
        })
    }

    /// Decide whether `own`'s constrained set, over this domain, is a
    /// subset of `other`'s, over `other_domain`.
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

    /// Decide whether some admissible value satisfies `side`'s constraints.
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
    /// effective values into a new domain with no constraint.
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
    /// carry both sides' constraints rescoped to `variable`.
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
            (Self::Custom(left), _) => left.is_structurally_equivalent(other),
            _ => false,
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
