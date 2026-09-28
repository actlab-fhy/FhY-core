//! Stories of the six domain kinds: their construction checks, what they
//! admit, which constraints they allow, what they imply, their profiles,
//! value-set subsets and equivalence, and bound expressions.

use fhy_core::constraint::Value;
use fhy_core::expression::{BigInt, Decimal, Expression, SymbolType};
use fhy_core::foreign::Part;
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, DomainError, DomainKind, IntegerDomain, IntervalIntegerDomain,
    IntervalProfile, OrdinalDomain, ParamBuildError, ParamDomain, PermutationDomain, RealDomain,
    is_bound_expression,
};
use fhy_core::param::{Inclusivity, Sign, ZeroInclusion};
use rstest::rstest;

use crate::support::constraint::ConstraintKey;
use crate::support::constraint::{TestOpaque, int, text};
use crate::support::param::{
    Level, above, at_least, boolean, describe_all, float, in_set, ints, less_than, literal,
    reference,
};

fn ordinal(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(OrdinalDomain::new(values).expect("an ordinal domain"))
}

fn categorical(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(CategoricalDomain::new(values).expect("a categorical domain"))
}

fn permutation(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(PermutationDomain::new(values).expect("a permutation domain"))
}

fn not_member_shaped() -> Value {
    TestOpaque {
        is_member_shaped: false,
        ..TestOpaque::token(1)
    }
    .into_value()
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[rstest]
#[case::ordinal(DomainKind::Ordinal)]
#[case::categorical(DomainKind::Categorical)]
#[case::permutation(DomainKind::Permutation)]
fn finite_domain_refuses_no_value(#[case] kind: DomainKind) {
    let result = match kind {
        DomainKind::Ordinal => OrdinalDomain::new(Vec::new()).map(ParamDomain::from),
        DomainKind::Categorical => CategoricalDomain::new(Vec::new()).map(ParamDomain::from),
        _ => PermutationDomain::new(Vec::new()).map(ParamDomain::from),
    };

    let error = result.expect_err("no value");

    assert!(matches!(error, DomainError::EmptyValues(refused) if refused == kind));
    assert!(error.to_string().contains("non-empty"));
}

#[rstest]
#[case::tuple(Value::Tuple(vec![int(1)]))]
#[case::frozen_set(Value::FrozenSet(vec![int(1)]))]
#[case::decimal(Value::Decimal("1.5".parse::<Decimal>().expect("a decimal")))]
#[case::opaque_that_is_no_member(not_member_shaped())]
fn finite_domain_refuses_a_value_that_is_no_leaf(#[case] value: Value) {
    let error = OrdinalDomain::new(vec![int(1), value]).expect_err("no leaf");

    assert!(matches!(
        error,
        DomainError::NotALeafValue {
            kind: DomainKind::Ordinal,
            index: 1
        }
    ));
}

#[test]
fn categorical_domain_refuses_a_float() {
    let error = CategoricalDomain::new(vec![text("a"), float(1.5)]).expect_err("a float");

    assert!(matches!(
        error,
        DomainError::NotALeafValue {
            kind: DomainKind::Categorical,
            index: 1
        }
    ));
}

#[rstest]
#[case::ordinal(DomainKind::Ordinal)]
#[case::permutation(DomainKind::Permutation)]
fn finite_domain_refuses_a_nan(#[case] kind: DomainKind) {
    let values = vec![float(1.0), float(f64::NAN)];
    let result = match kind {
        DomainKind::Ordinal => OrdinalDomain::new(values).map(ParamDomain::from),
        _ => PermutationDomain::new(values).map(ParamDomain::from),
    };

    let error = result.expect_err("a NaN");

    assert!(matches!(error, DomainError::NanValue(refused) if refused == kind));
    assert!(error.to_string().contains("NaN"));
}

#[test]
fn finite_domain_checks_each_value_s_kind_before_nan() {
    let error = PermutationDomain::new(vec![float(f64::NAN), Value::Tuple(Vec::new())])
        .expect_err("a NaN and a tuple");

    assert!(matches!(error, DomainError::NotALeafValue { index: 1, .. }));
}

#[test]
fn ordinal_domain_checks_order_before_uniqueness() {
    let error =
        OrdinalDomain::new(vec![int(1), int(1), text("a")]).expect_err("incomparable and equal");

    assert!(matches!(error, DomainError::IncomparableValues));
}

#[rstest]
#[case::ordinal(DomainKind::Ordinal)]
#[case::categorical(DomainKind::Categorical)]
#[case::permutation(DomainKind::Permutation)]
fn finite_domain_refuses_equal_values(#[case] kind: DomainKind) {
    let values = vec![int(1), int(2), int(1)];
    let result = match kind {
        DomainKind::Ordinal => OrdinalDomain::new(values).map(ParamDomain::from),
        DomainKind::Categorical => CategoricalDomain::new(values).map(ParamDomain::from),
        _ => PermutationDomain::new(values).map(ParamDomain::from),
    };

    let error = result.expect_err("equal values");

    assert!(matches!(error, DomainError::DuplicateValues(refused) if refused == kind));
    assert!(error.to_string().contains("unique"));
}

#[test]
fn finite_domain_keeps_values_equal_under_python_but_of_different_kinds() {
    let domain = PermutationDomain::new(vec![int(1), boolean(true), float(1.0)])
        .expect("three kinds are three values");

    assert_eq!(domain.values().len(), 3);
}

#[test]
fn categorical_domain_keeps_the_members_canonical_order() {
    let domain = CategoricalDomain::new(vec![text("b"), int(10), int(2), boolean(true), text("a")])
        .expect("a categorical domain");

    assert_eq!(
        describe_all(domain.values()),
        ["bool:true", "int:2", "int:10", "str:a", "str:b"]
    );
}

#[test]
fn permutation_domain_keeps_the_order_given() {
    let domain =
        PermutationDomain::new(vec![text("n"), text("c"), text("h")]).expect("a permutation");

    assert_eq!(describe_all(domain.values()), ["str:n", "str:c", "str:h"]);
}

#[test]
fn integer_domains_store_zero_included_without_non_negative() {
    assert_eq!(
        IntegerDomain::new(Sign::Any, ZeroInclusion::Excluded),
        IntegerDomain::new(Sign::Any, ZeroInclusion::Included)
    );
    assert!(IntegerDomain::new(Sign::Any, ZeroInclusion::Excluded).is_zero_included());
    assert!(!IntegerDomain::new(Sign::NonNegative, ZeroInclusion::Excluded).is_zero_included());
    assert_eq!(
        IntervalIntegerDomain::new(Inclusivity::Inclusive, Sign::Any, ZeroInclusion::Excluded),
        IntervalIntegerDomain::new(Inclusivity::Inclusive, Sign::Any, ZeroInclusion::Included)
    );
}

// ---------------------------------------------------------------------------
// Admissibility
// ---------------------------------------------------------------------------

#[rstest]
#[case::integer(int(3), true)]
#[case::big_integer(Value::Int(BigInt::from(10).pow(30)), true)]
#[case::boolean(boolean(true), false)]
#[case::float(float(3.0), false)]
#[case::string(text("3"), false)]
fn integer_domains_admit_integers_only(#[case] value: Value, #[case] is_admissible: bool) {
    for domain in [
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Inclusive,
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
    ] {
        assert_eq!(
            domain.is_value_admissible(&value).expect("native"),
            is_admissible
        );
    }
}

/// The domain's sign restriction holds at the domain level, not only once a
/// param folds it into an implied bound.
#[test]
fn non_negative_integer_domain_refuses_a_negative_integer() {
    let domain = ParamDomain::from(IntegerDomain::new(
        Sign::NonNegative,
        ZeroInclusion::Included,
    ));

    assert!(!domain.is_value_admissible(&int(-5)).expect("native"));
    assert!(domain.is_value_admissible(&int(0)).expect("native"));
}

#[rstest]
#[case::finite_float(float(1.5), true)]
#[case::infinity(float(f64::INFINITY), false)]
#[case::nan(float(f64::NAN), false)]
#[case::decimal_text(text("0.1"), true)]
#[case::integer_text(text("7"), true)]
#[case::signed_text(text("-1"), false)]
#[case::exponent_text(text("1e5"), false)]
#[case::nan_text(text("nan"), false)]
#[case::integer(int(1), false)]
#[case::boolean(boolean(true), false)]
fn real_domain_admits_finite_floats_and_literal_text(
    #[case] value: Value,
    #[case] is_admissible: bool,
) {
    let domain = ParamDomain::from(RealDomain);

    assert_eq!(
        domain.is_value_admissible(&value).expect("native"),
        is_admissible
    );
}

#[rstest]
#[case::member(int(2), true)]
#[case::float_equal_to_a_member(float(2.0), false)]
#[case::boolean_equal_to_a_member(boolean(true), false)]
#[case::non_member(int(5), false)]
#[case::tuple(Value::Tuple(vec![int(1)]), false)]
fn finite_domains_admit_their_members_type_strictly(
    #[case] value: Value,
    #[case] is_admissible: bool,
) {
    for domain in [ordinal(ints([1, 2, 3])), categorical(ints([1, 2, 3]))] {
        assert_eq!(
            domain.is_value_admissible(&value).expect("native"),
            is_admissible
        );
    }
}

#[test]
fn finite_domain_admits_an_opaque_value_equal_to_a_member() {
    let domain = ordinal(vec![Level::value(1), Level::value(2)]);

    assert!(
        domain
            .is_value_admissible(&Level::value(2))
            .expect("native")
    );
    assert!(
        !domain
            .is_value_admissible(&Level::grade(2))
            .expect("native")
    );
}

#[rstest]
#[case::permutation(vec![int(3), int(1), int(2)], true)]
#[case::identity(vec![int(1), int(2), int(3)], true)]
#[case::too_short(vec![int(1), int(2)], false)]
#[case::repeated(vec![int(1), int(1), int(2)], false)]
#[case::foreign(vec![int(1), int(2), int(4)], false)]
#[case::other_kind(vec![int(1), int(2), float(3.0)], false)]
fn permutation_domain_admits_tuples_of_each_member_once(
    #[case] elements: Vec<Value>,
    #[case] is_admissible: bool,
) {
    let domain = permutation(ints([1, 2, 3]));

    assert_eq!(
        domain
            .is_value_admissible(&Value::Tuple(elements))
            .expect("native"),
        is_admissible
    );
}

#[test]
fn permutation_domain_does_not_admit_a_member_itself() {
    let domain = permutation(ints([1]));

    assert!(!domain.is_value_admissible(&int(1)).expect("native"));
}

// ---------------------------------------------------------------------------
// Constraints, implied constraints, profiles
// ---------------------------------------------------------------------------

#[test]
fn integer_and_real_domains_allow_any_constraint() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    for domain in [
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        ParamDomain::from(RealDomain),
    ] {
        for constraint in [at_least(&x, 0), in_set(&x, ints([1])), less_than(&x, &y)] {
            domain
                .validate_constraint(&constraint, &x)
                .expect("allows any constraint");
        }
    }
}

#[test]
fn interval_domain_allows_bounds_only() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let domain = ParamDomain::from(IntervalIntegerDomain::new(
        Inclusivity::Inclusive,
        Sign::Any,
        ZeroInclusion::Included,
    ));

    domain
        .validate_constraint(&at_least(&x, 0), &x)
        .expect("allows a bound");
    assert!(matches!(
        domain.validate_constraint(&less_than(&x, &y), &x),
        Err(ParamBuildError::NotABound)
    ));
    let error = domain
        .validate_constraint(&in_set(&x, ints([1])), &x)
        .expect_err("a set constraint");
    assert!(matches!(
        error,
        ParamBuildError::ForbiddenConstraintKind(DomainKind::IntervalInteger)
    ));
}

#[test]
fn finite_domains_allow_set_constraints_only() {
    let x = Identifier::new("x");
    for domain in [
        ordinal(ints([1, 2])),
        categorical(ints([1, 2])),
        permutation(ints([1, 2])),
    ] {
        domain
            .validate_constraint(&in_set(&x, ints([1])), &x)
            .expect("allows a set constraint");
        let error = domain
            .validate_constraint(&at_least(&x, 0), &x)
            .expect_err("an equation");
        assert!(
            matches!(error, ParamBuildError::ForbiddenConstraintKind(kind) if kind == domain.kind())
        );
        assert!(error.to_string().contains("in-set and not-in-set"));
    }
}

#[rstest]
#[case::natural(true, true, Some(">="))]
#[case::positive(true, false, Some(">"))]
#[case::integer(false, true, None)]
fn non_negative_integer_domains_imply_a_sign_bound(
    #[case] non_negative: bool,
    #[case] zero_included: bool,
    #[case] operator: Option<&str>,
) {
    let x = Identifier::new("x");
    for domain in [
        ParamDomain::from(IntegerDomain::new(
            Sign::non_negative_if(non_negative),
            ZeroInclusion::included_if(zero_included),
        )),
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Inclusive,
            Sign::non_negative_if(non_negative),
            ZeroInclusion::included_if(zero_included),
        )),
    ] {
        let implied = domain.implied_constraints(&x).expect("native");
        let expected: Vec<String> = operator
            .map(|operator| {
                let bound = if operator == ">=" {
                    at_least(&x, 0)
                } else {
                    above(&x, 0)
                };
                bound.key()
            })
            .into_iter()
            .collect();
        assert_eq!(
            implied.iter().map(ConstraintKey::key).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn only_the_integer_domains_have_a_profile() {
    assert_eq!(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Excluded
        ))
        .interval_profile()
        .expect("native"),
        Some(IntervalProfile::new(
            Sign::NonNegative,
            ZeroInclusion::Excluded,
            Inclusivity::Inclusive
        ))
    );
    assert_eq!(
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Exclusive,
            Sign::Any,
            ZeroInclusion::Included
        ))
        .interval_profile()
        .expect("native"),
        Some(
            IntervalProfile::new(Sign::Any, ZeroInclusion::Included, Inclusivity::Exclusive)
                .with_only_bounds()
        )
    );
    for domain in [ParamDomain::from(RealDomain), ordinal(ints([1]))] {
        assert_eq!(domain.interval_profile().expect("native"), None);
    }
}

#[rstest]
fn an_interval_profile_is_built_and_read_through_its_api(
    #[values(Sign::Any, Sign::NonNegative)] sign: Sign,
    #[values(ZeroInclusion::Included, ZeroInclusion::Excluded)] zero: ZeroInclusion,
    #[values(Inclusivity::Inclusive, Inclusivity::Exclusive)] preferred: Inclusivity,
) {
    let profile = IntervalProfile::new(sign, zero, preferred);
    let bounds_only = profile.with_only_bounds();

    for built in [profile, bounds_only] {
        assert_eq!(built.is_non_negative(), sign == Sign::NonNegative);
        assert_eq!(built.is_zero_included(), zero == ZeroInclusion::Included);
        assert_eq!(
            built.is_inclusive_preferred(),
            preferred == Inclusivity::Inclusive
        );
    }
    assert!(!profile.is_bounds_only());
    assert!(bounds_only.is_bounds_only());
    assert_ne!(profile, bounds_only);
}

#[test]
fn symbol_types_follow_the_value_space() {
    assert_eq!(
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
            .symbol_type()
            .expect("native"),
        Some(SymbolType::Int)
    );
    assert_eq!(
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Inclusive,
            Sign::Any,
            ZeroInclusion::Included
        ))
        .symbol_type()
        .expect("native"),
        Some(SymbolType::Int)
    );
    assert_eq!(
        ParamDomain::from(RealDomain).symbol_type().expect("native"),
        Some(SymbolType::Real)
    );
    assert_eq!(categorical(ints([1])).symbol_type().expect("native"), None);
}

// ---------------------------------------------------------------------------
// Value-set subsets and equivalence
// ---------------------------------------------------------------------------

#[test]
fn numeric_value_sets_are_subsets_within_one_sort_and_no_stronger_restriction() {
    let integer = ParamDomain::from(IntegerDomain::new(
        Sign::NonNegative,
        ZeroInclusion::Included,
    ));
    let interval = ParamDomain::from(IntervalIntegerDomain::new(
        Inclusivity::Inclusive,
        Sign::Any,
        ZeroInclusion::Included,
    ));
    let real = ParamDomain::from(RealDomain);

    assert!(
        integer
            .is_value_set_subset(&interval, &native_context())
            .expect("native")
    );
    assert!(
        !interval
            .is_value_set_subset(&integer, &native_context())
            .expect("native"),
        "the unrestricted interval integers are no subset of the naturals"
    );
    assert!(
        !integer
            .is_value_set_subset(&real, &native_context())
            .expect("native")
    );
    assert!(
        !real
            .is_value_set_subset(&ordinal(ints([1])), &native_context())
            .expect("native")
    );
}

#[test]
fn finite_value_sets_are_subsets_by_type_strict_membership() {
    assert!(
        ordinal(ints([1, 2]))
            .is_value_set_subset(&ordinal(ints([1, 2, 3])), &native_context())
            .expect("native")
    );
    assert!(
        !ordinal(ints([1, 2]))
            .is_value_set_subset(&ordinal(vec![int(1), float(2.0)]), &native_context())
            .expect("native")
    );
    assert!(
        !ordinal(ints([1]))
            .is_value_set_subset(&categorical(ints([1])), &native_context())
            .expect("native")
    );
    assert!(
        permutation(ints([1, 2]))
            .is_value_set_subset(&permutation(ints([2, 1])), &native_context())
            .expect("native")
    );
    assert!(
        !permutation(ints([1, 2]))
            .is_value_set_subset(&permutation(ints([1, 2, 3])), &native_context())
            .expect("native")
    );
}

#[test]
fn domains_are_equivalent_by_kind_and_contents() {
    assert!(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Excluded
        ))
        .is_structurally_equivalent(&ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Excluded
        )))
    );
    assert!(
        !ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Excluded
        ))
        .is_structurally_equivalent(&ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included
        )))
    );
    assert!(
        !ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
            .is_structurally_equivalent(&ParamDomain::from(IntervalIntegerDomain::new(
                Inclusivity::Inclusive,
                Sign::Any,
                ZeroInclusion::Included
            )))
    );
    assert!(
        ParamDomain::from(RealDomain).is_structurally_equivalent(&ParamDomain::from(RealDomain))
    );
    assert!(ordinal(ints([2, 1])).is_structurally_equivalent(&ordinal(ints([1, 2]))));
    assert!(!ordinal(ints([1])).is_structurally_equivalent(&ordinal(vec![boolean(true)])));
    assert!(
        categorical(vec![text("a"), int(1)])
            .is_structurally_equivalent(&categorical(vec![int(1), text("a")]))
    );
    assert!(!permutation(ints([1, 2])).is_structurally_equivalent(&permutation(ints([2, 1]))));
}

#[test]
fn domains_share_their_values_when_cloned() {
    let domain = OrdinalDomain::new(vec![Value::Opaque(Part::new(TestOpaque::token(1)))])
        .expect("an ordinal domain");
    let clone = domain.clone();

    assert!(std::ptr::eq(domain.values(), clone.values()));
}

// ---------------------------------------------------------------------------
// Bound expressions
// ---------------------------------------------------------------------------

#[test]
fn bound_expressions_compare_an_identifier_with_an_integer_literal() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let x_reference = reference(&x);

    for bound in [
        x_reference.greater_equal(literal(0)),
        x_reference.greater(literal(0)),
        x_reference.less_equal(literal(0)),
        x_reference.less(literal(0)),
        literal(3).less_equal(x_reference.clone()),
    ] {
        assert!(is_bound_expression(&bound), "{bound}");
    }
    for other in [
        x_reference.equals(literal(0)),
        x_reference.less(reference(&y)),
        literal(1).less(literal(2)),
        x_reference.less(Expression::literal(
            fhy_core::expression::LiteralValue::Float(0.5),
        )),
        (&x_reference + 1).less(literal(3)),
        x_reference,
    ] {
        assert!(!is_bound_expression(&other), "{other}");
    }
}

/// Return the context of a solver with no backend, which a question about
/// built-in domains never asks.
fn native_context() -> fhy_core::param::ParamContext<'static> {
    static SOLVER: std::sync::LazyLock<fhy_core::solver::Solver> =
        std::sync::LazyLock::new(fhy_core::solver::Solver::new);
    fhy_core::param::ParamContext::new(&SOLVER)
}
