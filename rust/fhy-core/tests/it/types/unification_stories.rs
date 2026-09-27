//! Tests for the environment, template binding, substitution and
//! unification of the built-in types.
//!
//! Ported from `tests/types/test_unification.py`; the traceability table is
//! in `docs/design/python-switch.md`, "S11a.2 implementation notes".

use crate::support::expression::build_call_or_panic;
use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};
use crate::support::types::Equivalent;
use crate::support::types::{
    array, constrained_template, identifier_dimension, index, literal_dimension, scalar, template,
};

use fhy_core::expression::{Expression, LiteralValue, PiecewiseError};
use fhy_core::identifier::Identifier;
use fhy_core::types::{
    CoreDataType, DataType, Dimension, Type, TypeOperation, TypeUnificationEnvironment,
    UnificationError, unify_expressions,
};

fn empty() -> TypeUnificationEnvironment {
    TypeUnificationEnvironment::new()
}

fn int32() -> DataType {
    DataType::Primitive(CoreDataType::Int32)
}

fn float32() -> DataType {
    DataType::Primitive(CoreDataType::Float32)
}

fn reference(identifier: &Identifier) -> Expression {
    Expression::from(identifier.clone())
}

// =============================================================================
// The environment
// =============================================================================

#[test]
fn empty_environment_binds_nothing() {
    let environment = empty();
    let t = Identifier::new("T");

    assert_eq!(environment.data_type_binding(&t), None);
    assert_eq!(environment.type_binding(&t), None);
    assert_eq!(environment.expression_binding(&t), None);
    assert_eq!(environment.data_type_bindings().len(), 0);
}

#[test]
fn with_helpers_return_new_environments_and_leave_the_receiver_alone() {
    let (t, u, n) = (
        Identifier::new("T"),
        Identifier::new("U"),
        Identifier::new("N"),
    );
    let environment = empty();

    let with_data_type = environment.with_data_type_binding(t.clone(), int32());
    let with_type = environment.with_type_binding(u.clone(), scalar(CoreDataType::Int32));
    let with_expression = environment.with_expression_binding(n.clone(), Expression::from(4));

    assert_eq!(environment, empty());
    assert_eq!(with_data_type.data_type_binding(&t), Some(&int32()));
    assert_eq!(
        with_type.type_binding(&u),
        Some(&scalar(CoreDataType::Int32))
    );
    assert_eq!(
        with_expression.expression_binding(&n),
        Some(&Expression::from(4))
    );
    for extended in [&with_data_type, &with_type, &with_expression] {
        assert!(!environment.is_equivalent(extended));
    }
}

#[test]
fn environment_equivalence_compares_bindings_by_value() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let with_int = empty().with_data_type_binding(t.clone(), int32());
    let with_int_again = empty().with_data_type_binding(t.clone(), int32());
    let with_float = empty().with_data_type_binding(t, float32());
    let with_expression = empty().with_expression_binding(n, Expression::from(4));

    assert!(with_int.is_equivalent(&with_int_again));
    assert_eq!(with_int, with_int_again);
    assert!(!with_int.is_equivalent(&with_float));
    assert!(!with_int.is_equivalent(&with_expression));
}

/// Test `type_bindings` lists each type binding once, with its value, and
/// no data-type or expression binding (R2-030).
#[test]
fn type_bindings_lists_the_type_bindings_alone() {
    let (u, v, t, n) = (
        Identifier::new("U"),
        Identifier::new("V"),
        Identifier::new("T"),
        Identifier::new("N"),
    );
    let environment = empty()
        .with_type_binding(u.clone(), scalar(CoreDataType::Int32))
        .with_type_binding(v.clone(), scalar(CoreDataType::Float32))
        .with_data_type_binding(t, int32())
        .with_expression_binding(n, Expression::from(4));

    let mut bindings: Vec<(Identifier, Type)> = environment
        .type_bindings()
        .map(|(name, bound)| (name.clone(), bound.clone()))
        .collect();
    bindings.sort_by_key(|(name, _)| name.id());

    assert_eq!(environment.type_bindings().len(), 2);
    assert_eq!(
        bindings,
        [
            (u, scalar(CoreDataType::Int32)),
            (v, scalar(CoreDataType::Float32))
        ]
    );
    assert_eq!(empty().type_bindings().len(), 0);
}

#[test]
fn environment_equivalence_distinguishes_type_bindings() {
    let u = Identifier::new("U");
    let left = empty().with_type_binding(u.clone(), scalar(CoreDataType::Int32));
    let right = empty().with_type_binding(u, scalar(CoreDataType::Float32));

    assert!(!left.is_equivalent(&right));
}

#[test]
fn chained_with_helpers_produce_distinct_environments() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let first = empty().with_data_type_binding(t, int32());
    let second = first.with_expression_binding(n.clone(), Expression::from(4));
    let third = second.with_expression_binding(n, Expression::from(8));

    assert!(!first.is_equivalent(&second));
    assert!(!second.is_equivalent(&third));
    assert!(!first.is_equivalent(&third));
}

#[test]
fn equal_environments_hash_alike_whatever_the_binding_order() {
    let (m, n) = (Identifier::new("M"), Identifier::new("N"));
    let forward = empty()
        .with_expression_binding(m.clone(), Expression::from(1))
        .with_expression_binding(n.clone(), Expression::from(2));
    let backward = empty()
        .with_expression_binding(n, Expression::from(2))
        .with_expression_binding(m, Expression::from(1));

    assert_eq!(forward, backward);
    assert_eq!(
        crate::support::hashing::hash_of(&forward),
        crate::support::hashing::hash_of(&backward)
    );
}

// =============================================================================
// Binding
// =============================================================================

#[test]
fn full_type_wildcard_binds_the_whole_actual_and_its_data_type() {
    let t = Identifier::new("T");
    let pattern = array(template(&t), [Dimension::Wildcard]);
    let actual = array(
        int32(),
        [
            literal_dimension(4),
            literal_dimension(5),
            literal_dimension(6),
        ],
    );

    let environment = pattern.bind_template(&actual, &empty()).expect("binds");

    assert_eq!(environment.type_binding(&t), Some(&actual));
    assert_eq!(environment.data_type_binding(&t), Some(&int32()));
    assert_eq!(
        pattern
            .substitute_template(&environment)
            .expect("substitutes"),
        actual
    );
}

#[test]
fn wildcard_shape_over_a_concrete_data_type_accepts_any_shape() {
    let pattern = array(float32(), [Dimension::Wildcard]);
    let actual = array(float32(), [literal_dimension(1), literal_dimension(2)]);

    let environment = pattern.bind_template(&actual, &empty()).expect("binds");

    assert_eq!(environment, empty());
}

#[test]
fn full_type_binding_conflicting_with_an_existing_one_is_refused() {
    let t = Identifier::new("T");
    let pattern = array(template(&t), [Dimension::Wildcard]);
    let bound = array(int32(), [literal_dimension(2)]);
    let environment = empty().with_type_binding(t.clone(), bound.clone());
    let actual = array(int32(), [literal_dimension(3)]);

    let error = pattern
        .bind_template(&actual, &environment)
        .expect_err("conflicts");

    assert!(matches!(
        error,
        UnificationError::ConflictingTypeBinding { .. }
    ));
    assert_eq!(
        error.to_string(),
        format!(
            "conflicting full-type binding for T::{}: int32[2] vs int32[3]",
            t.id()
        )
    );
}

#[test]
fn shape_rank_mismatch_is_refused() {
    let pattern = array(float32(), [literal_dimension(4)]);
    let actual = array(float32(), [literal_dimension(4), literal_dimension(5)]);

    let error = pattern
        .bind_template(&actual, &empty())
        .expect_err("ranks differ");

    assert!(matches!(
        error,
        UnificationError::RankMismatch {
            operation: TypeOperation::Bind,
            expected: 1,
            actual: 2
        }
    ));
    assert_eq!(
        error.to_string(),
        "shape rank mismatch: pattern has 1 dimensions, actual has 2"
    );
}

#[test]
fn numerical_pattern_against_an_index_type_is_refused() {
    let error = scalar(CoreDataType::Float32)
        .bind_template(&index(0, 10, 1), &empty())
        .expect_err("kinds differ");

    assert_eq!(
        error.to_string(),
        "cannot bind NumericalType pattern against IndexType"
    );
}

#[test]
fn index_pattern_against_a_numerical_type_is_refused() {
    let error = index(0, 10, 1)
        .bind_template(&scalar(CoreDataType::Int8), &empty())
        .expect_err("kinds differ");

    assert_eq!(
        error.to_string(),
        "cannot bind IndexType pattern against NumericalType"
    );
}

#[test]
fn a_shape_variable_met_twice_with_different_extents_is_refused() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let pattern = array(
        template(&t),
        [identifier_dimension(&n), identifier_dimension(&n)],
    );
    let actual = array(int32(), [literal_dimension(4), literal_dimension(5)]);

    let error = pattern
        .bind_template(&actual, &empty())
        .expect_err("N is 4 and 5");

    assert_eq!(
        error.to_string(),
        format!(
            "conflicting binding for shape variable N::{}: 4 vs 5",
            n.id()
        )
    );
}

#[test]
fn a_concrete_pattern_dimension_must_equal_the_actual_one() {
    let error = array(int32(), [literal_dimension(4)])
        .bind_template(&array(int32(), [literal_dimension(5)]), &empty())
        .expect_err("4 is not 5");

    assert_eq!(error.to_string(), "shape dimension mismatch: 4 vs 5");
}

#[test]
fn a_wildcard_dimension_of_the_pattern_matches_any_one_dimension() {
    let n = Identifier::new("N");
    let pattern = array(int32(), [Dimension::Wildcard, identifier_dimension(&n)]);
    let actual = array(int32(), [literal_dimension(3), literal_dimension(7)]);

    let environment = pattern.bind_template(&actual, &empty()).expect("binds");

    assert_eq!(
        environment.expression_binding(&n),
        Some(&Expression::from(7))
    );
}

#[test]
fn a_wildcard_dimension_of_the_actual_is_refused() {
    let n = Identifier::new("N");
    let pattern = array(int32(), [identifier_dimension(&n), literal_dimension(2)]);
    let actual = array(int32(), [literal_dimension(3), Dimension::Wildcard]);

    let error = pattern
        .bind_template(&actual, &empty())
        .expect_err("a wildcard in the actual");

    assert!(matches!(error, UnificationError::WildcardInActual));
    assert_eq!(
        error.to_string(),
        "wildcard `...` cannot appear in `actual` during template binding"
    );
}

#[test]
fn index_pattern_binds_its_bounds_as_shape_variables() {
    let (lower, upper) = (Identifier::new("L"), Identifier::new("U"));
    let pattern = index(reference(&lower), reference(&upper), 1);

    let environment = pattern
        .bind_template(&index(0, 16, 1), &empty())
        .expect("binds");

    assert_eq!(
        environment.expression_binding(&lower),
        Some(&Expression::from(0))
    );
    assert_eq!(
        environment.expression_binding(&upper),
        Some(&Expression::from(16))
    );
}

#[test]
fn data_type_binding_conflicting_with_an_existing_one_is_refused() {
    let t = Identifier::new("T");
    let environment = template(&t)
        .bind_template(&int32(), &empty())
        .expect("binds");

    let error = template(&t)
        .bind_template(&float32(), &environment)
        .expect_err("conflicts");

    assert_eq!(
        error.to_string(),
        format!(
            "conflicting data-type binding for T::{}: int32 vs float32",
            t.id()
        )
    );
}

#[test]
fn repeated_consistent_data_type_binding_changes_nothing() {
    let t = Identifier::new("T");
    let environment = template(&t)
        .bind_template(&int32(), &empty())
        .expect("binds");

    let again = template(&t)
        .bind_template(&int32(), &environment)
        .expect("binds");

    assert!(environment.is_equivalent(&again));
}

#[test]
fn width_constraint_accepts_a_listed_width_in_any_order() {
    let t = Identifier::new("T");
    for widths in [&[8, 16][..], &[16, 8, 8, 16]] {
        let environment = constrained_template(&t, widths)
            .bind_template(&DataType::Primitive(CoreDataType::Int16), &empty())
            .expect("16 is listed");
        assert_eq!(
            environment.data_type_binding(&t),
            Some(&DataType::Primitive(CoreDataType::Int16))
        );
    }
}

#[test]
fn width_constraint_refuses_an_unlisted_width() {
    let t = Identifier::new("T");

    let error = constrained_template(&t, &[8, 16])
        .bind_template(&int32(), &empty())
        .expect_err("32 is not listed");

    assert_eq!(
        error.to_string(),
        format!(
            "width mismatch for template T::{}: actual int32 has width 32, not in [8, 16]",
            t.id()
        )
    );
}

#[test]
fn width_constraint_refuses_a_weak_type() {
    let t = Identifier::new("T");

    let weak = constrained_template(&t, &[8])
        .bind_template(&DataType::Primitive(CoreDataType::Int), &empty())
        .expect_err("a weak type has no width");

    assert!(weak.to_string().contains("has no width"));
}

#[test]
fn width_constraint_refuses_a_non_primitive_actual() {
    let (t, u) = (Identifier::new("T"), Identifier::new("U"));
    let pattern = array(constrained_template(&t, &[8]), [literal_dimension(1)]);
    let actual = array(constrained_template(&u, &[8]), [literal_dimension(1)]);

    let error = pattern
        .bind_template(&actual, &empty())
        .expect_err("distinct templates");

    assert!(error.to_string().contains("template"));
}

#[test]
fn unconstrained_template_binds_any_width() {
    let t = Identifier::new("T");

    let environment = template(&t)
        .bind_template(&DataType::Primitive(CoreDataType::Float64), &empty())
        .expect("binds");

    assert_eq!(
        environment.data_type_binding(&t),
        Some(&DataType::Primitive(CoreDataType::Float64))
    );
}

#[test]
fn template_against_another_template_is_refused_and_against_itself_binds_nothing() {
    let (t, u) = (Identifier::new("T"), Identifier::new("U"));

    let error = template(&t)
        .bind_template(&template(&u), &empty())
        .expect_err("distinct");
    let environment = template(&t)
        .bind_template(&template(&t), &empty())
        .expect("the same");

    assert_eq!(
        error.to_string(),
        format!(
            "cannot bind distinct template data types: T::{} vs U::{}",
            t.id(),
            u.id()
        )
    );
    assert_eq!(environment, empty());
}

#[test]
fn primitive_pattern_needs_the_same_primitive_actual() {
    let t = Identifier::new("T");

    let mismatch = int32()
        .bind_template(&float32(), &empty())
        .expect_err("differ");
    let kind = int32()
        .bind_template(&template(&t), &empty())
        .expect_err("a template");

    assert_eq!(
        mismatch.to_string(),
        "core data type mismatch: int32 vs float32"
    );
    assert_eq!(
        kind.to_string(),
        "cannot bind PrimitiveDataType pattern against TemplateDataType"
    );
    assert_eq!(
        int32().bind_template(&int32(), &empty()).expect("the same"),
        empty()
    );
}

#[test]
fn width_constraint_holds_through_a_numerical_pattern_and_unification() {
    let t = Identifier::new("T");
    let pattern = array(constrained_template(&t, &[8]), [literal_dimension(4)]);
    let actual = array(int32(), [literal_dimension(4)]);

    let bound = pattern
        .bind_template(&actual, &empty())
        .expect_err("32 is not 8");
    let unified = pattern.unify(&actual, &empty()).expect_err("32 is not 8");

    assert!(bound.to_string().contains("width"));
    assert!(unified.to_string().contains("width"));
}

// =============================================================================
// Substitution
// =============================================================================

#[test]
fn substitution_leaves_unbound_placeholders_and_returns_the_same_type() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let pattern = array(template(&t), [identifier_dimension(&n)]);

    let substituted = pattern.substitute_template(&empty()).expect("substitutes");

    assert!(Type::ptr_eq(&substituted, &pattern));
}

#[test]
fn substitution_walks_compound_shape_expressions() {
    let n = Identifier::new("N");
    let pattern = array(float32(), [Dimension::Expression(reference(&n) + 1)]);
    let environment = empty().with_expression_binding(n, Expression::from(8));

    let substituted = pattern
        .substitute_template(&environment)
        .expect("substitutes");

    assert_eq!(
        substituted,
        array(float32(), [Dimension::Expression(Expression::from(8) + 1)])
    );
}

#[test]
fn substitution_reaches_shape_variables_inside_calls_and_piecewise() {
    let m = Identifier::new("M");
    let call = build_call_or_panic("max", [reference(&m), Expression::from(1)]);
    let piecewise = Expression::piecewise([(reference(&m).greater(0), reference(&m))], 0)
        .expect("a valid piecewise");
    let pattern = array(
        int32(),
        [
            Dimension::Expression(call),
            Dimension::Expression(piecewise),
        ],
    );
    let environment = empty().with_expression_binding(m, Expression::from(4));

    let substituted = pattern
        .substitute_template(&environment)
        .expect("substitutes");

    let expected_call = build_call_or_panic("max", [Expression::from(4), Expression::from(1)]);
    let expected_piecewise =
        Expression::piecewise([(Expression::from(4).greater(0), Expression::from(4))], 0)
            .expect("a valid piecewise");
    assert_eq!(
        substituted,
        array(
            int32(),
            [
                Dimension::Expression(expected_call),
                Dimension::Expression(expected_piecewise)
            ]
        )
    );
}

#[test]
fn substitution_follows_a_chain_of_bindings_and_stops_at_a_cycle() {
    let (m, n) = (Identifier::new("M"), Identifier::new("N"));
    let chained = empty()
        .with_expression_binding(n.clone(), reference(&m) + 1)
        .with_expression_binding(m.clone(), Expression::from(2));
    let cyclic = empty()
        .with_expression_binding(n.clone(), reference(&m) + 1)
        .with_expression_binding(m.clone(), reference(&n));
    let pattern = array(int32(), [identifier_dimension(&n)]);

    let through_chain = pattern.substitute_template(&chained).expect("substitutes");
    let through_cycle = pattern.substitute_template(&cyclic).expect("substitutes");

    assert_eq!(
        through_chain,
        array(int32(), [Dimension::Expression(Expression::from(2) + 1)])
    );
    assert_eq!(
        through_cycle,
        array(int32(), [Dimension::Expression(reference(&n) + 1)])
    );
}

#[test]
fn substitution_keeps_a_wildcard_and_substitutes_the_data_type() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let pattern = array(
        template(&t),
        [Dimension::Wildcard, identifier_dimension(&n)],
    );
    let environment = empty()
        .with_data_type_binding(t, int32())
        .with_expression_binding(n, Expression::from(3));

    let substituted = pattern
        .substitute_template(&environment)
        .expect("substitutes");

    assert_eq!(
        substituted,
        array(int32(), [Dimension::Wildcard, literal_dimension(3)])
    );
}

#[test]
fn substitution_of_an_index_type_substitutes_its_bounds() {
    let n = Identifier::new("N");
    let environment = empty().with_expression_binding(n.clone(), Expression::from(9));

    let substituted = index(0, reference(&n), 1)
        .substitute_template(&environment)
        .expect("substitutes");

    assert_eq!(substituted, index(0, 9, 1));
}

#[test]
fn data_type_substitution_resolves_a_bound_template_and_leaves_others() {
    let (t, u) = (Identifier::new("T"), Identifier::new("U"));
    let environment = empty().with_data_type_binding(t.clone(), int32());

    assert_eq!(
        template(&t)
            .substitute_template(&environment)
            .expect("substitutes"),
        int32()
    );
    assert_eq!(
        template(&u)
            .substitute_template(&environment)
            .expect("substitutes"),
        template(&u)
    );
    assert_eq!(
        float32()
            .substitute_template(&environment)
            .expect("substitutes"),
        float32()
    );
}

// =============================================================================
// Unification
// =============================================================================

#[test]
fn unification_binds_a_placeholder_on_either_side() {
    let (n, m) = (Identifier::new("N"), Identifier::new("M"));
    let expected = array(
        float32(),
        [identifier_dimension(&n), identifier_dimension(&m)],
    );
    let actual = array(float32(), [literal_dimension(10), identifier_dimension(&m)]);

    let (unified, environment) = expected.unify(&actual, &empty()).expect("unifies");

    assert_eq!(
        unified,
        array(float32(), [literal_dimension(10), identifier_dimension(&m)])
    );
    assert_eq!(
        environment.expression_binding(&n),
        Some(&Expression::from(10))
    );
    assert_eq!(environment.expression_binding(&m), None);
}

#[test]
fn unification_fails_the_occurs_check() {
    let n = Identifier::new("N");
    let expected = array(float32(), [identifier_dimension(&n)]);
    let actual = array(float32(), [Dimension::Expression(reference(&n) + 1)]);

    let error = expected
        .unify(&actual, &empty())
        .expect_err("N occurs in N + 1");

    assert_eq!(
        error.to_string(),
        format!(
            "occurs check failed: identifier N::{id} appears in (N::{id} + 1) after substitution through existing bindings ((N::{id} + 1))",
            id = n.id()
        )
    );
}

#[test]
fn unification_of_equal_index_types_binds_nothing() {
    let (unified, environment) = index(0, 10, 1)
        .unify(&index(0, 10, 1), &empty())
        .expect("unifies");

    assert_eq!(unified, index(0, 10, 1));
    assert_eq!(environment, empty());
}

#[test]
fn unification_of_different_kinds_is_refused() {
    let error = scalar(CoreDataType::Float32)
        .unify(&index(0, 10, 1), &empty())
        .expect_err("kinds");
    let reverse = index(0, 10, 1)
        .unify(&scalar(CoreDataType::Float32), &empty())
        .expect_err("kinds");

    assert_eq!(
        error.to_string(),
        "cannot unify NumericalType with IndexType"
    );
    assert_eq!(
        reverse.to_string(),
        "cannot unify IndexType with NumericalType"
    );
}

#[test]
fn unification_checks_the_rank_before_the_data_types() {
    let error = array(int32(), [literal_dimension(1)])
        .unify(
            &array(float32(), [literal_dimension(1), literal_dimension(2)]),
            &empty(),
        )
        .expect_err("ranks differ");

    assert_eq!(
        error.to_string(),
        "shape rank mismatch during unification: 1 vs 2"
    );
}

#[test]
fn unification_binds_a_data_type_template_on_either_side() {
    let t = Identifier::new("T");
    let with_template = array(template(&t), [literal_dimension(4)]);
    let concrete = array(int32(), [literal_dimension(4)]);

    let (left_unified, left_environment) =
        with_template.unify(&concrete, &empty()).expect("unifies");
    let (right_unified, right_environment) =
        concrete.unify(&with_template, &empty()).expect("unifies");

    assert_eq!(left_unified, concrete);
    assert_eq!(right_unified, concrete);
    assert_eq!(left_environment.data_type_binding(&t), Some(&int32()));
    assert_eq!(right_environment.data_type_binding(&t), Some(&int32()));
}

#[test]
fn unification_refuses_distinct_templates_even_under_one_name() {
    let (t, same_name) = (Identifier::new("T"), Identifier::new("T"));

    let error = array(template(&t), [literal_dimension(4)])
        .unify(
            &array(template(&same_name), [literal_dimension(4)]),
            &empty(),
        )
        .expect_err("distinct identifiers");

    assert_eq!(
        error.to_string(),
        format!(
            "cannot unify distinct template data types: T::{} vs T::{}",
            t.id(),
            same_name.id()
        )
    );
}

#[test]
fn unification_accepts_a_template_with_itself() {
    let t = Identifier::new("T");
    let expected = array(template(&t), [literal_dimension(4)]);

    let (unified, environment) = expected.unify(&expected, &empty()).expect("unifies");

    assert_eq!(unified, expected);
    assert_eq!(environment, empty());
}

#[test]
fn unification_refuses_different_concrete_data_types() {
    let error = scalar(CoreDataType::Int32)
        .unify(&scalar(CoreDataType::Int16), &empty())
        .expect_err("differ");

    assert_eq!(
        error.to_string(),
        "data type mismatch during unification: int32 vs int16"
    );
}

#[test]
fn unification_refuses_a_wildcard_dimension() {
    let error = array(int32(), [Dimension::Wildcard])
        .unify(&array(int32(), [literal_dimension(1)]), &empty())
        .expect_err("a wildcard");

    assert!(matches!(error, UnificationError::WildcardInUnification));
    assert_eq!(
        error.to_string(),
        "wildcard `...` is not supported during unification"
    );
}

#[test]
fn unification_of_index_types_refuses_a_different_literal_bound() {
    let error = index(0, 10, 1)
        .unify(&index(0, 11, 1), &empty())
        .expect_err("10 is not 11");

    assert_eq!(error.to_string(), "cannot unify expressions 10 and 11");
}

// =============================================================================
// Unifying expressions
// =============================================================================

#[test]
fn equal_concrete_expressions_unify_unchanged() {
    let (unified, environment) =
        unify_expressions(&Expression::from(7), &Expression::from(7), &empty()).expect("unifies");

    assert_eq!(unified, Expression::from(7));
    assert_eq!(environment, empty());
}

#[test]
fn a_placeholder_on_either_side_binds_the_other_side() {
    let n = Identifier::new("N");

    let (left_unified, left) =
        unify_expressions(&reference(&n), &Expression::from(10), &empty()).expect("unifies");
    let (right_unified, right) =
        unify_expressions(&Expression::from(10), &reference(&n), &empty()).expect("unifies");

    assert_eq!(left_unified, Expression::from(10));
    assert_eq!(right_unified, Expression::from(10));
    assert_eq!(left.expression_binding(&n), Some(&Expression::from(10)));
    assert_eq!(right.expression_binding(&n), Some(&Expression::from(10)));
}

#[test]
fn a_bound_placeholder_resolves_to_its_binding_first() {
    let n = Identifier::new("N");
    let environment = empty().with_expression_binding(n.clone(), Expression::from(1) + 2);

    let (unified, next) =
        unify_expressions(&reference(&n), &(Expression::from(1) + 2), &environment)
            .expect("unifies");

    assert_eq!(unified, Expression::from(1) + 2);
    assert!(next.is_equivalent(&environment));
}

#[test]
fn distinct_concrete_expressions_do_not_unify() {
    let error = unify_expressions(&Expression::from(1), &Expression::from(2), &empty())
        .expect_err("differ");

    assert_eq!(error.to_string(), "cannot unify expressions 1 and 2");
}

#[test]
fn the_occurs_check_looks_inside_every_node_kind() {
    let n = Identifier::new("N");
    let inside = [
        reference(&n) + 1,
        Expression::from(LiteralValue::Bool(true)).and(reference(&n).less(1)),
        build_call_or_panic("max", [reference(&n), Expression::from(1)]),
        Expression::piecewise([(reference(&n).greater(0), Expression::from(1))], 0)
            .expect("a valid piecewise"),
    ];
    for expression in inside {
        let error = unify_expressions(&reference(&n), &expression, &empty()).expect_err("N occurs");
        assert!(
            matches!(error, UnificationError::OccursCheck { .. }),
            "{error}"
        );
    }
}

#[test]
fn the_occurs_check_follows_bindings_on_either_side() {
    let (m, n) = (Identifier::new("M"), Identifier::new("N"));
    let environment = empty().with_expression_binding(m.clone(), reference(&n));

    let into_logical = unify_expressions(
        &reference(&n),
        &Expression::from(LiteralValue::Bool(false)).or(reference(&m)),
        &environment,
    );
    let on_left = unify_expressions(&reference(&n), &(reference(&m) + 1), &environment);
    let on_right = unify_expressions(&(reference(&m) + 1), &reference(&n), &environment);

    assert!(matches!(
        into_logical,
        Err(UnificationError::OccursCheck { .. })
    ));
    assert!(matches!(on_left, Err(UnificationError::OccursCheck { .. })));
    assert!(matches!(
        on_right,
        Err(UnificationError::OccursCheck { .. })
    ));
}

/// The TYP probe's environment: `C := 5` and `Y := X + 1`, with the
/// piecewise `{Y if C; 0 otherwise}`, whose substitution puts the literal
/// `5` in a condition, which `Expression::substitute` refuses.
fn refused_substitution_case() -> (
    Identifier,
    TypeUnificationEnvironment,
    Expression,
    Expression,
) {
    let (c, x, y) = (
        Identifier::new("C"),
        Identifier::new("X"),
        Identifier::new("Y"),
    );
    let environment = empty()
        .with_expression_binding(c.clone(), Expression::from(5))
        .with_expression_binding(y.clone(), reference(&x) + 1);
    let piecewise = Expression::piecewise([(reference(&c), reference(&y))], 0)
        .expect("an identifier condition");
    (x, environment, piecewise, reference(&y) * 2)
}

#[test]
fn unifying_through_a_refused_substitution_is_an_error() {
    let (x, environment, piecewise, _) = refused_substitution_case();

    let error = unify_expressions(&reference(&x), &piecewise, &environment)
        .expect_err("the substitution is refused");

    let UnificationError::Substitution(source) = &error else {
        panic!("a substitution error, got {error}");
    };
    assert_eq!(
        *source,
        PiecewiseError::NonBooleanConditionLiteral { case_index: 0 }
    );
    assert_eq!(
        error.to_string(),
        "substituting the existing shape bindings was refused"
    );
    let chained = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<PiecewiseError>());
    assert_eq!(chained, Some(source));
}

#[test]
fn substitute_template_through_a_refused_substitution_is_an_error() {
    let (_, environment, piecewise, _) = refused_substitution_case();
    let pattern = array(int32(), [Dimension::Expression(piecewise.clone())]);
    let index_type = Type::Index(fhy_core::types::IndexType::new(
        Expression::from(0),
        piecewise,
        Expression::from(1),
    ));

    for value in [pattern, index_type] {
        let error = value
            .substitute_template(&environment)
            .expect_err("the substitution is refused");
        assert!(
            matches!(
                error,
                UnificationError::Substitution(PiecewiseError::NonBooleanConditionLiteral { .. })
            ),
            "{error}"
        );
    }
}

#[test]
fn a_placeholder_behind_a_bound_variable_still_fails_the_occurs_check() {
    let (x, environment, _, doubled) = refused_substitution_case();

    let error = unify_expressions(&reference(&x), &doubled, &environment)
        .expect_err("X occurs in Y * 2 through Y := X + 1");

    assert!(
        matches!(&error, UnificationError::OccursCheck { identifier, .. } if *identifier == x),
        "{error}"
    );
}

#[test]
fn a_concrete_binding_is_no_indirect_cycle() {
    let (m, n) = (Identifier::new("M"), Identifier::new("N"));
    let environment = empty().with_expression_binding(m.clone(), Expression::from(5));
    let right = reference(&m) + 1;

    let (unified, next) = unify_expressions(&reference(&n), &right, &environment).expect("unifies");

    assert_eq!(unified, right);
    assert_eq!(next.expression_binding(&n), Some(&right));
}

#[test]
fn unification_chains_through_an_existing_placeholder_binding() {
    let (x, y, z) = (
        Identifier::new("X"),
        Identifier::new("Y"),
        Identifier::new("Z"),
    );
    let environment = empty().with_expression_binding(x.clone(), reference(&y));

    let (_, next) =
        unify_expressions(&reference(&x), &reference(&z), &environment).expect("unifies");

    assert_eq!(next.expression_binding(&y), Some(&reference(&z)));
    assert_eq!(next.expression_binding(&x), Some(&reference(&y)));
}

#[test]
fn a_cycle_of_placeholder_bindings_resolves_to_where_it_closes() {
    let (m, n) = (Identifier::new("M"), Identifier::new("N"));
    let environment = empty()
        .with_expression_binding(m.clone(), reference(&n))
        .with_expression_binding(n.clone(), reference(&m));

    let (unified, next) =
        unify_expressions(&reference(&m), &reference(&m), &environment).expect("unifies");

    assert_eq!(unified, reference(&m));
    assert_eq!(next, environment);
}

/// Bindings shaped `N_i := N_{i+1} + N_{i+2}` reach each `N_i` along
/// Fibonacci-many paths: substitution must treat each once (F2-038). At
/// n = 64 an unmemoized substitution would not finish.
#[test]
fn fibonacci_bindings_substitute_in_linear_time() {
    const N: usize = 64;
    let names: Vec<Identifier> = (0..N + 2)
        .map(|index| Identifier::new(&format!("N{index}")))
        .collect();
    let environment = (0..N).fold(empty(), |environment, index| {
        environment.with_expression_binding(
            names[index].clone(),
            reference(&names[index + 1]) + reference(&names[index + 2]),
        )
    });
    let pattern = array(int32(), [identifier_dimension(&names[0])]);

    let start = std::time::Instant::now();
    let substituted = pattern
        .substitute_template(&environment)
        .expect("substitutes");
    let elapsed = start.elapsed();

    let Type::Numerical(numerical) = &substituted else {
        panic!("a numerical type");
    };
    let Dimension::Expression(dimension) = &numerical.shape()[0] else {
        panic!("an expression dimension");
    };
    let free = dimension.free_identifiers();
    assert_eq!(
        free,
        [names[N].clone(), names[N + 1].clone()]
            .into_iter()
            .collect()
    );
    assert!(elapsed < std::time::Duration::from_secs(2), "{elapsed:?}");
    let (x, occurs) = (&names[N + 1], reference(&names[0]) * 2);
    assert!(matches!(
        unify_expressions(&reference(x), &occurs, &environment),
        Err(UnificationError::OccursCheck { .. })
    ));
}

/// A chain `N_i := N_{i+1} + 1` of 100,000 bindings builds and substitutes
/// without recursing once per binding.
#[test]
fn a_100000_binding_chain_substitutes_on_a_small_stack() {
    run_on_small_stack(|| {
        let names: Vec<Identifier> = (0..=SMALL_STACK_DEPTH)
            .map(|index| Identifier::new(&format!("N{index}")))
            .collect();
        let environment = (0..SMALL_STACK_DEPTH).fold(empty(), |environment, index| {
            environment
                .with_expression_binding(names[index].clone(), reference(&names[index + 1]) + 1)
        });
        let pattern = array(int32(), [identifier_dimension(&names[0])]);

        let substituted = pattern
            .substitute_template(&environment)
            .expect("substitutes");

        let Type::Numerical(numerical) = &substituted else {
            panic!("a numerical type");
        };
        let Dimension::Expression(dimension) = &numerical.shape()[0] else {
            panic!("an expression dimension");
        };
        let expected: std::collections::HashSet<Identifier> =
            [names[SMALL_STACK_DEPTH].clone()].into_iter().collect();
        assert_eq!(dimension.free_identifiers(), expected);
        assert_eq!(environment.expression_bindings().len(), SMALL_STACK_DEPTH);
        let last = &names[SMALL_STACK_DEPTH];
        assert!(matches!(
            unify_expressions(&reference(last), &reference(&names[0]), &environment),
            Err(UnificationError::OccursCheck { .. })
        ));
    });
}

#[test]
fn deep_dimensions_bind_substitute_and_unify_on_a_small_stack() {
    run_on_small_stack(|| {
        let n = Identifier::new("N");
        let deep = (0..SMALL_STACK_DEPTH).fold(reference(&n), |tree, _| tree + 1);
        let pattern = array(int32(), [Dimension::Expression(deep.clone())]);
        let environment = empty().with_expression_binding(n.clone(), Expression::from(1));
        let substituted = pattern
            .substitute_template(&environment)
            .expect("substitutes");
        let (_, unified) = pattern.unify(&pattern, &environment).expect("unifies");
        assert!(!Type::ptr_eq(&substituted, &pattern));
        let bound = pattern.bind_template(&pattern, &empty()).expect("binds");
        assert_eq!(unified, environment);
        assert_eq!(bound, empty());
        let occurs = unify_expressions(&reference(&n), &deep, &empty());
        assert!(matches!(occurs, Err(UnificationError::OccursCheck { .. })));
    });
}

#[test]
fn from_bindings_holds_the_three_tables() {
    let (t, f, n) = (
        Identifier::new("T"),
        Identifier::new("F"),
        Identifier::new("N"),
    );
    let environment = TypeUnificationEnvironment::from_bindings(
        [(t.clone(), int32())].into_iter().collect(),
        [(f.clone(), scalar(CoreDataType::Bool))]
            .into_iter()
            .collect(),
        [(n.clone(), Expression::from(4))].into_iter().collect(),
    );
    let built = empty()
        .with_data_type_binding(t.clone(), int32())
        .with_type_binding(f.clone(), scalar(CoreDataType::Bool))
        .with_expression_binding(n.clone(), Expression::from(4));

    assert_eq!(environment.data_type_binding(&t), Some(&int32()));
    assert_eq!(
        environment.type_binding(&f),
        Some(&scalar(CoreDataType::Bool))
    );
    assert_eq!(
        environment.expression_binding(&n),
        Some(&Expression::from(4))
    );
    assert_eq!(environment.data_type_bindings().len(), 1);
    assert_eq!(environment, built);
    assert!(
        environment
            .is_structurally_equivalent(&built)
            .expect("no extension")
    );
    assert_eq!(
        TypeUnificationEnvironment::from_bindings(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            std::collections::HashMap::new()
        ),
        empty()
    );
}

#[test]
fn an_index_type_substitution_keeps_its_handle_unless_a_bound_changes() {
    let n = Identifier::new("N");
    let pattern = index(0, 8, 1);
    let symbolic = Type::Index(fhy_core::types::IndexType::new(
        Expression::from(0),
        reference(&n),
        Expression::from(1),
    ));
    let environment = empty().with_expression_binding(n.clone(), Expression::from(8));

    let unchanged = pattern
        .substitute_template(&environment)
        .expect("substitutes");
    let unbound = symbolic.substitute_template(&empty()).expect("substitutes");
    let changed = symbolic
        .substitute_template(&environment)
        .expect("substitutes");

    assert!(Type::ptr_eq(&unchanged, &pattern));
    assert!(Type::ptr_eq(&unbound, &symbolic));
    assert!(!Type::ptr_eq(&changed, &symbolic));
    assert_eq!(changed, pattern);
}
