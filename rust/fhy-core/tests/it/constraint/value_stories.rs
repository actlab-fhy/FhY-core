//! Tests for the values and members of `fhy_core::constraint`: type-strict
//! equality, the refused values, normalization, the canonical order,
//! deduplication, opaque values, lookups, and lifting.
//!
//! The cases mirror `test_member_validation.py` and the member
//! tests of `test_set_constraints.py`.

use fhy_core::constraint::{Member, MemberError, MemberKind, Value};
use fhy_core::expression::{Decimal, LiteralValue};
use rstest::rstest;

use crate::support::constraint::{
    Failing, FailingHook, TestOpaque, TestValueError, describe, int, int_set, member, member_set,
    text,
};

fn decimal(text: &str) -> Value {
    Value::Decimal(text.parse::<Decimal>().expect("a decimal text"))
}

#[test]
fn members_of_a_boolean_an_integer_and_a_float_with_one_value_are_three() {
    let members = member_set([Value::Bool(true), int(1), Value::Float(1.0)]);

    assert_eq!(members.len(), 3);
    assert_ne!(member(Value::Bool(true)), member(int(1)));
    assert_ne!(member(int(1)), member(Value::Float(1.0)));
}

#[test]
fn members_compare_type_strictly_inside_containers() {
    assert_ne!(
        member(Value::Tuple(vec![int(1)])),
        member(Value::Tuple(vec![Value::Bool(true)]))
    );
    assert_ne!(
        member(Value::FrozenSet(vec![int(1)])),
        member(Value::FrozenSet(vec![Value::Float(1.0)]))
    );
    assert_eq!(
        member(Value::Tuple(vec![int(1), text("a")])),
        member(Value::Tuple(vec![int(1), text("a")]))
    );
}

#[rstest]
#[case::bare(Value::Float(f64::NAN))]
#[case::in_a_tuple(Value::Tuple(vec![int(1), Value::Float(f64::NAN)]))]
#[case::in_a_frozen_set(Value::FrozenSet(vec![Value::Tuple(vec![Value::Float(f64::NAN)])]))]
fn member_refuses_a_nan_at_any_depth(#[case] value: Value) {
    let error = Member::try_from(value).expect_err("a NaN is no member");

    assert!(matches!(error, MemberError::Nan), "{error:?}");
    assert!(error.to_string().contains("NaN"), "{error}");
}

#[rstest]
#[case::bare(decimal("1.5"))]
#[case::in_a_tuple(Value::Tuple(vec![decimal("1")]))]
fn member_refuses_a_decimal_at_any_depth(#[case] value: Value) {
    let error = Member::try_from(value).expect_err("a decimal is no member");
    assert!(matches!(error, MemberError::Decimal), "{error:?}");
}

#[test]
fn member_refuses_an_opaque_value_that_is_not_member_shaped() {
    let value = TestOpaque {
        is_member_shaped: false,
        ..TestOpaque::token(1)
    }
    .into_value();

    let error = Member::try_from(value).expect_err("not member-shaped");

    assert!(
        matches!(&error, MemberError::NotMemberShaped { type_name } if type_name == "Token"),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "a member is, or holds, a value of type Token, which is no member kind"
    );
}

#[rstest]
#[case::bare(Value::Float(-0.0))]
#[case::nested(Value::Tuple(vec![Value::Float(-0.0)]))]
fn member_stores_a_negative_zero_as_positive_zero(#[case] value: Value) {
    let stored = member(value);

    let zero = match stored.kind() {
        MemberKind::Float(zero) => zero,
        MemberKind::Tuple(elements) => match elements[0].kind() {
            MemberKind::Float(zero) => zero,
            other => panic!("expected a float, got {other:?}"),
        },
        other => panic!("expected a float or a tuple, got {other:?}"),
    };
    assert!(zero.is_sign_positive(), "{zero}");
    assert_eq!(member(Value::Float(-0.0)), member(Value::Float(0.0)));
}

#[test]
fn member_set_keeps_one_of_equal_members() {
    let members = member_set([
        int(1),
        int(2),
        int(1),
        Value::Float(-0.0),
        Value::Float(0.0),
    ]);

    assert_eq!(members.len(), 3);
}

#[test]
fn member_set_orders_kinds_bool_float_frozenset_int_str_tuple_then_opaque() {
    let members = member_set([
        TestOpaque::token(1).into_value(),
        Value::Tuple(vec![int(2), int(3)]),
        text("1"),
        int(1),
        Value::FrozenSet(vec![int(5), int(4)]),
        Value::Float(1.0),
        Value::Bool(true),
    ]);

    let kinds: Vec<String> = members
        .iter()
        .map(|member| member.kind_name().into_owned())
        .collect();

    assert_eq!(
        kinds,
        ["bool", "float", "frozenset", "int", "str", "tuple", "Token"]
    );
}

#[test]
fn member_set_orders_values_within_a_kind() {
    assert_eq!(
        int_set([10, 2, -3])
            .iter()
            .map(|member| format!("{:?}", member.kind()))
            .collect::<Vec<_>>(),
        ["Int(-3)", "Int(2)", "Int(10)"]
    );
    assert_eq!(
        describe(&member_set([
            Value::Float(2.5),
            Value::Float(-1.0),
            Value::Float(0.5)
        ])),
        ["Float(-1.0)", "Float(0.5)", "Float(2.5)"]
    );
    assert_eq!(
        describe(&member_set([text("b"), text("B"), text("a"), text("")])),
        [r#"Str("")"#, r#"Str("B")"#, r#"Str("a")"#, r#"Str("b")"#]
    );
    assert_eq!(
        describe(&member_set([Value::Bool(true), Value::Bool(false)])),
        ["Bool(false)", "Bool(true)"]
    );
}

#[test]
fn member_set_orders_tuples_element_by_element_and_a_prefix_first() {
    let members = member_set([
        Value::Tuple(vec![int(1), int(2)]),
        Value::Tuple(vec![int(1)]),
        Value::Tuple(vec![int(0), int(9)]),
        Value::Tuple(vec![]),
    ]);

    let lengths_and_heads: Vec<(usize, Option<String>)> = members
        .iter()
        .map(|member| match member.kind() {
            MemberKind::Tuple(elements) => (
                elements.len(),
                elements.first().map(|head| format!("{:?}", head.kind())),
            ),
            other => panic!("expected a tuple, got {other:?}"),
        })
        .collect();

    assert_eq!(
        lengths_and_heads,
        [
            (0, None),
            (2, Some("Int(0)".to_owned())),
            (1, Some("Int(1)".to_owned())),
            (2, Some("Int(1)".to_owned())),
        ]
    );
}

#[test]
fn member_set_orders_frozen_sets_by_their_canonical_members() {
    let members = member_set([
        Value::FrozenSet(vec![int(3), int(1)]),
        Value::FrozenSet(vec![int(2)]),
        Value::FrozenSet(vec![int(1), int(2)]),
    ]);

    let contents: Vec<Vec<String>> = members
        .iter()
        .map(|member| match member.kind() {
            MemberKind::FrozenSet(set) => describe(set),
            other => panic!("expected a frozen set, got {other:?}"),
        })
        .collect();

    assert_eq!(
        contents,
        [
            vec!["Int(1)".to_owned(), "Int(2)".to_owned()],
            vec!["Int(1)".to_owned(), "Int(3)".to_owned()],
            vec!["Int(2)".to_owned()],
        ]
    );
}

#[test]
fn member_set_does_not_depend_on_the_order_members_are_given_in() {
    let forward = member_set([int(3), text("x"), Value::Float(0.5), Value::Bool(false)]);
    let backward = member_set([Value::Bool(false), Value::Float(0.5), text("x"), int(3)]);

    assert_eq!(describe(&forward), describe(&backward));
    assert_eq!(forward, backward);
}

#[test]
fn opaque_members_order_by_key_keeping_the_given_order_among_equal_keys() {
    let members = member_set([
        TestOpaque::colliding(2).into_value(),
        TestOpaque::token(9).into_value(),
        TestOpaque::colliding(1).into_value(),
        TestOpaque::colliding(2).into_value(),
    ]);

    let payloads: Vec<String> = members
        .iter()
        .map(|member| match member.kind() {
            MemberKind::Opaque(value) => value
                .get()
                .ordering_key()
                .expect("a test value has a key")
                .into_owned(),
            other => panic!("expected an opaque member, got {other:?}"),
        })
        .collect();

    assert_eq!(payloads, ["Colliding", "Colliding", "Token:9"]);
    assert!(members.contains(&member(TestOpaque::colliding(1).into_value())));
    assert!(members.contains(&member(TestOpaque::colliding(2).into_value())));
    assert!(!members.contains(&member(TestOpaque::colliding(3).into_value())));
}

#[test]
fn member_sets_of_colliding_opaque_members_are_equal_in_either_order() {
    let forward = member_set([
        TestOpaque::colliding(1).into_value(),
        TestOpaque::colliding(2).into_value(),
    ]);
    let backward = member_set([
        TestOpaque::colliding(2).into_value(),
        TestOpaque::colliding(1).into_value(),
    ]);
    let other = member_set([
        TestOpaque::colliding(1).into_value(),
        TestOpaque::colliding(3).into_value(),
    ]);

    assert_eq!(forward, backward);
    assert_ne!(forward, other);
}

#[rstest]
#[case::an_equal_integer(int(2), true)]
#[case::an_integer_not_held(int(4), false)]
#[case::a_float_equal_in_value(Value::Float(2.0), false)]
#[case::a_boolean_equal_in_value(Value::Bool(true), false)]
#[case::a_nan(Value::Float(f64::NAN), false)]
#[case::a_decimal(decimal("2"), false)]
#[case::an_equal_tuple(Value::Tuple(vec![int(1), text("a")]), true)]
#[case::a_frozen_set_with_a_repeat(Value::FrozenSet(vec![int(1), int(1), int(2)]), true)]
#[case::a_frozen_set_missing_a_member(Value::FrozenSet(vec![int(1)]), false)]
#[case::an_equal_opaque_value(TestOpaque::token(7).into_value(), true)]
#[case::an_opaque_value_not_held(TestOpaque::token(8).into_value(), false)]
fn member_set_contains_a_value_type_strictly(#[case] value: Value, #[case] expected: bool) {
    let members = member_set([
        int(1),
        int(2),
        Value::Tuple(vec![int(1), text("a")]),
        Value::FrozenSet(vec![int(2), int(1)]),
        TestOpaque::token(7).into_value(),
    ]);

    assert_eq!(members.contains_value(&value), expected);
}

#[test]
fn member_set_contains_a_negative_zero_value_as_zero() {
    assert!(member_set([Value::Float(0.0)]).contains_value(&Value::Float(-0.0)));
}

#[rstest]
#[case::a_boolean(Value::Bool(false), Some(LiteralValue::Bool(false)))]
#[case::an_integer(int(5), Some(LiteralValue::Int(5.into())))]
#[case::a_float(Value::Float(1.5), Some(LiteralValue::Float(1.5)))]
#[case::a_string(text("5"), None)]
#[case::a_tuple(Value::Tuple(vec![int(1)]), None)]
#[case::a_frozen_set(Value::FrozenSet(vec![int(1)]), None)]
#[case::an_opaque_value(TestOpaque::token(1).into_value(), None)]
fn member_lifts_to_a_literal_only_as_a_boolean_an_integer_or_a_float(
    #[case] value: Value,
    #[case] expected: Option<LiteralValue>,
) {
    let stored = member(value);

    assert_eq!(stored.lifts_to_expression(), expected.is_some());
    assert_eq!(stored.to_literal(), expected);
}

#[rstest]
#[case::a_boolean(Value::Bool(true), true)]
#[case::a_nan(Value::Float(f64::NAN), true)]
#[case::a_string(text("a"), true)]
#[case::a_decimal(decimal("1"), false)]
#[case::a_nested_decimal(Value::FrozenSet(vec![Value::Tuple(vec![decimal("1")])]), false)]
#[case::an_opaque_member(TestOpaque::token(1).into_value(), true)]
#[case::an_opaque_non_member(TestOpaque { is_member_shaped: false, ..TestOpaque::token(1) }.into_value(), false)]
fn value_is_member_shaped_when_everything_it_holds_could_be_a_member(
    #[case] value: Value,
    #[case] expected: bool,
) {
    assert_eq!(value.is_member_shaped(), expected);
}

#[test]
fn value_check_hashable_reports_the_first_unhashable_opaque_value() {
    let unhashable = TestOpaque {
        is_hashable: false,
        ..TestOpaque::token(1)
    };
    let value = Value::Tuple(vec![int(1), Value::Tuple(vec![unhashable.into_value()])]);

    let error = value
        .check_hashable()
        .expect_err("the opaque value is unhashable");

    assert_eq!(error.to_string(), "Token is unhashable");
    int(1)
        .check_hashable()
        .expect("an integer can be looked up");
}

#[test]
fn value_of_a_literal_keeps_its_kind() {
    assert!(matches!(
        Value::from(LiteralValue::Decimal("0.5".parse().expect("a decimal"))),
        Value::Decimal(_)
    ));
    assert!(matches!(
        Value::from(LiteralValue::Bool(true)),
        Value::Bool(true)
    ));
}

#[test]
fn a_failing_opaque_key_fails_member_construction() {
    let nested = Value::Tuple(vec![int(1), Failing(FailingHook::Key).into_value()]);

    let error = Member::try_from(nested).expect_err("the key fails");

    let MemberError::OrderingKey { type_name, source } = &error else {
        panic!("an ordering-key error, got {error:?}");
    };
    assert_eq!(type_name, "Failing");
    assert_eq!(source.to_string(), "the key failed");
    assert!(source.downcast_ref::<TestValueError>().is_some());
    assert_eq!(
        error.to_string(),
        "the ordering key of a member of type Failing failed"
    );
    assert!(std::error::Error::source(&error).is_some());
}
