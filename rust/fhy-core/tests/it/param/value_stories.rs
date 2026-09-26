//! Stories of the ordinal order: numbers across kinds, strings, opaque
//! values by their producer's order, ties, incomparable values, and a
//! comparison that is no order.

use fhy_core::constraint::Value;
use fhy_core::expression::BigInt;
use fhy_core::param::{OrdinalDomain, ParamError};
use rstest::rstest;

use crate::support::constraint::{int, text};
use crate::support::param::{Hand, Level, boolean, describe_all, float, ints};

/// Return the values of the ordinal domain of `values`, described.
fn ordered(values: Vec<Value>) -> Vec<String> {
    describe_all(
        OrdinalDomain::new(values)
            .expect("the values make an ordinal domain")
            .values(),
    )
}

#[test]
fn ordinal_domain_sorts_integers_ascending() {
    assert_eq!(
        ordered(ints([3, -1, 20, 2])),
        ["int:-1", "int:2", "int:3", "int:20"]
    );
}

#[test]
fn ordinal_domain_orders_numbers_numerically_across_kinds() {
    assert_eq!(
        ordered(vec![
            float(2.5),
            int(3),
            boolean(false),
            float(-0.5),
            int(1)
        ]),
        ["float:-0.5", "bool:false", "int:1", "float:2.5", "int:3"]
    );
}

#[test]
fn ordinal_domain_compares_big_integers_with_floats_exactly() {
    // 2**53 + 1 is not a float; it lies above the float 2**53.
    let above = BigInt::from(9_007_199_254_740_993_i64);
    assert_eq!(
        ordered(vec![Value::Int(above), float(9_007_199_254_740_992.0)]),
        ["float:9007199254740992", "int:9007199254740993"]
    );
}

#[test]
fn ordinal_domain_orders_infinities_around_every_integer() {
    let huge = BigInt::from(10).pow(400);
    assert_eq!(
        ordered(vec![
            float(f64::INFINITY),
            Value::Int(huge.clone()),
            float(f64::NEG_INFINITY),
            Value::Int(-huge)
        ]),
        [
            "float:-inf".to_owned(),
            format!("int:-{}", BigInt::from(10).pow(400)),
            format!("int:{}", BigInt::from(10).pow(400)),
            "float:inf".to_owned(),
        ]
    );
}

#[test]
fn ordinal_domain_breaks_ties_of_equal_numbers_by_kind() {
    assert_eq!(
        ordered(vec![int(1), float(1.0), boolean(true)]),
        ["bool:true", "float:1", "int:1"]
    );
}

#[test]
fn ordinal_domain_sorts_strings_by_code_point() {
    assert_eq!(
        ordered(vec![text("b"), text("B"), text("a"), text("é")]),
        ["str:B", "str:a", "str:b", "str:é"]
    );
}

#[test]
fn ordinal_domain_orders_opaque_values_by_their_producer() {
    assert_eq!(
        ordered(vec![Level::value(3), Level::value(1), Level::value(2)]),
        ["opaque:Level:1", "opaque:Level:2", "opaque:Level:3"]
    );
}

#[rstest]
#[case::number_and_string(vec![int(1), text("a")])]
#[case::float_and_string(vec![text("a"), float(0.5)])]
#[case::number_and_opaque(vec![int(1), Level::value(1)])]
#[case::string_and_opaque(vec![Level::value(1), text("a")])]
#[case::opaque_values_of_two_types(vec![Level::value(1), Level::grade(2)])]
fn ordinal_domain_refuses_values_that_do_not_order(#[case] values: Vec<Value>) {
    assert!(matches!(
        OrdinalDomain::new(values),
        Err(ParamError::IncomparableValues)
    ));
}

#[test]
fn ordinal_domain_sorts_under_a_comparison_that_is_no_order() {
    let hands: Vec<Value> = (0..9).map(Hand::value).collect();

    let result = OrdinalDomain::new(hands);

    // Three equal pairs: the values are not unique, and the sort did not
    // panic on the cycle.
    assert!(matches!(result, Err(ParamError::DuplicateValues(_))));
}

#[test]
fn ordinal_domain_keeps_a_cycle_of_distinct_values() {
    let domain = OrdinalDomain::new((0..3).map(Hand::value).collect())
        .expect("three distinct hands make a domain");

    assert_eq!(domain.values().len(), 3);
}

#[test]
fn ordinal_domain_order_is_independent_of_the_input_order() {
    let forward = ordered(vec![int(2), float(0.5), boolean(true), int(-7)]);
    let backward = ordered(vec![int(-7), boolean(true), float(0.5), int(2)]);

    assert_eq!(forward, backward);
}

#[test]
fn ordinal_domain_stores_a_negative_zero_as_zero() {
    assert_eq!(ordered(vec![float(-0.0)]), ["float:0"]);
}
