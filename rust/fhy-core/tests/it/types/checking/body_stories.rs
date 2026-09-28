//! Tests for the body checks and the sweep, ported from
//! `tests/types/checking/test_body_type_checker.py`,
//! `test_registry_body_sweep.py` and `test_builtin_bodies.py`.

use crate::support::expression::build_call_or_panic;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{FunctionDefinition, FunctionRegistry};
use fhy_core::expression::{Expression, FunctionName, FunctionSort, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::types::checking::{
    BodyCheck, BodyCheckError, FunctionLabel, FunctionSignature, SignatureError,
    check_all_function_bodies, check_function_body,
};

fn name(text: &str) -> FunctionName {
    FunctionName::new(text).expect("a name")
}

fn reference(identifier: &Identifier) -> Expression {
    Expression::from(identifier.clone())
}

/// Check `body` as the body of `name(x: parameter_sort) -> result_sort`.
fn check_body(
    registry: &FunctionRegistry,
    function: &str,
    x: &Identifier,
    parameter_sort: FunctionSort,
    result_sort: FunctionSort,
    body: &Expression,
    defer: bool,
) -> Result<BodyCheck, BodyCheckError> {
    let function = name(function);
    let parameters = [x.clone()];
    let sorts = [parameter_sort];
    let signature = FunctionSignature::new(&function, &parameters, &sorts, result_sort)
        .expect("a signature of matching lengths");
    check_function_body(&signature, body, registry, registry, defer)
}

#[test]
fn a_body_of_a_compatible_type_checks() {
    let x = Identifier::new("x");
    let registry = FunctionRegistry::new();

    for (parameter, result, body) in [
        (FunctionSort::Int, FunctionSort::Int, reference(&x) * 2 + 1),
        (FunctionSort::Nat, FunctionSort::Int, reference(&x) + 1),
        (FunctionSort::Real, FunctionSort::Real, reference(&x) / 3),
        (FunctionSort::Int, FunctionSort::Bool, reference(&x).less(4)),
        (
            FunctionSort::Real,
            FunctionSort::Real,
            reference(&x) * reference(BuiltinConstant::Pi.identifier()),
        ),
        (FunctionSort::Int, FunctionSort::Int, Expression::from(0)),
    ] {
        assert_eq!(
            check_body(&registry, "f", &x, parameter, result, &body, true).expect("checks"),
            BodyCheck::Checked
        );
    }
}

#[test]
fn a_body_of_an_incompatible_type_is_refused() {
    let x = Identifier::new("x");
    let registry = FunctionRegistry::new();

    let error = check_body(
        &registry,
        "lt_as_int",
        &x,
        FunctionSort::Int,
        FunctionSort::Int,
        &reference(&x).less(1),
        true,
    )
    .expect_err("a Boolean body for an int result");

    assert!(matches!(
        &error,
        BodyCheckError::IncompatibleResult {
            function: FunctionLabel::User(function),
            body: fhy_core::types::CoreDataType::Bool,
            sort: FunctionSort::Int,
        } if *function == name("lt_as_int")
    ));
    assert_eq!(
        error.to_string(),
        "function 'lt_as_int' body synthesized type bool is not compatible with the declared result sort int"
    );
}

#[test]
fn a_body_that_breaks_a_rule_or_is_unsupported_is_refused_naming_the_function() {
    let x = Identifier::new("x");
    let registry = FunctionRegistry::new();
    let decimal = Expression::from(LiteralValue::Decimal("1.5".parse().expect("a decimal")));

    let ill = check_body(
        &registry,
        "f",
        &x,
        FunctionSort::Int,
        FunctionSort::Int,
        &(reference(&x) + LiteralValue::Bool(true)),
        true,
    )
    .expect_err("a Boolean operand");
    let unsupported = check_body(
        &registry,
        "g",
        &x,
        FunctionSort::Real,
        FunctionSort::Real,
        &decimal,
        true,
    )
    .expect_err("a decimal");
    let unbound = check_body(
        &registry,
        "h",
        &x,
        FunctionSort::Int,
        FunctionSort::Int,
        &reference(&Identifier::new("y")),
        true,
    )
    .expect_err("an unbound identifier");

    assert!(
        ill.to_string()
            .starts_with("function 'f' body failed to type-check: type error while inferring")
    );
    assert!(
        unsupported.to_string().starts_with(
            "function 'g' body uses a construct the body type checker does not support"
        )
    );
    assert!(unbound.to_string().contains("is not bound"));
}

#[test]
fn a_body_whose_type_is_no_scalar_displays_the_type() {
    let error = BodyCheckError::NotScalar {
        function: FunctionLabel::User(name("f")),
        body_type: crate::support::types::index(0, 4, 1),
    };

    assert_eq!(
        error.to_string(),
        "function 'f' body must synthesize a scalar numerical type, but got index(0:4:1)"
    );
}

#[test]
fn an_unresolved_call_is_deferred_or_refused() {
    let x = Identifier::new("x");
    let registry = FunctionRegistry::new();
    let body = build_call_or_panic("not_yet", [reference(&x)]);

    assert_eq!(
        check_body(
            &registry,
            "f",
            &x,
            FunctionSort::Int,
            FunctionSort::Int,
            &body,
            true
        )
        .expect("deferred"),
        BodyCheck::Deferred
    );
    let error = check_body(
        &registry,
        "f",
        &x,
        FunctionSort::Int,
        FunctionSort::Int,
        &body,
        false,
    )
    .expect_err("refused");
    assert!(matches!(
        &error,
        BodyCheckError::UnknownCall {
            function: FunctionLabel::User(function),
            error: fhy_core::types::checking::CallTargetError::Unknown { name: callee, source: None, .. },
        } if *function == name("f") && callee == "not_yet"
    ));
    assert!(error.to_string().starts_with("function 'f' body calls a function that is not registered: no entry is registered under the name 'not_yet'"));
}

#[test]
fn a_self_recursive_body_resolves_its_own_call() {
    let x = Identifier::new("x");
    let mut registry = FunctionRegistry::new();
    let body = build_call_or_panic("countdown", [reference(&x) - 1]);
    registry
        .register_function(
            FunctionDefinition::new(
                name("countdown"),
                [x.clone()],
                [FunctionSort::Int],
                FunctionSort::Real,
                body.clone(),
            )
            .expect("a definition"),
        )
        .expect("registers");

    assert_eq!(
        check_body(
            &registry,
            "countdown",
            &x,
            FunctionSort::Int,
            FunctionSort::Real,
            &body,
            false
        )
        .expect("checks"),
        BodyCheck::Checked
    );
}

#[test]
fn the_sweep_checks_every_composed_builtin() {
    let sweep = check_all_function_bodies(&FunctionRegistry::new());

    let composed: Vec<FunctionLabel> = BuiltinFunction::iter()
        .filter(|function| function.composed().is_some())
        .map(FunctionLabel::Builtin)
        .collect();
    assert_eq!(composed.len(), 16);
    assert_eq!(sweep.checked(), composed);
    assert!(sweep.failures().is_empty(), "{:?}", sweep.failures());
}

#[test]
fn a_signature_with_mismatched_lengths_is_refused() {
    let function = name("f");
    let parameters = [Identifier::new("x"), Identifier::new("y")];

    let error = FunctionSignature::new(
        &function,
        &parameters,
        &[FunctionSort::Int],
        FunctionSort::Int,
    )
    .expect_err("two parameters, one sort");

    assert_eq!(
        error,
        SignatureError::LengthMismatch {
            function: FunctionLabel::User(function),
            parameters: 2,
            parameter_sorts: 1,
        }
    );
    assert_eq!(
        error.to_string(),
        "function 'f' has 2 parameter(s) but 1 parameter sort(s)"
    );
}

#[test]
fn a_function_label_displays_its_name() {
    assert_eq!(
        FunctionLabel::Builtin(BuiltinFunction::Max).to_string(),
        "max"
    );
    assert_eq!(FunctionLabel::User(name("scale")).to_string(), "scale");
    assert_eq!(
        FunctionLabel::from(BuiltinFunction::Relu),
        FunctionLabel::Builtin(BuiltinFunction::Relu)
    );
}

#[test]
fn the_sweep_reports_each_failing_user_body_in_registration_order() {
    let x = Identifier::new("x");
    let mut registry = FunctionRegistry::new();
    for (function, body) in [
        ("sweep_first", reference(&x).less(1)),
        ("sweep_fine", reference(&x) + 1),
        (
            "sweep_dangling",
            build_call_or_panic("sweep_never_registered", [reference(&x)]),
        ),
    ] {
        registry
            .register_function(
                FunctionDefinition::new(
                    name(function),
                    [x.clone()],
                    [FunctionSort::Int],
                    FunctionSort::Int,
                    body,
                )
                .expect("a definition"),
            )
            .expect("registers");
    }

    let sweep = check_all_function_bodies(&registry);

    let failures = sweep.failures();
    let names: Vec<String> = failures
        .iter()
        .map(|(function, _)| function.to_string())
        .collect();
    assert_eq!(names, ["sweep_first", "sweep_dangling"]);
    assert!(failures[1].1.to_string().contains("sweep_never_registered"));
    let users: Vec<String> = sweep
        .checked()
        .iter()
        .filter(|label| matches!(label, FunctionLabel::User(_)))
        .map(ToString::to_string)
        .collect();
    assert_eq!(users, ["sweep_first", "sweep_fine", "sweep_dangling"]);
}
