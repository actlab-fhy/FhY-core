//! Tests for `fhy_core::expression::registry`: the entries and their
//! checks, registration and its refusals, the lookups by name and by
//! identifier, registration order, the sort lookup the screens read, and
//! the registry as an owned, cloneable value.

use std::collections::HashSet;

use crate::support::expression as expression_support;
use crate::support::hashing as hashing_support;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{
    ConstantValueError, FunctionDefinition, FunctionDefinitionError, FunctionRegistry,
    NativeConstant, NativeFunction, RegistrationError, RegistryEntry,
};
use fhy_core::expression::{
    BigInt, Decimal, Expression, FunctionName, FunctionNameError, FunctionSort, LiteralValue,
    SortLookup,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;

use expression_support::build_identifier;
use hashing_support::hash_of;

fn name(text: &str) -> FunctionName {
    FunctionName::new(text).expect("the test names no built-in")
}

/// Return `name(x) = x + 1` over the reals.
fn define_increment(function: &str) -> FunctionDefinition {
    let (x, reference) = build_identifier("x");
    FunctionDefinition::new(
        name(function),
        [x],
        [FunctionSort::Real],
        FunctionSort::Real,
        reference + 1,
    )
    .expect("one parameter with one sort")
}

/// Return the function `name` of no parameter returning `body`.
fn define_constant_function(function: &str, body: Expression) -> FunctionDefinition {
    FunctionDefinition::new(name(function), [], [], FunctionSort::Real, body)
        .expect("no parameter needs no sort")
}

fn declare_native(function: &str) -> NativeFunction {
    NativeFunction::new(name(function), [FunctionSort::Real], FunctionSort::Real)
}

fn declare_constant(
    constant: &str,
    sort: FunctionSort,
    value: impl Into<LiteralValue>,
) -> NativeConstant {
    NativeConstant::new(name(constant), sort, value).expect("the sort accepts the value")
}

fn expect_function(entry: Option<RegistryEntry<'_>>) -> &FunctionDefinition {
    match entry {
        Some(RegistryEntry::Function(function)) => function,
        other => panic!("expected a function entry, got {other:?}"),
    }
}

fn names_in_order(registry: &FunctionRegistry) -> Vec<String> {
    registry
        .iter()
        .map(|entry| entry.name().as_str().to_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

#[test]
fn function_definition_keeps_its_fields() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let body = x_reference * y_reference;

    let function = FunctionDefinition::new(
        name("product"),
        [x.clone(), y.clone()],
        [FunctionSort::Int, FunctionSort::Real],
        FunctionSort::Real,
        body.clone(),
    )
    .expect("two parameters with two sorts");

    assert_eq!(function.name().as_str(), "product");
    assert_eq!(function.parameters(), [x, y]);
    assert_eq!(
        function.parameter_sorts(),
        [FunctionSort::Int, FunctionSort::Real]
    );
    assert_eq!(function.result_sort(), FunctionSort::Real);
    assert!(Expression::ptr_eq(function.body(), &body));
}

#[rstest]
#[case::fewer_sorts(2, 1, r#"function "f" has 2 parameters but 1 parameter sort"#)]
#[case::more_sorts(1, 2, r#"function "f" has 1 parameter but 2 parameter sorts"#)]
#[case::no_sorts(1, 0, r#"function "f" has 1 parameter but 0 parameter sorts"#)]
fn function_definition_refuses_a_sort_count_other_than_its_parameter_count(
    #[case] parameter_count: usize,
    #[case] sort_count: usize,
    #[case] message: &str,
) {
    let parameters: Vec<Identifier> = (0..parameter_count)
        .map(|index| Identifier::new(&format!("p{index}")))
        .collect();

    let error = FunctionDefinition::new(
        name("f"),
        parameters,
        vec![FunctionSort::Real; sort_count],
        FunctionSort::Real,
        Expression::literal(1),
    )
    .expect_err("the counts differ");

    assert_eq!(
        error,
        FunctionDefinitionError::SortCountMismatch {
            function: name("f"),
            parameters: parameter_count,
            sorts: sort_count,
        }
    );
    assert_eq!(error.to_string(), message);
}

#[test]
fn function_definition_refuses_a_repeated_parameter() {
    let (x, reference) = build_identifier("x");
    let y = Identifier::new("y");

    let error = FunctionDefinition::new(
        name("f"),
        [x.clone(), y, x.clone()],
        [FunctionSort::Real; 3],
        FunctionSort::Real,
        reference,
    )
    .expect_err("x is named twice");

    assert_eq!(
        error,
        FunctionDefinitionError::RepeatedParameter {
            function: name("f"),
            parameter: x,
        }
    );
    assert_eq!(
        error.to_string(),
        r#"function "f" repeats the parameter "x""#
    );
}

#[test]
fn function_definition_accepts_distinct_parameters_sharing_a_name_hint() {
    let first = Identifier::new("x");
    let second = Identifier::new("x");

    let function = FunctionDefinition::new(
        name("f"),
        [first, second],
        [FunctionSort::Real; 2],
        FunctionSort::Real,
        Expression::literal(0),
    );

    assert!(
        function.is_ok(),
        "identifiers are distinct by id: {function:?}"
    );
}

#[test]
fn function_definition_clone_shares_its_body() {
    let function = define_increment("f");

    let copy = function.clone();

    assert!(Expression::ptr_eq(copy.body(), function.body()));
}

#[test]
fn native_function_keeps_its_signature() {
    let function = NativeFunction::new(
        name("hypot"),
        [FunctionSort::Real, FunctionSort::Real],
        FunctionSort::Real,
    );

    assert_eq!(function.name().as_str(), "hypot");
    assert_eq!(
        function.parameter_sorts(),
        [FunctionSort::Real, FunctionSort::Real]
    );
    assert_eq!(function.result_sort(), FunctionSort::Real);
}

/// The sorts a constant's value may have: a Boolean only `bool`, a
/// non-negative integer `nat`, an integer `int`, and a number `real`. For
/// Python's `bool`, `int` and `float` these are
/// `is_python_value_compatible_with_sort`'s rules.
#[rstest]
#[case::true_is_bool(LiteralValue::from(true), FunctionSort::Bool, true)]
#[case::true_is_no_nat(LiteralValue::from(true), FunctionSort::Nat, false)]
#[case::true_is_no_int(LiteralValue::from(true), FunctionSort::Int, false)]
#[case::true_is_no_real(LiteralValue::from(true), FunctionSort::Real, false)]
#[case::zero_is_nat(LiteralValue::from(0), FunctionSort::Nat, true)]
#[case::positive_is_nat(LiteralValue::from(7), FunctionSort::Nat, true)]
#[case::negative_is_no_nat(LiteralValue::from(-1), FunctionSort::Nat, false)]
#[case::negative_is_int(LiteralValue::from(-1), FunctionSort::Int, true)]
#[case::integer_is_real(LiteralValue::from(-1), FunctionSort::Real, true)]
#[case::integer_is_no_bool(LiteralValue::from(1), FunctionSort::Bool, false)]
#[case::big_integer_is_int(LiteralValue::from(BigInt::from(u64::MAX) * 4), FunctionSort::Int, true)]
#[case::float_is_real(LiteralValue::from(1.5), FunctionSort::Real, true)]
#[case::nan_is_real(LiteralValue::from(f64::NAN), FunctionSort::Real, true)]
#[case::infinity_is_real(LiteralValue::from(f64::NEG_INFINITY), FunctionSort::Real, true)]
#[case::whole_float_is_no_int(LiteralValue::from(1.0), FunctionSort::Int, false)]
#[case::float_is_no_nat(LiteralValue::from(1.0), FunctionSort::Nat, false)]
#[case::float_is_no_bool(LiteralValue::from(0.0), FunctionSort::Bool, false)]
#[case::decimal_is_real(LiteralValue::Decimal("1.5".parse::<Decimal>().expect("decimal")), FunctionSort::Real, true)]
#[case::decimal_is_no_int(LiteralValue::Decimal("2".parse::<Decimal>().expect("decimal")), FunctionSort::Int, false)]
fn native_constant_accepts_exactly_the_values_of_its_sort(
    #[case] value: LiteralValue,
    #[case] sort: FunctionSort,
    #[case] is_accepted: bool,
) {
    let constant = NativeConstant::new(name("c"), sort, value.clone());

    match constant {
        Ok(constant) => {
            assert!(is_accepted, "{sort} accepted {value}");
            assert_eq!(constant.sort(), sort);
            assert_eq!(constant.value(), &value);
            assert_eq!(constant.name().as_str(), "c");
        }
        Err(error) => {
            assert!(!is_accepted, "{sort} refused {value}");
            assert_eq!(error.name().as_str(), "c");
            assert_eq!(error.sort(), sort);
            assert_eq!(error.value(), &value);
        }
    }
}

#[test]
fn equal_native_constants_hash_alike_and_collapse_in_a_set() {
    let zero = declare_constant("c", FunctionSort::Real, 0.0);
    let negative_zero = declare_constant("c", FunctionSort::Real, -0.0);
    let one = declare_constant("c", FunctionSort::Real, 1.0);
    let renamed = declare_constant("d", FunctionSort::Real, 0.0);

    assert_eq!(zero, negative_zero);
    assert_eq!(hash_of(&zero), hash_of(&negative_zero));
    let constants: HashSet<NativeConstant> = [zero, negative_zero, one, renamed].into();
    assert_eq!(constants.len(), 3);
}

#[rstest]
#[case::negative_nat(FunctionSort::Nat, LiteralValue::from(-1), r#"constant "c" of sort nat cannot hold -1"#)]
#[case::boolean_int(
    FunctionSort::Int,
    LiteralValue::from(true),
    r#"constant "c" of sort int cannot hold true"#
)]
#[case::float_bool(
    FunctionSort::Bool,
    LiteralValue::from(1.5),
    r#"constant "c" of sort bool cannot hold 1.5"#
)]
fn constant_value_error_displays_the_constant_its_sort_and_the_value(
    #[case] sort: FunctionSort,
    #[case] value: LiteralValue,
    #[case] message: &str,
) {
    let error: ConstantValueError =
        NativeConstant::new(name("c"), sort, value).expect_err("the sort refuses it");

    assert_eq!(error.to_string(), message);
}

// ---------------------------------------------------------------------------
// Registration and lookups
// ---------------------------------------------------------------------------

#[test]
fn new_registry_is_empty() {
    let registry = FunctionRegistry::new();

    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.iter().len(), 0);
    assert!(!registry.contains("f"));
    assert!(registry.entry("f").is_none());
}

#[test]
fn registered_function_is_found_by_name() {
    let mut registry = FunctionRegistry::new();
    let function = define_increment("increment");

    registry
        .register_function(function.clone())
        .expect("the name is free");

    let found = expect_function(registry.entry("increment"));
    assert!(Expression::ptr_eq(found.body(), function.body()));
    assert_eq!(found.parameters(), function.parameters());
    assert!(registry.contains("increment"));
    assert_eq!(registry.len(), 1);
}

#[test]
fn registered_native_function_is_found_by_name() {
    let mut registry = FunctionRegistry::new();

    registry
        .register_native_function(declare_native("softplus"))
        .expect("the name is free");

    match registry.entry("softplus") {
        Some(RegistryEntry::Native(function)) => {
            assert_eq!(function, &declare_native("softplus"));
        }
        other => panic!("expected a native entry, got {other:?}"),
    }
}

#[test]
fn registered_constant_is_found_by_name_and_by_its_minted_identifier() {
    let mut registry = FunctionRegistry::new();
    let constant = declare_constant("tau", FunctionSort::Real, std::f64::consts::TAU);

    let identifier = registry
        .register_constant(constant.clone())
        .expect("the name is free");

    assert_eq!(identifier.name_hint(), "tau");
    assert_eq!(registry.constant_identifier("tau"), Some(&identifier));
    assert_eq!(registry.constant(&identifier), Some(&constant));
    match registry.entry("tau") {
        Some(RegistryEntry::Constant(found, found_identifier)) => {
            assert_eq!(found, &constant);
            assert_eq!(found_identifier, &identifier);
        }
        other => panic!("expected a constant entry, got {other:?}"),
    }
}

#[test]
fn identifier_merely_named_like_a_constant_is_no_reference_to_it() {
    let mut registry = FunctionRegistry::new();
    let minted = registry
        .register_constant(declare_constant("tau", FunctionSort::Real, 6.5))
        .expect("the name is free");
    let look_alike = Identifier::new("tau");

    assert_ne!(look_alike, minted);
    assert!(registry.constant(&look_alike).is_none());
    assert!(registry.native_constant_sort(&look_alike).is_none());
}

#[test]
fn each_registered_constant_gets_a_new_identifier() {
    let mut first = FunctionRegistry::new();
    let mut second = FunctionRegistry::new();

    let in_first = first
        .register_constant(declare_constant("c", FunctionSort::Int, 1))
        .expect("the name is free");
    let in_second = second
        .register_constant(declare_constant("c", FunctionSort::Int, 1))
        .expect("the name is free");

    assert_ne!(in_first, in_second);
    assert!(first.constant(&in_second).is_none());
}

#[test]
fn constant_identifier_is_none_for_a_function_or_an_unknown_name() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define_increment("f"))
        .expect("the name is free");

    assert!(registry.constant_identifier("f").is_none());
    assert!(registry.constant_identifier("g").is_none());
}

#[test]
fn builtin_names_find_no_entry() {
    let registry = FunctionRegistry::new();

    for function in BuiltinFunction::iter() {
        assert!(registry.entry(function.name()).is_none(), "{function:?}");
        assert!(!registry.contains(function.name()), "{function:?}");
    }
    for constant in BuiltinConstant::iter() {
        assert!(registry.entry(constant.name()).is_none(), "{constant:?}");
        assert!(registry.constant_identifier(constant.name()).is_none());
        assert!(registry.constant(constant.identifier()).is_none());
    }
}

#[test]
fn builtin_function_names_are_no_function_names() {
    for function in BuiltinFunction::iter() {
        assert_eq!(
            FunctionName::new(function.name()),
            Err(FunctionNameError::Builtin(function))
        );
    }
}

#[test]
fn entries_iterate_in_registration_order() {
    let mut registry = FunctionRegistry::new();

    registry
        .register_constant(declare_constant("c", FunctionSort::Int, 1))
        .expect("free");
    registry
        .register_function(define_increment("b"))
        .expect("free");
    registry
        .register_native_function(declare_native("a"))
        .expect("free");
    registry
        .register_function(define_increment("d"))
        .expect("free");

    assert_eq!(names_in_order(&registry), ["c", "b", "a", "d"]);
    assert_eq!(registry.iter().len(), 4);
    assert_eq!(registry.len(), 4);
}

#[rstest]
#[case::function_over_function("function", "function")]
#[case::function_over_native("native", "function")]
#[case::function_over_constant("constant", "function")]
#[case::native_over_function("function", "native")]
#[case::constant_over_function("function", "constant")]
#[case::constant_over_constant("constant", "constant")]
fn registration_refuses_a_taken_name_whatever_the_kinds(
    #[case] first_kind: &str,
    #[case] second_kind: &str,
) {
    fn register(registry: &mut FunctionRegistry, kind: &str) -> Result<(), RegistrationError> {
        match kind {
            "function" => registry.register_function(define_increment("f")),
            "native" => registry.register_native_function(declare_native("f")),
            _ => registry
                .register_constant(declare_constant("f", FunctionSort::Int, 1))
                .map(|_| ()),
        }
    }
    let mut registry = FunctionRegistry::new();
    register(&mut registry, first_kind).expect("the name is free");

    let error = register(&mut registry, second_kind).expect_err("the name is taken");

    assert_eq!(error, RegistrationError::NameTaken(name("f")));
    assert_eq!(error.to_string(), r#""f" is already registered"#);
    assert_eq!(registry.len(), 1);
}

#[rstest]
fn registration_refuses_a_builtin_constant_name_for_every_kind(
    #[values(
        BuiltinConstant::Pi,
        BuiltinConstant::E,
        BuiltinConstant::Inf,
        BuiltinConstant::Nan
    )]
    constant: BuiltinConstant,
    #[values("function", "native", "constant")] kind: &str,
) {
    let mut registry = FunctionRegistry::new();
    let entry_name = constant.name();

    let result = match kind {
        "function" => {
            registry.register_function(define_constant_function(entry_name, Expression::literal(1)))
        }
        "native" => registry.register_native_function(declare_native(entry_name)),
        _ => registry
            .register_constant(declare_constant(entry_name, FunctionSort::Real, 1.0))
            .map(|_| ()),
    };

    let error = result.expect_err("a built-in constant's name is reserved");
    assert_eq!(error, RegistrationError::BuiltinConstantName(constant));
    assert_eq!(
        error.to_string(),
        format!("{entry_name:?} is the name of a built-in constant")
    );
    assert!(registry.is_empty());
}

#[test]
fn registration_refuses_a_body_capturing_free_identifiers() {
    let mut registry = FunctionRegistry::new();
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (z, z_reference) = build_identifier("z");
    let (w, w_reference) = build_identifier("w");
    let function = FunctionDefinition::new(
        name("f"),
        [w],
        [FunctionSort::Real],
        FunctionSort::Real,
        (z_reference + x_reference) * y_reference + w_reference,
    )
    .expect("one parameter with one sort");

    let error = registry
        .register_function(function)
        .expect_err("x, y and z are captured");

    assert_eq!(
        error,
        RegistrationError::CapturedIdentifiers {
            function: name("f"),
            identifiers: vec![x, y, z],
        }
    );
    assert_eq!(
        error.to_string(),
        r#"function "f" captures identifiers that are not its parameters: x, y, z"#
    );
    assert!(registry.is_empty());
}

#[test]
fn captured_identifiers_sharing_a_name_hint_are_ordered_by_id() {
    let mut registry = FunctionRegistry::new();
    let first = Identifier::new("x");
    let second = Identifier::new("x");

    let error = registry
        .register_function(define_constant_function(
            "f",
            Expression::from(second.clone()) + Expression::from(first.clone()),
        ))
        .expect_err("both are captured");

    assert_eq!(
        error,
        RegistrationError::CapturedIdentifiers {
            function: name("f"),
            identifiers: vec![first, second],
        }
    );
}

#[test]
fn capture_check_reads_the_constants_registered_so_far() {
    let mut registry = FunctionRegistry::new();
    let mut other = FunctionRegistry::new();
    let constant = declare_constant("scale", FunctionSort::Real, 2.0);
    let identifier = other
        .register_constant(constant.clone())
        .expect("the name is free");
    let body = Expression::from(identifier) * 3;

    let before = registry.register_function(define_constant_function("f", body));
    assert!(
        matches!(before, Err(RegistrationError::CapturedIdentifiers { .. })),
        "another registry's constant is captured: {before:?}"
    );

    let own = registry
        .register_constant(constant)
        .expect("the name is free here");
    let own_body = Expression::from(own) * 3;
    registry
        .register_function(define_constant_function("f", own_body))
        .expect("the registry's own constant is no capture");
    assert!(registry.contains("f"));
}

#[test]
fn capture_check_exempts_the_builtin_constants() {
    let mut registry = FunctionRegistry::new();
    let body = BuiltinConstant::iter()
        .map(|constant| Expression::from(constant.identifier().clone()))
        .reduce(|sum, reference| sum + reference)
        .expect("four constants");

    registry
        .register_function(define_constant_function("f", body))
        .expect("the built-in constants are no capture");
}

#[test]
fn capture_check_refuses_a_look_alike_of_a_builtin_constant() {
    let mut registry = FunctionRegistry::new();
    let look_alike = Identifier::new("pi");

    let error = registry
        .register_function(define_constant_function(
            "f",
            Expression::from(look_alike.clone()),
        ))
        .expect_err("an identifier merely named pi is captured");

    assert_eq!(
        error,
        RegistrationError::CapturedIdentifiers {
            function: name("f"),
            identifiers: vec![look_alike],
        }
    );
}

#[test]
fn registration_accepts_a_body_calling_an_unregistered_or_recursive_name() {
    let mut registry = FunctionRegistry::new();
    let (x, reference) = build_identifier("x");
    let body = Expression::call(name("later"), [reference.clone()])
        + Expression::call(name("f"), [reference]);

    registry
        .register_function(
            FunctionDefinition::new(
                name("f"),
                [x],
                [FunctionSort::Real],
                FunctionSort::Real,
                body,
            )
            .expect("one parameter with one sort"),
        )
        .expect("a call is a reference by name, checked when inlining");
}

#[test]
fn refused_registration_leaves_the_registry_unchanged() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define_increment("f"))
        .expect("free");
    let captured = define_constant_function("g", Expression::from(Identifier::new("free")));

    registry
        .register_function(captured)
        .expect_err("captures an identifier");
    registry
        .register_native_function(declare_native("f"))
        .expect_err("taken");

    assert_eq!(names_in_order(&registry), ["f"]);
    assert!(!registry.contains("g"));
}

#[test]
fn result_sort_answers_for_functions_only() {
    let mut registry = FunctionRegistry::new();
    let (x, reference) = build_identifier("x");
    registry
        .register_function(
            FunctionDefinition::new(
                name("positive"),
                [x],
                [FunctionSort::Real],
                FunctionSort::Bool,
                reference.greater(0),
            )
            .expect("one parameter with one sort"),
        )
        .expect("free");
    registry
        .register_native_function(NativeFunction::new(
            name("count"),
            [FunctionSort::Real],
            FunctionSort::Nat,
        ))
        .expect("free");
    registry
        .register_constant(declare_constant("c", FunctionSort::Int, 3))
        .expect("free");

    assert_eq!(
        registry.result_sort(&name("positive")),
        Some(FunctionSort::Bool)
    );
    assert_eq!(
        registry.result_sort(&name("count")),
        Some(FunctionSort::Nat)
    );
    assert_eq!(registry.result_sort(&name("c")), None);
    assert_eq!(registry.result_sort(&name("unknown")), None);
}

#[test]
fn registry_answers_the_screens_sort_lookup() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_native_function(NativeFunction::new(
            name("is_even"),
            [FunctionSort::Int],
            FunctionSort::Bool,
        ))
        .expect("free");
    let flag = registry
        .register_constant(declare_constant("flag", FunctionSort::Bool, true))
        .expect("free");

    assert_eq!(
        registry.native_constant_sort(&flag),
        Some(FunctionSort::Bool)
    );
    assert_eq!(
        registry.call_result_sort(&name("is_even")),
        Some(FunctionSort::Bool)
    );
    assert_eq!(registry.call_result_sort(&name("flag")), None);
    assert_eq!(registry.call_result_sort(&name("unknown")), None);
    assert_eq!(
        registry.native_constant_sort(BuiltinConstant::Pi.identifier()),
        None,
        "built-in constants are the catalogue's, not the registry's"
    );
}

#[test]
fn registry_clone_is_independent() {
    let mut original = FunctionRegistry::new();
    original
        .register_function(define_increment("shared"))
        .expect("free");

    let mut copy = original.clone();
    copy.register_function(define_increment("copied"))
        .expect("free");
    original
        .register_native_function(declare_native("original"))
        .expect("free");

    assert_eq!(names_in_order(&original), ["shared", "original"]);
    assert_eq!(names_in_order(&copy), ["shared", "copied"]);
    assert!(Expression::ptr_eq(
        expect_function(original.entry("shared")).body(),
        expect_function(copy.entry("shared")).body(),
    ));
}

#[test]
fn registry_clone_keeps_the_constants_identifiers() {
    let mut original = FunctionRegistry::new();
    let identifier = original
        .register_constant(declare_constant("c", FunctionSort::Int, 1))
        .expect("free");

    let copy = original.clone();

    assert_eq!(copy.constant_identifier("c"), Some(&identifier));
    assert_eq!(
        copy.constant(&identifier),
        Some(&declare_constant("c", FunctionSort::Int, 1))
    );
}

#[test]
fn registry_is_send_and_sync() {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FunctionRegistry>();
    assert_send_sync::<FunctionDefinition>();
    assert_send_sync::<NativeFunction>();
    assert_send_sync::<NativeConstant>();
}

#[test]
fn retain_keeps_the_chosen_entries_in_order_with_their_identifiers() {
    let mut registry = FunctionRegistry::new();
    let kept_constant = registry
        .register_constant(declare_constant("kept", FunctionSort::Int, 1))
        .expect("free");
    let dropped_constant = registry
        .register_constant(declare_constant("dropped", FunctionSort::Int, 2))
        .expect("free");
    registry
        .register_function(define_increment("f"))
        .expect("free");
    registry
        .register_native_function(declare_native("g"))
        .expect("free");

    registry.retain(|entry| !matches!(entry.name().as_str(), "dropped" | "g"));

    assert_eq!(names_in_order(&registry), ["kept", "f"]);
    assert_eq!(registry.constant_identifier("kept"), Some(&kept_constant));
    assert_eq!(
        registry.constant(&kept_constant),
        Some(&declare_constant("kept", FunctionSort::Int, 1))
    );
    assert!(registry.constant(&dropped_constant).is_none());
    assert!(registry.constant_identifier("dropped").is_none());
    assert!(registry.entry("g").is_none());
    registry
        .register_native_function(declare_native("g"))
        .expect("a dropped name is free again");
    assert_eq!(names_in_order(&registry), ["kept", "f", "g"]);
}

#[test]
fn constant_registered_again_after_retain_gets_a_new_identifier() {
    let mut registry = FunctionRegistry::new();
    let first = registry
        .register_constant(declare_constant("c", FunctionSort::Int, 1))
        .expect("free");
    registry.retain(|_| false);

    let second = registry
        .register_constant(declare_constant("c", FunctionSort::Int, 1))
        .expect("free again");

    assert_ne!(first, second);
    assert!(registry.constant(&first).is_none());
}
