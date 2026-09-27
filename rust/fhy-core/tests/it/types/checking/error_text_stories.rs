//! The text and the source of every variant of the checking errors
//! (F2-029): the checker's, the call-target lookup's, the signature's and
//! the body check's.

use std::collections::HashMap;

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Callee, Decimal, Expression, FunctionName, FunctionSort, LiteralValue};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::types::checking::{
    BodyCheckError, CallTarget, CallTargetError, CallTargets, FunctionLabel, FunctionSignature,
    IdentifierTypes, SignatureError, TypeCheckError, TypeChecker, TypeRuleKind,
    check_function_body,
};
use fhy_core::types::{CoreDataType, DataType, Dimension, Type, TypeQualifier};
use rstest::rstest;

use crate::support::error_text::{Source, assert_error_text, fixed, test_error};
use crate::support::expression::build_call_or_panic;
use crate::support::types::{array, index, scalar};

type Bindings = HashMap<Identifier, (Type, TypeQualifier)>;

fn params(pairs: &[(&Identifier, Type)]) -> Bindings {
    pairs
        .iter()
        .map(|(identifier, value)| ((*identifier).clone(), (value.clone(), TypeQualifier::Param)))
        .collect()
}

fn synthesize(bindings: &Bindings, expression: &Expression) -> TypeCheckError {
    let registry = FunctionRegistry::new();
    TypeChecker::new(bindings, &registry, &registry)
        .synthesize(expression)
        .expect_err("a broken rule")
}

/// A lookup that fails with the test error.
struct FailingIdentifiers;

impl IdentifierTypes for FailingIdentifiers {
    fn identifier_type(
        &self,
        _identifier: &Identifier,
    ) -> Result<Option<(Type, TypeQualifier)>, BoxError> {
        Err(test_error())
    }
}

/// Call targets that know no function, the lookup's own error the
/// source of each refusal, or that fail outright.
struct UnknownTargets {
    fails: bool,
}

impl CallTargets for UnknownTargets {
    fn call_target(&self, callee: &Callee) -> Result<CallTarget, CallTargetError> {
        if self.fails {
            return Err(CallTargetError::Callback(test_error()));
        }
        Err(CallTargetError::Unknown {
            name: callee.name().to_string(),
            message: format!("no entry is registered under the name '{}'", callee.name()),
            source: Some(test_error()),
        })
    }
}

#[test]
fn a_broken_rule_is_framed_by_the_root_and_the_sub_expression() {
    let i = fixed(61_720, "i");
    let bindings = params(&[(&i, index(0, 4, 1))]);
    let negation = -Expression::from(i.clone());

    let at_root = synthesize(&bindings, &negation);
    let below = synthesize(&bindings, &negation.positive());

    assert_error_text(
        &at_root,
        "type error while inferring the type of `(-i::61720)`: unary negation is not defined \
         for index types; the resulting bounds and stride cannot be inferred safely",
        Source::None,
    );
    assert_error_text(
        &below,
        "type error while inferring the type of `(+(-i::61720))` at sub-expression \
         `(-i::61720)`: unary negation is not defined for index types; the resulting bounds and \
         stride cannot be inferred safely",
        Source::None,
    );
    let TypeCheckError::Rule { rule, .. } = &below else {
        panic!("a rule error");
    };
    assert_eq!(rule.kind(), TypeRuleKind::Index);
    assert!(!rule.is_unsupported());
}

#[test]
fn an_unsupported_construct_is_a_rule_of_its_own_kind() {
    let decimal = Expression::from(LiteralValue::Decimal(
        "1.5".parse::<Decimal>().expect("a decimal"),
    ));

    let error = synthesize(&Bindings::new(), &decimal);

    assert_error_text(
        &error,
        "type error while inferring the type of `1.5`: decimal literals are not yet supported",
        Source::None,
    );
    let TypeCheckError::Rule { rule, .. } = &error else {
        panic!("a rule error");
    };
    assert!(rule.is_unsupported());
}

#[test]
fn a_deferred_unknown_call_and_a_failing_lookup_carry_the_lookup_s_error() {
    let x = fixed(61_721, "x");
    let targets = UnknownTargets { fails: false };
    let bindings = params(&[(&x, scalar(CoreDataType::Int32))]);
    let call = build_call_or_panic("later", [Expression::from(x.clone())]);

    let deferred = TypeChecker::new(&bindings, &targets, &FunctionRegistry::new())
        .with_deferred_unknown_calls()
        .synthesize(&call)
        .expect_err("unknown");
    let failing = TypeChecker::new(&FailingIdentifiers, &targets, &FunctionRegistry::new())
        .synthesize(&Expression::from(x))
        .expect_err("the lookup fails");

    assert_error_text(
        &deferred,
        "no entry is registered under the name 'later'",
        Source::TestValue,
    );
    assert_error_text(&failing, "a type-checking lookup failed", Source::TestValue);
}

#[rstest]
#[case::unknown_without_a_source(
    CallTargetError::Unknown {
        name: "f".to_owned(),
        message: "no entry is registered under the name 'f'".to_owned(),
        source: None,
    },
    "no entry is registered under the name 'f'",
    Source::None
)]
#[case::unknown_with_a_source(
    CallTargetError::Unknown {
        name: "f".to_owned(),
        message: "no entry is registered under the name 'f'".to_owned(),
        source: Some(test_error()),
    },
    "no entry is registered under the name 'f'",
    Source::TestValue
)]
#[case::callback(
    CallTargetError::Callback(test_error()),
    "the call target lookup failed",
    Source::TestValue
)]
fn call_target_error_text(
    #[case] error: CallTargetError,
    #[case] text: &str,
    #[case] source: Source,
) {
    assert_error_text(&error, text, source);
}

#[test]
fn signature_error_text() {
    let error = SignatureError::LengthMismatch {
        function: FunctionLabel::Builtin(BuiltinFunction::Clamp),
        parameters: 3,
        parameter_sorts: 2,
    };

    assert_error_text(
        &error,
        "function 'clamp' has 3 parameter(s) but 2 parameter sort(s)",
        Source::None,
    );
}

fn scale() -> FunctionLabel {
    FunctionLabel::User(FunctionName::new("scale").expect("a name"))
}

/// Return the error of checking `body` as the body of `scale(x: int) ->
/// result`, calls resolved by `targets`.
fn body_error(
    body: &Expression,
    x: &Identifier,
    result: FunctionSort,
    targets: &dyn CallTargets,
) -> BodyCheckError {
    let parameters = [x.clone()];
    let sorts = [FunctionSort::Int];
    let signature =
        FunctionSignature::new(scale(), &parameters, &sorts, result).expect("matching lengths");
    check_function_body(&signature, body, targets, &FunctionRegistry::new(), false)
        .expect_err("the body fails")
}

#[test]
fn each_body_check_failure_names_the_function() {
    let x = fixed(61_722, "x");
    let reference = || Expression::from(x.clone());
    let registry = FunctionRegistry::new();

    let unknown = body_error(
        &build_call_or_panic("g", [reference()]),
        &x,
        FunctionSort::Int,
        &registry,
    );
    let unsupported = body_error(
        &Expression::from(LiteralValue::Decimal("0.5".parse().expect("a decimal"))),
        &x,
        FunctionSort::Real,
        &registry,
    );
    let ill_typed = body_error(
        &(reference() + LiteralValue::Bool(true)),
        &x,
        FunctionSort::Int,
        &registry,
    );
    let incompatible = body_error(&reference().less(1), &x, FunctionSort::Int, &registry);
    let callback = body_error(
        &build_call_or_panic("g", [reference()]),
        &x,
        FunctionSort::Int,
        &UnknownTargets { fails: true },
    );

    assert_error_text(
        &unknown,
        "function 'scale' body calls a function that is not registered: no entry is registered \
         under the name 'g'; every call target must resolve by the time the body is held to its \
         declared result sort",
        Source::None,
    );
    assert!(matches!(
        &unknown,
        BodyCheckError::UnknownCall { function, error: CallTargetError::Unknown { name, .. } }
            if *function == scale() && name == "g"
    ));
    assert_error_text(
        &unsupported,
        "function 'scale' body uses a construct the body type checker does not support: type \
         error while inferring the type of `0.5`: decimal literals are not yet supported",
        Source::None,
    );
    assert!(matches!(
        &unsupported,
        BodyCheckError::Unsupported { function, error: TypeCheckError::Rule { .. } }
            if *function == scale()
    ));
    assert_error_text(
        &ill_typed,
        "function 'scale' body failed to type-check: type error while inferring the type of \
         `(x::61722 + true)`: the add operation is not defined for boolean operands",
        Source::None,
    );
    assert!(matches!(
        &ill_typed,
        BodyCheckError::IllTyped { function, error: TypeCheckError::Rule { rule, .. } }
            if *function == scale() && rule.kind() == TypeRuleKind::Boolean
    ));
    assert_error_text(
        &incompatible,
        "function 'scale' body synthesized type bool is not compatible with the declared result \
         sort int",
        Source::None,
    );
    assert!(matches!(
        &incompatible,
        BodyCheckError::IncompatibleResult { function, body: CoreDataType::Bool, sort: FunctionSort::Int }
            if *function == scale()
    ));
    assert_error_text(&callback, "a call target lookup failed", Source::TestValue);
}

#[rstest]
#[case::not_scalar(
    BodyCheckError::NotScalar {
        function: FunctionLabel::User(FunctionName::new("scale").expect("a name")),
        body_type: array(
            DataType::Primitive(CoreDataType::Int8),
            [Dimension::Expression(Expression::from(4))],
        ),
    },
    "function 'scale' body must synthesize a scalar numerical type, but got int8[4]",
    Source::None
)]
#[case::incompatible_builtin(
    BodyCheckError::IncompatibleResult {
        function: FunctionLabel::Builtin(BuiltinFunction::Max),
        body: CoreDataType::Bool,
        sort: FunctionSort::Real,
    },
    "function 'max' body synthesized type bool is not compatible with the declared result sort \
     real",
    Source::None
)]
#[case::signature(
    BodyCheckError::Signature(SignatureError::LengthMismatch {
        function: FunctionLabel::User(FunctionName::new("scale").expect("a name")),
        parameters: 1,
        parameter_sorts: 2,
    }),
    "function 'scale' has 1 parameter(s) but 2 parameter sort(s)",
    Source::None
)]
fn built_body_check_error_text(
    #[case] error: BodyCheckError,
    #[case] text: &str,
    #[case] source: Source,
) {
    assert_error_text(&error, text, source);
}
