//! Property tests for `FunctionRegistry::inline` over generated trees that
//! call composed built-ins, native built-ins and user functions, nested:
//! inlining leaves no call to inline, is idempotent, and keeps the meaning
//! a reference evaluation gives the calls.

use std::collections::HashMap;
use std::sync::LazyLock;

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::registry::{FunctionDefinition, FunctionRegistry};
use fhy_core::expression::{
    BinaryOperation, Callee, Expression, ExpressionKind, FunctionName, FunctionSort, LiteralValue,
    LogicalOperation, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use proptest::prelude::*;
use proptest::sample::select;

/// The Boolean identifiers the generated trees refer to.
static BOOLEANS: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("b0"),
        Identifier::new("b1"),
        Identifier::new("b2"),
    ]
});

/// The numeric identifiers the generated trees refer to.
static NUMBERS: LazyLock<[Identifier; 2]> =
    LazyLock::new(|| [Identifier::new("n0"), Identifier::new("n1")]);

fn name(text: &str) -> FunctionName {
    FunctionName::new(text).expect("the test names no built-in")
}

/// Return the registry of the user functions the trees call:
/// `double(x) = x * 2`, `majority(a, b, c) = (a && b) || (a && c) || (b &&
/// c)`, and `between(x, lo, hi) = clamp(x, lo, hi) == x`, which calls a
/// composed built-in.
fn build_registry() -> FunctionRegistry {
    fn parameters(names: &[&str]) -> (Vec<Identifier>, Vec<Expression>) {
        let identifiers: Vec<Identifier> = names.iter().map(|text| Identifier::new(text)).collect();
        let references = identifiers.iter().cloned().map(Expression::from).collect();
        (identifiers, references)
    }
    let mut registry = FunctionRegistry::new();
    let (x, references) = parameters(&["x"]);
    registry
        .register_function(
            FunctionDefinition::new(
                name("double"),
                x,
                [FunctionSort::Real],
                FunctionSort::Real,
                &references[0] * 2,
            )
            .expect("one sort per parameter"),
        )
        .expect("free");
    let (abc, references) = parameters(&["a", "b", "c"]);
    let [a, b, c] = [&references[0], &references[1], &references[2]];
    registry
        .register_function(
            FunctionDefinition::new(
                name("majority"),
                abc,
                [FunctionSort::Bool; 3],
                FunctionSort::Bool,
                Expression::any([a.and(b), a.and(c), b.and(c)]),
            )
            .expect("one sort per parameter"),
        )
        .expect("free");
    let (bounds, references) = parameters(&["x", "lo", "hi"]);
    registry
        .register_function(
            FunctionDefinition::new(
                name("between"),
                bounds,
                [FunctionSort::Real; 3],
                FunctionSort::Bool,
                Expression::call(BuiltinFunction::Clamp, references.iter()).equals(&references[0]),
            )
            .expect("one sort per parameter"),
        )
        .expect("free");
    registry
}

/// A value of the reference evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    Bool(bool),
    Number(f64),
}

impl Value {
    fn as_bool(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::Number(value) => panic!("expected a Boolean, got {value}"),
        }
    }

    fn as_number(self) -> f64 {
        match self {
            Self::Number(value) => value,
            Self::Bool(value) => panic!("expected a number, got {value}"),
        }
    }
}

fn maximum(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn minimum(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    minimum(maximum(x, lo), hi)
}

/// Return the value of the call of `callee` on `arguments`, by the
/// reference meaning of each function the trees call.
fn evaluate_call(callee: &Callee, arguments: &[Value]) -> Value {
    let number = |index: usize| arguments[index].as_number();
    let boolean = |index: usize| arguments[index].as_bool();
    match callee {
        Callee::Builtin(function) => match function {
            BuiltinFunction::Relu => Value::Number(maximum(number(0), 0.0)),
            BuiltinFunction::Max => Value::Number(maximum(number(0), number(1))),
            BuiltinFunction::Min => Value::Number(minimum(number(0), number(1))),
            BuiltinFunction::Abs => Value::Number(if number(0) >= 0.0 {
                number(0)
            } else {
                -number(0)
            }),
            BuiltinFunction::Sign => Value::Number(if number(0) > 0.0 {
                1.0
            } else if number(0) < 0.0 {
                -1.0
            } else {
                0.0
            }),
            BuiltinFunction::Clamp => Value::Number(clamp(number(0), number(1), number(2))),
            BuiltinFunction::Floor => Value::Number(number(0).floor()),
            BuiltinFunction::Xor => Value::Bool(boolean(0) != boolean(1)),
            BuiltinFunction::Nand => Value::Bool(!(boolean(0) && boolean(1))),
            BuiltinFunction::Nor => Value::Bool(!(boolean(0) || boolean(1))),
            BuiltinFunction::Implies => Value::Bool(!boolean(0) || boolean(1)),
            BuiltinFunction::Iff => Value::Bool(boolean(0) == boolean(1)),
            other => panic!("the trees never call {other:?}"),
        },
        Callee::Named(function) => match function.as_str() {
            "double" => Value::Number(number(0) * 2.0),
            "majority" => {
                let count = (0..3).filter(|&index| boolean(index)).count();
                Value::Bool(count >= 2)
            }
            "between" => Value::Bool(
                Value::Number(clamp(number(0), number(1), number(2))) == Value::Number(number(0)),
            ),
            other => panic!("the trees never call {other}"),
        },
    }
}

/// Return the value of `expression` with its identifiers bound by
/// `bindings`. The generated trees are shallow, so this recurses.
fn evaluate(expression: &Expression, bindings: &HashMap<Identifier, Value>) -> Value {
    match expression.kind() {
        ExpressionKind::Identifier(identifier) => bindings[identifier],
        ExpressionKind::Literal(LiteralValue::Bool(value)) => Value::Bool(*value),
        ExpressionKind::Literal(LiteralValue::Int(value)) => {
            Value::Number(value.to_string().parse().expect("a small integer"))
        }
        ExpressionKind::Literal(LiteralValue::Float(value)) => Value::Number(*value),
        ExpressionKind::Literal(literal) => panic!("the trees hold no {literal:?}"),
        ExpressionKind::Unary(node) => {
            let operand = evaluate(node.operand(), bindings);
            match node.operation() {
                UnaryOperation::Negate => Value::Number(-operand.as_number()),
                UnaryOperation::LogicalNot => Value::Bool(!operand.as_bool()),
                other @ UnaryOperation::Positive => panic!("the trees hold no {other:?}"),
            }
        }
        ExpressionKind::Binary(node) => {
            let left = evaluate(node.left(), bindings);
            let right = evaluate(node.right(), bindings);
            match node.operation() {
                BinaryOperation::Add => Value::Number(left.as_number() + right.as_number()),
                BinaryOperation::Subtract => Value::Number(left.as_number() - right.as_number()),
                BinaryOperation::Multiply => Value::Number(left.as_number() * right.as_number()),
                BinaryOperation::Greater => Value::Bool(left.as_number() > right.as_number()),
                BinaryOperation::GreaterEqual => Value::Bool(left.as_number() >= right.as_number()),
                BinaryOperation::Less => Value::Bool(left.as_number() < right.as_number()),
                BinaryOperation::Equal => Value::Bool(left == right),
                BinaryOperation::NotEqual => Value::Bool(left != right),
                other => panic!("the trees hold no {other:?}"),
            }
        }
        ExpressionKind::Logical(node) => {
            let mut values = node
                .operands()
                .iter()
                .map(|operand| evaluate(operand, bindings).as_bool());
            Value::Bool(match node.operation() {
                LogicalOperation::And => values.all(|value| value),
                LogicalOperation::Or => values.any(|value| value),
            })
        }
        ExpressionKind::Piecewise(node) => node
            .cases()
            .iter()
            .find(|(condition, _)| evaluate(condition, bindings).as_bool())
            .map_or_else(
                || evaluate(node.otherwise(), bindings),
                |(_, value)| evaluate(value, bindings),
            ),
        ExpressionKind::Call(node) => {
            let arguments: Vec<Value> = node
                .arguments()
                .iter()
                .map(|argument| evaluate(argument, bindings))
                .collect();
            evaluate_call(node.callee(), &arguments)
        }
    }
}

fn build_number_strategy() -> impl Strategy<Value = Expression> {
    let leaf = prop_oneof![
        select(NUMBERS.to_vec()).prop_map(Expression::from),
        (-3_i64..=3).prop_map(Expression::from),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        let named = |function: &'static str| {
            move |arguments: Vec<Expression>| Expression::call(name(function), arguments)
        };
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(left, right)| left + right),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| left - right),
            inner.clone().prop_map(|operand| -operand),
            prop::collection::vec(inner.clone(), 1)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Relu, arguments)),
            prop::collection::vec(inner.clone(), 1)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Abs, arguments)),
            prop::collection::vec(inner.clone(), 1)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Sign, arguments)),
            prop::collection::vec(inner.clone(), 1)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Floor, arguments)),
            prop::collection::vec(inner.clone(), 2)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Max, arguments)),
            prop::collection::vec(inner.clone(), 2)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Min, arguments)),
            prop::collection::vec(inner.clone(), 3)
                .prop_map(|arguments| Expression::call(BuiltinFunction::Clamp, arguments)),
            prop::collection::vec(inner, 1).prop_map(named("double")),
        ]
    })
}

fn build_boolean_strategy() -> impl Strategy<Value = Expression> {
    let comparison = (build_number_strategy(), build_number_strategy(), 0..3_u8).prop_map(
        |(left, right, operation)| match operation {
            0 => left.greater(right),
            1 => left.less(right),
            _ => left.equals(right),
        },
    );
    let leaf = prop_oneof![
        select(BOOLEANS.to_vec()).prop_map(Expression::from),
        any::<bool>().prop_map(Expression::literal),
        comparison,
        prop::collection::vec(build_number_strategy(), 3)
            .prop_map(|arguments| Expression::call(name("between"), arguments)),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        let connective = |function: BuiltinFunction| {
            move |arguments: Vec<Expression>| Expression::call(function, arguments)
        };
        prop_oneof![
            inner.clone().prop_map(|operand| !operand),
            prop::collection::vec(inner.clone(), 2..=3).prop_map(Expression::all),
            prop::collection::vec(inner.clone(), 2..=3).prop_map(Expression::any),
            prop::collection::vec(inner.clone(), 2).prop_map(connective(BuiltinFunction::Xor)),
            prop::collection::vec(inner.clone(), 2).prop_map(connective(BuiltinFunction::Nand)),
            prop::collection::vec(inner.clone(), 2).prop_map(connective(BuiltinFunction::Nor)),
            prop::collection::vec(inner.clone(), 2).prop_map(connective(BuiltinFunction::Implies)),
            prop::collection::vec(inner.clone(), 2).prop_map(connective(BuiltinFunction::Iff)),
            prop::collection::vec(inner.clone(), 3)
                .prop_map(|arguments| Expression::call(name("majority"), arguments)),
            (inner.clone(), inner.clone(), inner).prop_map(|(condition, value, otherwise)| {
                Expression::piecewise([(condition, value)], otherwise)
                    .expect("a condition that is no numeric literal")
            }),
        ]
    })
}

/// Return whether `expression` calls a composed built-in or a user
/// function anywhere.
fn calls_an_inlinable_function(expression: &Expression) -> bool {
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        if let ExpressionKind::Call(call) = node.kind() {
            match call.callee() {
                Callee::Builtin(function) if function.composed().is_some() => return true,
                Callee::Named(_) => return true,
                Callee::Builtin(_) => {}
            }
        }
        pending.extend(node.children());
    }
    false
}

proptest! {
    #[test]
    fn inline_leaves_no_call_of_a_composed_builtin_or_a_user_function(
        expression in build_boolean_strategy(),
    ) {
        let registry = build_registry();

        let inlined = registry.inline(&expression).expect("every callee is defined");

        prop_assert!(!calls_an_inlinable_function(&inlined), "{}", inlined);
    }

    #[test]
    fn inline_is_idempotent(expression in build_boolean_strategy()) {
        let registry = build_registry();
        let once = registry.inline(&expression).expect("every callee is defined");

        let twice = registry.inline(&once).expect("nothing is left to inline");

        prop_assert!(Expression::ptr_eq(&once, &twice));
    }

    #[test]
    fn inline_keeps_the_reference_meaning_of_the_calls(
        expression in build_boolean_strategy(),
        booleans in any::<[bool; 3]>(),
        numbers in any::<[i8; 2]>(),
    ) {
        let registry = build_registry();
        let bindings: HashMap<Identifier, Value> = BOOLEANS
            .iter()
            .cloned()
            .zip(booleans.map(Value::Bool))
            .chain(
                NUMBERS
                    .iter()
                    .cloned()
                    .zip(numbers.map(|number| Value::Number(f64::from(number)))),
            )
            .collect();

        let inlined = registry.inline(&expression).expect("every callee is defined");

        prop_assert_eq!(evaluate(&inlined, &bindings), evaluate(&expression, &bindings));
    }
}
