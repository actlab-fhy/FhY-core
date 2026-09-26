//! Properties of the solver: every screened-safe tree lowers to a
//! well-formed script, and, with a real SMT solver, ground scripts agree
//! with an exact rational evaluation and the three questions agree with
//! brute force over small integer domains.
//!
//! The solver-backed properties run on the z3 backend under the `z3`
//! feature, and otherwise when `FHY_SMT_SOLVER` names an SMT-LIB2
//! executable, such as `z3 -in`; without either, they pass trivially.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use fhy_core::expression::{
    BigInt, BinaryOperation, Expression, ExpressionKind, LiteralValue, NoRegisteredSorts,
    SymbolType, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    CheckLimits, QueryContext, Question, SatResult, SmtScript, SmtSolver, Solver,
};
use num_traits::ToPrimitive;
use proptest::prelude::*;

use crate::support::solver::build_symbol_types;

/// The smallest and largest value of the brute-force domain.
const DOMAIN: (i64, i64) = (-3, 3);

/// How long a solver-backed property's check may run. A check that runs
/// out of time answers `unknown`, which the properties accept: they pin
/// that every decided answer agrees with the reference, since a solver may
/// give up on nonlinear integer arithmetic.
const CHECK_TIMEOUT: Duration = Duration::from_secs(2);

/// Return the limits of a solver-backed property's check.
fn limits() -> CheckLimits {
    CheckLimits::new().with_timeout(CHECK_TIMEOUT)
}

/// Return the SMT solver the solver-backed properties run on, if any.
#[cfg(feature = "z3")]
#[expect(
    clippy::unnecessary_wraps,
    reason = "without the feature, there may be no backend"
)]
fn property_backend() -> Option<Arc<dyn SmtSolver>> {
    Some(Arc::new(fhy_core::solver::Z3Solver::new()))
}

/// Return the SMT solver the solver-backed properties run on, if any.
#[cfg(not(feature = "z3"))]
fn property_backend() -> Option<Arc<dyn SmtSolver>> {
    let configured = std::env::var("FHY_SMT_SOLVER").ok()?;
    let mut words = configured.split_whitespace();
    let program = words.next()?;
    Some(Arc::new(
        fhy_core::solver::SmtLib2Process::new(program).with_args(words),
    ))
}

// ---------------------------------------------------------------------------
// Integer trees over two identifiers
// ---------------------------------------------------------------------------

/// The identifiers of the generated integer trees.
struct Variables {
    x: Identifier,
    y: Identifier,
}

impl Variables {
    fn new() -> Self {
        Self {
            x: Identifier::new("x"),
            y: Identifier::new("y"),
        }
    }
}

/// A generated integer term: every operation the screen passes on
/// integers.
#[derive(Debug, Clone)]
enum IntTerm {
    X,
    Y,
    Literal(i64),
    Add(Box<IntTerm>, Box<IntTerm>),
    Subtract(Box<IntTerm>, Box<IntTerm>),
    Multiply(Box<IntTerm>, Box<IntTerm>),
    Negate(Box<IntTerm>),
    FloorDivide(Box<IntTerm>, i64),
    FloorMod(Box<IntTerm>, i64),
    Square(Box<IntTerm>),
}

/// A generated predicate over integer terms.
#[derive(Debug, Clone)]
enum Predicate {
    Compare(BinaryOperation, IntTerm, IntTerm),
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
}

fn int_term() -> impl Strategy<Value = IntTerm> {
    let leaf = prop_oneof![
        Just(IntTerm::X),
        Just(IntTerm::Y),
        (-3_i64..=3).prop_map(IntTerm::Literal),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| IntTerm::Add(a.into(), b.into())),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| IntTerm::Subtract(a.into(), b.into())),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| IntTerm::Multiply(a.into(), b.into())),
            inner.clone().prop_map(|a| IntTerm::Negate(a.into())),
            (inner.clone(), 1_i64..=3).prop_map(|(a, k)| IntTerm::FloorDivide(a.into(), k)),
            (inner.clone(), 1_i64..=3).prop_map(|(a, k)| IntTerm::FloorMod(a.into(), k)),
            inner.prop_map(|a| IntTerm::Square(a.into())),
        ]
    })
}

fn predicate() -> impl Strategy<Value = Predicate> {
    let comparison = prop_oneof![
        Just(BinaryOperation::Equal),
        Just(BinaryOperation::NotEqual),
        Just(BinaryOperation::Less),
        Just(BinaryOperation::LessEqual),
        Just(BinaryOperation::Greater),
        Just(BinaryOperation::GreaterEqual),
    ];
    let leaf = (comparison, int_term(), int_term())
        .prop_map(|(operation, left, right)| Predicate::Compare(operation, left, right));
    leaf.prop_recursive(2, 6, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Predicate::And(a.into(), b.into())),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Predicate::Or(a.into(), b.into())),
            inner.prop_map(|a| Predicate::Not(a.into())),
        ]
    })
}

impl IntTerm {
    fn build(&self, variables: &Variables) -> Expression {
        match self {
            Self::X => Expression::from(variables.x.clone()),
            Self::Y => Expression::from(variables.y.clone()),
            Self::Literal(value) => Expression::literal(*value),
            Self::Add(a, b) => a.build(variables) + b.build(variables),
            Self::Subtract(a, b) => a.build(variables) - b.build(variables),
            Self::Multiply(a, b) => a.build(variables) * b.build(variables),
            Self::Negate(a) => -a.build(variables),
            Self::FloorDivide(a, k) => a.build(variables).floor_divide(*k),
            Self::FloorMod(a, k) => a.build(variables).floor_mod(*k),
            Self::Square(a) => a.build(variables).power(2),
        }
    }

    fn evaluate(&self, x: i128, y: i128) -> i128 {
        match self {
            Self::X => x,
            Self::Y => y,
            Self::Literal(value) => i128::from(*value),
            Self::Add(a, b) => a.evaluate(x, y) + b.evaluate(x, y),
            Self::Subtract(a, b) => a.evaluate(x, y) - b.evaluate(x, y),
            Self::Multiply(a, b) => a.evaluate(x, y) * b.evaluate(x, y),
            Self::Negate(a) => -a.evaluate(x, y),
            Self::FloorDivide(a, k) => a.evaluate(x, y).div_euclid(i128::from(*k)),
            Self::FloorMod(a, k) => a.evaluate(x, y).rem_euclid(i128::from(*k)),
            Self::Square(a) => a.evaluate(x, y).pow(2),
        }
    }
}

impl Predicate {
    fn build(&self, variables: &Variables) -> Expression {
        match self {
            Self::Compare(operation, left, right) => {
                Expression::new_binary(*operation, left.build(variables), right.build(variables))
            }
            Self::And(a, b) => a.build(variables).and(b.build(variables)),
            Self::Or(a, b) => a.build(variables).or(b.build(variables)),
            Self::Not(a) => !a.build(variables),
        }
    }

    fn evaluate(&self, x: i128, y: i128) -> bool {
        match self {
            Self::Compare(operation, left, right) => {
                let ordering = left.evaluate(x, y).cmp(&right.evaluate(x, y));
                compare(*operation, ordering)
            }
            Self::And(a, b) => a.evaluate(x, y) && b.evaluate(x, y),
            Self::Or(a, b) => a.evaluate(x, y) || b.evaluate(x, y),
            Self::Not(a) => !a.evaluate(x, y),
        }
    }
}

fn compare(operation: BinaryOperation, ordering: Ordering) -> bool {
    match operation {
        BinaryOperation::Equal => ordering.is_eq(),
        BinaryOperation::NotEqual => ordering.is_ne(),
        BinaryOperation::Less => ordering.is_lt(),
        BinaryOperation::LessEqual => ordering.is_le(),
        BinaryOperation::Greater => ordering.is_gt(),
        BinaryOperation::GreaterEqual => ordering.is_ge(),
        other => panic!("{other:?} is no comparison"),
    }
}

/// Return `lower <= reference <= upper` for the brute-force domain.
fn bound(identifier: &Identifier) -> Expression {
    let reference = Expression::from(identifier.clone());
    Expression::all([
        reference.clone().greater_equal(DOMAIN.0),
        reference.less_equal(DOMAIN.1),
    ])
}

fn domain() -> impl Iterator<Item = i128> {
    i128::from(DOMAIN.0)..=i128::from(DOMAIN.1)
}

/// Return whether `text`'s parentheses balance outside quoted symbols.
fn is_balanced(text: &str) -> bool {
    let mut depth: i64 = 0;
    let mut is_quoted = false;
    for character in text.chars() {
        match character {
            '|' => is_quoted = !is_quoted,
            '(' if !is_quoted => depth += 1,
            ')' if !is_quoted => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0 && !is_quoted
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn screened_safe_tree_lowers_to_a_balanced_script_declaring_its_identifiers(
        generated in predicate(),
    ) {
        let variables = Variables::new();
        let expression = generated.build(&variables);
        let symbol_types = |_: &Identifier| Some(SymbolType::Int);

        let script = SmtScript::lower(&expression, &symbol_types, &NoRegisteredSorts)
            .expect("an integer predicate lowers");
        let text = script.to_string();
        let declared: HashSet<Identifier> = script
            .declarations()
            .iter()
            .map(|declaration| declaration.identifier().clone())
            .collect();

        prop_assert!(is_balanced(&text), "{}", text);
        prop_assert_eq!(declared, expression.free_identifiers());
        prop_assert_eq!(text.matches("(declare-const").count(), script.declarations().len());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn satisfiability_agrees_with_brute_force_over_a_small_domain(generated in predicate()) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let variables = Variables::new();
        let expression = Expression::all([
            bound(&variables.x),
            bound(&variables.y),
            generated.build(&variables),
        ]);
        let symbol_types = build_symbol_types(&[(&variables.x, SymbolType::Int), (&variables.y, SymbolType::Int)]);
        let expected = domain().any(|x| domain().any(|y| generated.evaluate(x, y)));

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::Satisfiability(&expression),
                &QueryContext::new(&symbol_types).with_limits(limits()),
            )
            .expect("answered");

        prop_assert!(answer.decided().is_none_or(|decided| decided == expected), "{:?} against {}", answer, expected);
    }

    #[test]
    fn implication_agrees_with_brute_force_over_a_small_domain(
        antecedent in predicate(),
        consequent in predicate(),
    ) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let variables = Variables::new();
        let bounded = Expression::all([
            bound(&variables.x),
            bound(&variables.y),
            antecedent.build(&variables),
        ]);
        let symbol_types = build_symbol_types(&[(&variables.x, SymbolType::Int), (&variables.y, SymbolType::Int)]);
        let expected = domain().all(|x| {
            domain().all(|y| !antecedent.evaluate(x, y) || consequent.evaluate(x, y))
        });

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::Implication { antecedent: &bounded, consequent: &consequent.build(&variables) },
                &QueryContext::new(&symbol_types).with_limits(limits()),
            )
            .expect("answered");

        prop_assert!(answer.decided().is_none_or(|decided| decided == expected), "{:?} against {}", answer, expected);
    }

    #[test]
    fn universal_validity_agrees_with_brute_force_over_a_small_domain(generated in predicate()) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let variables = Variables::new();
        let expression = Expression::any([
            !bound(&variables.x),
            Expression::all([bound(&variables.y), generated.build(&variables)]),
        ]);
        let considered = HashSet::from([variables.y.clone()]);
        let symbol_types = build_symbol_types(&[(&variables.x, SymbolType::Int), (&variables.y, SymbolType::Int)]);
        let expected = domain().all(|x| domain().any(|y| generated.evaluate(x, y)));

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::UniversalValidity { considered: &considered, expression: &expression },
                &QueryContext::new(&symbol_types).with_limits(limits()),
            )
            .expect("answered");

        prop_assert!(answer.decided().is_none_or(|decided| decided == expected), "{:?} against {}", answer, expected);
    }
}

// ---------------------------------------------------------------------------
// Ground trees against an exact rational evaluation
// ---------------------------------------------------------------------------

/// An exact rational, `numerator / denominator` with a positive denominator.
#[derive(Debug, Clone)]
struct Rational {
    numerator: BigInt,
    denominator: BigInt,
}

impl Rational {
    fn from_integer(value: BigInt) -> Self {
        Self {
            numerator: value,
            denominator: BigInt::from(1),
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            numerator: &self.numerator * &other.denominator + &other.numerator * &self.denominator,
            denominator: &self.denominator * &other.denominator,
        }
    }

    fn negate(&self) -> Self {
        Self {
            numerator: -&self.numerator,
            denominator: self.denominator.clone(),
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        Self {
            numerator: &self.numerator * &other.numerator,
            denominator: &self.denominator * &other.denominator,
        }
    }

    /// Return `self / other`, for a nonzero `other`.
    fn divide(&self, other: &Self) -> Self {
        let (numerator, denominator) = (
            &self.numerator * &other.denominator,
            &self.denominator * &other.numerator,
        );
        if denominator < BigInt::from(0) {
            Self {
                numerator: -numerator,
                denominator: -denominator,
            }
        } else {
            Self {
                numerator,
                denominator,
            }
        }
    }

    fn floor(&self) -> BigInt {
        let quotient = &self.numerator / &self.denominator;
        let remainder = &self.numerator % &self.denominator;
        if remainder < BigInt::from(0) {
            quotient - 1
        } else {
            quotient
        }
    }

    fn compare(&self, other: &Self) -> Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }
}

/// Return the exact value of a ground numeric `expression`, reading every
/// division as exact and every floor operation as rounding toward negative
/// infinity, the core's semantics.
fn evaluate_exactly(expression: &Expression) -> Rational {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Int(value)) => Rational::from_integer(value.clone()),
        ExpressionKind::Unary(node) if node.operation() == UnaryOperation::Negate => {
            evaluate_exactly(node.operand()).negate()
        }
        ExpressionKind::Binary(node) => {
            let left = evaluate_exactly(node.left());
            let right = evaluate_exactly(node.right());
            match node.operation() {
                BinaryOperation::Add => left.add(&right),
                BinaryOperation::Subtract => left.add(&right.negate()),
                BinaryOperation::Multiply => left.multiply(&right),
                BinaryOperation::Divide => left.divide(&right),
                BinaryOperation::FloorDivide => Rational::from_integer(left.divide(&right).floor()),
                BinaryOperation::FloorMod => left.add(
                    &right
                        .multiply(&Rational::from_integer(left.divide(&right).floor()))
                        .negate(),
                ),
                other => panic!("{other:?} is not generated"),
            }
        }
        _ => panic!("{expression} is not generated"),
    }
}

fn nonzero_literal() -> impl Strategy<Value = i64> {
    prop_oneof![-5_i64..=-1, 1_i64..=5]
}

fn ground_term() -> impl Strategy<Value = Expression> {
    let leaf = (-5_i64..=5).prop_map(Expression::literal);
    leaf.prop_recursive(4, 16, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
            inner.clone().prop_map(|a| -a),
            (inner.clone(), nonzero_literal()).prop_map(|(a, k)| a / k),
            (inner.clone(), nonzero_literal()).prop_map(|(a, k)| a.floor_divide(k)),
            (inner, nonzero_literal()).prop_map(|(a, k)| a.floor_mod(k)),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn ground_ordering_lowers_to_a_script_as_satisfiable_as_it_is_true(
        left in ground_term(),
        right in ground_term(),
        operation in prop_oneof![
            Just(BinaryOperation::Less),
            Just(BinaryOperation::LessEqual),
            Just(BinaryOperation::Greater),
            Just(BinaryOperation::GreaterEqual),
        ],
    ) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let expected = compare(operation, evaluate_exactly(&left).compare(&evaluate_exactly(&right)));
        let predicate = Expression::new_binary(operation, left, right);
        let script = SmtScript::lower(&predicate, &HashMap::<Identifier, SymbolType>::new(), &NoRegisteredSorts)
            .expect("a ground predicate lowers");

        let result = backend.check(&script, &limits()).expect("checked");

        prop_assert!(
            matches!(result, SatResult::Unknown { .. })
                || result == if expected { SatResult::Sat } else { SatResult::Unsat },
            "{:?} for {}", result, script
        );
    }
}

#[test]
fn exact_evaluation_floors_toward_negative_infinity() {
    let seven = Expression::literal(7);

    assert_eq!(
        evaluate_exactly(&seven.clone().floor_divide(-2))
            .numerator
            .to_i64(),
        Some(-4)
    );
    assert_eq!(
        evaluate_exactly(&seven.floor_mod(-2)).numerator.to_i64(),
        Some(-1)
    );
}
