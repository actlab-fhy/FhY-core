//! Properties of the solver: every screened-safe tree lowers to a
//! well-formed script, and, with a real SMT solver, ground scripts agree
//! with an exact rational evaluation, the three questions agree with brute
//! force over small integer and Boolean domains, and, under the `z3`
//! feature, the z3 backend agrees with the process backend.
//!
//! The solver-backed properties run on the z3 backend under the `z3`
//! feature, and otherwise on the process backend `FHY_SMT_SOLVER` names,
//! such as `z3 -in`. Without one, they fail when `CI` is set, and otherwise
//! say on standard error that they were skipped.
//!
//! They cannot pass vacuously: the generators draw only trees the hazard
//! screen admits, so a refused answer fails, and a backend may give up on
//! at most [`GAVE_UP_BUDGET`] of a property's cases.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{
    BigInt, BinaryOperation, Expression, ExpressionKind, LiteralValue, NoRegisteredSorts,
    SymbolType, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    Answer, CheckLimits, QueryContext, Question, SatResult, SmtScript, SmtSolver, Solver,
    UnknownReason,
};
use num_traits::ToPrimitive;
use proptest::prelude::*;

use crate::support::solver::build_symbol_types;

/// The smallest and largest value of the brute-force domain.
const DOMAIN: (i64, i64) = (-3, 3);

/// How long a solver-backed property's check may run. A check that runs
/// out of time answers `unknown`, which the properties accept for at most
/// [`GAVE_UP_BUDGET`] cases, since a solver may give up on nonlinear
/// integer arithmetic.
const CHECK_TIMEOUT: Duration = Duration::from_secs(2);

/// How many of a property's cases a backend may give up on.
const GAVE_UP_BUDGET: usize = 4;

/// Return the limits of a solver-backed property's check.
fn limits() -> CheckLimits {
    CheckLimits::new().with_timeout(CHECK_TIMEOUT)
}

/// Return the process backend `FHY_SMT_SOLVER` names, as a program followed
/// by its arguments, or `None` when it is unset.
fn configured_process_backend() -> Option<Arc<dyn SmtSolver>> {
    let configured = std::env::var("FHY_SMT_SOLVER").ok()?;
    let mut words = configured.split_whitespace();
    let program = words.next()?;
    Some(Arc::new(
        fhy_core::solver::SmtLib2Process::new(program).with_args(words),
    ))
}

/// Return `backend`, or, when it is missing, fail under `CI` and otherwise
/// say once on standard error that `what` is skipped.
fn required(backend: Option<Arc<dyn SmtSolver>>, what: &str) -> Option<Arc<dyn SmtSolver>> {
    static SKIPPED: Once = Once::new();
    if backend.is_none() {
        assert!(
            std::env::var_os("CI").is_none(),
            "{what}: FHY_SMT_SOLVER must name an SMT-LIB2 solver, such as `z3 -in`, when CI is set"
        );
        SKIPPED.call_once(|| {
            #[expect(
                clippy::print_stderr,
                reason = "a skipped property must say so where the test runner shows it"
            )]
            {
                eprintln!("skipping the solver properties: FHY_SMT_SOLVER is not set");
            }
        });
    }
    backend
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
    required(configured_process_backend(), "the solver properties")
}

/// Check that `answer`, to a question the screen admits, is decided or,
/// within the property's `gave_up` budget, given up; a refusal never is.
fn check_not_vacuous(answer: &Answer, gave_up: &AtomicUsize) -> Result<(), TestCaseError> {
    match answer {
        Answer::Yes | Answer::No => Ok(()),
        Answer::Unknown(UnknownReason::Refused(hazard)) => Err(TestCaseError::fail(format!(
            "a generated question was refused: {hazard}"
        ))),
        Answer::Unknown(UnknownReason::GaveUp { reason }) => {
            let count = gave_up.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if count > GAVE_UP_BUDGET {
                return Err(TestCaseError::fail(format!(
                    "the backend gave up on {count} cases, the last with {reason:?}"
                )));
            }
            Ok(())
        }
        other @ Answer::Unknown(_) => {
            Err(TestCaseError::fail(format!("unexpected answer {other:?}")))
        }
    }
}

// ---------------------------------------------------------------------------
// Integer and Boolean trees over two integers and a flag
// ---------------------------------------------------------------------------

/// The identifiers of the generated trees: two integers and a Boolean.
struct Variables {
    x: Identifier,
    y: Identifier,
    p: Identifier,
}

impl Variables {
    fn new() -> Self {
        Self {
            x: Identifier::new("x"),
            y: Identifier::new("y"),
            p: Identifier::new("p"),
        }
    }

    /// Return the symbol types of the three identifiers.
    fn symbol_types(&self) -> HashMap<Identifier, SymbolType> {
        build_symbol_types(&[
            (&self.x, SymbolType::Int),
            (&self.y, SymbolType::Int),
            (&self.p, SymbolType::Bool),
        ])
    }
}

/// A point of the brute-force domain: the two integers and the flag.
#[derive(Debug, Clone, Copy)]
struct Point {
    x: i128,
    y: i128,
    p: bool,
}

/// A generated integer term: every operation the screen passes on
/// integers, and a piecewise leaf chosen by a simple condition.
#[derive(Debug, Clone)]
enum IntTerm {
    X,
    Y,
    Literal(i64),
    Add(Box<Self>, Box<Self>),
    Subtract(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Negate(Box<Self>),
    FloorDivide(Box<Self>, i64),
    FloorMod(Box<Self>, i64),
    Square(Box<Self>),
    Piecewise(Box<Predicate>, Box<Self>, Box<Self>),
}

/// A generated predicate over integer terms and the flag.
#[derive(Debug, Clone)]
enum Predicate {
    Compare(BinaryOperation, IntTerm, IntTerm),
    Flag,
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
}

fn comparison() -> impl Strategy<Value = BinaryOperation> {
    prop_oneof![
        Just(BinaryOperation::Equal),
        Just(BinaryOperation::NotEqual),
        Just(BinaryOperation::Less),
        Just(BinaryOperation::LessEqual),
        Just(BinaryOperation::Greater),
        Just(BinaryOperation::GreaterEqual),
    ]
}

/// Return a leaf term: `x`, `y` or a literal.
fn leaf_term() -> impl Strategy<Value = IntTerm> {
    prop_oneof![
        Just(IntTerm::X),
        Just(IntTerm::Y),
        (-3_i64..=3).prop_map(IntTerm::Literal),
    ]
}

/// Return the condition of a piecewise leaf: the flag, or a comparison of
/// two leaf terms.
fn simple_condition() -> impl Strategy<Value = Predicate> {
    prop_oneof![
        Just(Predicate::Flag),
        (comparison(), leaf_term(), leaf_term())
            .prop_map(|(operation, left, right)| Predicate::Compare(operation, left, right)),
    ]
}

fn int_term() -> impl Strategy<Value = IntTerm> {
    let leaf = prop_oneof![
        3 => leaf_term(),
        1 => (simple_condition(), leaf_term(), leaf_term()).prop_map(|(condition, a, b)| {
            IntTerm::Piecewise(condition.into(), a.into(), b.into())
        }),
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
    let leaf = prop_oneof![
        4 => (comparison(), int_term(), int_term())
            .prop_map(|(operation, left, right)| Predicate::Compare(operation, left, right)),
        1 => Just(Predicate::Flag),
    ];
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
            Self::Piecewise(condition, a, b) => Expression::piecewise(
                [(condition.build(variables), a.build(variables))],
                b.build(variables),
            )
            .expect("a piecewise"),
        }
    }

    fn evaluate(&self, point: Point) -> i128 {
        match self {
            Self::X => point.x,
            Self::Y => point.y,
            Self::Literal(value) => i128::from(*value),
            Self::Add(a, b) => a.evaluate(point) + b.evaluate(point),
            Self::Subtract(a, b) => a.evaluate(point) - b.evaluate(point),
            Self::Multiply(a, b) => a.evaluate(point) * b.evaluate(point),
            Self::Negate(a) => -a.evaluate(point),
            Self::FloorDivide(a, k) => a.evaluate(point).div_euclid(i128::from(*k)),
            Self::FloorMod(a, k) => a.evaluate(point).rem_euclid(i128::from(*k)),
            Self::Square(a) => a.evaluate(point).pow(2),
            Self::Piecewise(condition, a, b) => {
                if condition.evaluate(point) {
                    a.evaluate(point)
                } else {
                    b.evaluate(point)
                }
            }
        }
    }
}

impl Predicate {
    fn build(&self, variables: &Variables) -> Expression {
        match self {
            Self::Compare(operation, left, right) => {
                Expression::new_binary(*operation, left.build(variables), right.build(variables))
            }
            Self::Flag => Expression::from(variables.p.clone()),
            Self::And(a, b) => a.build(variables).and(b.build(variables)),
            Self::Or(a, b) => a.build(variables).or(b.build(variables)),
            Self::Not(a) => !a.build(variables),
        }
    }

    fn evaluate(&self, point: Point) -> bool {
        match self {
            Self::Compare(operation, left, right) => {
                let ordering = left.evaluate(point).cmp(&right.evaluate(point));
                compare(*operation, ordering)
            }
            Self::Flag => point.p,
            Self::And(a, b) => a.evaluate(point) && b.evaluate(point),
            Self::Or(a, b) => a.evaluate(point) || b.evaluate(point),
            Self::Not(a) => !a.evaluate(point),
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

/// Return whether some flag and `y` of the domain make `holds` true at `x`.
fn exists_y_and_flag(x: i128, holds: impl Fn(Point) -> bool) -> bool {
    [false, true]
        .into_iter()
        .any(|p| domain().any(|y| holds(Point { x, y, p })))
}

/// Return every point of the domain.
fn points() -> impl Iterator<Item = Point> {
    domain().flat_map(|x| {
        domain().flat_map(move |y| [false, true].into_iter().map(move |p| Point { x, y, p }))
    })
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
        let symbol_types = variables.symbol_types();

        let script = SmtScript::lower(&expression, &symbol_types, &NoRegisteredSorts)
            .expect("a generated predicate lowers");
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

/// The given-up answers of each solver-backed property so far.
static SATISFIABILITY_GAVE_UP: AtomicUsize = AtomicUsize::new(0);
static IMPLICATION_GAVE_UP: AtomicUsize = AtomicUsize::new(0);
static VALIDITY_GAVE_UP: AtomicUsize = AtomicUsize::new(0);
static GROUND_GAVE_UP: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "z3")]
static AGREEMENT_GAVE_UP: AtomicUsize = AtomicUsize::new(0);

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
        let expected = points().any(|point| generated.evaluate(point));

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::Satisfiability(&expression),
                &QueryContext::new(&variables.symbol_types()).with_limits(limits()),
            )
            .expect("answered");

        check_not_vacuous(&answer, &SATISFIABILITY_GAVE_UP)?;
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
        let expected =
            points().all(|point| !antecedent.evaluate(point) || consequent.evaluate(point));

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::Implication { antecedent: &bounded, consequent: &consequent.build(&variables) },
                &QueryContext::new(&variables.symbol_types()).with_limits(limits()),
            )
            .expect("answered");

        check_not_vacuous(&answer, &IMPLICATION_GAVE_UP)?;
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
        let considered = HashSet::from([variables.y.clone(), variables.p.clone()]);
        let expected = domain().all(|x| exists_y_and_flag(x, |point| generated.evaluate(point)));

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::UniversalValidity { considered: &considered, expression: &expression },
                &QueryContext::new(&variables.symbol_types()).with_limits(limits()),
            )
            .expect("answered");

        check_not_vacuous(&answer, &VALIDITY_GAVE_UP)?;
        prop_assert!(answer.decided().is_none_or(|decided| decided == expected), "{:?} against {}", answer, expected);
    }
}

#[cfg(feature = "z3")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn z3_agrees_with_the_process_backend(
        antecedent in predicate(),
        consequent in predicate(),
    ) {
        let Some(process) = required(configured_process_backend(), "z3_agrees_with_the_process_backend") else {
            return Ok(());
        };
        let variables = Variables::new();
        let bounded = Expression::all([
            bound(&variables.x),
            bound(&variables.y),
            antecedent.build(&variables),
        ]);
        let consequent = consequent.build(&variables);
        let question = Question::Implication { antecedent: &bounded, consequent: &consequent };
        let symbol_types = variables.symbol_types();
        let context = QueryContext::new(&symbol_types).with_limits(limits());

        let from_z3 = Solver::new()
            .with_smt_solver(fhy_core::solver::Z3Solver::new())
            .ask(&question, &context)
            .expect("answered");
        let from_process = Solver::new()
            .with_shared_smt_solver(process)
            .ask(&question, &context)
            .expect("answered");

        check_not_vacuous(&from_z3, &AGREEMENT_GAVE_UP)?;
        check_not_vacuous(&from_process, &AGREEMENT_GAVE_UP)?;
        if let (Some(z3), Some(process)) = (from_z3.decided(), from_process.decided()) {
            prop_assert_eq!(z3, process);
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed int/real equalities against the evaluator
// ---------------------------------------------------------------------------

/// A literal an integer term is compared with: an integer, the float of an
/// integer, or a float halfway between two integers.
fn mixed_literal() -> impl Strategy<Value = Expression> {
    prop_oneof![
        (-4_i64..=4).prop_map(Expression::literal),
        (-4_i32..=4).prop_map(|value| Expression::literal(f64::from(value))),
        (-4_i32..=4).prop_map(|value| Expression::literal(f64::from(value) + 0.5)),
    ]
}

/// Return whether `comparison` holds at `point`, by the crate's evaluator.
fn evaluate_at(comparison: &Expression, variables: &Variables, point: Point) -> bool {
    let registry = FunctionRegistry::new();
    let bindings = HashMap::from([
        (
            variables.x.clone(),
            Scalar::Int(i64::try_from(point.x).expect("in the domain")),
        ),
        (
            variables.y.clone(),
            Scalar::Int(i64::try_from(point.y).expect("in the domain")),
        ),
        (variables.p.clone(), Scalar::Bool(point.p)),
    ]);
    match Evaluator::new(&registry).evaluate(comparison, &bindings) {
        Ok(Scalar::Bool(value)) => value,
        other => panic!("{comparison} evaluates to {other:?}"),
    }
}

static MIXED_GAVE_UP: AtomicUsize = AtomicUsize::new(0);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn a_mixed_equality_is_answered_as_the_evaluator_decides_it(
        term in int_term(),
        literal in mixed_literal(),
        is_equal in any::<bool>(),
        is_literal_on_the_left in any::<bool>(),
    ) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let variables = Variables::new();
        let operation = if is_equal { BinaryOperation::Equal } else { BinaryOperation::NotEqual };
        let comparison = if is_literal_on_the_left {
            Expression::new_binary(operation, literal, term.build(&variables))
        } else {
            Expression::new_binary(operation, term.build(&variables), literal)
        };
        let expression = Expression::all([bound(&variables.x), bound(&variables.y), comparison.clone()]);
        let expected = points().any(|point| evaluate_at(&comparison, &variables, point));

        let answer = Solver::new()
            .with_shared_smt_solver(backend)
            .ask(
                &Question::Satisfiability(&expression),
                &QueryContext::new(&variables.symbol_types()).with_limits(limits()),
            )
            .expect("answered");

        check_not_vacuous(&answer, &MIXED_GAVE_UP)?;
        prop_assert!(answer.decided().is_none_or(|decided| decided == expected), "{:?} for {}", answer, comparison);
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

        if let SatResult::Unknown { reason } = &result {
            let count = GROUND_GAVE_UP.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            prop_assert!(count <= GAVE_UP_BUDGET, "the backend gave up {} times, the last with {:?}", count, reason);
        } else {
            prop_assert_eq!(&result, &if expected { SatResult::Sat } else { SatResult::Unsat }, "for {}", script);
        }
    }
}

#[test]
fn exact_evaluation_floors_toward_negative_infinity() {
    let seven = Expression::literal(7);

    assert_eq!(
        evaluate_exactly(&seven.floor_divide(-2)).numerator.to_i64(),
        Some(-4)
    );
    assert_eq!(
        evaluate_exactly(&seven.floor_mod(-2)).numerator.to_i64(),
        Some(-1)
    );
}
