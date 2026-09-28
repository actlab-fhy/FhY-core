//! Tests for `ConstraintSystem`: the canonical order, the conjunction and
//! its events, the expression, the three questions and the order of their
//! checks, the decided set members, custom members, and equivalence.
//!
//! The cases are ported from `test_constraint_system.py`; a recording fake
//! backend stands in for z3.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, ConstraintError, ConstraintSystem,
    EquationConstraint, Outcome, Polarity, SetConstraint, UnusableBindingReason, Value,
};
use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{CheckLimits, QueryKind, SatResult, Solver};
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};

use crate::support::constraint::{ConstraintKey, Failing, FailingHook, TestValueError};
use crate::support::constraint::{
    RecordedEvent, RecordingObserver, TestCustom, bind, int, int_set, member_set, text,
};
use crate::support::lambda::Alpha;
use crate::support::solver::{RecordingSmtSolver, build_symbol_types, quoted_symbol};

fn equation(expression: Expression) -> Constraint {
    Constraint::from(EquationConstraint::new(expression))
}

fn in_set(variable: &Identifier, members: &[i64]) -> Constraint {
    Constraint::from(SetConstraint::new(
        variable.clone(),
        int_set(members.iter().copied()),
        Polarity::In,
    ))
}

fn not_in_set(variable: &Identifier, members: &[i64]) -> Constraint {
    Constraint::from(SetConstraint::new(
        variable.clone(),
        int_set(members.iter().copied()),
        Polarity::NotIn,
    ))
}

/// A system's questions asked of a backend answering one result, with the
/// events and checks they caused.
struct Harness {
    backend: Arc<RecordingSmtSolver>,
    solver: Solver,
    observer: RecordingObserver,
}

impl Harness {
    fn answering(answer: SatResult) -> Self {
        let backend = RecordingSmtSolver::answering(answer);
        let solver = backend.solver();
        Self {
            backend,
            solver,
            observer: RecordingObserver::default(),
        }
    }

    fn context(&self) -> ConstraintContext<'_> {
        ConstraintContext::new(&self.solver).with_observer(&self.observer)
    }

    fn scripts(&self) -> Vec<String> {
        self.backend
            .checks()
            .into_iter()
            .map(|(script, _)| script)
            .collect()
    }
}

fn assertion_lines(script: &str) -> Vec<String> {
    script
        .lines()
        .filter(|line| line.starts_with("(assert"))
        .map(str::to_owned)
        .collect()
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn members_are_sorted_by_key_keeping_duplicates() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let first = not_in_set(&y, &[1]);
    let second = in_set(&x, &[2]);
    let third = equation(Expression::from(x).less(1));

    let system = ConstraintSystem::new([first, second.clone(), third, second])
        .expect("every member has a key");

    let keys: Vec<String> = system
        .constraints()
        .iter()
        .map(ConstraintKey::key)
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
    assert_eq!(system.constraints().len(), 4);
}

#[test]
fn a_custom_member_sorts_among_the_built_in_kinds_by_its_key() {
    let x = Identifier::new("x");
    let log = Arc::new(Mutex::new(Vec::new()));
    let custom = TestCustom::build("probe", Expression::literal(true), Outcome::Satisfied, &log);

    let system = ConstraintSystem::new([
        in_set(&x, &[1]),
        custom,
        equation(Expression::literal(true)),
    ])
    .expect("every member has a key");

    let keys: Vec<String> = system
        .constraints()
        .iter()
        .map(ConstraintKey::key)
        .collect();
    assert_eq!(keys[0], "custom|probe");
    assert!(keys[1].starts_with("equation|"));
    assert!(keys[2].starts_with("in_set|"));
}

// ---------------------------------------------------------------------------
// The conjunction
// ---------------------------------------------------------------------------

#[test]
fn evaluation_stops_at_the_first_violated_member_in_order() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let system = ConstraintSystem::new([
        TestCustom::build(
            "a_first",
            Expression::literal(true),
            Outcome::Violated,
            &log,
        ),
        TestCustom::build(
            "b_second",
            Expression::literal(true),
            Outcome::Satisfied,
            &log,
        ),
    ])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.evaluate(&Bindings::new(), &harness.context());

    assert_eq!(outcome.expect("decided"), Outcome::Violated);
    assert_eq!(*log.lock().expect("the log"), ["a_first:none"]);
}

#[test]
fn an_undecided_member_makes_the_system_undecided_unless_a_later_one_is_violated() {
    let x = Identifier::new("x");
    let log = Arc::new(Mutex::new(Vec::new()));
    let undecided = ConstraintSystem::new([
        in_set(&x, &[1]),
        TestCustom::build(
            "z_last",
            Expression::literal(true),
            Outcome::Satisfied,
            &log,
        ),
    ])
    .expect("every member has a key");
    let violated = ConstraintSystem::new([
        in_set(&x, &[1]),
        TestCustom::build("z_last", Expression::literal(true), Outcome::Violated, &log),
    ])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let first = undecided.evaluate(&Bindings::new(), &harness.context());
    let second = violated.evaluate(&Bindings::new(), &harness.context());

    assert_eq!(first.expect("decided"), Outcome::Undecided);
    assert_eq!(second.expect("decided"), Outcome::Violated);
    let set_index = undecided
        .constraints()
        .iter()
        .position(|member| matches!(member, Constraint::Set(_)))
        .expect("a set member");
    assert_eq!(
        harness.observer.events()[..2],
        [
            RecordedEvent::InMember(set_index, Box::new(RecordedEvent::Unbound(x))),
            RecordedEvent::UndecidedMember(set_index),
        ]
    );
}

#[test]
fn every_member_satisfied_satisfies_the_system() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([in_set(&x, &[1, 2]), not_in_set(&x, &[3])])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.evaluate(&bind([(x, Binding::Value(int(2)))]), &harness.context());

    assert_eq!(outcome.expect("decided"), Outcome::Satisfied);
    assert!(harness.observer.events().is_empty());
}

#[test]
fn a_member_error_is_the_system_error() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([in_set(&x, &[1])]).expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);
    let unusable = Value::Decimal("1".parse().expect("a decimal"));

    let outcome = system.evaluate(&bind([(x, Binding::Value(unusable))]), &harness.context());

    assert!(matches!(
        outcome,
        Err(ConstraintError::UnusableBinding {
            reason: UnusableBindingReason::NotMemberShaped,
            ..
        })
    ));
}

#[test]
fn a_custom_member_receives_the_bindings_with_their_source() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let system = ConstraintSystem::new([TestCustom::build(
        "probe",
        Expression::literal(true),
        Outcome::Satisfied,
        &log,
    )])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);
    let bindings = Bindings::new().with_source(Arc::new("the caller's".to_owned()));

    system
        .evaluate(&bindings, &harness.context())
        .expect("decided");

    assert_eq!(*log.lock().expect("the log"), ["probe:the caller's"]);
}

// ---------------------------------------------------------------------------
// The expression
// ---------------------------------------------------------------------------

#[test]
fn the_empty_system_is_true_one_member_is_itself_and_several_their_conjunction() {
    let x = Identifier::new("x");
    let bound = Expression::from(x.clone()).less(1);
    let other = Expression::from(x).greater(0);

    let empty = ConstraintSystem::new([])
        .expect("every member has a key")
        .to_expression()
        .expect("converts");
    let one = ConstraintSystem::new([equation(bound.clone())])
        .expect("every member has a key")
        .to_expression()
        .expect("converts");
    let two = ConstraintSystem::new([equation(bound.clone()), equation(other.clone())])
        .expect("every member has a key")
        .to_expression()
        .expect("converts");

    assert_eq!(empty, Expression::literal(true));
    assert!(Expression::ptr_eq(&one, &bound));
    assert!(
        two == Expression::all([bound.clone(), other.clone()])
            || two == Expression::all([other, bound])
    );
}

// ---------------------------------------------------------------------------
// Satisfiability
// ---------------------------------------------------------------------------

#[test]
fn the_empty_system_is_satisfiable_without_asking_the_solver() {
    let harness = Harness::answering(SatResult::Unsat);

    let outcome = ConstraintSystem::new([])
        .expect("every member has a key")
        .check_satisfiability(&HashMap::new(), CheckLimits::new(), &harness.context());

    assert_eq!(outcome.expect("decided"), Outcome::Satisfied);
    assert!(harness.scripts().is_empty());
}

#[test]
fn satisfiability_asks_about_the_conjunction_and_reads_each_answer() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([
        equation(Expression::from(x.clone()).greater(0)),
        in_set(&x, &[1, 2]),
    ])
    .expect("every member has a key");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);

    for (answer, expected) in [
        (SatResult::Sat, Outcome::Satisfied),
        (SatResult::Unsat, Outcome::Violated),
    ] {
        let harness = Harness::answering(answer);
        let outcome = system
            .check_satisfiability(&symbol_types, CheckLimits::new(), &harness.context())
            .expect("decided");
        assert_eq!(outcome, expected);
        let scripts = harness.scripts();
        assert_eq!(scripts.len(), 1);
        let assertion = &assertion_lines(&scripts[0])[0];
        assert!(assertion.starts_with("(assert (and"), "{assertion}");
        assert!(assertion.contains(&quoted_symbol(&x)), "{assertion}");
    }
}

#[test]
fn an_unknown_answer_is_undecided_and_reported() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([in_set(&x, &[1])]).expect("every member has a key");
    let harness = Harness::answering(SatResult::Unknown {
        reason: "timeout".to_owned(),
    });

    let outcome = system.check_satisfiability(
        &build_symbol_types(&[(&x, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("undecided"), Outcome::Undecided);
    assert_eq!(
        harness.observer.events(),
        [RecordedEvent::GaveUp(
            QueryKind::Satisfiability,
            "timeout".to_owned()
        )]
    );
}

#[test]
fn a_hazard_is_undecided_and_reported_without_asking_the_backend() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let system = ConstraintSystem::new([equation(
        (&Expression::from(x.clone()) / Expression::from(y.clone())).greater(0),
    )])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability(
        &build_symbol_types(&[(&x, SymbolType::Real), (&y, SymbolType::Real)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("undecided"), Outcome::Undecided);
    assert_eq!(
        harness.observer.events(),
        [RecordedEvent::Refused(QueryKind::Satisfiability)]
    );
    assert!(harness.scripts().is_empty());
}

// ---------------------------------------------------------------------------
// Mixed int/real equalities
// ---------------------------------------------------------------------------

/// Return the set constraint of `variable` against the float members
/// `members`.
fn float_set(variable: &Identifier, members: &[f64], polarity: Polarity) -> Constraint {
    Constraint::from(SetConstraint::new(
        variable.clone(),
        member_set(members.iter().map(|&member| Value::Float(member))),
        polarity,
    ))
}

#[test]
fn an_equation_mixing_int_and_float_is_asked_of_the_backend() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([equation(Expression::from(x.clone()).equals(1.0))])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability(
        &build_symbol_types(&[(&x, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("answered"), Outcome::Satisfied);
    assert_eq!(harness.scripts().len(), 1);
}

#[test]
fn a_set_member_of_the_other_numeric_kind_is_refused_in_every_question() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let int_types = build_symbol_types(&[(&x, SymbolType::Int), (&y, SymbolType::Int)]);
    let excluded = ConstraintSystem::new([float_set(&x, &[2.0], Polarity::NotIn)])
        .expect("every member has a key");
    let included = ConstraintSystem::new([float_set(&x, &[3.0], Polarity::In)])
        .expect("every member has a key");
    let unequal = ConstraintSystem::new([equation(Expression::from(x.clone()).not_equals(2))])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);
    let context = harness.context();

    let satisfiability = excluded.check_satisfiability(&int_types, CheckLimits::new(), &context);
    let implication =
        excluded.check_implication(&unequal, &int_types, CheckLimits::new(), &context);
    let residual = included.check_satisfiability_with_bindings(
        &bind([(x, Expression::from(y) + 1)]),
        &int_types,
        CheckLimits::new(),
        &context,
    );

    assert_eq!(satisfiability.expect("undecided"), Outcome::Undecided);
    assert_eq!(implication.expect("undecided"), Outcome::Undecided);
    assert_eq!(residual.expect("undecided"), Outcome::Undecided);
    assert_eq!(
        harness.observer.events(),
        [
            RecordedEvent::Refused(QueryKind::Satisfiability),
            RecordedEvent::Refused(QueryKind::Implication),
            RecordedEvent::Refused(QueryKind::Satisfiability),
        ]
    );
    assert!(harness.scripts().is_empty());
}

#[test]
fn a_set_member_of_the_same_numeric_kind_is_asked_of_the_backend() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([float_set(&x, &[2.0], Polarity::NotIn)])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability(
        &build_symbol_types(&[(&x, SymbolType::Real)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("answered"), Outcome::Satisfied);
    assert_eq!(harness.scripts().len(), 1);
}

#[test]
fn a_conversion_error_comes_before_missing_symbol_types() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([Constraint::from(SetConstraint::new(
        x,
        member_set([text("a")]),
        Polarity::In,
    ))])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome =
        system.check_satisfiability(&HashMap::new(), CheckLimits::new(), &harness.context());

    assert!(
        matches!(outcome, Err(ConstraintError::UnliftableMember(_))),
        "{outcome:?}"
    );
}

#[test]
fn missing_symbol_types_come_before_ill_typedness_and_skip_native_constants() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());
    let system = ConstraintSystem::new([
        equation(&Expression::from(y.clone()) + 1),
        equation(Expression::from(x.clone()).less(pi)),
    ])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome =
        system.check_satisfiability(&HashMap::new(), CheckLimits::new(), &harness.context());

    let error = outcome.expect_err("missing symbol types");
    assert!(
        matches!(&error, ConstraintError::MissingSymbolTypes(identifiers) if *identifiers == vec![x.clone(), y.clone()]),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!("symbol_types is missing an entry for free identifier(s): {x:?}, {y:?}")
    );
}

#[test]
fn an_ill_typed_member_is_refused_naming_its_own_expression() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([equation(&Expression::from(x.clone()) + 1)])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability(
        &build_symbol_types(&[(&x, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert!(
        matches!(outcome, Err(ConstraintError::IllTyped(_))),
        "{outcome:?}"
    );
    assert!(harness.scripts().is_empty());
}

#[test]
fn the_limits_reach_the_backend() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([in_set(&x, &[1])]).expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);
    let limits = CheckLimits::new().with_timeout(Duration::from_millis(2500));

    system
        .check_satisfiability(
            &build_symbol_types(&[(&x, SymbolType::Int)]),
            limits,
            &harness.context(),
        )
        .expect("decided");

    assert_eq!(harness.backend.checks()[0].1, limits);
}

// ---------------------------------------------------------------------------
// Satisfiability with bindings
// ---------------------------------------------------------------------------

#[test]
fn a_violated_decided_set_member_answers_without_the_solver() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let system = ConstraintSystem::new([
        in_set(&x, &[1, 2]),
        equation(Expression::from(y.clone()).greater(0)),
    ])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability_with_bindings(
        &bind([(x, Binding::Value(int(5)))]),
        &build_symbol_types(&[(&y, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("decided"), Outcome::Violated);
    assert!(harness.scripts().is_empty());
}

#[test]
fn decided_set_members_alone_answer_their_fold() {
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([in_set(&x, &[1, 2]), not_in_set(&x, &[3])])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Unsat);

    let outcome = system.check_satisfiability_with_bindings(
        &bind([(x, Binding::Value(int(2)))]),
        &HashMap::new(),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("decided"), Outcome::Satisfied);
    assert!(harness.scripts().is_empty());
}

#[test]
fn the_residual_is_substituted_and_needs_symbol_types_for_what_it_leaves_free() {
    let (x, y, z) = (
        Identifier::new("x"),
        Identifier::new("y"),
        Identifier::new("z"),
    );
    let system = ConstraintSystem::new([equation(
        Expression::from(x.clone()).less(Expression::from(y.clone())),
    )])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);
    let bindings = bind([
        (x.clone(), Binding::Value(int(1))),
        (
            y.clone(),
            Binding::Expression(&Expression::from(z.clone()) + 1),
        ),
    ]);

    let missing = system.check_satisfiability_with_bindings(
        &bindings,
        &build_symbol_types(&[(&x, SymbolType::Int), (&y, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );
    let decided = system.check_satisfiability_with_bindings(
        &bindings,
        &build_symbol_types(&[(&z, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert!(
        matches!(&missing, Err(ConstraintError::MissingSymbolTypes(identifiers)) if *identifiers == vec![z.clone()]),
        "{missing:?}"
    );
    assert_eq!(decided.expect("decided"), Outcome::Satisfied);
    let script = &harness.scripts()[0];
    assert!(!script.contains(&quoted_symbol(&x)), "{script}");
    assert!(script.contains(&quoted_symbol(&z)), "{script}");
}

#[test]
fn every_binding_must_be_an_expression_or_a_literal() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let system = ConstraintSystem::new([in_set(&x, &[1])]).expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability_with_bindings(
        &bind([(y.clone(), Binding::Value(Value::Tuple(vec![])))]),
        &HashMap::new(),
        CheckLimits::new(),
        &harness.context(),
    );

    assert!(matches!(
        outcome,
        Err(ConstraintError::UnusableBinding { identifier, reason: UnusableBindingReason::NotALiteral }) if identifier == y
    ));
}

#[test]
fn a_bound_native_constant_the_system_refers_to_is_undecided_and_reported() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let x = Identifier::new("x");
    let system = ConstraintSystem::new([equation(
        Expression::from(x.clone()).less(Expression::from(pi.clone())),
    )])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Sat);

    let outcome = system.check_satisfiability_with_bindings(
        &bind([(pi.clone(), Binding::Value(int(3)))]),
        &build_symbol_types(&[(&x, SymbolType::Real)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("undecided"), Outcome::Undecided);
    assert_eq!(
        harness.observer.events(),
        [RecordedEvent::BoundNativeConstants(vec![pi])]
    );
    assert!(harness.scripts().is_empty());
}

#[test]
fn satisfied_leaves_and_an_undecided_residual_are_undecided() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let system = ConstraintSystem::new([
        in_set(&x, &[1]),
        equation(Expression::from(y.clone()).greater(0)),
    ])
    .expect("every member has a key");
    let harness = Harness::answering(SatResult::Unknown {
        reason: "incomplete".to_owned(),
    });

    let outcome = system.check_satisfiability_with_bindings(
        &bind([(x, Binding::Value(int(1)))]),
        &build_symbol_types(&[(&y, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert_eq!(outcome.expect("undecided"), Outcome::Undecided);
}

#[test]
fn an_empty_system_with_bindings_is_satisfiable_without_reading_them() {
    let x = Identifier::new("x");
    let harness = Harness::answering(SatResult::Unsat);

    let outcome = ConstraintSystem::new([])
        .expect("every member has a key")
        .check_satisfiability_with_bindings(
            &bind([(x, Binding::Value(Value::Tuple(vec![])))]),
            &HashMap::new(),
            CheckLimits::new(),
            &harness.context(),
        );

    assert_eq!(outcome.expect("decided"), Outcome::Satisfied);
}

// ---------------------------------------------------------------------------
// Implication
// ---------------------------------------------------------------------------

#[test]
fn implication_asks_about_a_counterexample_and_reads_each_answer() {
    let x = Identifier::new("x");
    let antecedent =
        ConstraintSystem::new([equation(Expression::from(x.clone()).greater_equal(1))])
            .expect("every member has a key");
    let consequent =
        ConstraintSystem::new([equation(Expression::from(x.clone()).greater_equal(0))])
            .expect("every member has a key");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);

    for (answer, expected) in [
        (SatResult::Unsat, Outcome::Satisfied),
        (SatResult::Sat, Outcome::Violated),
    ] {
        let harness = Harness::answering(answer);
        let outcome = antecedent
            .check_implication(
                &consequent,
                &symbol_types,
                CheckLimits::new(),
                &harness.context(),
            )
            .expect("decided");
        assert_eq!(outcome, expected);
        let assertion = &assertion_lines(&harness.scripts()[0])[0];
        assert!(assertion.contains("(not"), "{assertion}");
    }
}

#[test]
fn implication_screens_every_member_of_this_system_before_the_other() {
    let x = Identifier::new("x");
    let antecedent = ConstraintSystem::new([equation(&Expression::from(x.clone()) + 1)])
        .expect("every member has a key");
    let consequent = ConstraintSystem::new([equation(&Expression::from(x.clone()) + 2)])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Unsat);

    let outcome = antecedent.check_implication(
        &consequent,
        &build_symbol_types(&[(&x, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    let Err(ConstraintError::IllTyped(error)) = outcome else {
        panic!("expected an ill-typed member, got {outcome:?}");
    };
    assert_eq!(error.operand(), &(&Expression::from(x) + 1));
}

#[test]
fn implication_needs_symbol_types_for_both_sides() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let antecedent = ConstraintSystem::new([equation(Expression::from(x.clone()).greater(0))])
        .expect("every member has a key");
    let consequent = ConstraintSystem::new([equation(Expression::from(y.clone()).greater(0))])
        .expect("every member has a key");
    let harness = Harness::answering(SatResult::Unsat);

    let outcome = antecedent.check_implication(
        &consequent,
        &build_symbol_types(&[(&x, SymbolType::Int)]),
        CheckLimits::new(),
        &harness.context(),
    );

    assert!(
        matches!(&outcome, Err(ConstraintError::MissingSymbolTypes(identifiers)) if *identifiers == vec![y]),
        "{outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// Equivalence
// ---------------------------------------------------------------------------

#[test]
fn systems_are_equivalent_member_by_member_in_canonical_order() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let left = ConstraintSystem::new([
        in_set(&x, &[1]),
        equation(Expression::from(x.clone()).less(3)),
    ])
    .expect("every member has a key");
    let right = ConstraintSystem::new([
        equation(Expression::from(x.clone()).less(3)),
        in_set(&x, &[1]),
    ])
    .expect("every member has a key");
    let renamed = ConstraintSystem::new([
        in_set(&y, &[1]),
        equation(Expression::from(y.clone()).less(3)),
    ])
    .expect("every member has a key");
    let shorter = ConstraintSystem::new([in_set(&x, &[1])]).expect("every member has a key");
    let renaming = AlphaRenaming::new(HashMap::from([(x, y)])).expect("injective");

    assert!(left.is_structurally_equivalent(&right));
    assert!(!left.is_structurally_equivalent(&shorter));
    assert!(!left.is_structurally_equivalent(&renamed));
    assert!(left.alpha_equivalent_under(&renamed, &renaming));
    assert!(!left.alpha_equivalent(&renamed));
}

#[test]
fn a_failing_custom_key_fails_system_construction() {
    let x = Identifier::new("x");

    let error = ConstraintSystem::new([
        in_set(&x, &[1]),
        Failing(FailingHook::Key).into_constraint(),
    ])
    .expect_err("the key fails");

    let ConstraintError::Custom(source) = &error else {
        panic!("a custom error, got {error:?}");
    };
    assert!(source.downcast_ref::<TestValueError>().is_some());
    assert_eq!(source.to_string(), "the key failed");
}

#[test]
fn a_failing_custom_scope_is_an_error_not_a_closed_constraint() {
    let constraint = Failing(FailingHook::Scope).into_constraint();

    let error = constraint.free_identifiers().expect_err("the scope fails");

    let ConstraintError::Custom(source) = &error else {
        panic!("a custom error, got {error:?}");
    };
    assert_eq!(source.to_string(), "the scope failed");
}

// =============================================================================
// Colliding keys
// =============================================================================

/// Return `x in {value}` over one opaque value whose ordering key is the
/// same for every payload.
fn colliding_member(variable: &Identifier, payload: i64) -> Constraint {
    Constraint::from(SetConstraint::new(
        variable.clone(),
        member_set([crate::support::constraint::TestOpaque::colliding(payload).into_value()]),
        Polarity::In,
    ))
}

fn hash_of(system: &ConstraintSystem) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    system.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn systems_with_colliding_opaque_keys_are_equivalent_in_either_order() {
    let x = Identifier::new("x");
    let (one, two) = (colliding_member(&x, 1), colliding_member(&x, 2));
    assert_eq!(one.key(), two.key());
    assert_ne!(one, two);

    let forward = ConstraintSystem::new([one.clone(), two.clone()]).expect("a system");
    let backward = ConstraintSystem::new([two, one.clone()]).expect("a system");
    let other = ConstraintSystem::new([one.clone(), one]).expect("a system");

    assert!(forward.is_structurally_equivalent(&backward));
    assert_eq!(forward, backward);
    assert_eq!(hash_of(&forward), hash_of(&backward));
    assert!(
        forward
            .is_alpha_equivalent_under(&backward, &AlphaRenaming::default())
            .expect("no custom member fails")
    );
    assert_ne!(forward, other);
}

#[test]
fn equivalent_members_of_a_tie_run_are_grouped_by_first_appearance() {
    let x = Identifier::new("x");
    let (one, two) = (colliding_member(&x, 1), colliding_member(&x, 2));

    let system = ConstraintSystem::new([
        two.clone(),
        one.clone(),
        two.clone(),
        one.clone(),
        two.clone(),
    ])
    .expect("a system");

    assert_eq!(
        system.constraints(),
        [two.clone(), two.clone(), two, one.clone(), one]
    );
}

#[test]
fn params_over_colliding_members_are_equivalent_in_either_order() {
    use fhy_core::param::{Param, ParamContext, ParamDomain, RealDomain};
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let x = Identifier::new("x");
    let (one, two) = (colliding_member(&x, 1), colliding_member(&x, 2));
    let build = |constraints: [Constraint; 2]| {
        Param::new(
            ParamDomain::from(RealDomain),
            x.clone(),
            constraints,
            &context,
        )
        .expect("a param")
    };

    let forward = build([one.clone(), two.clone()]);
    let backward = build([two, one]);

    assert!(forward.is_structurally_equivalent(&backward));
    assert_eq!(forward, backward);
}

// =============================================================================
// Ord for constraints
// =============================================================================

#[test]
fn constraints_order_by_their_keys_in_a_b_tree_set() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let constraints = [
        not_in_set(&x, &[3]),
        equation(Expression::from(y.clone()).less(2)),
        in_set(&x, &[1, 2]),
        in_set(&x, &[2, 1]),
        equation(Expression::from(y).less(2)),
    ];

    let set: std::collections::BTreeSet<Constraint> = constraints.iter().cloned().collect();

    assert_eq!(set.len(), 3, "equivalent constraints are one element");
    let keys: Vec<String> = set.iter().map(ConstraintKey::key).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
    let system = ConstraintSystem::new(constraints).expect("no key fails");
    let mut members = system.constraints().to_vec();
    members.dedup();
    assert_eq!(members, set.into_iter().collect::<Vec<_>>());
}

#[test]
fn a_custom_constraint_orders_by_its_key_and_one_whose_key_fails_orders_last() {
    let x = Identifier::new("x");
    let log = Arc::new(Mutex::new(Vec::new()));
    let custom = TestCustom::build("probe", Expression::literal(true), Outcome::Satisfied, &log);
    let failing = Failing(FailingHook::Key).into_constraint();
    let built_in = in_set(&x, &[1]);

    assert_eq!(custom.cmp(&built_in), custom.key().cmp(&built_in.key()));
    assert_eq!(failing.cmp(&built_in), std::cmp::Ordering::Greater);
    assert_eq!(failing.cmp(&custom), std::cmp::Ordering::Greater);
}
