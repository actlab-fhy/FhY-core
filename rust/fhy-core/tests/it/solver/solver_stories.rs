//! Stories for the facade, [`Solver`]: capabilities, the order of checks,
//! the script of each question and the answer each `check-sat` result
//! gives, limits, backend failures, and simplification.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
use fhy_core::expression::{Expression, FunctionName, FunctionSort, NoRegisteredSorts, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    Answer, CheckLimits, Hazard, LoweringError, QueryContext, QueryKind, Question, SatResult,
    SimplifyContext, SimplifyLimits, SolveError, Solver, UnknownReason,
};
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};
use crate::support::solver::{
    FakeBackendError, RecordingSimplifier, RecordingSmtSolver, build_symbol_types, quoted_symbol,
};

/// Ask `question` of a solver whose backend answers `answer`, returning
/// the answer and the one script checked.
fn ask_answering(
    answer: SatResult,
    question: &Question<'_>,
    symbol_types: &HashMap<Identifier, SymbolType>,
) -> (Answer, String) {
    let backend = RecordingSmtSolver::answering(answer);
    let answer = backend
        .solver()
        .ask(question, &QueryContext::new(symbol_types))
        .expect("the question is answered");
    (answer, backend.only_script())
}

/// Return the error asking `question` of a solver answering `sat`.
fn ask_error(
    question: &Question<'_>,
    symbol_types: &HashMap<Identifier, SymbolType>,
) -> SolveError {
    RecordingSmtSolver::answering(SatResult::Sat)
        .solver()
        .ask(question, &QueryContext::new(symbol_types))
        .expect_err("the question is refused")
}

/// Return the assertion lines of `script`.
fn assertions(script: &str) -> Vec<&str> {
    script
        .lines()
        .filter(|line| line.starts_with("(assert"))
        .collect()
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

#[test]
fn capabilities_follow_the_backends_a_solver_holds() {
    let kinds = [
        QueryKind::Simplification,
        QueryKind::Satisfiability,
        QueryKind::Implication,
        QueryKind::UniversalValidity,
    ];
    let answers = |solver: &Solver| kinds.map(|kind| solver.can_answer(kind));

    assert_eq!(answers(&Solver::new()), [false; 4]);
    assert_eq!(
        answers(&RecordingSmtSolver::answering(SatResult::Sat).solver()),
        [false, true, true, true]
    );
    assert_eq!(
        answers(&RecordingSimplifier::identity().solver()),
        [true, false, false, false]
    );
}

#[test]
fn missing_backend_is_reported_before_every_other_check() {
    let (_, reference) = build_identifier("x");
    let ill_typed_and_undeclared = Expression::all([reference, build_literal(2)]);

    let logical = Solver::new().ask(
        &Question::Satisfiability(&ill_typed_and_undeclared),
        &QueryContext::new(&HashMap::new()),
    );
    let simplification = RecordingSmtSolver::answering(SatResult::Sat)
        .solver()
        .simplify(
            &ill_typed_and_undeclared,
            &HashMap::new(),
            &SimplifyContext::new(&NoRegisteredSorts),
        );

    assert!(matches!(
        logical,
        Err(SolveError::NoCapableBackend(QueryKind::Satisfiability))
    ));
    assert!(matches!(
        simplification,
        Err(SolveError::NoCapableBackend(QueryKind::Simplification))
    ));
}

#[test]
fn solver_clone_shares_its_backends() {
    let solver = RecordingSmtSolver::answering(SatResult::Sat).solver();
    let clone = solver.clone();

    assert!(Arc::ptr_eq(
        solver.smt_solver().expect("a backend"),
        clone.smt_solver().expect("a backend")
    ));
    assert!(clone.simplifier().is_none());
}

// ---------------------------------------------------------------------------
// The order of checks
// ---------------------------------------------------------------------------

#[test]
fn missing_symbol_types_are_reported_before_ill_typedness_naming_every_identifier() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (u, _) = build_identifier("u");
    let considered = HashSet::from([u.clone()]);
    let expression = Expression::all([build_literal(2), x_reference.less(y_reference)]);

    let error = ask_error(
        &Question::UniversalValidity {
            considered: &considered,
            expression: &expression,
        },
        &HashMap::new(),
    );

    assert!(matches!(error, SolveError::MissingSymbolTypes(ids) if ids == vec![x, y, u]));
}

#[test]
fn missing_symbol_types_cover_both_sides_of_an_implication_but_no_constant() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());
    let pi_identifier = BuiltinConstant::Pi.identifier().clone();
    let considered = HashSet::from([pi_identifier]);

    let implication = ask_error(
        &Question::Implication {
            antecedent: &pi.clone().less(x_reference),
            consequent: &y_reference.greater(0),
        },
        &HashMap::new(),
    );
    let (answer, _) = ask_answering(
        SatResult::Unsat,
        &Question::UniversalValidity {
            considered: &considered,
            expression: &build_literal(true),
        },
        &HashMap::new(),
    );

    assert!(matches!(implication, SolveError::MissingSymbolTypes(ids) if ids == vec![x, y]));
    assert_eq!(
        answer,
        Answer::Yes,
        "a considered constant needs no symbol type"
    );
}

#[test]
fn ill_typedness_is_reported_before_a_hazard_on_either_side() {
    let hazardous = build_literal(true).equals(1);
    let ill_typed = Expression::any([build_literal(2), build_literal(4)]);
    let both = Expression::all([build_literal(2), build_literal(4)]).equals(1);

    let behind = ask_error(
        &Question::Implication {
            antecedent: &hazardous,
            consequent: &ill_typed,
        },
        &HashMap::new(),
    );
    let within = ask_error(&Question::Satisfiability(&both), &HashMap::new());

    assert!(matches!(behind, SolveError::IllTyped(_)));
    assert!(matches!(within, SolveError::IllTyped(_)));
}

#[rstest]
#[case::literal(build_literal(1))]
#[case::arithmetic(build_literal(1) + 2)]
fn numeric_root_is_ill_typed(#[case] root: Expression) {
    assert!(matches!(
        ask_error(&Question::Satisfiability(&root), &HashMap::new()),
        SolveError::IllTyped(_)
    ));
}

#[test]
fn symbol_typed_operand_in_a_boolean_position_is_ill_typed_despite_a_hazard() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let expression = Expression::all([x_reference, (y_reference / 0).equals(1)]);
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int), (&y, SymbolType::Real)]);

    assert!(matches!(
        ask_error(&Question::Satisfiability(&expression), &symbol_types),
        SolveError::IllTyped(_)
    ));
}

#[test]
fn hazard_answers_unknown_without_asking_the_backend() {
    let (x, reference) = build_identifier("x");
    let hazard = reference.clone().equals(build_literal(true));
    let backend = RecordingSmtSolver::answering(SatResult::Sat);
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);

    let answer = backend
        .solver()
        .ask(
            &Question::Implication {
                antecedent: &hazard,
                consequent: &reference.greater(0),
            },
            &QueryContext::new(&symbol_types),
        )
        .expect("answered");

    assert_eq!(
        answer,
        Answer::Unknown(UnknownReason::Refused(Hazard::BooleanCoercion(hazard)))
    );
    assert!(backend.checks().is_empty());
}

#[test]
fn each_expression_is_screened_on_its_own_antecedent_first() {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
    let equality = reference.clone().equals(1.5);
    let coercion = reference.clone().equals(build_literal(true));

    let answer = RecordingSmtSolver::answering(SatResult::Sat)
        .solver()
        .ask(
            &Question::Implication {
                antecedent: &equality,
                consequent: &coercion,
            },
            &QueryContext::new(&symbol_types),
        )
        .expect("answered");
    let consequent_only = RecordingSmtSolver::answering(SatResult::Sat)
        .solver()
        .ask(
            &Question::Implication {
                antecedent: &reference.greater(0),
                consequent: &coercion,
            },
            &QueryContext::new(&symbol_types),
        )
        .expect("answered");

    assert_eq!(
        answer,
        Answer::Unknown(UnknownReason::Refused(Hazard::MixedIntRealEquality(
            equality
        )))
    );
    assert_eq!(
        consequent_only,
        Answer::Unknown(UnknownReason::Refused(Hazard::BooleanCoercion(coercion)))
    );
}

#[test]
fn native_constant_is_refused_by_the_screen_needing_no_symbol_type() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let expression = Expression::from(pi.clone()).greater(4);

    let (answer, checks) = {
        let backend = RecordingSmtSolver::answering(SatResult::Sat);
        let answer = backend
            .solver()
            .ask(
                &Question::Satisfiability(&expression),
                &QueryContext::new(&HashMap::new()),
            )
            .expect("answered");
        (answer, backend.checks())
    };

    assert_eq!(
        answer,
        Answer::Unknown(UnknownReason::Refused(Hazard::NativeConstant(vec![pi])))
    );
    assert!(checks.is_empty());
}

#[test]
fn user_constant_is_read_from_the_sorts_of_the_context() {
    let mut registry = FunctionRegistry::new();
    let answer_identifier = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("a new name");
    let expression = Expression::from(answer_identifier.clone()).equals(42);
    let symbol_types = HashMap::new();

    let answer = RecordingSmtSolver::answering(SatResult::Sat)
        .solver()
        .ask(
            &Question::Satisfiability(&expression),
            &QueryContext::new(&symbol_types).with_sorts(&registry),
        )
        .expect("answered");

    assert_eq!(
        answer,
        Answer::Unknown(UnknownReason::Refused(Hazard::NativeConstant(vec![
            answer_identifier
        ])))
    );
}

#[test]
fn call_has_no_lowering_after_the_screens_pass() {
    let call = Expression::call(FunctionName::new("f").expect("a name"), [build_literal(1)]);

    let error = ask_error(
        &Question::Satisfiability(&call.clone().greater(0)),
        &HashMap::new(),
    );

    assert!(matches!(error, SolveError::Lowering(LoweringError::Call(node)) if node == call));
}

// ---------------------------------------------------------------------------
// Scripts and answers
// ---------------------------------------------------------------------------

#[rstest]
#[case::sat(SatResult::Sat, Answer::Yes)]
#[case::unsat(SatResult::Unsat, Answer::No)]
fn satisfiability_asserts_the_expression_and_is_yes_when_sat(
    #[case] result: SatResult,
    #[case] expected: Answer,
) {
    let (x, reference) = build_identifier("x");
    let expression = reference.greater(0);

    let (answer, script) = ask_answering(
        result,
        &Question::Satisfiability(&expression),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert_eq!(answer, expected);
    assert_eq!(
        script,
        format!(
            "(set-logic QF_LIA)\n(declare-const {x} Int)\n(assert (> {x} 0))\n",
            x = quoted_symbol(&x)
        )
    );
}

#[rstest]
#[case::unsat(SatResult::Unsat, Answer::Yes)]
#[case::sat(SatResult::Sat, Answer::No)]
fn implication_asserts_a_counterexample_and_is_yes_when_unsat(
    #[case] result: SatResult,
    #[case] expected: Answer,
) {
    let (x, reference) = build_identifier("x");
    let xs = quoted_symbol(&x);

    let (answer, script) = ask_answering(
        result,
        &Question::Implication {
            antecedent: &reference.clone().greater_equal(5),
            consequent: &reference.greater(3),
        },
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert_eq!(answer, expected);
    assert_eq!(
        assertions(&script),
        [format!("(assert (and (>= {xs} 5) (not (> {xs} 3))))").as_str()]
    );
}

#[rstest]
#[case::unsat(SatResult::Unsat, Answer::Yes)]
#[case::sat(SatResult::Sat, Answer::No)]
fn universal_validity_quantifies_the_considered_under_the_free(
    #[case] result: SatResult,
    #[case] expected: Answer,
) {
    let (x, x_reference) = build_identifier("x");
    let (n, n_reference) = build_identifier("n");
    let considered = HashSet::from([x.clone()]);
    let expression = Expression::all([
        x_reference.clone().less(n_reference.clone()),
        x_reference.greater(n_reference),
    ]);
    let (xs, ns) = (quoted_symbol(&x), quoted_symbol(&n));

    let (answer, script) = ask_answering(
        result,
        &Question::UniversalValidity {
            considered: &considered,
            expression: &expression,
        },
        &build_symbol_types(&[(&x, SymbolType::Int), (&n, SymbolType::Int)]),
    );

    assert_eq!(answer, expected);
    assert_eq!(
        script,
        format!(
            "(set-logic LIA)\n(declare-const {ns} Int)\n\
             (assert (forall (({xs} Int)) (not (and (< {xs} {ns}) (> {xs} {ns})))))\n"
        )
    );
}

#[rstest]
#[case::unsat(SatResult::Unsat, Answer::Yes)]
#[case::sat(SatResult::Sat, Answer::No)]
fn universal_validity_without_considered_identifiers_asserts_the_negation(
    #[case] result: SatResult,
    #[case] expected: Answer,
) {
    let (x, reference) = build_identifier("x");
    let (u, _) = build_identifier("u");
    let considered = HashSet::from([u.clone()]);
    let expression = (reference.clone() * reference).greater_equal(0.0);
    let xs = quoted_symbol(&x);

    let (answer, script) = ask_answering(
        result,
        &Question::UniversalValidity {
            considered: &considered,
            expression: &expression,
        },
        &build_symbol_types(&[(&x, SymbolType::Real), (&u, SymbolType::Real)]),
    );

    assert_eq!(answer, expected);
    assert_eq!(
        script,
        format!(
            "(set-logic QF_NRA)\n(declare-const {xs} Real)\n(assert (not (>= (* {xs} {xs}) 0.0)))\n"
        ),
        "a considered identifier the expression does not mention quantifies nothing"
    );
}

#[rstest]
#[case::sat(SatResult::Sat, Answer::Yes)]
#[case::unsat(SatResult::Unsat, Answer::No)]
fn universal_validity_without_free_identifiers_asserts_the_expression(
    #[case] result: SatResult,
    #[case] expected: Answer,
) {
    let (x, reference) = build_identifier("x");
    let considered = HashSet::from([x.clone()]);
    let expression = reference.equals(5);

    let (answer, script) = ask_answering(
        result,
        &Question::UniversalValidity {
            considered: &considered,
            expression: &expression,
        },
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert_eq!(answer, expected);
    assert_eq!(
        assertions(&script),
        [format!("(assert (= {} 5))", quoted_symbol(&x)).as_str()]
    );
}

#[rstest]
#[case::satisfiability(0)]
#[case::implication(1)]
#[case::universal_validity(2)]
fn unknown_answers_unknown_with_the_reason_the_backend_gave(#[case] which: usize) {
    let (x, reference) = build_identifier("x");
    let expression = reference.greater(0);
    let considered = HashSet::new();
    let questions = [
        Question::Satisfiability(&expression),
        Question::Implication {
            antecedent: &expression,
            consequent: &expression,
        },
        Question::UniversalValidity {
            considered: &considered,
            expression: &expression,
        },
    ];

    let (answer, _) = ask_answering(
        SatResult::Unknown {
            reason: "timeout".to_owned(),
        },
        &questions[which],
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );

    assert_eq!(
        answer,
        Answer::Unknown(UnknownReason::GaveUp {
            reason: "timeout".to_owned()
        })
    );
    assert_eq!(answer.decided(), None);
}

#[test]
fn limits_reach_the_backend() {
    let backend = RecordingSmtSolver::answering(SatResult::Sat);
    let limits = CheckLimits::new().with_timeout(Duration::from_millis(2_500));

    backend
        .solver()
        .ask(
            &Question::Satisfiability(&build_literal(true)),
            &QueryContext::new(&HashMap::new()).with_limits(limits),
        )
        .expect("answered");

    assert_eq!(backend.checks()[0].1, limits);
}

#[test]
fn backend_failure_names_the_backend_and_keeps_its_error() {
    let error = RecordingSmtSolver::failing("the solver crashed")
        .solver()
        .ask(
            &Question::Satisfiability(&build_literal(true)),
            &QueryContext::new(&HashMap::new()),
        )
        .expect_err("the backend fails");

    assert_eq!(error.to_string(), r#"the backend "recording" failed"#);
    let source = error.source().expect("a source");
    assert_eq!(
        source.downcast_ref::<FakeBackendError>(),
        Some(&FakeBackendError("the solver crashed".to_owned()))
    );
    assert!(matches!(error, SolveError::Backend { backend, .. } if backend == "recording"));
}

#[test]
fn answer_decided_is_the_tri_state() {
    assert_eq!(Answer::Yes.decided(), Some(true));
    assert_eq!(Answer::No.decided(), Some(false));
    assert_eq!(
        Answer::Unknown(UnknownReason::GaveUp {
            reason: String::new()
        })
        .decided(),
        None
    );
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

#[test]
fn simplification_substitutes_the_environment_before_the_simplifier() {
    let (x, reference) = build_identifier("x");
    let simplifier = RecordingSimplifier::returning(build_literal(5));
    let expression = reference + 2;

    let result = simplifier
        .solver()
        .simplify(
            &expression,
            &HashMap::from([(x, build_literal(3))]),
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect("simplified");

    assert_eq!(result, build_literal(5));
    assert_eq!(simplifier.inputs(), vec![build_literal(3) + 2]);
}

#[test]
fn simplifier_receives_the_expression_itself_when_nothing_is_bound() {
    let (_, reference) = build_identifier("x");
    let (y, _) = build_identifier("y");
    let simplifier = RecordingSimplifier::identity();
    let expression = reference + 0;

    let unbound = simplifier
        .solver()
        .simplify(
            &expression,
            &HashMap::new(),
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect("simplified");
    let unrelated = simplifier
        .solver()
        .simplify(
            &expression,
            &HashMap::from([(y, build_literal(1))]),
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect("simplified");

    let inputs = simplifier.inputs();
    assert!(Expression::ptr_eq(&inputs[0], &expression));
    assert!(Expression::ptr_eq(&inputs[1], &expression));
    assert!(Expression::ptr_eq(&unbound, &expression));
    assert!(Expression::ptr_eq(&unrelated, &expression));
}

#[test]
fn simplification_screens_with_the_environment() {
    let (b, reference) = build_identifier("b");
    let simplifier = RecordingSimplifier::identity();

    let error = simplifier
        .solver()
        .simplify(
            &reference.and(build_literal(true)),
            &HashMap::from([(b, build_literal(1))]),
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect_err("a number bound into a connective");

    assert!(matches!(error, SolveError::IllTyped(_)));
    assert!(simplifier.inputs().is_empty());
}

#[test]
fn simplification_refuses_binding_a_referenced_native_constant() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let e = BuiltinConstant::E.identifier().clone();
    let simplifier = RecordingSimplifier::identity();
    let expression = Expression::from(pi.clone()).greater(3);
    let environment = HashMap::from([(pi.clone(), build_literal(1)), (e, build_literal(2))]);

    let error = simplifier
        .solver()
        .simplify(
            &expression,
            &environment,
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect_err("refused");

    assert!(matches!(error, SolveError::BoundNativeConstant(ids) if ids == vec![pi]));
}

#[test]
fn simplification_accepts_binding_an_unreferenced_native_constant() {
    let (x, reference) = build_identifier("x");
    let e = BuiltinConstant::E.identifier().clone();
    let simplifier = RecordingSimplifier::identity();

    let result = simplifier.solver().simplify(
        &reference.greater(0),
        &HashMap::from([(e, build_literal(2)), (x, build_literal(1))]),
        &SimplifyContext::new(&NoRegisteredSorts),
    );

    assert_eq!(result.expect("simplified"), build_literal(1).greater(0));
}

#[test]
fn simplification_screens_with_the_sorts_of_a_registry_context() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("registered");
    let (b, reference) = build_identifier("b");
    let simplifier = RecordingSimplifier::identity();

    let error = simplifier
        .solver()
        .simplify(
            &reference.and(Expression::from(answer)),
            &HashMap::from([(b, build_literal(true))]),
            &SimplifyContext::from_registry(&registry),
        )
        .expect_err("an integer constant in a connective");

    assert!(matches!(error, SolveError::IllTyped(_)));
    assert!(simplifier.inputs().is_empty());
}

#[test]
fn simplifier_receives_the_registry_of_the_context() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("registered");
    let simplifier = RecordingSimplifier::identity();

    simplifier
        .solver()
        .simplify(
            &build_literal(1),
            &HashMap::new(),
            &SimplifyContext::from_registry(&registry),
        )
        .expect("simplified");
    simplifier
        .solver()
        .simplify(
            &build_literal(1),
            &HashMap::new(),
            &SimplifyContext::new(&registry),
        )
        .expect("simplified");

    assert_eq!(simplifier.registry_sizes(), vec![Some(1), None]);
}

#[test]
fn simplify_context_reads_sorts_from_its_registry() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Bool,
                true,
            )
            .expect("a constant"),
        )
        .expect("registered");

    let context = SimplifyContext::from_registry(&registry);

    assert_eq!(
        context.sorts().native_constant_sort(&answer),
        Some(FunctionSort::Bool)
    );
    assert!(context.registry().is_some_and(|held| held.len() == 1));
    assert!(SimplifyContext::default().registry().is_none());
}

#[test]
fn simplify_hands_its_limits_to_the_simplifier() {
    let simplifier = RecordingSimplifier::identity();
    let solver = simplifier.solver();
    let bounded = SimplifyLimits::new().with_timeout(Duration::from_millis(250));

    solver
        .simplify(
            &build_literal(1),
            &HashMap::new(),
            &SimplifyContext::new(&NoRegisteredSorts).with_limits(bounded),
        )
        .expect("simplified");
    solver
        .simplify(
            &build_literal(1),
            &HashMap::new(),
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect("simplified");

    assert_eq!(simplifier.limits(), vec![bounded, SimplifyLimits::new()]);
    assert_eq!(
        simplifier
            .limits()
            .first()
            .and_then(SimplifyLimits::timeout),
        Some(Duration::from_millis(250))
    );
}

#[test]
fn simplifier_failure_is_a_backend_error() {
    let error = RecordingSimplifier::failing("no")
        .solver()
        .simplify(
            &build_literal(1),
            &HashMap::new(),
            &SimplifyContext::new(&NoRegisteredSorts),
        )
        .expect_err("fails");

    assert!(matches!(error, SolveError::Backend { backend, .. } if backend == "recording"));
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[test]
fn solve_errors_display_one_lowercase_line() {
    let (x, _) = build_identifier("x");
    let pi = BuiltinConstant::Pi.identifier().clone();

    assert_eq!(
        SolveError::NoCapableBackend(QueryKind::UniversalValidity).to_string(),
        "the backends of this solver cannot answer universal validity queries"
    );
    assert_eq!(
        SolveError::MissingSymbolTypes(vec![x.clone()]).to_string(),
        format!(
            "symbol_types is missing entries for identifiers: x::{}",
            x.id()
        )
    );
    assert_eq!(
        SolveError::BoundNativeConstant(vec![pi]).to_string(),
        "the environment binds native constants the expression refers to, whose values are \
         fixed: pi::48"
    );
    assert_eq!(
        SolveError::Lowering(LoweringError::MissingSymbolTypes(vec![x])).to_string(),
        "the expression has no smt-lib2 lowering"
    );
}

#[test]
fn query_kinds_have_snake_case_names_and_word_text() {
    assert_eq!(QueryKind::UniversalValidity.as_str(), "universal_validity");
    assert_eq!(
        QueryKind::UniversalValidity.to_string(),
        "universal validity"
    );
    assert_eq!(QueryKind::Simplification.as_str(), "simplification");
    assert_eq!(
        Question::Satisfiability(&build_literal(true)).kind(),
        QueryKind::Satisfiability
    );
}
