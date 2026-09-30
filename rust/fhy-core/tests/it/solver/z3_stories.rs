//! Stories for the z3 backend, [`Z3Solver`], under the `z3` feature: the
//! decided cases of the Python solver tests decided the same way, a
//! timeout, concurrent checks, and nothing kept between checks.

use std::collections::{HashMap, HashSet};
use std::thread;
use std::time::Duration;

use fhy_core::expression::{Expression, LiteralValue, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    Answer, CheckLimits, QueryContext, Question, SmtSolver, Solver, UnknownReason, Z3Solver,
};
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};
use crate::support::solver::build_symbol_types;

/// Return the solver holding the z3 backend.
fn z3() -> Solver {
    Solver::new().with_smt_solver(Z3Solver::new())
}

/// Return whether some assignment of integer `x` and `y`, and real `r`,
/// satisfies the expression `build` returns.
fn is_satisfiable(build: fn(&Expression, &Expression, &Expression) -> Expression) -> Answer {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (r, r_reference) = build_identifier("r");
    let symbol_types = build_symbol_types(&[
        (&x, SymbolType::Int),
        (&y, SymbolType::Int),
        (&r, SymbolType::Real),
    ]);
    z3().ask(
        &Question::Satisfiability(&build(&x_reference, &y_reference, &r_reference)),
        &QueryContext::new(&symbol_types),
    )
    .expect("answered")
}

fn parse(text: &str) -> Expression {
    build_literal(LiteralValue::parse_text(text).expect("a literal text"))
}

#[rstest]
#[case::positive_int(|x: &Expression, _: &Expression, _: &Expression| x.greater(0), true)]
#[case::empty_interval(|x: &Expression, _: &Expression, _: &Expression| Expression::all([x.greater(10), x.less(5)]), false)]
#[case::no_int_between_zero_and_one(|x: &Expression, _: &Expression, _: &Expression| Expression::all([build_literal(0).less(x.clone()), x.less(1)]), false)]
#[case::a_real_between_zero_and_one(|_: &Expression, _: &Expression, r: &Expression| Expression::all([build_literal(0).less(r.clone()), r.less(1)]), true)]
#[case::true_literal(|_: &Expression, _: &Expression, _: &Expression| build_literal(true), true)]
#[case::false_literal(|_: &Expression, _: &Expression, _: &Expression| build_literal(false), false)]
#[case::even(|x: &Expression, _: &Expression, _: &Expression| x.floor_mod(2).equals(0), true)]
#[case::floor_division(|_: &Expression, _: &Expression, _: &Expression| build_literal(7).floor_divide(2).equals(3), true)]
#[case::negative_float_divisor(|_: &Expression, _: &Expression, _: &Expression| (build_literal(7.0) / -2.0).equals(-3.5), true)]
#[case::real_divisor(|_: &Expression, _: &Expression, _: &Expression| (build_literal(7) / 2.0).equals(3.5), true)]
#[case::square(|x: &Expression, _: &Expression, _: &Expression| x.power(2).equals(4), true)]
#[case::real_equal_to_a_whole_float(|_: &Expression, _: &Expression, r: &Expression| r.equals(1.0), true)]
#[case::real_below_an_int(|_: &Expression, _: &Expression, r: &Expression| r.less(1), true)]
#[case::real_equal_to_a_float(|_: &Expression, _: &Expression, r: &Expression| r.equals(1.5), true)]
#[case::int_below_a_float(|x: &Expression, _: &Expression, _: &Expression| x.less(1.5), true)]
#[case::int_sum(|_: &Expression, y: &Expression, _: &Expression| (y.clone() + 1).equals(3), true)]
#[case::int_times_a_float(|_: &Expression, y: &Expression, _: &Expression| (y.clone() * 1.0).equals(3.0), true)]
#[case::real_sum(|_: &Expression, _: &Expression, r: &Expression| (r.clone() + 1).equals(3.0), true)]
#[case::int_floor_division(|_: &Expression, y: &Expression, _: &Expression| y.floor_divide(2).equals(3), true)]
#[case::int_sum_below_a_float(|_: &Expression, y: &Expression, _: &Expression| (y.clone() + 1).less(3.0), true)]
#[case::int_equal_to_a_real(|_: &Expression, y: &Expression, r: &Expression| y.equals(r.clone()), true)]
#[case::parsed_int_equal(|_: &Expression, _: &Expression, _: &Expression| parse("1").equals(1), true)]
#[case::parsed_int_unequal(|_: &Expression, _: &Expression, _: &Expression| parse("1").equals(2), false)]
#[case::padded_int_equal(|_: &Expression, _: &Expression, _: &Expression| parse("1").equals(parse("01")), true)]
#[case::exponent_spellings(|x: &Expression, _: &Expression, _: &Expression| x.power(parse("02")).equals(4), true)]
fn z3_decides_the_python_solver_cases_the_same_way(
    #[case] build: fn(&Expression, &Expression, &Expression) -> Expression,
    #[case] expected: bool,
) {
    assert_eq!(is_satisfiable(build).decided(), Some(expected));
}

#[test]
fn z3_decides_float_arithmetic_in_exact_rationals() {
    let ground = (build_literal(1e16) + 1.0).equals(1e16);
    let tenths = (build_literal(0.1) + 0.2).equals(0.3);
    let decimal_tenths = (parse("0.1") + parse("0.2")).equals(parse("0.3"));

    let ask = |expression: &Expression| {
        z3().ask(
            &Question::Satisfiability(expression),
            &QueryContext::new(&HashMap::new()),
        )
        .expect("answered")
    };

    assert_eq!(ask(&ground), Answer::No, "1e16 + 1 is exact");
    assert_eq!(ask(&tenths), Answer::No, "binary tenths do not add up");
    assert_eq!(ask(&decimal_tenths), Answer::Yes, "decimal tenths do");
}

#[test]
fn z3_decides_implication_and_universal_validity() {
    let (x, x_reference) = build_identifier("x");
    let (n, n_reference) = build_identifier("n");
    let (u, _) = build_identifier("u");
    let int_types = build_symbol_types(&[
        (&x, SymbolType::Int),
        (&n, SymbolType::Int),
        (&u, SymbolType::Int),
    ]);
    let real_types = build_symbol_types(&[(&x, SymbolType::Real)]);
    let solver = z3();
    let context = QueryContext::new(&int_types);
    let none = HashSet::new();
    let only_x = HashSet::from([x.clone()]);
    let x_and_unused = HashSet::from([x.clone(), u]);
    let no_witness = Expression::all([
        x_reference.clone().less(n_reference.clone()),
        x_reference.clone().greater(n_reference),
    ]);
    let square = (x_reference.clone() * x_reference.clone()).greater_equal(0);

    let implies = |antecedent: Expression, consequent: Expression| {
        solver
            .ask(
                &Question::Implication {
                    antecedent: &antecedent,
                    consequent: &consequent,
                },
                &context,
            )
            .expect("answered")
    };

    assert_eq!(
        implies(
            x_reference.clone().greater_equal(5),
            x_reference.clone().greater(3)
        ),
        Answer::Yes
    );
    assert_eq!(
        implies(
            x_reference.clone().greater_equal(5),
            x_reference.clone().greater(10)
        ),
        Answer::No
    );
    assert_eq!(
        solver
            .ask(
                &Question::UniversalValidity {
                    considered: &none,
                    expression: &square,
                },
                &QueryContext::new(&real_types),
            )
            .expect("answered"),
        Answer::Yes
    );
    assert_eq!(
        solver
            .ask(
                &Question::UniversalValidity {
                    considered: &only_x,
                    expression: &no_witness,
                },
                &context,
            )
            .expect("answered"),
        Answer::No
    );
    assert_eq!(
        solver
            .ask(
                &Question::UniversalValidity {
                    considered: &x_and_unused,
                    expression: &x_reference.equals(5),
                },
                &context,
            )
            .expect("answered"),
        Answer::Yes
    );
}

#[test]
fn z3_answers_unknown_at_the_timeout() {
    let names = ["a", "b", "c"].map(build_identifier);
    let [(a, a_reference), (b, b_reference), (c, c_reference)] = names;
    let cube = |reference: &Expression| reference.clone().power(3);
    let fermat = Expression::all([
        a_reference.clone().greater(0),
        b_reference.clone().greater(0),
        c_reference.clone().greater(0),
        (cube(&a_reference) + cube(&b_reference)).equals(cube(&c_reference)),
    ]);
    let symbol_types = build_symbol_types(&[
        (&a, SymbolType::Int),
        (&b, SymbolType::Int),
        (&c, SymbolType::Int),
    ]);

    let answer = z3()
        .ask(
            &Question::Satisfiability(&fermat),
            &QueryContext::new(&symbol_types)
                .with_limits(CheckLimits::new().with_timeout(Duration::from_millis(50))),
        )
        .expect("answered");

    assert!(
        matches!(answer, Answer::Unknown(UnknownReason::GaveUp { .. })),
        "{answer:?}"
    );
}

#[test]
fn z3_checks_on_several_threads_at_once() {
    let answers: Vec<Option<bool>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..8_i64)
            .map(|index| {
                scope.spawn(move || {
                    let (x, reference) = build_identifier("x");
                    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
                    let expression = Expression::all([
                        reference.clone().greater(index),
                        reference.less(index + 2 - index % 2),
                    ]);
                    z3().ask(
                        &Question::Satisfiability(&expression),
                        &QueryContext::new(&symbol_types),
                    )
                    .expect("answered")
                    .decided()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("the thread answers"))
            .collect()
    });

    let expected: Vec<Option<bool>> = (0..8).map(|index| Some(index % 2 == 0)).collect();
    assert_eq!(answers, expected);
}

#[test]
fn z3_keeps_nothing_between_checks() {
    let (x, reference) = build_identifier("x");
    let between = Expression::all([build_literal(0).less(reference.clone()), reference.less(1)]);
    let as_int = build_symbol_types(&[(&x, SymbolType::Int)]);
    let as_real = build_symbol_types(&[(&x, SymbolType::Real)]);
    let ask = |symbol_types: &HashMap<Identifier, SymbolType>| {
        z3().ask(
            &Question::Satisfiability(&between),
            &QueryContext::new(symbol_types),
        )
        .expect("answered")
    };

    assert_eq!(ask(&as_int), Answer::No);
    assert_eq!(
        ask(&as_real),
        Answer::Yes,
        "the same symbol, redeclared real"
    );
    assert_eq!(ask(&as_int), Answer::No);
}

#[rstest]
#[case::implied_by_its_int_spelling(1, 1.0, Answer::Yes)]
#[case::a_fraction_no_int_equals(1, 1.5, Answer::No)]
fn z3_answers_an_int_equal_to_a_float_as_the_evaluator_does(
    #[case] value: i64,
    #[case] float: f64,
    #[case] expected: Answer,
) {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
    let antecedent = reference.clone().equals(value);
    let consequent = reference.equals(float);

    let answer = z3()
        .ask(
            &Question::Implication {
                antecedent: &antecedent,
                consequent: &consequent,
            },
            &QueryContext::new(&symbol_types),
        )
        .expect("answered");

    assert_eq!(answer, expected);
}

#[test]
fn a_nul_name_hint_is_answered_not_a_panic() {
    let (x, reference) = build_identifier("nul\0x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
    let expression = Expression::all([reference.clone().greater(0), reference.less(2)]);

    let answer = z3()
        .ask(
            &Question::Satisfiability(&expression),
            &QueryContext::new(&symbol_types),
        )
        .expect("answered");

    assert_eq!(answer, Answer::Yes);
}

#[test]
fn z3_backend_is_named_z3() {
    assert_eq!(Z3Solver::new().name(), "z3");
}
