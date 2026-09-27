//! Stories for the process backend, [`SmtLib2Process`]: the protocol for
//! each answer and reason, the failures, and the timeout, with `sh` scripts
//! as fake solvers. The stories against a real solver run when
//! `FHY_SMT_SOLVER` names one, as a program followed by its arguments, such
//! as `z3 -in`.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use fhy_core::expression::{Expression, NoRegisteredSorts, SymbolType};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::solver::{
    Answer, CheckLimits, ProcessError, QueryContext, Question, SatResult, SmtLib2Process,
    SmtScript, SmtSolver, Solver, UnknownReason,
};

use crate::support::expression::{build_identifier, build_literal};
use crate::support::solver::{build_symbol_types, quoted_symbol};

/// Return a fake solver running the `sh` program `program`.
fn build_fake(program: &str) -> SmtLib2Process {
    SmtLib2Process::new("/bin/sh").with_args(["-c", program])
}

/// Return a fake solver that answers `answer` to `(check-sat)`, `reason`
/// to `(get-info :reason-unknown)`, and exits on `(exit)`.
fn build_answering_fake(answer: &str, reason: &str) -> SmtLib2Process {
    build_fake(&format!(
        r#"while read -r line; do case "$line" in "(check-sat)") echo '{answer}';; "(get-info :reason-unknown)") echo '{reason}';; "(exit)") exit 0;; esac; done"#
    ))
}

/// Return the script asserting `x > 0` over an integer `x`.
fn build_script() -> (Identifier, SmtScript) {
    let (x, reference) = build_identifier("x");
    let script = SmtScript::lower(
        &reference.greater(0),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
        &NoRegisteredSorts,
    )
    .expect("lowers");
    (x, script)
}

/// Check the script of `x > 0` with `backend` and no limits.
fn check(backend: &SmtLib2Process) -> Result<SatResult, BoxError> {
    backend.check(&build_script().1, &CheckLimits::new())
}

/// Return the process error `result` fails with.
fn expect_process_error(result: Result<SatResult, BoxError>) -> ProcessError {
    let error = result.expect_err("the check fails");
    *error
        .downcast::<ProcessError>()
        .expect("the backend fails with a process error")
}

#[test]
fn process_reads_sat_and_unsat() {
    assert_eq!(
        check(&build_answering_fake("sat", "")).expect("an answer"),
        SatResult::Sat
    );
    assert_eq!(
        check(&build_answering_fake("unsat", "")).expect("an answer"),
        SatResult::Unsat
    );
}

#[test]
fn process_asks_the_reason_of_unknown() {
    let quoted = build_answering_fake("unknown", r#"(:reason-unknown "incomplete")"#);
    let bare = build_answering_fake("unknown", "(:reason-unknown timeout)");
    let refused = build_answering_fake("unknown", r#"(error "no reason")"#);

    assert_eq!(
        check(&quoted).expect("an answer"),
        SatResult::Unknown {
            reason: "incomplete".to_owned()
        }
    );
    assert_eq!(
        check(&bare).expect("an answer"),
        SatResult::Unknown {
            reason: "timeout".to_owned()
        }
    );
    assert_eq!(
        check(&refused).expect("an answer"),
        SatResult::Unknown {
            reason: String::new()
        }
    );
}

#[test]
fn process_writes_the_script_before_check_sat() {
    let (x, script) = build_script();
    let declaration = format!("(declare-const {} Int)", quoted_symbol(&x));
    let backend = build_fake(&format!(
        r#"seen=; while read -r line; do case "$line" in "{declaration}") seen=1;; "(check-sat)") if [ -n "$seen" ]; then echo sat; else echo unsat; fi;; "(exit)") exit 0;; esac; done"#
    ));

    assert_eq!(
        backend
            .check(&script, &CheckLimits::new())
            .expect("an answer"),
        SatResult::Sat
    );
}

#[test]
fn process_reports_an_error_line() {
    let backend = build_fake(
        r#"read -r line; echo '(error "line 1 column 1: unknown constant")'; cat > /dev/null"#,
    );

    let error = expect_process_error(check(&backend));

    assert!(
        matches!(&error, ProcessError::Solver(line) if line == r#"(error "line 1 column 1: unknown constant")"#),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        r#"the solver reported (error "line 1 column 1: unknown constant")"#
    );
}

#[test]
fn process_reports_an_unexpected_answer() {
    let error = expect_process_error(check(&build_answering_fake("maybe", "")));

    assert!(matches!(&error, ProcessError::UnexpectedAnswer(line) if line == "maybe"));
    assert_eq!(
        error.to_string(),
        r#"the solver answered "maybe", not sat, unsat or unknown"#
    );
}

#[test]
fn process_reports_a_missing_program() {
    let error = expect_process_error(check(&SmtLib2Process::new("/nonexistent/fhy-smt-solver")));

    assert!(matches!(error, ProcessError::Spawn { .. }));
    assert_eq!(
        error.to_string(),
        r#"cannot start the solver "/nonexistent/fhy-smt-solver""#
    );
}

#[test]
fn process_reports_a_solver_that_exits_before_answering() {
    let error = expect_process_error(check(&build_fake("exit 3")));

    assert!(matches!(error, ProcessError::Exited(Some(status)) if status.code() == Some(3)));
}

#[test]
fn process_is_killed_at_the_timeout_and_answers_unknown() {
    let backend = build_fake("exec sleep 30");
    let started = Instant::now();

    let result = backend.check(
        &build_script().1,
        &CheckLimits::new().with_timeout(Duration::from_millis(200)),
    );

    assert_eq!(
        result.expect("an answer"),
        SatResult::Unknown {
            reason: "timeout".to_owned()
        }
    );
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[test]
fn process_is_named_by_its_program_file_name() {
    assert_eq!(SmtLib2Process::new("/usr/local/bin/cvc5").name(), "cvc5");
    assert_eq!(SmtLib2Process::new("z3").with_args(["-in"]).name(), "z3");
    assert_eq!(SmtLib2Process::new("z3").with_args(["-in"]).args(), ["-in"]);
}

#[test]
fn process_answers_through_a_solver() {
    let solver = Solver::new().with_smt_solver(build_answering_fake("unsat", ""));
    let answer = solver
        .ask(
            &Question::Satisfiability(&build_literal(false)),
            &QueryContext::new(&HashMap::new()),
        )
        .expect("answered");

    assert_eq!(answer, Answer::No);
}

// ---------------------------------------------------------------------------
// A real solver, named by FHY_SMT_SOLVER
// ---------------------------------------------------------------------------

/// Return the solver `FHY_SMT_SOLVER` names, or `None` when it is unset.
fn configured_solver() -> Option<Solver> {
    let configured = std::env::var("FHY_SMT_SOLVER").ok()?;
    let mut words = configured.split_whitespace();
    let program = words.next()?;
    Some(Solver::new().with_smt_solver(SmtLib2Process::new(program).with_args(words)))
}

#[test]
fn real_solver_decides_the_three_questions() {
    let Some(solver) = configured_solver() else {
        return;
    };
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int), (&y, SymbolType::Int)]);
    let context = QueryContext::new(&symbol_types);
    let positive = x_reference.clone().greater(0);
    let contradiction = Expression::all([positive.clone(), x_reference.clone().less(0)]);
    let considered = HashSet::from([y.clone()]);
    let witness = y_reference.greater(x_reference.clone());

    let ask = |question: Question<'_>| solver.ask(&question, &context).expect("answered");

    assert_eq!(ask(Question::Satisfiability(&positive)), Answer::Yes);
    assert_eq!(ask(Question::Satisfiability(&contradiction)), Answer::No);
    assert_eq!(
        ask(Question::Implication {
            antecedent: &x_reference.clone().greater_equal(5),
            consequent: &x_reference.greater(3),
        }),
        Answer::Yes
    );
    assert_eq!(
        ask(Question::UniversalValidity {
            considered: &considered,
            expression: &witness,
        }),
        Answer::Yes
    );
}

#[test]
fn real_solver_answers_unknown_at_the_timeout() {
    let Some(solver) = configured_solver() else {
        return;
    };
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

    let answer = solver
        .ask(
            &Question::Satisfiability(&fermat),
            &QueryContext::new(&symbol_types)
                .with_limits(CheckLimits::new().with_timeout(Duration::from_millis(200))),
        )
        .expect("answered");

    assert!(
        matches!(answer, Answer::Unknown(UnknownReason::GaveUp { .. })),
        "{answer:?}"
    );
}
