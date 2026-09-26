//! The SymPy backend without an interpreter, and without SymPy.
//!
//! This binary holds a single test because the test needs a process in
//! which no Python interpreter has started yet, and then one in which
//! `sympy` cannot be imported; the `it` binary starts its interpreter and
//! loads SymPy once for all its stories. Do not add a second test.

use fhy_core::expression::Expression;
use fhy_core::solver::{
    Simplifier, SimplifyContext, SympyError, SympyErrorKind, SympySimplifier, SympyUnavailableError,
};
use pyo3::prelude::*;

/// Run the Python statements `source` in the interpreter.
fn run(source: &std::ffi::CStr) {
    Python::attach(|py| py.run(source, None, None)).expect("the statements run");
}

/// Test the backend refuses to run without an interpreter, reports a
/// missing SymPy once one runs, and loads once SymPy is importable, never
/// keeping a failed load.
#[test]
fn backend_is_unavailable_without_an_interpreter_or_sympy_and_loads_once_both_are_there() {
    let backend = SympySimplifier::new();

    assert!(matches!(
        backend.load(),
        Err(SympyUnavailableError::NoInterpreter)
    ));
    let error = backend
        .simplify(&Expression::literal(1), &SimplifyContext::default())
        .expect_err("no interpreter");
    let error = error.downcast::<SympyError>().expect("a sympy error");
    assert!(matches!(
        error.kind(),
        SympyErrorKind::Unavailable(SympyUnavailableError::NoInterpreter)
    ));

    Python::initialize();
    run(c"import sys\nsys.modules['sympy'] = None\n");
    assert!(matches!(
        backend.load(),
        Err(SympyUnavailableError::MissingSympy(_))
    ));

    run(c"import sys\ndel sys.modules['sympy']\n");
    backend
        .load()
        .expect("loading needs SymPy on PYTHONPATH; see D-S12-13 of docs/design/python-switch.md");
    assert_eq!(
        backend
            .simplify(&Expression::literal(1), &SimplifyContext::default())
            .expect("simplified"),
        Expression::literal(1)
    );
}
