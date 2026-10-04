//! The stories of the infallible-hook helper of the util module, run in an
//! interpreter this crate's tests embed.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::ask;
use crate::util::pending::{capture_pending_errors, has_pending_error};
use crate::util::testing::{define, entry, with_framework};

#[test]
fn an_answer_is_returned_as_the_hook_gave_it() {
    with_framework(|py| {
        let namespace = define(
            py,
            "class Hook:\n    def same(self, other):\n        return other == 3\n",
        );
        let hook = entry(&namespace, "Hook").call0().unwrap().unbind();

        let (answer, raised) = capture_pending_errors(|| {
            ask(false, |py| {
                hook.bind(py).call_method1("same", (3,))?.is_truthy()
            })
        });

        assert!(answer);
        assert!(raised.is_none());
    });
}

#[test]
fn an_exception_answers_the_fallback_and_is_kept_pending() {
    with_framework(|py| {
        let (answer, raised) = capture_pending_errors(|| {
            let answer = ask(7, |_py| Err(PyValueError::new_err("the hook raised")));
            assert!(has_pending_error());
            answer
        });

        assert_eq!(answer, 7);
        let raised = raised.expect("the exception is kept");
        assert!(raised.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn once_an_exception_is_pending_python_is_not_called_again() {
    with_framework(|py| {
        let mut called = 0;
        let (answers, raised) = capture_pending_errors(|| {
            let first = ask(1, |_py| Err(PyValueError::new_err("first")));
            let second = ask(2, |_py| {
                called += 1;
                Err(PyTypeError::new_err("second"))
            });
            (first, second)
        });

        assert_eq!(answers, (1, 2));
        assert_eq!(called, 0);
        assert!(raised.expect("kept").is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn a_hook_that_answers_the_wrong_type_is_a_kept_exception() {
    with_framework(|py| {
        let namespace = define(
            py,
            "class Hook:\n    def answer(self):\n        return 'text'\n",
        );
        let hook = entry(&namespace, "Hook").call0().unwrap().unbind();

        let (answer, raised) = capture_pending_errors(|| {
            ask(0_u32, |py| hook.bind(py).call_method0("answer")?.extract())
        });

        assert_eq!(answer, 0);
        assert!(raised.expect("kept").is_instance_of::<PyTypeError>(py));
    });
}
