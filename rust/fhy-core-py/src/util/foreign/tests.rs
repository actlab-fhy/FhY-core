//! The stories of [`foreign_of`], run in an interpreter this test binary
//! embeds, against the stand-in of `fhy_core.serialization._foreign_payload`.

use std::error::Error;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};

use crate::util::pending::{capture_pending_errors, has_pending_error};
use crate::util::testing::{define, entry, with_framework};

use super::*;

/// Parts of the stand-in framework: `Foreign(type_id, data, error=None)`.
const PARTS: &str = "
import fhy_core.serialization as framework

class Part:
    def __init__(self, error=None):
        self.type_id = 'tests.Part'
        self.data = '{\"size\": 3}'
        self.error = error

class NotAPair:
    def __init__(self):
        self.type_id = 'tests.NotAPair'
        self.data = None
        self.error = None

ok = Part()
broken = Part(ValueError('the hook failed'))
interrupted = Part(KeyboardInterrupt())
";

#[test]
fn a_part_gives_its_type_id_and_data_as_a_family_member_or_whole() {
    with_framework(|py| {
        let namespace = define(py, PARTS);
        let part = entry(&namespace, "ok").unbind();

        let family = foreign_of(&part, true).expect("a foreign part");
        let whole = foreign_of(&part, false).expect("a foreign part");

        assert_eq!(family.type_id(), "tests.Part:family");
        assert_eq!(whole.type_id(), "tests.Part:whole");
        assert_eq!(family.data(), "{\"size\": 3}");
        assert_eq!(whole.data(), family.data());
        assert!(!has_pending_error());
    });
}

#[test]
fn a_hook_that_raises_fails_the_part_and_keeps_the_exception_pending() {
    with_framework(|py| {
        let namespace = define(py, PARTS);
        let part = entry(&namespace, "broken").unbind();

        let (result, pending) = capture_pending_errors(|| foreign_of(&part, true));

        let Err(ForeignError::Failed { type_id, source }) = result else {
            panic!("expected a failed foreign part");
        };
        assert_eq!(type_id, "Part");
        assert_eq!(source.to_string(), "the hook failed");
        let pending = pending.expect("the exception is kept");
        assert!(pending.is_instance_of::<PyValueError>(py));
        assert_eq!(pending.value(py).to_string(), "the hook failed");
    });
}

#[test]
fn an_answer_that_is_not_a_pair_of_strings_fails_the_part_with_a_type_error() {
    with_framework(|py| {
        let namespace = define(py, PARTS);
        let part = entry(&namespace, "NotAPair")
            .call0()
            .expect("an instance")
            .unbind();

        let (result, pending) = capture_pending_errors(|| foreign_of(&part, false));

        let Err(ForeignError::Failed { type_id, .. }) = result else {
            panic!("expected a failed foreign part");
        };
        assert_eq!(type_id, "NotAPair");
        assert!(pending.expect("kept").is_instance_of::<PyTypeError>(py));
    });
}

#[test]
fn the_first_failure_stays_pending_unless_an_interrupt_follows() {
    with_framework(|py| {
        let namespace = define(py, PARTS);
        let broken = entry(&namespace, "broken").unbind();
        let interrupted = entry(&namespace, "interrupted").unbind();

        let (_results, pending) = capture_pending_errors(|| {
            let first = foreign_of(&broken, true);
            let second = foreign_of(&interrupted, true);
            (first.is_err(), second.is_err())
        });

        let pending = pending.expect("kept");
        assert!(pending.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py));
    });
}

#[test]
fn a_failure_keeps_the_given_exception_and_the_core_sees_its_message() {
    with_framework(|py| {
        let (error, pending) =
            capture_pending_errors(|| foreign_failure(py, "tests.Id", PyKeyError::new_err("gone")));

        let ForeignError::Failed { type_id, source } = error else {
            panic!("expected Failed");
        };
        assert_eq!(type_id, "tests.Id");
        assert_eq!(source.to_string(), "'gone'");
        assert!(pending.expect("kept").is_instance_of::<PyKeyError>(py));
    });
}

#[test]
fn a_raised_error_displays_its_message_as_an_error() {
    let error = RaisedError::new("refused by the part".to_owned());

    assert_eq!(error.to_string(), "refused by the part");
    assert_eq!(format!("{error:?}"), "RaisedError(\"refused by the part\")");
    let boxed: Box<dyn Error + Send + Sync> = Box::new(error);
    assert!(boxed.source().is_none());
}
