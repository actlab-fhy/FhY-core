//! The stories of the context-frame helper of the util module, run in an
//! interpreter this crate's tests embed.

use pyo3::exceptions::PyRuntimeError;

use super::Frames;
use crate::util::pending::{capture_pending_errors, has_pending_error};
use crate::util::scoped::ScopedStack;
use crate::util::testing::with_framework;

thread_local! {
    static STACK: ScopedStack<u32> = const { ScopedStack::new() };
}

static NUMBERS: Frames<u32> = Frames::new(&STACK, "the numbers");

#[test]
fn a_read_returns_the_innermost_frame_and_pops_with_its_guard() {
    with_framework(|_py| {
        let ((), raised) = capture_pending_errors(|| {
            let outer = NUMBERS.push(1);
            {
                let _inner = NUMBERS.push(2);
                assert_eq!(NUMBERS.read(|number| *number), Some(2));
                assert_eq!(NUMBERS.cloned(), Some(2));
            }
            assert_eq!(NUMBERS.read_or(0, |number| *number), 1);
            drop(outer);
            assert!(!NUMBERS.is_pushed());
        });

        assert!(raised.is_none());
    });
}

#[test]
fn a_miss_is_a_pending_runtime_error_naming_what_was_asked() {
    with_framework(|py| {
        let (read, raised) = capture_pending_errors(|| {
            let read = NUMBERS.read_or(99, |number| *number);
            assert!(has_pending_error());
            read
        });

        assert_eq!(read, 99);
        let raised = raised.expect("a miss is recorded");
        assert!(raised.is_instance_of::<PyRuntimeError>(py));
        assert_eq!(
            raised.value(py).to_string(),
            "a hook asked for the numbers outside an entry point that provides it"
        );
    });
}

#[test]
fn a_miss_does_not_call_the_reader_and_is_none() {
    with_framework(|_py| {
        let mut called = false;
        let (read, raised) = capture_pending_errors(|| {
            NUMBERS.read(|_number| {
                called = true;
            })
        });

        assert!(read.is_none());
        assert!(!called);
        assert!(raised.is_some());
    });
}

#[test]
fn a_hit_records_nothing() {
    with_framework(|_py| {
        let ((), raised) = capture_pending_errors(|| {
            let _guard = NUMBERS.push(5);
            let _ = NUMBERS.read(|number| *number);
            assert!(!has_pending_error());
        });

        assert!(raised.is_none());
    });
}

#[test]
fn a_panic_inside_an_entry_point_pops_its_frame() {
    with_framework(|_py| {
        let ((), raised) = capture_pending_errors(|| {
            let unwound = std::panic::catch_unwind(|| {
                let _guard = NUMBERS.push(8);
                panic!("inside an entry point");
            });
            let _panic = unwound.unwrap_err();

            assert!(!NUMBERS.is_pushed());
        });

        assert!(raised.is_none());
    });
}

#[test]
fn is_pushed_is_silent_on_a_miss() {
    with_framework(|_py| {
        let (pushed, raised) = capture_pending_errors(|| NUMBERS.is_pushed());

        assert!(!pushed);
        assert!(raised.is_none());
    });
}
