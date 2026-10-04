//! The stories of [`PublicClass`], run in an interpreter this test binary
//! embeds.

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyType;

use crate::kit::testing::{evaluate, with_framework};

use super::*;

/// Return a new class named `name`.
fn class_named<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyType> {
    evaluate(py, &format!("type({name:?}, (), {{}})"))
        .cast_into::<PyType>()
        .expect("a class")
}

#[test]
fn a_registered_class_is_the_one_get_returns() {
    with_framework(|py| {
        let slot = PublicClass::new("Thing");
        let class = class_named(py, "PublicThing");

        slot.register(&class).expect("register");

        assert!(slot.get(py).expect("registered").is(&class));
    });
}

#[test]
fn registering_the_registered_class_again_does_nothing() {
    with_framework(|py| {
        let slot = PublicClass::new("Thing");
        let class = class_named(py, "PublicThing");

        slot.register(&class).expect("first");
        slot.register(&class).expect("again");

        assert!(slot.get(py).expect("registered").is(&class));
    });
}

#[test]
fn registering_another_class_is_a_runtime_error_naming_the_first() {
    with_framework(|py| {
        let slot = PublicClass::new("Thing");
        let first = class_named(py, "First");
        let second = class_named(py, "Second");
        slot.register(&first).expect("first");

        let error = slot.register(&second).expect_err("another class");

        assert!(error.is_instance_of::<PyRuntimeError>(py));
        assert_eq!(
            error.value(py).to_string(),
            "the public class of fhy_core._rs.Thing is registered already, as First"
        );
        assert!(slot.get(py).expect("still the first").is(&first));
    });
}

#[test]
fn an_empty_slot_is_a_runtime_error_naming_the_class() {
    with_framework(|py| {
        let error = PublicClass::new("Thing")
            .get(py)
            .expect_err("nothing registered");

        assert!(error.is_instance_of::<PyRuntimeError>(py));
        assert_eq!(
            error.value(py).to_string(),
            "no public class is registered for fhy_core._rs.Thing"
        );
    });
}

#[test]
fn a_slot_of_another_module_names_that_module() {
    with_framework(|py| {
        let slot = PublicClass::in_module("moga._rs", "Rule");
        let error = slot.get(py).expect_err("nothing registered");
        assert_eq!(
            error.value(py).to_string(),
            "no public class is registered for moga._rs.Rule"
        );

        let first = class_named(py, "First");
        slot.register(&first).expect("first");
        let error = slot
            .register(&class_named(py, "Second"))
            .expect_err("another");
        assert_eq!(
            error.value(py).to_string(),
            "the public class of moga._rs.Rule is registered already, as First"
        );
    });
}

#[test]
fn a_slot_is_a_debug_value_naming_its_class() {
    let slot = PublicClass::in_module("moga._rs", "Rule");

    let text = format!("{slot:?}");

    assert!(text.contains("moga._rs") && text.contains("Rule"));
}
