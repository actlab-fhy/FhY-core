//! The stories of the test stand-ins of the util module, run in an
//! interpreter this crate's tests embed.

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use super::{define, entry, evaluate, install_module, with_stand_ins};

#[test]
fn the_stand_ins_of_fhy_core_modules_are_importable() {
    with_stand_ins(|py| {
        for (module, name) in [
            ("fhy_core.serialization", "SerializationError"),
            ("fhy_core.serialization", "Foreign"),
            ("fhy_core.traits.frozen", "FrozenMutationError"),
            (
                "fhy_core.term.derived_equivalence",
                "EquivalenceDerivationError",
            ),
        ] {
            let found = py.import(module).and_then(|module| module.getattr(name));
            assert!(found.is_ok(), "{module}.{name} is a stand-in");
        }
    });
}

#[test]
fn the_stand_ins_are_installed_once_and_stay_the_same_objects() {
    let first = with_stand_ins(|py| {
        py.import("fhy_core.serialization")
            .and_then(|module| module.getattr("SerializationError"))
            .unwrap()
            .unbind()
    });

    with_stand_ins(|py| {
        let again = py
            .import("fhy_core.serialization")
            .and_then(|module| module.getattr("SerializationError"))
            .unwrap();
        assert!(again.is(first.bind(py)));
    });
}

#[test]
fn the_stand_in_serialization_errors_share_one_root() {
    with_stand_ins(|py| {
        let module = py.import("fhy_core.serialization").unwrap();
        let root = module.getattr("SerializationError").unwrap();
        let root = root.cast::<PyType>().unwrap();

        for name in ["DeserializationValueError", "MalformedPayloadError"] {
            let class = module.getattr(name).unwrap();
            assert!(class.cast::<PyType>().unwrap().is_subclass(root).unwrap());
        }
    });
}

#[test]
fn install_module_creates_missing_parent_packages_and_links_the_leaf() {
    with_stand_ins(|py| {
        install_module(
            py,
            "fhy_core_util_testing_story.inner.leaf",
            "VALUE = 41 + 1\n",
        )
        .unwrap();

        let leaf = evaluate(
            py,
            "__import__('fhy_core_util_testing_story.inner.leaf', fromlist=['x'])",
        );
        assert_eq!(leaf.getattr("VALUE").unwrap().extract::<i32>().unwrap(), 42);
        let through_parents = evaluate(
            py,
            "__import__('sys').modules['fhy_core_util_testing_story'].inner.leaf.VALUE",
        );
        assert_eq!(through_parents.extract::<i32>().unwrap(), 42);
    });
}

#[test]
fn install_module_returns_the_exception_its_source_raises() {
    with_stand_ins(|py| {
        let error = install_module(py, "fhy_core_util_testing_raises", "raise KeyError('x')\n")
            .unwrap_err();

        assert!(error.is_instance_of::<PyKeyError>(py));
    });
}

#[test]
fn define_runs_source_in_a_new_namespace_that_entry_reads() {
    with_stand_ins(|py| {
        let namespace = define(py, "value = 3\ndef double(n):\n    return 2 * n\n");

        assert_eq!(entry(&namespace, "value").extract::<i32>().unwrap(), 3);
        assert_eq!(
            entry(&namespace, "double")
                .call1((4,))
                .unwrap()
                .extract::<i32>()
                .unwrap(),
            8
        );
    });
}

#[test]
#[should_panic(expected = "the namespace has no \"missing\"")]
fn entry_panics_for_a_name_the_namespace_lacks() {
    with_stand_ins(|py| {
        let namespace = define(py, "present = 1\n");

        drop(entry(&namespace, "missing"));
    });
}

#[test]
#[should_panic(expected = "evaluating")]
fn evaluate_panics_when_the_expression_raises() {
    with_stand_ins(|py| {
        drop(evaluate(py, "1 / 0"));
    });
}
