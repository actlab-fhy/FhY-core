//! The stories of the Python helpers of the util module, run in an interpreter this
//! test binary embeds; they need no package but the standard library.

use pyo3::exceptions::{PyAttributeError, PyImportError, PyModuleNotFoundError, PyTypeError};
use pyo3::types::{PyList, PyType};

use crate::util::testing::{evaluate, with_stand_ins};

use super::*;

static LEN: ImportedAttr = ImportedAttr::new("builtins", "len");
static VALUE_ERROR: ImportedAttr<PyType> = ImportedAttr::new("builtins", "ValueError");
static LEN_AS_A_TYPE: ImportedAttr<PyType> = ImportedAttr::new("builtins", "len");
static NO_MODULE: ImportedAttr = ImportedAttr::new("fhy_core_util_no_such_module", "Thing");
static NO_ATTRIBUTE: ImportedAttr = ImportedAttr::new("builtins", "fhy_core_util_no_such_name");
static LATE: ImportedAttr = ImportedAttr::new("fhy_core_util_late_module", "Value");

#[test]
fn a_seed_is_taken_once_and_the_second_take_names_its_kind() {
    with_stand_ins(|_py| {
        let seed = Seed::new(vec![1, 2]);

        assert_eq!(seed.take("a list").expect("first take"), vec![1, 2]);
        let error = seed.take("a list").expect_err("second take");

        assert_eq!(error.to_string(), "RuntimeError: a list seed is used once");
    });
}

#[test]
fn a_seed_whose_lock_was_poisoned_still_gives_its_contents() {
    with_stand_ins(|_py| {
        let seed = Seed::new(7_u32);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = seed.0.lock().expect("lock");
            panic!("poison the lock");
        }));
        poisoned.unwrap_err();

        assert_eq!(seed.take("a number").expect("take"), 7);
    });
}

#[test]
fn an_imported_attribute_is_the_one_object_every_time() {
    with_stand_ins(|py| {
        let first = LEN.get(py).expect("len");
        let second = LEN.get(py).expect("len");

        assert!(first.is(second));
        assert!(first.is(evaluate(py, "len")));
        assert_eq!(
            first
                .call1((PyList::empty(py),))
                .expect("call")
                .extract::<usize>()
                .unwrap(),
            0
        );
    });
}

#[test]
fn a_typed_attribute_is_checked_against_its_type() {
    with_stand_ins(|py| {
        let class = VALUE_ERROR.get(py).expect("a class");
        assert!(class.is(evaluate(py, "ValueError")));

        let error = LEN_AS_A_TYPE.get(py).expect_err("len is no type");
        assert!(error.is_instance_of::<PyTypeError>(py));
    });
}

#[test]
fn an_attribute_that_cannot_be_imported_raises_what_the_import_raises() {
    with_stand_ins(|py| {
        let error = NO_MODULE.get(py).expect_err("no module");
        assert!(error.is_instance_of::<PyModuleNotFoundError>(py));

        let error = NO_ATTRIBUTE.get(py).expect_err("no attribute");
        assert!(
            error.is_instance_of::<PyAttributeError>(py)
                || error.is_instance_of::<PyImportError>(py)
        );
    });
}

#[test]
fn a_failed_import_is_not_kept() {
    with_stand_ins(|py| {
        LATE.get(py).expect_err("the module does not exist yet");
        py.run(
            c"import sys, types\nm = types.ModuleType('fhy_core_util_late_module')\nm.Value = 5\nsys.modules['fhy_core_util_late_module'] = m",
            None,
            None,
        )
        .expect("install");

        assert_eq!(
            LATE.get(py)
                .expect("now it imports")
                .extract::<i32>()
                .unwrap(),
            5
        );
    });
}

#[test]
fn cached_attr_imports_for_its_call_site_with_or_without_a_type() {
    with_stand_ins(|py| {
        let lookup = |py| crate::cached_attr!(py, "builtins", "len");
        let first = lookup(py).expect("len");
        let second = lookup(py).expect("len");
        assert!(first.is(second));

        let class = crate::cached_attr!(py, "builtins", "ValueError" => PyType).expect("class");
        assert!(class.is(evaluate(py, "ValueError")));
        let error = crate::cached_attr!(py, "builtins", "len" => PyType).expect_err("no type");
        assert!(error.is_instance_of::<PyTypeError>(py));
        // The re-export in this module names the same macro.
        let again = cached_attr!(py, "builtins", "len").expect("len");
        assert!(again.is(first));
    });
}

#[test]
fn type_name_is_the_name_of_the_type() {
    with_stand_ins(|py| {
        assert_eq!(read_type_name(&evaluate(py, "3")), "int");
        assert_eq!(read_type_name(&evaluate(py, "[]")), "list");
        assert_eq!(
            read_type_name(&evaluate(py, "type('Custom', (), {})()")),
            "Custom"
        );
        assert_eq!(read_type_name(&evaluate(py, "int")), "type");
    });
}
