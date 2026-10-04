//! The stories of the dataclass helpers of the util module, run in an interpreter
//! this test binary embeds; they need no package but the standard library.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::wrap_pyfunction;

use crate::util::testing::{define, entry, evaluate, with_framework};

use super::*;

/// A frozen class of two numbers, which compares as a dataclass.
#[pyclass(frozen, subclass, module = "fhy_core_util_tests")]
struct Pair {
    left: i32,
    right: i32,
}

#[pymethods]
impl Pair {
    #[new]
    fn new(left: i32, right: i32) -> Self {
        Self { left, right }
    }

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |a, b| a.left == b.left && a.right == b.right)
    }

    fn compare_failing<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |_a, _b| -> PyResult<bool> {
            Err(PyValueError::new_err("cannot compare"))
        })
    }
}

/// Take an argument that may be omitted, and describe it.
#[pyfunction]
#[pyo3(signature = (value = OptionalArgument::Omitted))]
fn describe(value: OptionalArgument<'_>) -> PyResult<String> {
    Ok(match value {
        OptionalArgument::Omitted => "omitted".to_owned(),
        OptionalArgument::Given(object) => format!("given {}", object.repr()?),
    })
}

fn pair(py: Python<'_>, left: i32, right: i32) -> Bound<'_, Pair> {
    Bound::new(py, Pair::new(left, right)).expect("a pair")
}

#[test]
fn an_argument_type_error_names_the_owner_field_and_type_found() {
    with_framework(|py| {
        let value = evaluate(py, "3.5");

        let error = build_argument_type_error("Span", "start", "an int", &value).expect("builds");

        assert!(error.is_instance_of::<PyTypeError>(py));
        assert_eq!(
            error.value(py).to_string(),
            "Span start must be an int, got float."
        );
    });
}

#[test]
fn read_str_returns_a_str_and_refuses_anything_else() {
    with_framework(|py| {
        let text = PyString::new(py, "x").into_any();
        assert_eq!(
            read_str(&text, "Note", "text")
                .expect("a str")
                .to_str()
                .unwrap(),
            "x"
        );

        let number = evaluate(py, "3");
        let error = read_str(&number, "Note", "text").expect_err("not a str");
        assert!(error.is_instance_of::<PyTypeError>(py));
        assert_eq!(
            error.value(py).to_string(),
            "Note text must be a str, got int."
        );
    });
}

#[test]
fn equal_values_hash_equally_and_different_ones_differ() {
    assert_eq!(hash_value(&("a", 1_u8)), hash_value(&("a", 1_u8)));
    assert_ne!(hash_value(&("a", 1_u8)), hash_value(&("a", 2_u8)));
}

#[test]
fn a_dataclass_compares_by_value_within_its_own_class() {
    with_framework(|py| {
        let a = pair(py, 1, 2);
        let same = pair(py, 1, 2);
        let other = pair(py, 1, 3);

        let equal = Pair::__eq__(&a, same.as_any()).expect("compares");
        let unequal = Pair::__eq__(&a, other.as_any()).expect("compares");

        assert!(equal.is_truthy().unwrap() && equal.is_instance_of::<pyo3::types::PyBool>());
        assert!(!unequal.is_truthy().unwrap());
    });
}

#[test]
fn another_class_is_not_implemented_even_a_subclass() {
    with_framework(|py| {
        let a = pair(py, 1, 2);
        let namespace = define(py, "");
        namespace
            .set_item("Pair", py.get_type::<Pair>())
            .expect("set");
        py.run(
            c"class Sub(Pair):\n    pass\nsub = Sub(1, 2)",
            Some(&namespace),
            None,
        )
        .expect("subclass");
        let sub = entry(&namespace, "sub");

        for other in [sub, evaluate(py, "3"), py.None().into_bound(py)] {
            let result = Pair::__eq__(&a, &other).expect("compares");
            assert!(result.is(py.NotImplemented()));
        }
    });
}

#[test]
fn a_comparison_that_raises_is_the_error() {
    with_framework(|py| {
        let a = pair(py, 1, 2);

        let error = Pair::compare_failing(&a, pair(py, 1, 2).as_any()).expect_err("raises");
        assert!(error.is_instance_of::<PyValueError>(py));

        let skipped = Pair::compare_failing(&a, &evaluate(py, "1")).expect("not the same class");
        assert!(skipped.is(py.NotImplemented()));
    });
}

#[test]
fn an_outcome_is_a_value_or_the_exception() {
    with_framework(|py| {
        assert!(Outcome::<bool>::into_result(true).unwrap());
        assert_eq!(Outcome::<u64>::into_result(5_u64).unwrap(), 5);
        assert_eq!(Outcome::into_result(Ok::<_, PyErr>(7_u8)).unwrap(), 7);
        let error = Outcome::<u8>::into_result(Err(PyValueError::new_err("no"))).expect_err("err");
        assert!(error.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn fields_are_the_same_when_one_object_or_equal() {
    with_framework(|py| {
        let nan = evaluate(py, "float('nan')");
        assert!(is_same_or_equal(&nan, &nan).unwrap());
        assert!(!nan.eq(&nan).unwrap());
        assert!(is_same_or_equal(&evaluate(py, "[1, 2]"), &evaluate(py, "[1, 2]")).unwrap());
        assert!(!is_same_or_equal(&evaluate(py, "[1, 2]"), &evaluate(py, "[1, 3]")).unwrap());
    });
}

#[test]
fn a_field_whose_equality_raises_is_the_error() {
    with_framework(|py| {
        let namespace = define(
            py,
            "class Angry:\n    def __eq__(self, other):\n        raise ValueError('angry')\na = Angry()\nb = Angry()",
        );

        let error =
            is_same_or_equal(&entry(&namespace, "a"), &entry(&namespace, "b")).expect_err("raises");
        assert!(error.is_instance_of::<PyValueError>(py));
        assert!(is_same_or_equal(&entry(&namespace, "a"), &entry(&namespace, "a")).unwrap());
    });
}

#[test]
fn a_tuple_is_returned_as_it_is_and_anything_else_is_collected() {
    with_framework(|py| {
        let tuple = evaluate(py, "(1, 2)");
        assert!(collect_tuple(&tuple).unwrap().is(&tuple));

        let list = PyList::new(py, [1, 2, 3]).unwrap().into_any();
        assert_eq!(collect_tuple(&list).unwrap().to_string(), "(1, 2, 3)");
        let generated = evaluate(py, "(x * x for x in range(3))");
        assert_eq!(collect_tuple(&generated).unwrap().to_string(), "(0, 1, 4)");
        let subclass = evaluate(py, "type('T', (tuple,), {})((5,))");
        let collected = collect_tuple(&subclass).unwrap();
        assert!(
            collected.is_exact_instance_of::<pyo3::types::PyTuple>() && !collected.is(&subclass)
        );
    });
}

#[test]
fn collecting_what_is_not_iterable_or_fails_while_iterating_raises() {
    with_framework(|py| {
        let error = collect_tuple(&evaluate(py, "3")).expect_err("not iterable");
        assert!(error.is_instance_of::<PyTypeError>(py));

        let failing = evaluate(py, "(1 // 0 for _ in range(1))");
        let error = collect_tuple(&failing).expect_err("fails while iterating");
        assert!(error.is_instance_of::<pyo3::exceptions::PyZeroDivisionError>(py));
    });
}

#[test]
fn a_repr_lists_the_fields_as_a_dataclass_does() {
    with_framework(|py| {
        let class = evaluate(py, "type('Point', (), {})")
            .cast_into::<pyo3::types::PyType>()
            .unwrap();
        let x = evaluate(py, "1");
        let y = evaluate(py, "'a'");

        assert_eq!(
            format_dataclass_repr(&class, &[("x", &x), ("y", &y)]).unwrap(),
            "Point(x=1, y='a')"
        );
        assert_eq!(format_dataclass_repr(&class, &[]).unwrap(), "Point()");
    });
}

#[test]
fn an_argument_that_may_be_omitted_tells_none_from_omitted() {
    with_framework(|py| {
        let function = wrap_pyfunction!(describe, py).expect("function");
        let namespace = PyDict::new(py);
        namespace.set_item("describe", function).expect("set");
        let run = |source: &str| {
            py.eval(
                &std::ffi::CString::new(source).unwrap(),
                Some(&namespace),
                None,
            )
            .unwrap()
            .to_string()
        };

        assert_eq!(run("describe()"), "omitted");
        assert_eq!(run("describe(None)"), "given None");
        assert_eq!(run("describe(0)"), "given 0");
    });
}
