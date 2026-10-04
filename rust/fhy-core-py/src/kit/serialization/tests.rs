//! The stories of the serialization helpers of the kit, run in an
//! interpreter this test binary embeds, against the stand-in of
//! `fhy_core.serialization`: the structure of the framework's exceptions and
//! the payload check are the stand-in's, and the Python suite covers the
//! behavior against the real framework.

use pyo3::exceptions::{
    PyAttributeError, PyKeyError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};

use crate::kit::exceptions::{
    DESERIALIZATION_DICT_STRUCTURE_ERROR, DESERIALIZATION_VALUE_ERROR, SERIALIZATION_ERROR,
};
use crate::kit::testing::{define, entry, evaluate, with_framework};

use super::*;

/// Return a new class named `Thing`.
fn thing(py: Python<'_>) -> Bound<'_, PyType> {
    evaluate(py, "type('Thing', (), {})")
        .cast_into::<PyType>()
        .expect("a class")
}

/// Return the structure error `read_payload_fields` raised, as its
/// `(expected, data)` arguments.
fn structure_error<'py>(
    py: Python<'py>,
    error: &PyErr,
    cls: &Bound<'py, PyType>,
) -> (String, Bound<'py, PyAny>) {
    assert!(DESERIALIZATION_DICT_STRUCTURE_ERROR.is_instance_of(py, error));
    let value = error.value(py);
    assert!(value.getattr("cls").unwrap().is(cls));
    (
        value.getattr("expected").unwrap().to_string(),
        value.getattr("data").unwrap(),
    )
}

/// What each shape accepts and refuses, and the type it names when refusing.
const SHAPES: &[(FieldShape, &[&str], &[&str], &str)] = &[
    (
        FieldShape::Payload,
        &["{'a': 1}", "{}"],
        &["[]", "None", "{1: 2}", "'x'"],
        "<class 'dict'>",
    ),
    (
        FieldShape::Str,
        &["'x'", "''"],
        &["1", "None", "b'x'"],
        "<class 'str'>",
    ),
    (
        FieldShape::Int,
        &["0", "-3", "2 ** 70"],
        &["True", "1.0", "None", "'1'"],
        "<class 'int'>",
    ),
    (
        FieldShape::OptionalPayload,
        &["None", "{'a': 1}"],
        &["[]", "{1: 2}", "1"],
        "dict | None",
    ),
    (
        FieldShape::OptionalStr,
        &["None", "'x'"],
        &["1", "[]"],
        "str | None",
    ),
    (
        FieldShape::OptionalInt,
        &["None", "4"],
        &["False", "'x'", "1.5"],
        "int | None",
    ),
    (
        FieldShape::PayloadList,
        &["[]", "[{'a': 1}, {}]"],
        &["None", "[1]", "[{1: 2}]", "({},)", "{}"],
        "<class 'list'>",
    ),
    (
        FieldShape::Literal,
        &["'x'", "1.5", "3", "True"],
        &["None", "[]", "b'x'"],
        "str | float | int | bool",
    ),
    (
        FieldShape::OptionalIntList,
        &["None", "[]", "[1, 2]"],
        &["[True]", "[1.5]", "(1,)", "1"],
        "list[int] | None",
    ),
    (
        FieldShape::Bool,
        &["True", "False"],
        &["0", "None", "'x'"],
        "<class 'bool'>",
    ),
    (
        FieldShape::IntList,
        &["[]", "[1, 2]"],
        &["None", "[True]", "(1,)", "[1, 'a']"],
        "<class 'list'>",
    ),
    (
        FieldShape::Object,
        &[
            "{}",
            "{1: 2}",
            "__import__('types').MappingProxyType({'a': 1})",
        ],
        &["[]", "None", "'x'"],
        "<class 'dict'>",
    ),
    (
        FieldShape::OptionalObject,
        &["None", "{}", "{1: 2}"],
        &["[]", "1"],
        "dict | None",
    ),
    (
        FieldShape::List,
        &["[]", "[1, 'a', None]"],
        &["None", "(1,)", "{}"],
        "<class 'list'>",
    ),
    (
        FieldShape::ObjectList,
        &["[]", "[{}, {1: 2}]"],
        &["None", "[1]", "[{}, None]", "({},)"],
        "<class 'list'>",
    ),
    (
        FieldShape::ObjectMap,
        &["{}", "{'a': {}, 'b': {1: 2}}"],
        &["[]", "{'a': 1}", "{'a': {}, 'b': []}", "None"],
        "<class 'dict'>",
    ),
    (
        FieldShape::IntListList,
        &["[]", "[[], [1, 2]]"],
        &["[1]", "[[True]]", "None", "[(1,)]"],
        "<class 'list'>",
    ),
    (
        FieldShape::ObjectListList,
        &["[]", "[[], [{}, {1: 2}]]"],
        &["[{}]", "[[1]]", "None", "[({},)]"],
        "<class 'list'>",
    ),
    (
        FieldShape::Any,
        &["None", "1", "[]", "object()"],
        &[],
        "<class 'object'>",
    ),
];

#[test]
fn every_shape_accepts_what_it_describes_and_refuses_the_rest() {
    with_framework(|py| {
        let cls = thing(py);
        for (shape, accepted, refused, _expected) in SHAPES {
            for source in *accepted {
                let data = define(py, &format!("data = {{'field': {source}}}"));
                let data = entry(&data, "data");
                let [value] = read_payload_fields(&cls, &data, [("field", *shape)])
                    .unwrap_or_else(|error| panic!("{shape:?} refused {source}: {error}"));
                assert!(value.is(data.get_item("field").unwrap()));
            }
            for source in *refused {
                let data = define(py, &format!("data = {{'field': {source}}}"));
                let data = entry(&data, "data");
                let error = read_payload_fields(&cls, &data, [("field", *shape)])
                    .expect_err(&format!("{shape:?} accepted {source}"));
                structure_error(py, &error, &cls);
            }
        }
    });
}

#[test]
fn a_refused_payload_names_the_expected_type_of_each_shape() {
    with_framework(|py| {
        let cls = thing(py);
        // `None` is refused by every shape but the optional ones and `Any`.
        let data = evaluate(py, "'not a mapping'");
        for (shape, _accepted, _refused, expected) in SHAPES {
            let error = read_payload_fields(&cls, &data, [("field", *shape)]).expect_err("refused");
            let (named, seen) = structure_error(py, &error, &cls);
            // The `int` list's own spelling differs by Python version.
            let wanted = format!("{{'field': {expected}}}");
            assert!(
                named == wanted
                    || (*shape == FieldShape::OptionalIntList && named.contains("| None")),
                "{shape:?} named {named}, wanted {wanted}"
            );
            assert!(seen.is(&data));
        }
    });
}

#[test]
fn the_values_come_back_in_the_order_of_the_fields_not_the_payload() {
    with_framework(|py| {
        let cls = thing(py);
        let data = evaluate(py, "{'b': 2, 'a': 'x'}");

        let [a, b] = read_payload_fields(
            &cls,
            &data,
            [("a", FieldShape::Str), ("b", FieldShape::Int)],
        )
        .expect("reads");

        assert_eq!(a.to_string(), "x");
        assert_eq!(b.to_string(), "2");
    });
}

#[test]
fn a_missing_extra_or_misshapen_field_is_a_structure_error_naming_the_expected_fields() {
    with_framework(|py| {
        let cls = thing(py);
        let fields = [("a", FieldShape::Str), ("b", FieldShape::OptionalInt)];
        for source in [
            "{'a': 'x'}",
            "{'a': 'x', 'b': None, 'c': 1}",
            "{'a': 1, 'b': None}",
            "[]",
            "None",
        ] {
            let data = evaluate(py, source);

            let error = read_payload_fields(&cls, &data, fields).expect_err("refused");

            let (expected, seen) = structure_error(py, &error, &cls);
            assert_eq!(expected, "{'a': <class 'str'>, 'b': int | None}");
            assert!(seen.is(&data));
        }
    });
}

#[test]
fn a_payload_that_allows_extra_keys_ignores_them_but_still_checks_its_own() {
    with_framework(|py| {
        let cls = thing(py);
        let fields =
            PayloadFields::allowing_extra([("a", FieldShape::Str), ("b", FieldShape::Int)]);

        let data = evaluate(py, "{'a': 'x', 'b': 1, 'extra': [1], 'more': None}");
        let [a, b] = read_payload_fields(&cls, &data, fields).expect("extra keys are tolerated");
        assert_eq!(
            (a.to_string(), b.to_string()),
            ("x".to_owned(), "1".to_owned())
        );

        for source in [
            "{'a': 'x', 'extra': 1}",
            "{'a': 1, 'b': 1, 'extra': 1}",
            "[1]",
        ] {
            let error =
                read_payload_fields(&cls, &evaluate(py, source), fields).expect_err("refused");
            structure_error(py, &error, &cls);
        }
    });
}

#[test]
fn an_exact_payload_refuses_what_the_array_form_refuses() {
    with_framework(|py| {
        let cls = thing(py);
        let data = evaluate(py, "{'a': 'x', 'extra': 1}");
        let exact = PayloadFields::exact([("a", FieldShape::Str)]);

        read_payload_fields(&cls, &data, exact).expect_err("extra key");
        read_payload_fields(&cls, &data, [("a", FieldShape::Str)]).expect_err("extra key");
        read_payload_fields(
            &cls,
            &data,
            PayloadFields::allowing_extra([("a", FieldShape::Str)]),
        )
        .expect("tolerated");
    });
}

#[test]
fn a_payload_of_no_fields_is_an_empty_mapping() {
    with_framework(|py| {
        let cls = thing(py);

        let [] = read_payload_fields(&cls, &evaluate(py, "{}"), []).expect("empty");
        read_payload_fields(&cls, &evaluate(py, "{'a': 1}"), []).expect_err("not empty");
        let [] = read_payload_fields(
            &cls,
            &evaluate(py, "{'a': 1}"),
            PayloadFields::allowing_extra([]),
        )
        .expect("extra tolerated");
    });
}

#[test]
fn the_constructor_fields_come_back_in_order() {
    with_framework(|py| {
        let cls = thing(py);
        let fields = evaluate(py, "{'b': 2, 'a': 1}");

        let [a, b] = read_constructor_fields(&cls, &fields, ["a", "b"], 0).expect("reads");

        assert_eq!(
            (a.to_string(), b.to_string()),
            ("1".to_owned(), "2".to_owned())
        );
    });
}

#[test]
fn the_constructor_fields_refuse_a_missing_extra_or_non_mapping_input_as_a_type_error() {
    with_framework(|py| {
        let cls = thing(py);

        let error = read_constructor_fields(&cls, &evaluate(py, "{'a': 1}"), ["a", "b"], 0)
            .expect_err("missing");
        assert!(error.is_instance_of::<PyTypeError>(py));
        assert_eq!(
            error.value(py).to_string(),
            "Thing.construct_from_fields() missing field 'b'"
        );

        let error = read_constructor_fields(
            &cls,
            &evaluate(py, "{'a': 1, 'b': 2, 'c': 3}"),
            ["a", "b"],
            0,
        )
        .expect_err("extra");
        assert!(error.is_instance_of::<PyTypeError>(py));
        assert!(
            error
                .value(py)
                .to_string()
                .starts_with("Thing.construct_from_fields() takes only the fields 'a', 'b', got ")
        );

        let error = read_constructor_fields(&cls, &evaluate(py, "[1]"), ["a"], 0)
            .expect_err("not a mapping");
        assert!(error.is_instance_of::<PyTypeError>(py));
    });
}

#[test]
fn the_last_optional_constructor_fields_may_be_missing_and_read_as_none() {
    with_framework(|py| {
        let cls = thing(py);

        let [a, b, c] =
            read_constructor_fields(&cls, &evaluate(py, "{'a': 1}"), ["a", "b", "c"], 2)
                .expect("reads");
        assert_eq!(a.to_string(), "1");
        assert!(b.is_none() && c.is_none());

        let [_a, b, c] =
            read_constructor_fields(&cls, &evaluate(py, "{'a': 1, 'b': 2}"), ["a", "b", "c"], 2)
                .expect("reads");
        assert_eq!(b.to_string(), "2");
        assert!(c.is_none());

        let error = read_constructor_fields(&cls, &evaluate(py, "{'b': 2}"), ["a", "b", "c"], 2)
            .expect_err("a is required");
        assert!(error.is_instance_of::<PyTypeError>(py));
    });
}

#[test]
fn a_mapping_whose_lookup_fails_for_another_reason_raises_that() {
    with_framework(|py| {
        let namespace = define(
            py,
            "import collections.abc\nclass Broken(collections.abc.Mapping):\n    def __getitem__(self, key):\n        raise RuntimeError('broken')\n    def __iter__(self):\n        return iter(())\n    def __len__(self):\n        return 0\nbroken = Broken()",
        );
        let cls = thing(py);

        let error = read_constructor_fields(&cls, &entry(&namespace, "broken"), ["a"], 0)
            .expect_err("raises");

        assert!(error.is_instance_of::<PyRuntimeError>(py));
    });
}

#[test]
fn keeping_fields_returns_them_unchanged() {
    with_framework(|py| {
        let values = [evaluate(py, "1"), evaluate(py, "[]")];

        let kept = keep_fields(&thing(py), values.clone()).expect("never fails");

        assert!(kept[0].is(&values[0]) && kept[1].is(&values[1]));
    });
}

#[test]
fn a_nested_value_is_read_by_its_class() {
    with_framework(|py| {
        let namespace = define(
            py,
            "class Leaf:\n    def __init__(self, data):\n        self.data = data\n    @classmethod\n    def deserialize_from_dict(cls, data):\n        if data is None:\n            raise ValueError('no data')\n        return cls(data)\n",
        );
        let leaf = entry(&namespace, "Leaf").cast_into::<PyType>().unwrap();
        let payload = evaluate(py, "{'x': 1}");

        let value = read_nested_value(&leaf, &payload).expect("reads");
        assert!(value.getattr("data").unwrap().is(&payload));

        let list = read_nested_list(&leaf, &evaluate(py, "[{'x': 1}, {'x': 2}]")).expect("reads");
        assert_eq!(list.len(), 2);
        assert_eq!(
            list.get_item(1)
                .unwrap()
                .getattr("data")
                .unwrap()
                .to_string(),
            "{'x': 2}"
        );

        let error = read_nested_value(&leaf, &py.None().into_bound(py)).expect_err("refused");
        assert!(error.is_instance_of::<PyValueError>(py));
        let error =
            read_nested_list(&leaf, &evaluate(py, "[{'x': 1}, None]")).expect_err("refused");
        assert!(error.is_instance_of::<PyValueError>(py));
        let error = read_nested_list(&leaf, &evaluate(py, "3")).expect_err("not iterable");
        assert!(error.is_instance_of::<PyTypeError>(py));
        assert_eq!(
            read_nested_list(&leaf, &evaluate(py, "[]"))
                .expect("empty")
                .len(),
            0
        );
    });
}

/// A class whose `construct_from_fields` raises what `fields['raise']` names.
const CONSTRUCTOR: &str = "
import fhy_core.serialization as framework

class Built:
    def __init__(self, fields):
        self.fields = fields

    @classmethod
    def construct_from_fields(cls, fields):
        kind = fields.get('raise')
        if kind == 'value':
            raise ValueError('bad value')
        if kind == 'type':
            raise TypeError('bad type')
        if kind == 'overflow':
            raise OverflowError('too large')
        if kind == 'key':
            raise KeyError('missing')
        if kind == 'serialization':
            raise framework.DeserializationValueError('already framed')
        if kind == 'base':
            raise framework.SerializationError('framework')
        return cls(fields)
";

#[test]
fn a_construction_that_succeeds_returns_the_instance() {
    with_framework(|py| {
        let namespace = define(py, CONSTRUCTOR);
        let built = entry(&namespace, "Built").cast_into::<PyType>().unwrap();
        let fields = PyDict::new(py);

        for construct in [
            construct_from_decoded_fields,
            construct_from_decoded_fields_reporting_overflow,
        ] {
            let instance = construct(&built, &fields).expect("constructs");
            assert!(instance.getattr("fields").unwrap().is(&fields));
        }
    });
}

#[test]
fn a_refused_value_becomes_a_deserialization_value_error_caused_by_it() {
    with_framework(|py| {
        let namespace = define(py, CONSTRUCTOR);
        let built = entry(&namespace, "Built").cast_into::<PyType>().unwrap();

        for (kind, message, class) in [
            ("value", "bad value", "ValueError"),
            ("type", "bad type", "TypeError"),
        ] {
            for construct in [
                construct_from_decoded_fields,
                construct_from_decoded_fields_reporting_overflow,
            ] {
                let fields = PyDict::new(py);
                fields.set_item("raise", kind).unwrap();

                let error = construct(&built, &fields).expect_err("refused");

                assert!(DESERIALIZATION_VALUE_ERROR.is_instance_of(py, &error));
                assert_eq!(error.value(py).to_string(), message);
                let cause = error.cause(py).expect("caused by the refusal");
                assert_eq!(cause.get_type(py).name().unwrap().to_string(), class);
            }
        }
    });
}

#[test]
fn an_overflow_is_a_value_error_only_when_asked() {
    with_framework(|py| {
        let namespace = define(py, CONSTRUCTOR);
        let built = entry(&namespace, "Built").cast_into::<PyType>().unwrap();
        let fields = PyDict::new(py);
        fields.set_item("raise", "overflow").unwrap();

        let plain = construct_from_decoded_fields(&built, &fields).expect_err("raises");
        assert!(plain.is_instance_of::<PyOverflowError>(py));
        assert!(!DESERIALIZATION_VALUE_ERROR.is_instance_of(py, &plain));

        let reported =
            construct_from_decoded_fields_reporting_overflow(&built, &fields).expect_err("raises");
        assert!(DESERIALIZATION_VALUE_ERROR.is_instance_of(py, &reported));
        assert_eq!(reported.value(py).to_string(), "too large");
        assert!(
            reported
                .cause(py)
                .expect("cause")
                .is_instance_of::<PyOverflowError>(py)
        );
    });
}

#[test]
fn any_other_exception_passes_through_unchanged() {
    with_framework(|py| {
        let namespace = define(py, CONSTRUCTOR);
        let built = entry(&namespace, "Built").cast_into::<PyType>().unwrap();

        for construct in [
            construct_from_decoded_fields,
            construct_from_decoded_fields_reporting_overflow,
        ] {
            let fields = PyDict::new(py);
            fields.set_item("raise", "key").unwrap();
            let error = construct(&built, &fields).expect_err("raises");
            assert!(error.is_instance_of::<PyKeyError>(py));

            fields.set_item("raise", "serialization").unwrap();
            let error = construct(&built, &fields).expect_err("raises");
            assert!(DESERIALIZATION_VALUE_ERROR.is_instance_of(py, &error));
            assert_eq!(error.value(py).to_string(), "already framed");
            assert!(error.cause(py).is_none());

            fields.set_item("raise", "base").unwrap();
            let error = construct(&built, &fields).expect_err("raises");
            assert!(SERIALIZATION_ERROR.is_instance_of(py, &error));
            assert!(!DESERIALIZATION_VALUE_ERROR.is_instance_of(py, &error));
        }
    });
}

#[test]
fn a_nested_value_is_serialized_by_its_own_method_and_none_stays_none() {
    with_framework(|py| {
        let namespace = define(
            py,
            "class Leaf:\n    def serialize_to_dict(self):\n        return {'leaf': 1}\nleaf = Leaf()",
        );

        let payload = serialize_nested(&entry(&namespace, "leaf")).expect("serializes");
        assert_eq!(payload.to_string(), "{'leaf': 1}");

        let none = py.None().into_bound(py);
        assert!(serialize_nested(&none).expect("none").is_none());
        let error = serialize_nested(&evaluate(py, "3")).expect_err("no method");
        assert!(error.is_instance_of::<PyAttributeError>(py));
    });
}

#[test]
fn a_payload_dict_is_what_the_framework_says() {
    with_framework(|py| {
        assert!(is_serialized_dict(&evaluate(py, "{'a': 1}")).unwrap());
        assert!(!is_serialized_dict(&evaluate(py, "{1: 2}")).unwrap());
        assert!(!is_serialized_dict(&evaluate(py, "[]")).unwrap());
    });
}
