//! The stories of the unsigned-integer readers of the util module, run in
//! an interpreter this crate's tests embed.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use rstest::rstest;

use super::{
    Label, Minimum, Reading, build_too_large_error, classify_unsigned, read_unsigned,
    read_unsigned_lenient,
};
use crate::util::testing::{define, entry, evaluate, with_stand_ins};

/// Return the message of the exception `error`.
fn message(py: Python<'_>, error: &PyErr) -> String {
    error.value(py).to_string()
}

#[rstest]
#[case("0", 0)]
#[case("7", 7)]
#[case("2**64 - 1", u64::MAX)]
fn an_integer_in_range_is_read_as_the_value(#[case] source: &str, #[case] expected: u64) {
    with_stand_ins(|py| {
        let object = evaluate(py, source);
        let label = Label::argument("count");

        assert_eq!(
            read_unsigned::<u64>(&object, &label, Minimum::NonNegative).unwrap(),
            expected
        );
        assert_eq!(
            classify_unsigned::<u64>(&object, &label).unwrap(),
            Reading::Value(expected)
        );
        assert_eq!(
            read_unsigned_lenient::<u64>(&object, &label).unwrap(),
            Some(expected)
        );
    });
}

#[rstest]
#[case("True")]
#[case("False")]
#[case("2.0")]
#[case("'3'")]
#[case("None")]
fn a_bool_a_float_or_a_non_integer_is_a_type_error(#[case] source: &str) {
    with_stand_ins(|py| {
        let object = evaluate(py, source);
        let label = Label::argument("count");

        let strict = read_unsigned::<u64>(&object, &label, Minimum::NonNegative).unwrap_err();
        let lenient = read_unsigned_lenient::<u64>(&object, &label).unwrap_err();

        for error in [strict, lenient] {
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(message(py, &error).starts_with("count must be an integer, but got "));
        }
    });
}

#[rstest]
#[case(Minimum::Positive, "extent must be a positive integer, but got -1")]
#[case(
    Minimum::NonNegative,
    "extent must be a non-negative integer, but got -1"
)]
fn a_negative_is_a_value_error_worded_by_the_minimum(
    #[case] minimum: Minimum,
    #[case] expected: &str,
) {
    with_stand_ins(|py| {
        let object = evaluate(py, "-1");
        let label = Label::argument("extent");

        let error = read_unsigned::<u64>(&object, &label, minimum).unwrap_err();

        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(message(py, &error), expected);
    });
}

#[test]
fn zero_is_accepted_under_the_non_negative_minimum() {
    with_stand_ins(|py| {
        let zero = evaluate(py, "0");
        let label = Label::argument("extent");

        assert_eq!(
            read_unsigned::<u64>(&zero, &label, Minimum::NonNegative).unwrap(),
            0
        );
    });
}

#[test]
fn zero_is_a_value_error_under_the_positive_minimum() {
    with_stand_ins(|py| {
        let zero = evaluate(py, "0");
        let label = Label::argument("extent");

        let error = read_unsigned::<u64>(&zero, &label, Minimum::Positive).unwrap_err();

        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(
            message(py, &error),
            "extent must be a positive integer, but got 0"
        );
    });
}

#[test]
fn an_int_subclass_is_read_as_its_value_and_worded_by_its_repr() {
    with_stand_ins(|py| {
        let namespace = define(
            py,
            "import enum\n\
             class Rank(enum.IntEnum):\n    ZERO = 0\n    THREE = 3\n\
             class Offset(int):\n    pass\n\
             negative = Offset(-4)\n",
        );
        let label = Label::argument("rank");
        let three = entry(&namespace, "Rank").getattr("THREE").unwrap();
        let zero = entry(&namespace, "Rank").getattr("ZERO").unwrap();
        let negative = entry(&namespace, "negative");

        assert_eq!(
            read_unsigned::<u64>(&three, &label, Minimum::Positive).unwrap(),
            3
        );
        assert_eq!(
            read_unsigned::<u64>(&zero, &label, Minimum::NonNegative).unwrap(),
            0
        );
        assert_eq!(
            message(
                py,
                &read_unsigned::<u64>(&zero, &label, Minimum::Positive).unwrap_err()
            ),
            "rank must be a positive integer, but got <Rank.ZERO: 0>"
        );
        assert_eq!(
            classify_unsigned::<u64>(&negative, &label).unwrap(),
            Reading::Negative
        );
    });
}

#[test]
fn an_integer_above_the_maximum_is_an_overflow_error_naming_the_width() {
    with_stand_ins(|py| {
        let object = evaluate(py, "2**64");
        let label = Label::entry("origin", 1);

        let error = read_unsigned::<u64>(&object, &label, Minimum::NonNegative).unwrap_err();

        assert!(error.is_instance_of::<PyOverflowError>(py));
        assert_eq!(
            message(py, &error),
            "origin[1] is too large: it must fit in 64 bits"
        );
        assert_eq!(
            message(py, &build_too_large_error::<u64>(&label)),
            message(py, &error)
        );
    });
}

#[test]
fn the_lenient_reader_answers_none_for_an_integer_out_of_every_range() {
    with_stand_ins(|py| {
        let label = Label::argument("offset");

        for source in ["-1", "-(2**100)", "2**64", "2**100"] {
            let object = evaluate(py, source);
            assert_eq!(read_unsigned_lenient::<u64>(&object, &label).unwrap(), None);
        }
    });
}

#[test]
fn classify_separates_negative_from_too_large() {
    with_stand_ins(|py| {
        let label = Label::argument("n");

        assert_eq!(
            classify_unsigned::<u64>(&evaluate(py, "-(2**70)"), &label).unwrap(),
            Reading::Negative
        );
        assert_eq!(
            classify_unsigned::<u64>(&evaluate(py, "2**70"), &label).unwrap(),
            Reading::TooLarge
        );
    });
}

#[rstest]
#[case("255", "256", 8)]
fn the_target_type_sets_the_maximum_and_the_width_in_the_message(
    #[case] fits: &str,
    #[case] too_large: &str,
    #[case] bits: u32,
) {
    with_stand_ins(|py| {
        let label = Label::argument("byte");

        assert_eq!(
            read_unsigned::<u8>(&evaluate(py, fits), &label, Minimum::NonNegative).unwrap(),
            255
        );
        let error = read_unsigned::<u8>(&evaluate(py, too_large), &label, Minimum::NonNegative)
            .unwrap_err();
        assert!(error.is_instance_of::<PyOverflowError>(py));
        assert_eq!(
            message(py, &error),
            format!("byte is too large: it must fit in {bits} bits")
        );
    });
}

#[test]
fn the_other_unsigned_targets_read_their_ranges() {
    with_stand_ins(|py| {
        let label = Label::argument("n");
        let minimum = Minimum::NonNegative;

        assert_eq!(
            read_unsigned::<u16>(&evaluate(py, "65535"), &label, minimum).unwrap(),
            65535
        );
        assert_eq!(
            read_unsigned::<u32>(&evaluate(py, "2**32 - 1"), &label, minimum).unwrap(),
            u32::MAX
        );
        assert_eq!(
            read_unsigned::<u128>(&evaluate(py, "2**128 - 1"), &label, minimum).unwrap(),
            u128::MAX
        );
        assert_eq!(
            read_unsigned::<usize>(&evaluate(py, "12"), &label, minimum).unwrap(),
            12
        );
        assert!(
            read_unsigned::<u32>(&evaluate(py, "2**32"), &label, minimum)
                .unwrap_err()
                .is_instance_of::<PyOverflowError>(py)
        );
    });
}

#[rstest]
#[case(Label::argument("shape"), "shape")]
#[case(Label::component("Tile shape axis", 1), "Tile shape axis 1")]
#[case(Label::entry("origin", 2), "origin[2]")]
fn a_label_displays_its_name_and_position(#[case] label: Label<'_>, #[case] expected: &str) {
    assert_eq!(label.to_string(), expected);
}
