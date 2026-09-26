//! Tests for `CoreDataType` and `TypeQualifier`: widths, families,
//! promotion against the promotion lattices, literal resolution, text and
//! serde.
//!
//! Ported from the promotion, literal and qualifier tests of
//! `tests/types/test_core.py`; the traceability table is in
//! `docs/design/python-switch.md`, "S11a.2 implementation notes".

use fhy_core::expression::{BigInt, Decimal, LiteralValue};
use fhy_core::lattice::Lattice;
use fhy_core::types::{
    CoreDataType, IntegerFamily, LiteralTypeError, PromotionError, TypeQualifier,
};
use rstest::rstest;

use CoreDataType::{
    Bool, Complex32, Complex64, Complex128, Float, Float16, Float32, Float64, Int, Int8, Int16,
    Int32, Int64, Uint, Uint8, Uint16, Uint32,
};

/// The integer promotion order as a lattice, built from its covering pairs.
fn build_integer_lattice() -> Lattice<CoreDataType> {
    build_lattice(
        &[Uint, Uint8, Uint16, Uint32, Int, Int8, Int16, Int32, Int64],
        &[
            (Uint, Uint8),
            (Uint8, Uint16),
            (Uint16, Uint32),
            (Int, Int8),
            (Int8, Int16),
            (Int16, Int32),
            (Int32, Int64),
            (Uint, Int),
            (Uint8, Int16),
            (Uint16, Int32),
            (Uint32, Int64),
        ],
    )
}

/// The float and complex promotion order as a lattice.
fn build_float_complex_lattice() -> Lattice<CoreDataType> {
    build_lattice(
        &[
            Float, Float16, Float32, Float64, Complex32, Complex64, Complex128,
        ],
        &[
            (Float, Float16),
            (Float16, Float32),
            (Float32, Float64),
            (Float16, Complex32),
            (Float32, Complex64),
            (Float64, Complex128),
            (Complex32, Complex64),
            (Complex64, Complex128),
        ],
    )
}

fn build_lattice(
    elements: &[CoreDataType],
    orders: &[(CoreDataType, CoreDataType)],
) -> Lattice<CoreDataType> {
    let mut lattice = Lattice::new();
    for &element in elements {
        lattice.add_element(element).expect("each type is new");
    }
    for (lower, upper) in orders {
        lattice
            .add_order(lower, upper)
            .expect("the order is acyclic");
    }
    lattice
}

#[rstest]
#[case(Uint, None)]
#[case(Int, None)]
#[case(Float, None)]
#[case(Uint8, Some(8))]
#[case(Uint16, Some(16))]
#[case(Uint32, Some(32))]
#[case(Int8, Some(8))]
#[case(Int16, Some(16))]
#[case(Int32, Some(32))]
#[case(Int64, Some(64))]
#[case(Float16, Some(16))]
#[case(Float32, Some(32))]
#[case(Float64, Some(64))]
#[case(Complex32, Some(32))]
#[case(Complex64, Some(64))]
#[case(Complex128, Some(128))]
#[case(Bool, Some(1))]
fn bit_width_of_each_core_data_type(#[case] data_type: CoreDataType, #[case] width: Option<u32>) {
    assert_eq!(data_type.bit_width(), width);
}

#[test]
fn only_the_three_literal_families_are_weak() {
    let weak: Vec<CoreDataType> = CoreDataType::all()
        .filter(|data_type| data_type.is_weak())
        .collect();

    assert_eq!(weak, [Uint, Int, Float]);
}

#[test]
fn every_type_but_bool_belongs_to_exactly_one_promotion_family() {
    for data_type in CoreDataType::all() {
        let families =
            usize::from(data_type.is_integral()) + usize::from(data_type.is_float_like());
        let expected = usize::from(data_type != Bool);
        assert_eq!(families, expected, "{data_type}");
    }
    assert_eq!(CoreDataType::all().len(), 17);
}

#[test]
fn family_predicates_partition_the_numbers() {
    for data_type in CoreDataType::all() {
        assert_eq!(
            data_type.is_integral(),
            data_type.is_unsigned() || data_type.is_signed()
        );
        assert!(!(data_type.is_unsigned() && data_type.is_signed()));
        assert_eq!(
            data_type.is_float_like(),
            data_type.is_real_float() || data_type.is_complex()
        );
    }
    assert!(
        Uint.is_unsigned() && Int.is_signed() && Float.is_real_float() && Complex64.is_complex()
    );
}

#[rstest]
#[case(Uint8, Uint8, Uint8)]
#[case(Uint8, Uint16, Uint16)]
#[case(Uint16, Uint8, Uint16)]
#[case(Uint, Uint8, Uint8)]
#[case(Int, Int16, Int16)]
#[case(Float, Float16, Float16)]
#[case(Int32, Int64, Int64)]
#[case(Float16, Float32, Float32)]
#[case(Float64, Float16, Float64)]
#[case(Complex32, Complex64, Complex64)]
#[case(Float32, Complex32, Complex64)]
#[case(Uint, Int, Int)]
#[case(Int, Uint, Int)]
#[case(Uint8, Int8, Int16)]
#[case(Uint16, Int16, Int32)]
#[case(Uint32, Int32, Int64)]
#[case(Uint, Int32, Int32)]
#[case(Uint16, Int8, Int32)]
#[case(Uint32, Int8, Int64)]
#[case(Bool, Bool, Bool)]
fn promotion_joins_two_types_of_one_family(
    #[case] left: CoreDataType,
    #[case] right: CoreDataType,
    #[case] promoted: CoreDataType,
) {
    assert_eq!(left.promote(right), Ok(promoted));
}

#[rstest]
#[case(Int32, Float32)]
#[case(Uint8, Complex64)]
#[case(Float, Int)]
fn promotion_across_families_is_refused(#[case] left: CoreDataType, #[case] right: CoreDataType) {
    assert_eq!(
        left.promote(right),
        Err(PromotionError::AcrossFamilies(left, right))
    );
}

#[rstest]
#[case(Bool, Int32)]
#[case(Uint8, Bool)]
#[case(Bool, Float64)]
#[case(Complex64, Bool)]
#[case(Bool, Uint)]
fn promotion_of_bool_with_any_other_type_is_refused(
    #[case] left: CoreDataType,
    #[case] right: CoreDataType,
) {
    assert_eq!(
        left.promote(right),
        Err(PromotionError::Boolean(left, right))
    );
}

#[test]
fn both_promotion_orders_are_lattices_whose_joins_are_the_promotions() {
    for lattice in [build_integer_lattice(), build_float_complex_lattice()] {
        assert!(lattice.is_lattice());
        for left in lattice.poset().iter() {
            for right in lattice.poset().iter() {
                let join = lattice.join(left, right).expect("members").copied();
                assert_eq!(left.promote(*right).ok(), join, "{left} and {right}");
            }
        }
    }
}

#[test]
fn promotion_errors_display_one_lowercase_line() {
    assert_eq!(
        PromotionError::AcrossFamilies(Int32, Float32).to_string(),
        "unsupported primitive data type promotion: int32, float32"
    );
    assert_eq!(
        PromotionError::Boolean(Bool, Int8).to_string(),
        "unsupported primitive data type promotion involving boolean: bool, int8"
    );
}

#[rstest]
#[case(LiteralValue::from(0), Uint, Uint8)]
#[case(LiteralValue::from(255), Uint, Uint8)]
#[case(LiteralValue::from(256), Uint, Uint16)]
#[case(LiteralValue::from(1), Int32, Int32)]
#[case(LiteralValue::from(1), Float32, Float32)]
#[case(LiteralValue::from(1), Complex64, Complex64)]
#[case(LiteralValue::from(255), Int8, Int16)]
#[case(LiteralValue::from(-1), Int, Int8)]
#[case(LiteralValue::from(-129), Int, Int16)]
#[case(LiteralValue::from(-1), Float64, Float64)]
#[case(LiteralValue::from(1.5), Float, Float64)]
#[case(LiteralValue::from(std::f64::consts::PI), Float, Float64)]
#[case(LiteralValue::from(2_i64.pow(31)), Uint32, Uint32)]
#[case(LiteralValue::from(1.5), Complex128, Complex128)]
#[case(LiteralValue::from(i64::MIN), Int, Int64)]
#[case(LiteralValue::from(u32::MAX), Uint, Uint32)]
#[case(LiteralValue::from(true), Bool, Bool)]
#[case(LiteralValue::from(false), Bool, Bool)]
fn literal_resolves_to_its_narrowest_type_in_the_context(
    #[case] literal: LiteralValue,
    #[case] context: CoreDataType,
    #[case] resolved: CoreDataType,
) {
    assert_eq!(
        CoreDataType::resolve_literal(&literal, context),
        Ok(resolved)
    );
}

#[rstest]
#[case(Int32)]
#[case(Uint8)]
#[case(Float64)]
#[case(Complex64)]
fn boolean_literal_is_refused_outside_the_bool_context(#[case] context: CoreDataType) {
    assert_eq!(
        CoreDataType::resolve_literal(&LiteralValue::from(true), context),
        Err(LiteralTypeError::BooleanIncompatible {
            value: true,
            context
        })
    );
}

#[rstest]
#[case(LiteralValue::from(0))]
#[case(LiteralValue::from(1))]
#[case(LiteralValue::from(-1))]
#[case(LiteralValue::from(2.5))]
fn number_is_refused_in_the_bool_context(#[case] literal: LiteralValue) {
    assert_eq!(
        CoreDataType::resolve_literal(&literal, Bool),
        Err(LiteralTypeError::NonBooleanInBooleanContext { literal })
    );
}

#[test]
fn float_literal_is_refused_in_an_integer_context() {
    let literal = LiteralValue::from(1.5);

    let error = CoreDataType::resolve_literal(&literal, Int32).expect_err("a float is no integer");

    assert_eq!(
        error,
        LiteralTypeError::Incompatible {
            literal,
            context: Int32
        }
    );
    assert_eq!(
        error.to_string(),
        "float literal 1.5 is incompatible with int32"
    );
}

#[test]
fn negative_integer_is_refused_in_an_unsigned_context() {
    let literal = LiteralValue::from(-1);

    let error = CoreDataType::resolve_literal(&literal, Uint16).expect_err("uint16 is unsigned");

    assert_eq!(error.to_string(), "literal -1 is incompatible with uint16");
}

#[rstest]
#[case(LiteralValue::from(2_i64.pow(32)), Uint, IntegerFamily::Unsigned)]
#[case(LiteralValue::from(BigInt::from(1) << 63_u32), Int, IntegerFamily::Signed)]
#[case(LiteralValue::from(-(BigInt::from(1) << 63_u32) - 1), Int, IntegerFamily::Signed)]
#[case(LiteralValue::from(BigInt::from(1) << 70_u32), Int, IntegerFamily::Signed)]
fn integer_outside_every_width_of_its_family_is_refused(
    #[case] literal: LiteralValue,
    #[case] context: CoreDataType,
    #[case] family: IntegerFamily,
) {
    let LiteralValue::Int(value) = &literal else {
        unreachable!("an integer literal")
    };
    assert_eq!(
        CoreDataType::resolve_literal(&literal, context),
        Err(LiteralTypeError::OutOfRange {
            value: value.clone(),
            family
        })
    );
}

#[test]
fn out_of_range_error_names_the_family() {
    let error = CoreDataType::resolve_literal(&LiteralValue::from(300_000_000_000_i64), Uint)
        .expect_err("beyond uint32");

    assert_eq!(
        error.to_string(),
        "literal 300000000000 does not fit in a supported uint type"
    );
}

#[test]
fn decimal_literal_has_no_core_data_type_yet() {
    let literal = LiteralValue::Decimal("1.5".parse::<Decimal>().expect("a decimal"));

    let error =
        CoreDataType::resolve_literal(&literal, Float).expect_err("decimals are unsupported");

    assert_eq!(error, LiteralTypeError::UnsupportedDecimal);
    assert!(error.is_unsupported());
    assert_eq!(error.to_string(), "decimal literals are not yet supported");
}

#[test]
fn literal_errors_display_one_lowercase_line() {
    assert_eq!(
        LiteralTypeError::BooleanIncompatible {
            value: true,
            context: Int32
        }
        .to_string(),
        "boolean literal true is incompatible with int32"
    );
    assert_eq!(
        LiteralTypeError::NonBooleanInBooleanContext {
            literal: LiteralValue::from(1)
        }
        .to_string(),
        "non-boolean literal 1 is incompatible with the bool context"
    );
}

#[rstest]
#[case(TypeQualifier::Input, TypeQualifier::Input, TypeQualifier::Temp)]
#[case(TypeQualifier::State, TypeQualifier::Param, TypeQualifier::Temp)]
#[case(TypeQualifier::Param, TypeQualifier::Temp, TypeQualifier::Temp)]
#[case(TypeQualifier::Param, TypeQualifier::Param, TypeQualifier::Param)]
#[case(TypeQualifier::Output, TypeQualifier::Param, TypeQualifier::Temp)]
#[case(TypeQualifier::Output, TypeQualifier::Output, TypeQualifier::Temp)]
fn qualifiers_promote_to_param_only_from_two_params(
    #[case] left: TypeQualifier,
    #[case] right: TypeQualifier,
    #[case] promoted: TypeQualifier,
) {
    assert_eq!(left.promote(right), promoted);
}

#[test]
fn core_data_types_display_and_parse_as_their_names() {
    for data_type in CoreDataType::all() {
        assert_eq!(data_type.to_string().parse::<CoreDataType>(), Ok(data_type));
        assert_eq!(data_type.to_string(), data_type.as_str());
    }
    assert_eq!(Complex128.to_string(), "complex128");
    let _unknown = "INT32"
        .parse::<CoreDataType>()
        .expect_err("names are lowercase");
    assert_eq!(
        "int33"
            .parse::<CoreDataType>()
            .expect_err("no such type")
            .to_string(),
        "unknown core data type `int33`"
    );
}

#[test]
fn qualifiers_display_and_parse_as_their_names() {
    for (qualifier, name) in [
        (TypeQualifier::Input, "input"),
        (TypeQualifier::Output, "output"),
        (TypeQualifier::State, "state"),
        (TypeQualifier::Param, "param"),
        (TypeQualifier::Temp, "temp"),
    ] {
        assert_eq!(qualifier.to_string(), name);
        assert_eq!(name.parse::<TypeQualifier>(), Ok(qualifier));
    }
}

#[test]
fn core_data_types_and_qualifiers_round_trip_through_json_and_postcard() {
    for data_type in CoreDataType::all() {
        let json = serde_json::to_string(&data_type).expect("serializes");
        assert_eq!(json, format!("\"{data_type}\""));
        assert_eq!(
            serde_json::from_str::<CoreDataType>(&json).expect("decodes"),
            data_type
        );
        let bytes = postcard::to_allocvec(&data_type).expect("serializes");
        assert_eq!(
            postcard::from_bytes::<CoreDataType>(&bytes).expect("decodes"),
            data_type
        );
    }
    let bytes = postcard::to_allocvec(&TypeQualifier::State).expect("serializes");
    assert_eq!(
        postcard::from_bytes::<TypeQualifier>(&bytes),
        Ok(TypeQualifier::State)
    );
    assert_eq!(
        serde_json::to_string(&TypeQualifier::Param).expect("serializes"),
        "\"param\""
    );
}
