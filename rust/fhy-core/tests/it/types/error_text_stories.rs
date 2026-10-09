//! The text and the source of every variant of the `types` errors:
//! promotion, literal resolution, template widths, and binding, substitution
//! and unification.

use fhy_core::expression::{BigInt, Expression, LiteralValue, PiecewiseError};
use fhy_core::types::{
    CoreDataType, DataType, IntegerFamily, LiteralTypeError, PromotionError, TemplateDataType,
    TypeOperation, UnificationError,
};
use rstest::rstest;

use crate::support::error_text::{Source, assert_error_text, fixed, test_error};
use crate::support::foreign::{NamedDataType, NamedType};
use crate::support::types::{array, scalar};

#[rstest]
#[case::boolean(
    PromotionError::Boolean(CoreDataType::Bool, CoreDataType::Int8),
    "unsupported primitive data type promotion involving boolean: bool, int8"
)]
#[case::across_families(
    PromotionError::AcrossFamilies(CoreDataType::Int8, CoreDataType::Float32),
    "unsupported primitive data type promotion: int8, float32"
)]
fn promotion_error_text(#[case] error: PromotionError, #[case] text: &str) {
    assert_error_text(&error, text, Source::None);
}

#[rstest]
#[case::boolean_incompatible(
    LiteralTypeError::BooleanIncompatible { value: true, context: CoreDataType::Int32 },
    "boolean literal true is incompatible with int32"
)]
#[case::non_boolean_in_boolean_context(
    LiteralTypeError::NonBooleanInBooleanContext { literal: LiteralValue::from(3) },
    "non-boolean literal 3 is incompatible with the bool context"
)]
#[case::incompatible_float(
    LiteralTypeError::Incompatible { literal: LiteralValue::from(1.5), context: CoreDataType::Int8 },
    "float literal 1.5 is incompatible with int8"
)]
#[case::incompatible_integer(
    LiteralTypeError::Incompatible { literal: LiteralValue::from(-1), context: CoreDataType::Uint8 },
    "literal -1 is incompatible with uint8"
)]
#[case::out_of_unsigned_range(
    LiteralTypeError::OutOfRange { value: BigInt::from(1) << 70_u32, family: IntegerFamily::Unsigned },
    "literal 1180591620717411303424 does not fit in a supported uint type"
)]
#[case::out_of_signed_range(
    LiteralTypeError::OutOfRange { value: BigInt::from(1) << 70_u32, family: IntegerFamily::Signed },
    "literal 1180591620717411303424 does not fit in a supported int type"
)]
#[case::unsupported_decimal(
    LiteralTypeError::UnsupportedDecimal,
    "decimal literals are not yet supported"
)]
fn literal_type_error_text(#[case] error: LiteralTypeError, #[case] text: &str) {
    assert_error_text(&error, text, Source::None);
    assert_eq!(
        error.is_unsupported(),
        matches!(error, LiteralTypeError::UnsupportedDecimal)
    );
}

#[test]
fn template_width_error_text() {
    let t = fixed(61_701, "T");

    let zero = TemplateDataType::with_widths(t.clone(), [0]).expect_err("a zero width");
    let empty = TemplateDataType::with_widths(t, []).expect_err("no width");

    assert_error_text(
        &zero,
        "template data type widths must be positive, but got 0",
        Source::None,
    );
    assert!(!zero.is_empty_list());
    assert_error_text(
        &empty,
        "template data type widths must not be empty",
        Source::None,
    );
    assert!(empty.is_empty_list());
}

fn template(id: u64, name: &str) -> TemplateDataType {
    TemplateDataType::new(fixed(id, name))
}

fn widths_8_16() -> TemplateDataType {
    TemplateDataType::with_widths(fixed(61_701, "T"), [16, 8]).expect("positive widths")
}

fn n() -> Expression {
    Expression::from(fixed(61_700, "N"))
}

#[rstest]
#[case::type_mismatch_bind(
    UnificationError::TypeMismatch {
        operation: TypeOperation::Bind,
        expected: scalar(CoreDataType::Int32),
        actual: scalar(CoreDataType::Float32),
    },
    "cannot bind int32[] against float32[]: structural mismatch",
    Source::None
)]
#[case::type_mismatch_unify(
    UnificationError::TypeMismatch {
        operation: TypeOperation::Unify,
        expected: scalar(CoreDataType::Int32),
        actual: scalar(CoreDataType::Float32),
    },
    "cannot unify int32[] with float32[]: structural mismatch",
    Source::None
)]
#[case::type_mismatch_of_an_extension(
    UnificationError::TypeMismatch {
        operation: TypeOperation::Bind,
        expected: NamedType::build("a"),
        actual: array(DataType::Primitive(CoreDataType::Int8), []),
    },
    "cannot bind named<a> against int8[]: structural mismatch",
    Source::None
)]
#[case::data_type_mismatch_bind(
    UnificationError::DataTypeMismatch {
        operation: TypeOperation::Bind,
        expected: DataType::Template(template(61_701, "T")),
        actual: DataType::Primitive(CoreDataType::Bool),
    },
    "cannot bind data type T::61701 against bool: structural mismatch",
    Source::None
)]
#[case::data_type_mismatch_of_an_extension(
    UnificationError::DataTypeMismatch {
        operation: TypeOperation::Bind,
        expected: NamedDataType::build("a"),
        actual: DataType::Primitive(CoreDataType::Bool),
    },
    "cannot bind data type named_data<a> against bool: structural mismatch",
    Source::None
)]
#[case::data_type_mismatch_unify(
    UnificationError::DataTypeMismatch {
        operation: TypeOperation::Unify,
        expected: DataType::Primitive(CoreDataType::Int8),
        actual: DataType::Primitive(CoreDataType::Bool),
    },
    "data type mismatch during unification: int8 vs bool",
    Source::None
)]
#[case::kind_mismatch_bind(
    UnificationError::KindMismatch {
        operation: TypeOperation::Bind,
        expected: "numerical".to_owned(),
        actual: "index".to_owned(),
    },
    "cannot bind numerical pattern against index",
    Source::None
)]
#[case::kind_mismatch_unify(
    UnificationError::KindMismatch {
        operation: TypeOperation::Unify,
        expected: "numerical".to_owned(),
        actual: "index".to_owned(),
    },
    "cannot unify numerical with index",
    Source::None
)]
#[case::core_data_type_mismatch(
    UnificationError::CoreDataTypeMismatch {
        expected: CoreDataType::Int8,
        actual: CoreDataType::Int16,
    },
    "core data type mismatch: int8 vs int16",
    Source::None
)]
#[case::rank_mismatch_bind(
    UnificationError::RankMismatch { operation: TypeOperation::Bind, expected: 2, actual: 1 },
    "shape rank mismatch: pattern has 2 dimensions, actual has 1",
    Source::None
)]
#[case::rank_mismatch_unify(
    UnificationError::RankMismatch { operation: TypeOperation::Unify, expected: 2, actual: 1 },
    "shape rank mismatch during unification: 2 vs 1",
    Source::None
)]
#[case::dimension_mismatch(
    UnificationError::DimensionMismatch { expected: Expression::from(4), actual: n() + 1 },
    "shape dimension mismatch: 4 vs (N::61700 + 1)",
    Source::None
)]
#[case::conflicting_expression_binding(
    UnificationError::ConflictingExpressionBinding {
        identifier: fixed(61_700, "N"),
        bound: Expression::from(4),
        actual: Expression::from(5),
    },
    "conflicting binding for shape variable N::61700: 4 vs 5",
    Source::None
)]
#[case::conflicting_data_type_binding(
    UnificationError::ConflictingDataTypeBinding {
        identifier: fixed(61_701, "T"),
        bound: DataType::Primitive(CoreDataType::Int8),
        actual: DataType::Template(template(61_702, "U")),
    },
    "conflicting data-type binding for T::61701: int8 vs U::61702",
    Source::None
)]
#[case::conflicting_type_binding(
    UnificationError::ConflictingTypeBinding {
        identifier: fixed(61_701, "T"),
        bound: scalar(CoreDataType::Int32),
        actual: scalar(CoreDataType::Float32),
    },
    "conflicting full-type binding for T::61701: int32[] vs float32[]",
    Source::None
)]
#[case::distinct_templates_bind(
    UnificationError::DistinctTemplates {
        operation: TypeOperation::Bind,
        expected: template(61_701, "T"),
        actual: template(61_702, "U"),
    },
    "cannot bind distinct template data types: T::61701 vs U::61702",
    Source::None
)]
#[case::distinct_templates_unify(
    UnificationError::DistinctTemplates {
        operation: TypeOperation::Unify,
        expected: template(61_701, "T"),
        actual: template(61_702, "U"),
    },
    "cannot unify distinct template data types: T::61701 vs U::61702",
    Source::None
)]
#[case::width_on_non_primitive(
    UnificationError::WidthOnNonPrimitive {
        template: widths_8_16(),
        actual: DataType::Template(template(61_702, "U")),
    },
    "cannot satisfy width constraint [8, 16] on T::61701 with non-primitive actual U::61702",
    Source::None
)]
#[case::width_mismatch(
    UnificationError::WidthMismatch { template: widths_8_16(), actual: CoreDataType::Int32 },
    "width mismatch for template T::61701: actual int32 has width 32, not in [8, 16]",
    Source::None
)]
#[case::width_mismatch_of_a_weak_type(
    UnificationError::WidthMismatch { template: widths_8_16(), actual: CoreDataType::Int },
    "width mismatch for template T::61701: actual int has no width, not in [8, 16]",
    Source::None
)]
#[case::occurs_check(
    UnificationError::OccursCheck {
        identifier: fixed(61_700, "N"),
        expression: n() + 1,
        substituted: n() + 1,
    },
    "occurs check failed: identifier N::61700 appears in (N::61700 + 1) after substitution \
     through existing bindings ((N::61700 + 1))",
    Source::None
)]
#[case::expression_mismatch(
    UnificationError::ExpressionMismatch { left: Expression::from(4), right: Expression::from(5) },
    "cannot unify expressions 4 and 5",
    Source::None
)]
#[case::wildcard_in_actual(
    UnificationError::WildcardInActual,
    "wildcard `...` cannot appear in `actual` during template binding",
    Source::None
)]
#[case::wildcard_in_unification(
    UnificationError::WildcardInUnification,
    "wildcard `...` is not supported during unification",
    Source::None
)]
#[case::extension(
    UnificationError::Extension(test_error()),
    "a type defined outside the core failed",
    Source::TestValue
)]
#[case::substitution(
    UnificationError::Substitution(PiecewiseError::NonBooleanConditionLiteral { case_index: 0 }),
    "substituting the existing shape bindings was refused",
    Source::Piecewise
)]
fn unification_error_text(
    #[case] error: UnificationError,
    #[case] text: &str,
    #[case] source: Source,
) {
    assert_error_text(&error, text, source);
}
