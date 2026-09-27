//! Tests for the type values: construction, the accessors, `==` and `Hash`
//! as structural equality, structural equivalence, and `Display`.

use crate::support::hashing::hash_of;
use crate::support::types::Equivalent;
use crate::support::types::{
    array, constrained_template, identifier_dimension, index, literal_dimension, scalar, template,
};

use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::types::{
    CoreDataType, DataType, Dimension, NumericalType, TemplateDataType, TemplateWidthError, Type,
};

#[test]
fn numerical_type_holds_its_data_type_and_shape() {
    let n = Identifier::new("N");
    let numerical = NumericalType::new(
        CoreDataType::Int32,
        [identifier_dimension(&n), literal_dimension(4)],
    );

    assert_eq!(
        numerical.data_type(),
        &DataType::Primitive(CoreDataType::Int32)
    );
    assert_eq!(
        numerical.shape(),
        [identifier_dimension(&n), literal_dimension(4)]
    );
    assert!(!numerical.is_scalar());
    assert!(!numerical.is_wildcard_shape());
}

#[test]
fn scalar_has_the_empty_shape() {
    let numerical = NumericalType::scalar(CoreDataType::Float32);

    assert!(numerical.is_scalar());
    assert!(numerical.shape().is_empty());
}

#[test]
fn only_a_lone_wildcard_is_the_full_shape_wildcard() {
    assert!(NumericalType::new(CoreDataType::Int8, [Dimension::Wildcard]).is_wildcard_shape());
    assert!(
        !NumericalType::new(
            CoreDataType::Int8,
            [Dimension::Wildcard, Dimension::Wildcard]
        )
        .is_wildcard_shape()
    );
    assert!(
        !NumericalType::new(
            CoreDataType::Int8,
            [Dimension::Wildcard, literal_dimension(2)]
        )
        .is_wildcard_shape()
    );
}

#[test]
fn template_keeps_its_identifier_and_widths() {
    let t = Identifier::new("T");
    let unconstrained = TemplateDataType::new(t.clone());
    let constrained = TemplateDataType::with_widths(t.clone(), [8, 16]).expect("positive widths");

    assert_eq!(unconstrained.identifier(), &t);
    assert_eq!(unconstrained.widths(), None);
    assert_eq!(constrained.widths(), Some(&[8, 16][..]));
    assert_eq!(
        TemplateDataType::with_widths(t, [])
            .expect("no width")
            .widths(),
        Some(&[][..])
    );
}

#[test]
fn template_with_a_zero_width_is_refused() {
    let error =
        TemplateDataType::with_widths(Identifier::new("T"), [8, 0]).expect_err("zero is no width");

    let _: &TemplateWidthError = &error;
    assert_eq!(
        error.to_string(),
        "template data type widths must be positive, but got 0"
    );
}

#[test]
fn index_type_holds_its_bounds_and_stride() {
    let n = Identifier::new("N");
    let Type::Index(range) = index(0, n.clone(), 1) else {
        unreachable!("an index type")
    };

    assert_eq!(range.lower_bound(), &Expression::from(0));
    assert_eq!(range.upper_bound(), &Expression::from(n));
    assert_eq!(range.stride(), &Expression::from(1));
}

#[test]
fn separately_built_equal_types_are_equal_and_hash_alike() {
    let n = Identifier::new("N");
    let pairs = [
        (
            array(
                CoreDataType::Int32,
                [literal_dimension(4), identifier_dimension(&n)],
            ),
            array(
                CoreDataType::Int32,
                [literal_dimension(4), identifier_dimension(&n)],
            ),
        ),
        (index(0, n.clone(), 1), index(0, n.clone(), 1)),
        (
            array(template(&n), [Dimension::Wildcard]),
            array(template(&n), [Dimension::Wildcard]),
        ),
    ];
    for (left, right) in pairs {
        assert_eq!(left, right);
        assert_eq!(hash_of(&left), hash_of(&right));
        assert!(left.is_equivalent(&right));
    }
}

#[test]
fn types_differing_anywhere_are_unequal_and_not_equivalent() {
    let n = Identifier::new("N");
    let base = array(CoreDataType::Int32, [literal_dimension(4)]);
    let others = [
        array(CoreDataType::Int16, [literal_dimension(4)]),
        array(CoreDataType::Int32, [literal_dimension(8)]),
        array(
            CoreDataType::Int32,
            [literal_dimension(4), literal_dimension(4)],
        ),
        array(CoreDataType::Int32, [Dimension::Wildcard]),
        array(template(&n), [literal_dimension(4)]),
        index(0, 4, 1),
    ];
    for other in others {
        assert_ne!(base, other);
        assert!(!base.is_equivalent(&other));
        assert!(!other.is_equivalent(&base));
    }
}

#[test]
fn index_types_differing_in_the_stride_are_not_equivalent() {
    assert!(!index(0, 10, 1).is_equivalent(&index(0, 10, 2)));
}

#[test]
fn templates_are_equivalent_only_with_the_same_identifier_and_widths() {
    let t = Identifier::new("T");
    let same_name = Identifier::new("T");

    assert!(template(&t).is_equivalent(&template(&t)));
    assert!(!template(&t).is_equivalent(&template(&same_name)));
    assert!(!template(&t).is_equivalent(&constrained_template(&t, &[8])));
    assert!(constrained_template(&t, &[8]).is_equivalent(&constrained_template(&t, &[8])));
    assert!(!DataType::Primitive(CoreDataType::Int8).is_equivalent(&template(&t)));
}

#[test]
fn clones_share_the_type() {
    let numerical = array(CoreDataType::Int32, [literal_dimension(4)]);

    assert!(Type::ptr_eq(&numerical, &numerical.clone()));
    assert!(!Type::ptr_eq(
        &numerical,
        &array(CoreDataType::Int32, [literal_dimension(4)])
    ));
}

#[test]
fn kind_names_are_the_class_names() {
    assert_eq!(scalar(CoreDataType::Int8).kind_name(), "NumericalType");
    assert_eq!(index(0, 1, 1).kind_name(), "IndexType");
    assert_eq!(
        DataType::Primitive(CoreDataType::Int8).kind_name(),
        "PrimitiveDataType"
    );
    assert_eq!(
        template(&Identifier::new("T")).kind_name(),
        "TemplateDataType"
    );
}

#[test]
fn types_display_with_identifier_ids() {
    let n = Identifier::new("N");
    let t = Identifier::new("T");
    let shape = [
        Dimension::Expression(Expression::from(n.clone()) + 1),
        Dimension::Wildcard,
    ];

    assert_eq!(
        array(template(&t), shape).to_string(),
        format!("T[(N::{} + 1), ...]", n.id())
    );
    assert_eq!(scalar(CoreDataType::Int32).to_string(), "int32[]");
    assert_eq!(
        array(
            CoreDataType::Int32,
            [literal_dimension(4), literal_dimension(8)]
        )
        .to_string(),
        "int32[4, 8]"
    );
    assert_eq!(
        index(0, n.clone(), 1).to_string(),
        format!("index(0:N::{}:1)", n.id())
    );
    assert_eq!(
        DataType::Primitive(CoreDataType::Complex64).to_string(),
        "complex64"
    );
    assert_eq!(template(&t).to_string(), "T");
}
