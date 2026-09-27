//! Tests for the serde form of the types (`fhy_core::types::wire`): the
//! shape of every variant, JSON and postcard round trips, the refusals, and
//! extension parts through a resolver.

use crate::support::foreign::{NamedDataType, NamedType, REFUSED, SilentType, TestResolver};
use crate::support::types::{array, constrained_template, index, scalar, template};

use fhy_core::expression::Expression;
use fhy_core::foreign::{BuildError, ForeignError};
use fhy_core::identifier::Identifier;
use fhy_core::types::wire::{DataTypeData, TypeData};
use fhy_core::types::{CoreDataType, DataType, Dimension, NumericalType, TemplateDataType, Type};
use proptest::prelude::*;
use rstest::rstest;
use serde::Serialize;
use serde::de::DeserializeOwned;

fn restored(id: u64, name: &str) -> Identifier {
    Identifier::try_restore(id, name).expect("the id is below the cap")
}

fn assert_round_trips<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("encodes as JSON");
    assert_eq!(
        &serde_json::from_str::<T>(&json).expect("decodes"),
        value,
        "{json}"
    );
    let bytes = postcard::to_allocvec(value).expect("encodes as postcard");
    assert_eq!(&postcard::from_bytes::<T>(&bytes).expect("decodes"), value);
}

#[rstest]
#[case::scalar(
    scalar(CoreDataType::Float32),
    r#"{"numerical":{"data_type":{"primitive":"float32"},"shape":[]}}"#
)]
#[case::shaped(
    array(CoreDataType::Int32, [
        Dimension::Expression(Expression::from(4)),
        Dimension::Expression(Expression::from(restored(61_300, "N"))),
    ]),
    concat!(
        r#"{"numerical":{"data_type":{"primitive":"int32"},"shape":["#,
        r#"{"expression":{"nodes":[{"literal":{"int":"4"}}]}},"#,
        r#"{"expression":{"nodes":[{"identifier":{"id":61300,"name_hint":"N"}}]}}]}}"#,
    )
)]
#[case::wildcard(
    array(template(&restored(61_301, "T")), [Dimension::Wildcard]),
    concat!(
        r#"{"numerical":{"data_type":{"template":{"identifier":{"id":61301,"name_hint":"T"},"#,
        r#""widths":null}},"shape":["wildcard"]}}"#,
    )
)]
#[case::widths(
    array(constrained_template(&restored(61_302, "T"), &[16, 32]), []),
    concat!(
        r#"{"numerical":{"data_type":{"template":{"identifier":{"id":61302,"name_hint":"T"},"#,
        r#""widths":[16,32]}},"shape":[]}}"#,
    )
)]
#[case::index(
    index(Expression::from(0), Expression::from(8), Expression::from(2)),
    concat!(
        r#"{"index":{"lower_bound":{"nodes":[{"literal":{"int":"0"}}]},"#,
        r#""upper_bound":{"nodes":[{"literal":{"int":"8"}}]},"#,
        r#""stride":{"nodes":[{"literal":{"int":"2"}}]}}}"#,
    )
)]
fn a_type_serializes_in_its_variant_form_and_round_trips(#[case] ty: Type, #[case] text: &str) {
    assert_eq!(serde_json::to_string(&ty).expect("encodes"), text);
    assert_round_trips(&ty);
}

#[test]
fn every_primitive_data_type_serializes_by_its_name() {
    let data_type = DataType::Primitive(CoreDataType::Uint8);

    assert_eq!(
        serde_json::to_string(&data_type).expect("encodes"),
        r#"{"primitive":"uint8"}"#
    );
    assert_round_trips(&data_type);
}

#[test]
fn a_template_data_type_refuses_a_zero_width_with_its_constructor_error() {
    let refused = serde_json::from_str::<TemplateDataType>(
        r#"{"identifier":{"id":61303,"name_hint":"T"},"widths":[8,0]}"#,
    );

    let message = refused.expect_err("a zero width is refused").to_string();
    assert!(
        message.starts_with(
            &TemplateDataType::with_widths(restored(61_304, "U"), [0])
                .expect_err("a zero width is refused")
                .to_string()
        ),
        "{message}"
    );
}

#[test]
fn a_numerical_type_and_an_index_type_serialize_on_their_own() {
    let Type::Numerical(numerical) = scalar(CoreDataType::Bool) else {
        unreachable!("a scalar is numerical")
    };

    assert_eq!(
        serde_json::to_string(&numerical).expect("encodes"),
        r#"{"data_type":{"primitive":"bool"},"shape":[]}"#
    );
    assert_round_trips(&numerical);
    let decoded: NumericalType =
        postcard::from_bytes(&postcard::to_allocvec(&numerical).expect("encodes"))
            .expect("decodes");
    assert_eq!(decoded, numerical);
}

#[test]
fn a_type_refuses_unknown_variants_and_fields() {
    serde_json::from_str::<Type>(r#"{"tensor":{}}"#).unwrap_err();
    serde_json::from_str::<Type>(
        r#"{"numerical":{"data_type":{"primitive":"int32"},"shape":[],"x":1}}"#,
    )
    .unwrap_err();
}

#[test]
fn an_extension_type_serializes_as_its_foreign_part() {
    let ty = NamedType::build("q");

    assert_eq!(
        serde_json::to_string(&ty).expect("encodes"),
        r#"{"extension":{"type_id":"test.named_type","data":"q"}}"#
    );
}

#[test]
fn plain_deserialization_refuses_an_extension_part_by_its_type_id() {
    let text = serde_json::to_string(&NamedType::build("q")).expect("encodes");

    let message = serde_json::from_str::<Type>(&text)
        .expect_err("no resolver is given")
        .to_string();

    assert!(
        message.starts_with("no implementation for the foreign part `test.named_type`"),
        "{message}"
    );
}

#[test]
fn a_resolver_builds_extension_parts_at_any_depth() {
    let ty = array(NamedDataType::build("fixed"), [Dimension::Wildcard]);
    let text = serde_json::to_string(&ty).expect("encodes");

    let decoded = serde_json::from_str::<TypeData>(&text)
        .expect("the wire form reads")
        .build(&TestResolver)
        .expect("the resolver knows the part");

    assert_eq!(decoded, ty);
    let extension = NamedType::build("q");
    let data: TypeData =
        postcard::from_bytes(&postcard::to_allocvec(&extension).expect("encodes")).expect("reads");
    assert!(data.foreign().is_some());
    assert_eq!(data.build(&TestResolver).expect("builds"), extension);
}

#[test]
fn a_data_type_wire_form_resolves_its_extension() {
    let data_type = NamedDataType::build("fixed");
    let text = serde_json::to_string(&data_type).expect("encodes");

    let data: DataTypeData = serde_json::from_str(&text).expect("reads");

    assert_eq!(
        data.foreign().map(fhy_core::foreign::Foreign::data),
        Some("fixed")
    );
    assert_eq!(data.build(&TestResolver).expect("builds"), data_type);
}

#[test]
fn a_part_the_resolver_refuses_fails_the_build() {
    let text = r#"{"extension":{"type_id":"test.named_type","data":"refused"}}"#;

    let error = serde_json::from_str::<TypeData>(text)
        .expect("reads")
        .build(&TestResolver)
        .expect_err("the payload is refused");

    assert!(matches!(
        error,
        BuildError::Foreign(ForeignError::Failed { .. })
    ));
}

#[test]
fn an_extension_without_a_wire_form_fails_to_serialize_with_its_name() {
    let ty = Type::Extension(fhy_core::foreign::Part::new(SilentType));

    let message = serde_json::to_string(&ty)
        .expect_err("it has no wire form")
        .to_string();

    assert_eq!(message, "`SilentType` has no wire form");
    postcard::to_allocvec(&ty).unwrap_err();
}

#[test]
fn an_extension_that_fails_to_give_its_part_fails_the_serializer() {
    let message = serde_json::to_string(&NamedType::build(REFUSED))
        .expect_err("the part fails")
        .to_string();

    assert_eq!(message, "the foreign part `test.named_type` failed");
}

/// Return a strategy for shape expressions over three shape variables:
/// literals, variables, sums and products.
fn shape_expression_strategy(variables: Vec<Identifier>) -> BoxedStrategy<Expression> {
    let leaf = prop_oneof![
        (1_i64..9).prop_map(Expression::from),
        prop::sample::select(variables).prop_map(Expression::from),
    ];
    leaf.prop_recursive(2, 6, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(left, right)| left + right),
            (inner.clone(), inner).prop_map(|(left, right)| left * right),
        ]
    })
    .boxed()
}

/// Return a strategy for data types: primitives, and templates with or
/// without widths.
fn data_type_strategy(templates: Vec<Identifier>) -> BoxedStrategy<DataType> {
    let primitive = prop::sample::select(vec![
        CoreDataType::Int8,
        CoreDataType::Uint16,
        CoreDataType::Int32,
        CoreDataType::Float32,
        CoreDataType::Float64,
        CoreDataType::Complex64,
        CoreDataType::Bool,
        CoreDataType::Int,
    ])
    .prop_map(DataType::Primitive);
    let template_type = (
        prop::sample::select(templates),
        prop::option::of(prop::collection::vec(
            prop::sample::select(vec![8_u32, 16, 32, 64]),
            1..4,
        )),
    )
        .prop_map(|(identifier, widths)| match widths {
            None => DataType::Template(TemplateDataType::new(identifier)),
            Some(widths) => DataType::Template(
                TemplateDataType::with_widths(identifier, widths).expect("positive widths"),
            ),
        });
    prop_oneof![primitive, template_type].boxed()
}

/// Return a strategy for types: numerical types over data types and
/// shapes of expressions and wildcards, and index types.
fn type_strategy() -> impl Strategy<Value = Type> {
    let variables: Vec<Identifier> = ["N", "M", "K"].into_iter().map(Identifier::new).collect();
    let templates: Vec<Identifier> = ["T", "U"].into_iter().map(Identifier::new).collect();
    let dimension = prop_oneof![
        4 => shape_expression_strategy(variables.clone()).prop_map(Dimension::Expression),
        1 => Just(Dimension::Wildcard),
    ];
    let expression = move || shape_expression_strategy(variables.clone());
    prop_oneof![
        (
            data_type_strategy(templates),
            prop::collection::vec(dimension, 0..4)
        )
            .prop_map(|(data_type, shape)| Type::Numerical(NumericalType::new(data_type, shape))),
        (expression(), expression(), expression()).prop_map(|(lower, upper, stride)| {
            Type::Index(fhy_core::types::IndexType::new(lower, upper, stride))
        }),
    ]
}

proptest::proptest! {
    /// Every built-in type round-trips through JSON and postcard, and its
    /// JSON re-encodes byte-identically (F2-046).
    #[test]
    fn a_type_round_trips_through_serde(value in type_strategy()) {
        crate::support::serde::check_serde_round_trip(&value)?;
        if let Type::Numerical(numerical) = &value {
            crate::support::serde::check_serde_round_trip(numerical.data_type())?;
        }
    }
}
