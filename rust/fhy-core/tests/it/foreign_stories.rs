//! Tests for `fhy_core::foreign`: the foreign part, the resolvers and their
//! errors, and the default wire form of the open traits.

use std::error::Error;
use std::sync::Arc;

use crate::support::constraint::{TestCustom, TestOpaque};
use crate::support::error_text::{Source, assert_error_text, test_error};
use crate::support::foreign::{
    NAMED_TYPE, NamedType, REFUSED, SilentDataType, SilentType, TestResolver,
};
use crate::support::param::EvenDomain;

use fhy_core::constraint::{Constraint, CustomConstraint, OpaqueValue, Outcome, Value};
use fhy_core::expression::Expression;
use fhy_core::foreign::{BuildError, Foreign, ForeignError, ForeignPart, NoForeign, Part, Resolve};
use fhy_core::param::{CustomDomain, ParamDomain};
use fhy_core::types::{DataTypeExtension, Type, TypeExtension};
use rstest::rstest;

#[test]
fn a_foreign_part_serializes_as_its_type_id_and_text() {
    let part = Foreign::new("pkg.even", r#"{"modulus":2}"#);

    let text = serde_json::to_string(&part).expect("a part encodes");

    assert_eq!(text, r#"{"type_id":"pkg.even","data":"{\"modulus\":2}"}"#);
    assert_eq!(part.type_id(), "pkg.even");
    assert_eq!(part.data(), r#"{"modulus":2}"#);
}

#[test]
fn a_foreign_part_round_trips_through_json_and_postcard() {
    let part = Foreign::new("pkg.even", "é \"quoted\" \n text");

    let from_json: Foreign =
        serde_json::from_str(&serde_json::to_string(&part).expect("encodes")).expect("decodes");
    let from_postcard: Foreign =
        postcard::from_bytes(&postcard::to_allocvec(&part).expect("encodes")).expect("decodes");

    assert_eq!(from_json, part);
    assert_eq!(from_postcard, part);
}

#[test]
fn a_foreign_part_refuses_unknown_and_missing_fields() {
    let extra = serde_json::from_str::<Foreign>(r#"{"type_id":"a","data":"b","other":1}"#);
    let missing = serde_json::from_str::<Foreign>(r#"{"type_id":"a"}"#);

    extra.unwrap_err();
    missing.unwrap_err();
}

#[test]
fn no_foreign_refuses_every_part_by_its_type_id() {
    let refused: Result<u8, ForeignError> = NoForeign.resolve(&Foreign::new("pkg.even", ""));

    let error = refused.expect_err("no implementation is known");
    assert!(matches!(&error, ForeignError::Unresolved { type_id } if type_id == "pkg.even"));
    assert_eq!(
        error.to_string(),
        "no implementation for the foreign part `pkg.even`"
    );
    assert!(error.source().is_none());
}

#[test]
fn a_failed_part_names_its_type_id_and_keeps_its_cause() {
    let refused: Result<Part<dyn TypeExtension>, ForeignError> =
        TestResolver.resolve(&Foreign::new(NAMED_TYPE, REFUSED));

    let error = refused.expect_err("the payload is refused");
    assert_eq!(
        error.to_string(),
        "the foreign part `test.named_type` failed"
    );
    assert_eq!(
        error.source().map(ToString::to_string).as_deref(),
        Some("test.named_type failed")
    );
}

fn zero_width_error() -> fhy_core::types::TemplateWidthError {
    fhy_core::types::TemplateDataType::with_widths(fhy_core::identifier::Identifier::new("t"), [0])
        .expect_err("a zero width is refused")
}

#[test]
fn a_build_error_shows_its_underlying_error() {
    let foreign = BuildError::from(ForeignError::Unresolved {
        type_id: "x".to_owned(),
    });
    let width_error = zero_width_error();
    let expected = width_error.to_string();
    let invalid = BuildError::invalid(width_error);

    assert_eq!(
        foreign.to_string(),
        "no implementation for the foreign part `x`"
    );
    assert_eq!(invalid.to_string(), expected);
    assert!(matches!(invalid, BuildError::Invalid(_)));
}

#[test]
fn a_type_extension_has_no_wire_form_by_default() {
    let error = ForeignPart::to_foreign(&SilentType).expect_err("the default has none");

    assert!(matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "SilentType"));
    assert_eq!(error.to_string(), "`SilentType` has no wire form");
}

#[test]
fn a_data_type_extension_has_no_wire_form_by_default() {
    let error = ForeignPart::to_foreign(&SilentDataType).expect_err("the default has none");

    assert!(
        matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "SilentDataType")
    );
}

#[test]
fn an_opaque_value_has_no_wire_form_by_default() {
    let error = ForeignPart::to_foreign(&TestOpaque::token(1)).expect_err("the default has none");

    assert!(matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "Token"));
}

#[test]
fn no_wire_form_names_the_custom_type() {
    let log = Arc::default();
    let Constraint::Custom(custom) =
        TestCustom::build("c", Expression::literal(true), Outcome::Satisfied, &log)
    else {
        unreachable!("the builder returns a custom constraint")
    };
    let (ParamDomain::Custom(domain), _) = EvenDomain::build(false) else {
        unreachable!("the builder returns a custom domain")
    };

    let constraint_error = custom.get().to_foreign().expect_err("the default has none");
    let domain_error = domain.get().to_foreign().expect_err("the default has none");

    assert!(
        matches!(&constraint_error, ForeignError::NoWireForm { type_name } if type_name == "TestCustom")
    );
    assert_eq!(
        constraint_error.to_string(),
        "`TestCustom` has no wire form"
    );
    assert!(
        matches!(&domain_error, ForeignError::NoWireForm { type_name } if type_name == "EvenDomain")
    );
}

#[test]
fn as_any_on_a_part_downcasts_to_the_implementation() {
    let Type::Extension(part) = NamedType::build("n") else {
        unreachable!("the builder returns an extension type")
    };
    let Value::Opaque(opaque) = TestOpaque::token(3).into_value() else {
        unreachable!("the builder returns an opaque value")
    };

    let named = part.get().as_any().downcast_ref::<NamedType>();
    let through_the_trait =
        fhy_core::foreign::AsAny::as_any(part.get()).downcast_ref::<NamedType>();
    let token = opaque.get().as_any().downcast_ref::<TestOpaque>();

    assert_eq!(named.map(|named| named.0.as_str()), Some("n"));
    assert!(through_the_trait.is_some());
    assert_eq!(token.map(|token| token.payload), Some(3));
    // The handle's own `Any` is the `Part`, not the implementation.
    assert!(
        fhy_core::foreign::AsAny::as_any(&part)
            .downcast_ref::<NamedType>()
            .is_none()
    );
}

#[test]
fn every_part_handle_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Part<dyn OpaqueValue>>();
    assert_send_sync::<Part<dyn CustomConstraint>>();
    assert_send_sync::<Part<dyn CustomDomain>>();
    assert_send_sync::<Part<dyn TypeExtension>>();
    assert_send_sync::<Part<dyn DataTypeExtension>>();
}

#[rstest]
#[case::unresolved(
    ForeignError::Unresolved { type_id: "pkg.even".to_owned() },
    "no implementation for the foreign part `pkg.even`",
    Source::None
)]
#[case::no_wire_form(
    ForeignError::NoWireForm { type_name: "Handle".to_owned() },
    "`Handle` has no wire form",
    Source::None
)]
#[case::failed(
    ForeignError::Failed { type_id: "pkg.even".to_owned(), source: test_error() },
    "the foreign part `pkg.even` failed",
    Source::TestValue
)]
fn foreign_error_text(#[case] error: ForeignError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}

#[rstest]
#[case::foreign(
    BuildError::Foreign(ForeignError::Unresolved { type_id: "pkg.even".to_owned() }),
    "no implementation for the foreign part `pkg.even`",
    Source::None
)]
#[case::failed_foreign(
    BuildError::Foreign(ForeignError::Failed {
        type_id: "pkg.even".to_owned(),
        source: test_error(),
    }),
    "the foreign part `pkg.even` failed",
    Source::TestValue
)]
#[case::invalid(
    BuildError::invalid(zero_width_error()),
    "template data type widths must be positive, but got 0",
    Source::None
)]
#[case::invalid_with_a_cause(
    BuildError::invalid(ForeignError::Failed { type_id: "t".to_owned(), source: test_error() }),
    "the foreign part `t` failed",
    Source::TestValue
)]
fn build_error_text(#[case] error: BuildError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}
