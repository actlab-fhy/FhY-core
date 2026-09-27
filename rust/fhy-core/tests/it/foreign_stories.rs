//! Tests for `fhy_core::foreign`: the foreign part, the resolvers and their
//! errors, and the default wire form of the open traits.

use std::error::Error;
use std::sync::Arc;

use crate::support::constraint::{TestCustom, TestOpaque};
use crate::support::foreign::{NAMED_TYPE, REFUSED, SilentDataType, SilentType, TestResolver};
use crate::support::param::EvenDomain;

use fhy_core::constraint::{Constraint, CustomConstraint, Outcome};
use fhy_core::expression::Expression;
use fhy_core::foreign::{BuildError, Foreign, ForeignError, NoForeign, Resolve};
use fhy_core::param::ParamDomain;
use fhy_core::types::{DataTypeExtension, TypeExtension};

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
    let refused: Result<Arc<dyn TypeExtension>, ForeignError> =
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
    let error = SilentType.to_foreign().expect_err("the default has none");

    assert!(matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "SilentType"));
    assert_eq!(error.to_string(), "`SilentType` has no wire form");
}

#[test]
fn a_data_type_extension_has_no_wire_form_by_default() {
    let error = SilentDataType
        .to_foreign()
        .expect_err("the default has none");

    assert!(
        matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "SilentDataType")
    );
}

#[test]
fn an_opaque_value_has_no_wire_form_by_default() {
    let error = fhy_core::constraint::OpaqueValue::to_foreign(&TestOpaque::token(1))
        .expect_err("the default has none");

    assert!(matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "Token"));
}

#[test]
fn a_custom_constraint_has_no_wire_form_by_default() {
    let log = Arc::default();
    let Constraint::Custom(custom) =
        TestCustom::build("c", Expression::literal(true), Outcome::Satisfied, &log)
    else {
        unreachable!("the builder returns a custom constraint")
    };

    let error = CustomConstraint::to_foreign(custom.as_ref()).expect_err("the default has none");

    assert!(
        matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "custom constraint")
    );
}

#[test]
fn a_custom_domain_has_no_wire_form_by_default() {
    let (ParamDomain::Custom(domain), _) = EvenDomain::build(false) else {
        unreachable!("the builder returns a custom domain")
    };

    let error = domain.to_foreign().expect_err("the default has none");

    assert!(
        matches!(&error, ForeignError::NoWireForm { type_name } if type_name == "custom domain")
    );
}
