//! Tests for types and data types defined outside the crate: a tagged
//! wrapper type with rules of its own through every operation, and bare
//! extensions that take the core's default rules.
//!
//! Ported from `tests/types/test_extension.py` and the dispatcher-default
//! tests of `tests/types/test_unification.py`.

use crate::support::hashing::hash_of;
use crate::support::types::{
    array, constrained_template, identifier_dimension, literal_dimension, template,
};

use std::any::Any;
use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::types::{
    CoreDataType, DataType, DataTypeExtension, Type, TypeExtension, TypeOperation,
    TypeUnificationEnvironment, UnificationError,
};

/// A type wrapping another type under a tag, with rules of its own.
#[derive(Debug)]
struct Tagged {
    tag: &'static str,
    inner: Type,
}

/// The error of two tags that differ.
#[derive(Debug)]
struct TagMismatch;

impl fmt::Display for TagMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tag mismatch")
    }
}

impl Error for TagMismatch {}

fn tagged(tag: &'static str, inner: Type) -> Type {
    Type::Extension(Arc::new(Tagged { tag, inner }))
}

/// Return the `Tagged` a type is, if it is one.
fn as_tagged(value: &Type) -> Option<&Tagged> {
    match value {
        Type::Extension(extension) => extension.as_any().downcast_ref::<Tagged>(),
        _ => None,
    }
}

/// Return the inner type of `actual` if it is a `Tagged` of `tag`, or the
/// tag mismatch.
fn matching_inner<'a>(tag: &str, actual: &'a Type) -> Result<&'a Type, UnificationError> {
    match as_tagged(actual) {
        Some(other) if other.tag == tag => Ok(&other.inner),
        _ => Err(UnificationError::Extension(Box::new(TagMismatch))),
    }
}

impl fmt::Display for Tagged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}<{}>", self.tag, self.inner)
    }
}

impl TypeExtension for Tagged {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Tagged")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_structurally_equivalent(&self, other: &Type) -> bool {
        as_tagged(other).is_some_and(|other| {
            other.tag == self.tag && self.inner.is_structurally_equivalent(&other.inner)
        })
    }

    fn eq_extension(&self, other: &dyn TypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Tagged>()
            .is_some_and(|other| other.tag == self.tag && other.inner == self.inner)
    }

    fn hash_extension(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.tag.as_bytes());
        state.write_u64(hash_of(&self.inner));
    }

    fn bind_template(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<TypeUnificationEnvironment, UnificationError>> {
        Some(
            matching_inner(self.tag, actual)
                .and_then(|inner| self.inner.bind_template(inner, environment)),
        )
    }

    fn substitute_template(
        &self,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<Type, UnificationError>> {
        Some(
            self.inner
                .substitute_template(environment)
                .map(|inner| tagged(self.tag, inner)),
        )
    }

    fn unify(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<(Type, TypeUnificationEnvironment), UnificationError>> {
        Some(matching_inner(self.tag, actual).and_then(|inner| {
            self.inner
                .unify(inner, environment)
                .map(|(unified, environment)| (tagged(self.tag, unified), environment))
        }))
    }
}

/// A type with no rules of its own.
#[derive(Debug)]
struct Bare(&'static str);

impl fmt::Display for Bare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bare {}", self.0)
    }
}

impl TypeExtension for Bare {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Bare")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A data type with no rules of its own.
#[derive(Debug)]
struct BareData;

impl fmt::Display for BareData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bare data")
    }
}

impl DataTypeExtension for BareData {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("BareData")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn empty() -> TypeUnificationEnvironment {
    TypeUnificationEnvironment::new()
}

fn int32() -> DataType {
    DataType::Primitive(CoreDataType::Int32)
}

#[test]
fn extension_takes_part_in_structural_equivalence_and_equality() {
    let first = tagged("dense", array(int32(), [literal_dimension(2)]));
    let duplicate = tagged("dense", array(int32(), [literal_dimension(2)]));
    let other_tag = tagged("sparse", array(int32(), [literal_dimension(2)]));
    let other_inner = tagged("dense", array(int32(), [literal_dimension(3)]));
    let plain = array(int32(), [literal_dimension(2)]);

    assert!(first.is_structurally_equivalent(&duplicate));
    assert_eq!(first, duplicate);
    assert_eq!(hash_of(&first), hash_of(&duplicate));
    assert!(!first.is_structurally_equivalent(&other_tag));
    assert!(!first.is_structurally_equivalent(&other_inner));
    assert!(!first.is_structurally_equivalent(&plain));
    assert!(!plain.is_structurally_equivalent(&first));
    assert_ne!(first, plain);
}

#[test]
fn extension_binds_then_substitutes_through_its_inner_type() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let pattern = tagged("dense", array(template(&t), [identifier_dimension(&n)]));
    let actual = tagged("dense", array(int32(), [literal_dimension(8)]));

    let environment = pattern.bind_template(&actual, &empty()).expect("binds");
    let substituted = pattern
        .substitute_template(&environment)
        .expect("substitutes");

    assert!(substituted.is_structurally_equivalent(&actual));
    assert_eq!(substituted, actual);
}

#[test]
fn extension_unifies_through_its_inner_type_recording_the_inner_bindings() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let expected = tagged("dense", array(template(&t), [identifier_dimension(&n)]));
    let actual = tagged("dense", array(int32(), [literal_dimension(8)]));

    let (unified, environment) = expected.unify(&actual, &empty()).expect("unifies");

    assert!(unified.is_structurally_equivalent(&actual));
    assert_eq!(environment.data_type_binding(&t), Some(&int32()));
    assert_eq!(
        environment.expression_binding(&n),
        Some(&Expression::from(8))
    );
}

#[test]
fn extension_errors_propagate_from_bind_and_unify() {
    let inner = array(int32(), []);
    let expected = tagged("dense", inner.clone());
    let actual = tagged("sparse", inner);

    let bound = expected
        .bind_template(&actual, &empty())
        .expect_err("tags differ");
    let unified = expected.unify(&actual, &empty()).expect_err("tags differ");

    for error in [bound, unified] {
        let UnificationError::Extension(source) = &error else {
            panic!("an extension error, got {error}");
        };
        assert!(source.downcast_ref::<TagMismatch>().is_some());
        assert!(error.source().is_some());
    }
}

#[test]
fn a_width_violation_inside_an_extension_surfaces_as_the_core_error() {
    let t = Identifier::new("T");
    let pattern = tagged(
        "dense",
        array(constrained_template(&t, &[8]), [literal_dimension(4)]),
    );
    let actual = tagged("dense", array(int32(), [literal_dimension(4)]));

    let bound = pattern
        .bind_template(&actual, &empty())
        .expect_err("32 is not 8");
    let unified = pattern.unify(&actual, &empty()).expect_err("32 is not 8");

    assert!(matches!(bound, UnificationError::WidthMismatch { .. }));
    assert!(matches!(unified, UnificationError::WidthMismatch { .. }));
}

#[test]
fn a_built_in_pattern_refuses_an_extension_by_its_kind_name() {
    let error = array(int32(), [])
        .bind_template(&tagged("dense", array(int32(), [])), &empty())
        .expect_err("kinds differ");

    assert_eq!(
        error.to_string(),
        "cannot bind NumericalType pattern against Tagged"
    );
}

#[test]
fn an_extension_without_rules_takes_the_default_rules() {
    let first: Type = Type::Extension(Arc::new(Bare("a")));
    let second: Type = Type::Extension(Arc::new(Bare("b")));

    let bound = first
        .bind_template(&second, &empty())
        .expect_err("not equivalent");
    let unified = first.unify(&second, &empty()).expect_err("not equivalent");
    let substituted = first.substitute_template(&empty()).expect("substitutes");

    assert!(!first.is_structurally_equivalent(&first));
    assert!(matches!(
        bound,
        UnificationError::TypeMismatch {
            operation: TypeOperation::Bind,
            ..
        }
    ));
    assert_eq!(
        bound.to_string(),
        "cannot bind bare a against bare b: structural mismatch"
    );
    assert_eq!(
        unified.to_string(),
        "cannot unify bare a with bare b: structural mismatch"
    );
    assert!(Type::ptr_eq(&substituted, &first));
}

#[test]
fn an_extension_without_rules_is_equal_only_to_itself() {
    let first: Type = Type::Extension(Arc::new(Bare("a")));
    let same_text: Type = Type::Extension(Arc::new(Bare("a")));

    assert_eq!(first, first.clone());
    assert_ne!(first, same_text);
    assert_eq!(first.kind_name(), "Bare");
}

#[test]
fn a_data_type_extension_without_rules_takes_the_default_rules() {
    let first = DataType::Extension(Arc::new(BareData));
    let second = DataType::Extension(Arc::new(BareData));

    let bound = first
        .bind_template(&second, &empty())
        .expect_err("not equivalent");
    let substituted = first.substitute_template(&empty()).expect("substitutes");

    assert_eq!(
        bound.to_string(),
        "cannot bind data type bare data against bare data: structural mismatch"
    );
    assert_eq!(substituted, first);
    assert_ne!(first, second);
    assert_eq!(first.kind_name(), "BareData");
}

#[test]
fn a_numerical_type_over_a_data_type_extension_binds_it_by_the_default_rule() {
    let data_type = DataType::Extension(Arc::new(BareData));
    let pattern = array(data_type.clone(), [literal_dimension(1)]);

    let error = pattern
        .bind_template(&pattern, &empty())
        .expect_err("the extension is not equivalent to itself");
    let t = Identifier::new("T");
    let environment = array(template(&t), [literal_dimension(1)])
        .bind_template(&pattern, &empty())
        .expect("a template binds any data type");

    assert!(matches!(error, UnificationError::DataTypeMismatch { .. }));
    assert_eq!(environment.data_type_binding(&t), Some(&data_type));
}
