//! Tests for types and data types defined outside the crate: a tagged
//! wrapper type with rules of its own through every operation, and bare
//! extensions that take the core's default rules.
//!
//! Ported from `tests/types/test_extension.py` and the dispatcher-default
//! tests of `tests/types/test_unification.py`.

use crate::support::hashing::hash_of;
use crate::support::types::Equivalent;
use crate::support::types::{
    array, constrained_template, identifier_dimension, literal_dimension, template,
};

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use fhy_core::expression::Expression;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
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
    Type::Extension(Part::new(Tagged { tag, inner }))
}

/// Return the `Tagged` a type is, if it is one.
fn as_tagged(value: &Type) -> Option<&Tagged> {
    match value {
        Type::Extension(extension) => extension.get().as_any().downcast_ref::<Tagged>(),
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

impl ForeignPart for Tagged {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Tagged")
    }
}

impl TypeExtension for Tagged {
    fn is_structurally_equivalent(&self, other: &Type) -> Result<bool, BoxError> {
        Ok(match as_tagged(other) {
            Some(other) => {
                other.tag == self.tag && self.inner.is_structurally_equivalent(&other.inner)?
            }
            None => false,
        })
    }

    fn eq_part(&self, other: &dyn TypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.tag == self.tag && other.inner == self.inner)
    }

    fn hash_part(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.tag.as_bytes());
        state.write_u64(hash_of(&self.inner));
    }

    fn bind_template(
        &self,
        _this: &Type,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<TypeUnificationEnvironment, UnificationError> {
        matching_inner(self.tag, actual)
            .and_then(|inner| self.inner.bind_template(inner, environment))
    }

    fn substitute_template(
        &self,
        _this: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<Type, UnificationError> {
        self.inner
            .substitute_template(environment)
            .map(|inner| tagged(self.tag, inner))
    }

    fn unify(
        &self,
        _this: &Type,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Result<(Type, TypeUnificationEnvironment), UnificationError> {
        matching_inner(self.tag, actual).and_then(|inner| {
            self.inner
                .unify(inner, environment)
                .map(|(unified, environment)| (tagged(self.tag, unified), environment))
        })
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

impl ForeignPart for Bare {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Bare")
    }
}

impl TypeExtension for Bare {}

/// A data type with no rules of its own.
#[derive(Debug)]
struct BareData;

impl fmt::Display for BareData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bare data")
    }
}

impl ForeignPart for BareData {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("BareData")
    }
}

impl DataTypeExtension for BareData {}

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

    assert!(first.is_equivalent(&duplicate));
    assert_eq!(first, duplicate);
    assert_eq!(hash_of(&first), hash_of(&duplicate));
    assert!(!first.is_equivalent(&other_tag));
    assert!(!first.is_equivalent(&other_inner));
    assert!(!first.is_equivalent(&plain));
    assert!(!plain.is_equivalent(&first));
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

    assert!(substituted.is_equivalent(&actual));
    assert_eq!(substituted, actual);
}

#[test]
fn extension_unifies_through_its_inner_type_recording_the_inner_bindings() {
    let (t, n) = (Identifier::new("T"), Identifier::new("N"));
    let expected = tagged("dense", array(template(&t), [identifier_dimension(&n)]));
    let actual = tagged("dense", array(int32(), [literal_dimension(8)]));

    let (unified, environment) = expected.unify(&actual, &empty()).expect("unifies");

    assert!(unified.is_equivalent(&actual));
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

    for error in [bound, unified] {
        let UnificationError::WidthMismatch { template, actual } = &error else {
            panic!("a width mismatch, got {error}");
        };
        assert_eq!(template.identifier(), &t);
        assert_eq!(template.widths(), Some(&[8][..]));
        assert_eq!(*actual, CoreDataType::Int32);
    }
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
    let first: Type = Type::Extension(Part::new(Bare("a")));
    let second: Type = Type::Extension(Part::new(Bare("b")));

    let bound = first
        .bind_template(&second, &empty())
        .expect_err("not equivalent");
    let unified = first.unify(&second, &empty()).expect_err("not equivalent");
    let substituted = first.substitute_template(&empty()).expect("substitutes");

    assert!(!first.is_equivalent(&second));
    assert!(!second.is_equivalent(&first));
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
    let first: Type = Type::Extension(Part::new(Bare("a")));
    let same_text: Type = Type::Extension(Part::new(Bare("a")));

    assert_eq!(first, first.clone());
    assert_ne!(first, same_text);
    assert_eq!(first.kind_name(), "Bare");
}

#[test]
fn a_data_type_extension_without_rules_takes_the_default_rules() {
    let first = DataType::Extension(Part::new(BareData));
    let second = DataType::Extension(Part::new(BareData));

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
    let data_type = DataType::Extension(Part::new(BareData));
    let other = DataType::Extension(Part::new(BareData));
    let pattern = array(data_type.clone(), [literal_dimension(1)]);

    let error = pattern
        .bind_template(&array(other, [literal_dimension(1)]), &empty())
        .expect_err("two bare extensions are not equivalent");
    let t = Identifier::new("T");
    let environment = array(template(&t), [literal_dimension(1)])
        .bind_template(&pattern, &empty())
        .expect("a template binds any data type");

    let UnificationError::DataTypeMismatch {
        operation,
        expected,
        actual,
    } = &error
    else {
        panic!("a data type mismatch, got {error}");
    };
    assert_eq!(*operation, TypeOperation::Bind);
    let is_same_part = |left: &DataType, right: &DataType| matches!((left, right), (DataType::Extension(left), DataType::Extension(right)) if Part::ptr_eq(left, right));
    assert!(is_same_part(expected, &data_type));
    assert!(!is_same_part(actual, &data_type));
    assert_eq!(environment.data_type_binding(&t), Some(&data_type));
}

/// A type that stands for another type: equivalent to it, and to an alias
/// of an equivalent type.
#[derive(Debug)]
struct Alias(Type);

impl fmt::Display for Alias {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alias of {}", self.0)
    }
}

impl ForeignPart for Alias {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Alias")
    }
}

impl TypeExtension for Alias {
    fn is_structurally_equivalent(&self, other: &Type) -> Result<bool, BoxError> {
        Ok(match other {
            Type::Extension(extension) => match extension.get().as_any().downcast_ref::<Self>() {
                Some(other) => self.0.is_structurally_equivalent(&other.0)?,
                None => self.0.is_structurally_equivalent(other)?,
            },
            _ => self.0.is_structurally_equivalent(other)?,
        })
    }
}

/// A data type that stands for another data type, as [`Alias`] does.
#[derive(Debug)]
struct DataAlias(DataType);

impl fmt::Display for DataAlias {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alias of {}", self.0)
    }
}

impl ForeignPart for DataAlias {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("DataAlias")
    }
}

impl DataTypeExtension for DataAlias {
    fn is_structurally_equivalent(&self, other: &DataType) -> Result<bool, BoxError> {
        Ok(match other {
            DataType::Extension(extension) => {
                match extension.get().as_any().downcast_ref::<Self>() {
                    Some(other) => self.0.is_structurally_equivalent(&other.0)?,
                    None => self.0.is_structurally_equivalent(other)?,
                }
            }
            _ => self.0.is_structurally_equivalent(other)?,
        })
    }
}

#[test]
fn an_extension_without_overrides_is_equivalent_to_itself() {
    let bare: Type = Type::Extension(Part::new(Bare("a")));
    let bare_data = DataType::Extension(Part::new(BareData));

    assert!(bare.is_equivalent(&bare));
    assert!(bare.is_equivalent(&bare.clone()));
    assert!(bare_data.is_equivalent(&bare_data.clone()));
    assert!(!bare.is_equivalent(&Type::Extension(Part::new(Bare("a")))));
}

#[test]
fn an_extension_without_overrides_unifies_with_itself() {
    let bare: Type = Type::Extension(Part::new(Bare("a")));

    let environment = bare
        .bind_template(&bare.clone(), &empty())
        .expect("binds itself");
    let (unified, unified_environment) = bare.unify(&bare.clone(), &empty()).expect("unifies");

    assert_eq!(environment, empty());
    assert!(Type::ptr_eq(&unified, &bare));
    assert_eq!(unified_environment, empty());
}

#[test]
fn equal_extension_types_bind_as_templates() {
    let data_type = DataType::Extension(Part::new(BareData));
    let left = array(data_type.clone(), [literal_dimension(4)]);
    let right = array(data_type, [literal_dimension(4)]);

    assert_eq!(left, right);
    let environment = left
        .bind_template(&right, &empty())
        .expect("equal types bind");
    let (unified, _) = left.unify(&right, &empty()).expect("equal types unify");

    assert_eq!(environment, empty());
    assert_eq!(unified, right);
}

#[test]
fn a_numerical_type_against_an_extension_asks_the_extension() {
    let plain = array(int32(), [literal_dimension(2)]);
    let alias: Type = Type::Extension(Part::new(Alias(plain.clone())));
    let data_alias = DataType::Extension(Part::new(DataAlias(int32())));

    assert!(alias.is_equivalent(&plain));
    assert!(plain.is_equivalent(&alias));
    assert!(int32().is_equivalent(&data_alias));
    assert!(data_alias.is_equivalent(&int32()));
    assert!(
        array(int32(), [literal_dimension(2)])
            .is_equivalent(&array(data_alias, [literal_dimension(2)]))
    );
    assert!(!plain.is_equivalent(&Type::Extension(Part::new(Bare("a")))));
}

/// Return a strategy of data types: primitives and aliases of them, and
/// bare extensions. An alias stands only for a primitive, since a bare
/// extension, knowing nothing of aliases, could not answer symmetrically.
fn data_type_strategy() -> impl proptest::strategy::Strategy<Value = DataType> {
    use proptest::prelude::*;
    let bare = DataType::Extension(Part::new(BareData));
    let primitive = prop_oneof![
        Just(int32()),
        Just(DataType::Primitive(CoreDataType::Float32)),
    ]
    .prop_recursive(2, 4, 1, |inner| {
        inner.prop_map(|data_type| DataType::Extension(Part::new(DataAlias(data_type))))
    });
    prop_oneof![
        primitive,
        Just(bare),
        Just(DataType::Extension(Part::new(BareData))),
    ]
}

/// Return a strategy of types: numerical types over
/// [`data_type_strategy`] and aliases of them, and bare and tagged
/// extensions.
fn type_strategy() -> impl proptest::strategy::Strategy<Value = Type> {
    use proptest::prelude::*;
    let numerical = (data_type_strategy(), 0_usize..2)
        .prop_map(|(data_type, rank)| array(data_type, (0..rank).map(|_| literal_dimension(2))))
        .prop_recursive(2, 4, 1, |inner| {
            inner.prop_map(|inner| Type::Extension(Part::new(Alias(inner))))
        });
    prop_oneof![
        numerical,
        Just(Type::Extension(Part::new(Bare("shared")))),
        Just(Type::Extension(Part::new(Bare("fresh")))),
        Just(tagged("dense", array(int32(), []))),
    ]
}

proptest::proptest! {
    #[test]
    fn structural_equivalence_is_symmetric_over_extensions(
        left in type_strategy(),
        right in type_strategy(),
    ) {
        proptest::prop_assert_eq!(
            left.is_equivalent(&right),
            right.is_equivalent(&left),
            "{} against {}", left, right
        );
        proptest::prop_assert!(left.is_equivalent(&left.clone()));
    }

    #[test]
    fn data_type_equivalence_is_symmetric_over_extensions(
        left in data_type_strategy(),
        right in data_type_strategy(),
    ) {
        proptest::prop_assert_eq!(
            left.is_equivalent(&right),
            right.is_equivalent(&left),
            "{} against {}", left, right
        );
    }
}

/// A type whose structural equivalence fails.
#[derive(Debug)]
struct Failing;

impl fmt::Display for Failing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("failing")
    }
}

impl ForeignPart for Failing {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Failing")
    }
}

impl TypeExtension for Failing {
    fn is_structurally_equivalent(&self, _other: &Type) -> Result<bool, BoxError> {
        Err(Box::new(TagMismatch))
    }
}

#[test]
fn a_failing_extension_equivalence_is_a_unification_error() {
    let failing: Type = Type::Extension(Part::new(Failing));
    let plain = array(int32(), []);
    let t = Identifier::new("T");
    let full = array(template(&t), [fhy_core::types::Dimension::Wildcard]);
    let bound = empty().with_type_binding(t, failing.clone());

    let errors = [
        failing
            .is_structurally_equivalent(&plain)
            .expect_err("the hook fails"),
        plain
            .is_structurally_equivalent(&failing)
            .expect_err("the hook fails on the right too"),
        failing
            .bind_template(&plain, &empty())
            .expect_err("the default rule asks the hook"),
        failing
            .unify(&plain, &empty())
            .expect_err("the default rule asks the hook"),
        full.bind_template(&plain, &bound)
            .expect_err("the conflicting binding asks the hook"),
    ];

    for error in errors {
        let UnificationError::Extension(source) = &error else {
            panic!("an extension error, got {error}");
        };
        assert!(source.downcast_ref::<TagMismatch>().is_some());
    }
}
