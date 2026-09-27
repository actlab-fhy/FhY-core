//! Test implementations of the open traits that have a wire form, and a
//! resolver that reads them back.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hasher;
use std::sync::Arc;

use fhy_core::constraint::{
    Bindings, Constraint, CustomConstraint, Opaque, OpaqueValue, Outcome, Value,
};
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::foreign::BoxError;
use fhy_core::foreign::{Foreign, ForeignError, Resolve};
use fhy_core::identifier::Identifier;
use fhy_core::param::{CustomDomain, IntervalProfile, ParamDomain, Side};
use fhy_core::term::AlphaRenaming;
use fhy_core::types::{DataType, DataTypeExtension, Type, TypeExtension};

use super::constraint::TestValueError;

/// The type id of [`NamedType`].
pub(crate) const NAMED_TYPE: &str = "test.named_type";
/// The type id of [`NamedDataType`].
pub(crate) const NAMED_DATA_TYPE: &str = "test.named_data_type";
/// The type id of [`WireToken`].
pub(crate) const TOKEN: &str = "test.token";
/// The type id of [`WireCustom`].
pub(crate) const CUSTOM: &str = "test.custom";
/// The type id of [`WireDomain`].
pub(crate) const DOMAIN: &str = "test.domain";
/// A payload the resolver refuses with [`ForeignError::Failed`].
pub(crate) const REFUSED: &str = "refused";

/// The error of a part that fails.
fn failed(type_id: &str) -> ForeignError {
    ForeignError::Failed {
        type_id: type_id.to_owned(),
        source: Box::new(TestValueError(format!("{type_id} failed"))),
    }
}

/// A type named by its label, whose foreign part is its label; a label of
/// [`REFUSED`] fails to give it.
#[derive(Debug)]
pub(crate) struct NamedType(pub(crate) String);

impl NamedType {
    /// Return the extension type labelled `label`.
    pub(crate) fn build(label: &str) -> Type {
        Type::Extension(Arc::new(Self(label.to_owned())))
    }
}

impl fmt::Display for NamedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "named<{}>", self.0)
    }
}

impl TypeExtension for NamedType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("NamedType")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq_extension(&self, other: &dyn TypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn hash_extension(&self, state: &mut dyn Hasher) {
        state.write(self.0.as_bytes());
    }

    fn is_structurally_equivalent(&self, other: &Type) -> bool {
        matches!(other, Type::Extension(other) if self.eq_extension(other.as_ref()))
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        if self.0 == REFUSED {
            return Err(failed(NAMED_TYPE));
        }
        Ok(Foreign::new(NAMED_TYPE, self.0.as_str()))
    }
}

/// A data type named by its label, whose foreign part is its label.
#[derive(Debug)]
pub(crate) struct NamedDataType(pub(crate) String);

impl NamedDataType {
    /// Return the extension data type labelled `label`.
    pub(crate) fn build(label: &str) -> DataType {
        DataType::Extension(Arc::new(Self(label.to_owned())))
    }
}

impl fmt::Display for NamedDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "named_data<{}>", self.0)
    }
}

impl DataTypeExtension for NamedDataType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("NamedDataType")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq_extension(&self, other: &dyn DataTypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn hash_extension(&self, state: &mut dyn Hasher) {
        state.write(self.0.as_bytes());
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(NAMED_DATA_TYPE, self.0.as_str()))
    }
}

/// A type with no wire form: its trait's default `to_foreign`.
#[derive(Debug)]
pub(crate) struct SilentType;

impl fmt::Display for SilentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("silent")
    }
}

impl TypeExtension for SilentType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("SilentType")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A data type with no wire form: its trait's default `to_foreign`.
#[derive(Debug)]
pub(crate) struct SilentDataType;

impl fmt::Display for SilentDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("silent_data")
    }
}

impl DataTypeExtension for SilentDataType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("SilentDataType")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An opaque value equal to another of the same payload, whose foreign
/// part is its payload's digits.
#[derive(Debug)]
pub(crate) struct WireToken(pub(crate) i64);

impl WireToken {
    /// Return the opaque value of `payload`.
    pub(crate) fn value(payload: i64) -> Value {
        Value::Opaque(Opaque::new(Self(payload)))
    }
}

impl OpaqueValue for WireToken {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("WireToken")
    }

    fn is_member_shaped(&self) -> bool {
        true
    }

    fn is_equal(&self, other: &dyn OpaqueValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        Ok(())
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("WireToken:{}", self.0))
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(TOKEN, self.0.to_string()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A custom constraint over no identifiers, keyed by its label, whose
/// foreign part is its label.
#[derive(Debug)]
pub(crate) struct WireCustom(pub(crate) String);

impl WireCustom {
    /// Return the custom constraint labelled `label`.
    pub(crate) fn build(label: &str) -> Constraint {
        Constraint::Custom(Arc::new(Self(label.to_owned())))
    }
}

impl CustomConstraint for WireCustom {
    fn free_identifiers(&self) -> HashSet<Identifier> {
        HashSet::new()
    }

    fn evaluate(&self, _bindings: &Bindings) -> Result<Outcome, BoxError> {
        Ok(Outcome::Satisfied)
    }

    fn to_expression(&self) -> Result<Expression, BoxError> {
        Ok(Expression::literal(true))
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("wire|{}", self.0))
    }

    fn is_structurally_equivalent(&self, other: &dyn CustomConstraint) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        _renaming: &AlphaRenaming,
    ) -> bool {
        self.is_structurally_equivalent(other)
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(CUSTOM, self.0.as_str()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A custom domain of every integer that admits every constraint, whose
/// foreign part is its label.
#[derive(Debug)]
pub(crate) struct WireDomain(pub(crate) String);

impl WireDomain {
    /// Return the custom domain labelled `label`.
    pub(crate) fn build(label: &str) -> ParamDomain {
        ParamDomain::Custom(Arc::new(Self(label.to_owned())))
    }
}

impl CustomDomain for WireDomain {
    fn symbol_type(&self) -> Result<Option<SymbolType>, BoxError> {
        Ok(Some(SymbolType::Int))
    }

    fn is_value_admissible(&self, value: &Value) -> Result<bool, BoxError> {
        Ok(matches!(value, Value::Int(_)))
    }

    fn validate_constraint(
        &self,
        _constraint: &Constraint,
        _variable: &Identifier,
    ) -> Result<(), BoxError> {
        Ok(())
    }

    fn implied_constraints(&self, _variable: &Identifier) -> Result<Vec<Constraint>, BoxError> {
        Ok(Vec::new())
    }

    fn interval_profile(&self) -> Result<Option<IntervalProfile>, BoxError> {
        Ok(None)
    }

    fn is_value_set_subset(&self, _other: &ParamDomain) -> Result<bool, BoxError> {
        Ok(false)
    }

    fn feasibility_subset(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
    ) -> Result<Outcome, BoxError> {
        Ok(Outcome::Undecided)
    }

    fn has_feasible_value(&self, _side: Side<'_>) -> Result<Outcome, BoxError> {
        Ok(Outcome::Satisfied)
    }

    fn union(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _variable: &Identifier,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, BoxError> {
        Ok(None)
    }

    fn intersection(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _variable: &Identifier,
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError> {
        Ok((Self::build(&self.0), Vec::new()))
    }

    fn is_structurally_equivalent(&self, other: &ParamDomain) -> bool {
        matches!(
            other,
            ParamDomain::Custom(other)
                if other.as_any().downcast_ref::<Self>().is_some_and(|other| other.0 == self.0)
        )
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(DOMAIN, self.0.as_str()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The resolver of the parts above, by type id; it refuses any other id,
/// and fails on the payload [`REFUSED`].
#[derive(Debug, Default)]
pub(crate) struct TestResolver;

/// Check `foreign` is of `type_id` and not refused, returning its payload.
fn read<'a>(foreign: &'a Foreign, type_id: &str) -> Result<&'a str, ForeignError> {
    if foreign.type_id() != type_id {
        return Err(ForeignError::Unresolved {
            type_id: foreign.type_id().to_owned(),
        });
    }
    if foreign.data() == REFUSED {
        return Err(failed(type_id));
    }
    Ok(foreign.data())
}

impl Resolve<Arc<dyn TypeExtension>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn TypeExtension>, ForeignError> {
        Ok(Arc::new(NamedType(read(foreign, NAMED_TYPE)?.to_owned())))
    }
}

impl Resolve<Arc<dyn DataTypeExtension>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn DataTypeExtension>, ForeignError> {
        Ok(Arc::new(NamedDataType(
            read(foreign, NAMED_DATA_TYPE)?.to_owned(),
        )))
    }
}

impl Resolve<Opaque> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Opaque, ForeignError> {
        let payload = read(foreign, TOKEN)?;
        let payload = payload.parse().map_err(|_refused| failed(TOKEN))?;
        Ok(Opaque::new(WireToken(payload)))
    }
}

impl Resolve<Arc<dyn CustomConstraint>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn CustomConstraint>, ForeignError> {
        Ok(Arc::new(WireCustom(read(foreign, CUSTOM)?.to_owned())))
    }
}

impl Resolve<Arc<dyn CustomDomain>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn CustomDomain>, ForeignError> {
        Ok(Arc::new(WireDomain(read(foreign, DOMAIN)?.to_owned())))
    }
}
