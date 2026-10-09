//! Test implementations of the open traits that have a wire form, and a
//! resolver that reads them back.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hasher;

use fhy_core::constraint::{
    Bindings, Constraint, ConstraintContext, CustomConstraint, OpaqueValue, Outcome, Value,
};
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::foreign::{BoxError, Foreign, ForeignError, ForeignPart, Part, Resolve};
use fhy_core::identifier::Identifier;
use fhy_core::param::{CustomDomain, IntervalProfile, ParamContext, ParamDomain, Side};
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
        Type::Extension(Part::new(Self(label.to_owned())))
    }
}

impl fmt::Display for NamedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "named<{}>", self.0)
    }
}

impl ForeignPart for NamedType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("NamedType")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        if self.0 == REFUSED {
            return Err(failed(NAMED_TYPE));
        }
        Ok(Foreign::new(NAMED_TYPE, self.0.as_str()))
    }
}

impl TypeExtension for NamedType {
    fn eq_part(&self, other: &dyn TypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        state.write(self.0.as_bytes());
    }
}

/// A data type named by its label, whose foreign part is its label.
#[derive(Debug)]
pub(crate) struct NamedDataType(pub(crate) String);

impl NamedDataType {
    /// Return the extension data type labelled `label`.
    pub(crate) fn build(label: &str) -> DataType {
        DataType::Extension(Part::new(Self(label.to_owned())))
    }
}

impl fmt::Display for NamedDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "named_data<{}>", self.0)
    }
}

impl ForeignPart for NamedDataType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("NamedDataType")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(NAMED_DATA_TYPE, self.0.as_str()))
    }
}

impl DataTypeExtension for NamedDataType {
    fn eq_part(&self, other: &dyn DataTypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        state.write(self.0.as_bytes());
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

impl ForeignPart for SilentType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("SilentType")
    }
}

impl TypeExtension for SilentType {}

/// A data type with no wire form: its trait's default `to_foreign`.
#[derive(Debug)]
pub(crate) struct SilentDataType;

impl fmt::Display for SilentDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("silent_data")
    }
}

impl ForeignPart for SilentDataType {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("SilentDataType")
    }
}

impl DataTypeExtension for SilentDataType {}

/// An opaque value equal to another of the same payload, whose foreign
/// part is its payload's digits.
#[derive(Debug)]
pub(crate) struct WireToken(pub(crate) i64);

impl WireToken {
    /// Return the opaque value of `payload`.
    pub(crate) fn value(payload: i64) -> Value {
        Value::Opaque(Part::new(Self(payload)))
    }
}

impl ForeignPart for WireToken {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("WireToken")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(TOKEN, self.0.to_string()))
    }
}

impl OpaqueValue for WireToken {
    fn is_member_shaped(&self) -> bool {
        true
    }

    fn eq_part(&self, other: &dyn OpaqueValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        Ok(())
    }

    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        Ok(Cow::Owned(format!("WireToken:{}", self.0)))
    }
}

/// A custom constraint over no identifiers, keyed by its label, whose
/// foreign part is its label.
#[derive(Debug)]
pub(crate) struct WireCustom(pub(crate) String);

impl WireCustom {
    /// Return the custom constraint labelled `label`.
    pub(crate) fn build(label: &str) -> Constraint {
        Constraint::Custom(Part::new(Self(label.to_owned())))
    }
}

impl ForeignPart for WireCustom {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("WireCustom")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(CUSTOM, self.0.as_str()))
    }
}

impl CustomConstraint for WireCustom {
    fn free_identifiers(&self) -> Result<HashSet<Identifier>, BoxError> {
        Ok(HashSet::new())
    }

    fn evaluate(
        &self,
        _bindings: &Bindings,
        _context: &ConstraintContext<'_>,
    ) -> Result<Outcome, BoxError> {
        Ok(Outcome::Satisfied)
    }

    fn to_expression(&self) -> Result<Expression, BoxError> {
        Ok(Expression::literal(true))
    }

    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        Ok(Cow::Owned(format!("wire|{}", self.0)))
    }

    fn eq_part(&self, other: &dyn CustomConstraint) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        _renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        Ok(self.eq_part(other))
    }
}

/// A custom domain of every integer that admits every constraint, whose
/// foreign part is its label.
#[derive(Debug)]
pub(crate) struct WireDomain(pub(crate) String);

impl WireDomain {
    /// Return the custom domain labelled `label`.
    pub(crate) fn build(label: &str) -> ParamDomain {
        ParamDomain::Custom(Part::new(Self(label.to_owned())))
    }
}

impl ForeignPart for WireDomain {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("WireDomain")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Ok(Foreign::new(DOMAIN, self.0.as_str()))
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

    fn is_value_set_subset(
        &self,
        _other: &ParamDomain,
        _context: &ParamContext<'_>,
    ) -> Result<bool, BoxError> {
        Ok(false)
    }

    fn feasibility_subset(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError> {
        Ok(Outcome::Undecided)
    }

    fn has_feasible_value(
        &self,
        _side: Side<'_>,
        _context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError> {
        Ok(Outcome::Satisfied)
    }

    fn union(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _variable: &Identifier,
        _context: &ParamContext<'_>,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, BoxError> {
        Ok(None)
    }

    fn intersection(
        &self,
        _own: Side<'_>,
        _other_domain: &ParamDomain,
        _other: Side<'_>,
        _variable: &Identifier,
        _context: &ParamContext<'_>,
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError> {
        Ok((Self::build(&self.0), Vec::new()))
    }

    fn eq_part(&self, other: &dyn CustomDomain) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
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

impl Resolve<Part<dyn TypeExtension>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn TypeExtension>, ForeignError> {
        Ok(Part::new(NamedType(read(foreign, NAMED_TYPE)?.to_owned())))
    }
}

impl Resolve<Part<dyn DataTypeExtension>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn DataTypeExtension>, ForeignError> {
        Ok(Part::new(NamedDataType(
            read(foreign, NAMED_DATA_TYPE)?.to_owned(),
        )))
    }
}

impl Resolve<Part<dyn OpaqueValue>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
        let payload = read(foreign, TOKEN)?;
        let payload = payload.parse().map_err(|_refused| failed(TOKEN))?;
        Ok(Part::new(WireToken(payload)))
    }
}

impl Resolve<Part<dyn CustomConstraint>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
        Ok(Part::new(WireCustom(read(foreign, CUSTOM)?.to_owned())))
    }
}

impl Resolve<Part<dyn CustomDomain>> for TestResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
        Ok(Part::new(WireDomain(read(foreign, DOMAIN)?.to_owned())))
    }
}

/// The type id 0.2.0 writes an identifier member's opaque part under.
pub(crate) const LEGACY_IDENTIFIER: &str = "id";

/// An identifier as 0.2.0 held it, an opaque value whose foreign part is
/// the identifier's payload, `{"id": .., "name_hint": ..}`, and which
/// reports the identifier it stands for.
#[derive(Debug)]
pub(crate) struct LegacyIdentifier(pub(crate) Identifier);

impl LegacyIdentifier {
    /// Return the opaque value of `identifier`.
    pub(crate) fn value(identifier: &Identifier) -> Value {
        Value::Opaque(Part::new(Self(identifier.clone())))
    }

    /// Return the legacy wire form of `identifier`, as 0.2.0 writes it.
    pub(crate) fn wire(identifier: &Identifier) -> String {
        let payload = serde_json::to_string(identifier).expect("an identifier encodes");
        serde_json::json!({"opaque": {"type_id": LEGACY_IDENTIFIER, "data": payload}}).to_string()
    }
}

impl ForeignPart for LegacyIdentifier {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Identifier")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        let payload = serde_json::to_string(&self.0).map_err(|error| ForeignError::Failed {
            type_id: LEGACY_IDENTIFIER.to_owned(),
            source: Box::new(error),
        })?;
        Ok(Foreign::new(LEGACY_IDENTIFIER, payload))
    }
}

impl OpaqueValue for LegacyIdentifier {
    fn is_member_shaped(&self) -> bool {
        true
    }

    fn eq_part(&self, other: &dyn OpaqueValue) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.0 == self.0)
    }

    fn hash_part(&self, mut state: &mut dyn Hasher) {
        std::hash::Hash::hash(&self.0, &mut state);
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        Ok(())
    }

    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        Ok(Cow::Owned(format!("Identifier:{}", self.0.id())))
    }

    fn identifier(&self) -> Option<Identifier> {
        Some(self.0.clone())
    }
}

/// The resolver of 0.2.0's payloads: it reads a [`LEGACY_IDENTIFIER`] part
/// as a [`LegacyIdentifier`], and every other part as [`TestResolver`]
/// does.
#[derive(Debug, Default)]
pub(crate) struct LegacyResolver;

impl Resolve<Part<dyn OpaqueValue>> for LegacyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
        if foreign.type_id() != LEGACY_IDENTIFIER {
            return TestResolver.resolve(foreign);
        }
        let identifier: Identifier =
            serde_json::from_str(foreign.data()).map_err(|error| ForeignError::Failed {
                type_id: LEGACY_IDENTIFIER.to_owned(),
                source: Box::new(error),
            })?;
        Ok(Part::new(LegacyIdentifier(identifier)))
    }
}

impl Resolve<Part<dyn CustomConstraint>> for LegacyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
        TestResolver.resolve(foreign)
    }
}

impl Resolve<Part<dyn CustomDomain>> for LegacyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
        TestResolver.resolve(foreign)
    }
}
