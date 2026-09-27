//! The binding environment of template binding, substitution and
//! unification.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::expression::Expression;
use crate::identifier::Identifier;

use super::data_type::DataType;
use super::error::UnificationError;
use super::ty::Type;

/// The placeholder bindings learned while binding, substituting or unifying
/// types, in three tables keyed by [`Identifier`]:
///
/// - data-type bindings, of [`TemplateDataType`](super::TemplateDataType)
///   placeholders;
/// - type bindings, of full-type placeholders: a template data type over the
///   full-shape wildcard, which captures a whole type;
/// - expression bindings, of shape variables.
///
/// An environment is a value: the `with_*` methods return a new one, and
/// cloning shares the tables. `==` compares the tables, and `Hash` does not
/// depend on the order of their entries.
#[expect(
    clippy::struct_field_names,
    reason = "the three tables are named after the three kinds of binding"
)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeUnificationEnvironment {
    data_type_bindings: Arc<HashMap<Identifier, DataType>>,
    type_bindings: Arc<HashMap<Identifier, Type>>,
    expression_bindings: Arc<HashMap<Identifier, Expression>>,
}

impl TypeUnificationEnvironment {
    /// Return the environment with no bindings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the environment of the three tables.
    #[must_use]
    pub fn from_bindings(
        data_type_bindings: HashMap<Identifier, DataType>,
        type_bindings: HashMap<Identifier, Type>,
        expression_bindings: HashMap<Identifier, Expression>,
    ) -> Self {
        Self {
            data_type_bindings: Arc::new(data_type_bindings),
            type_bindings: Arc::new(type_bindings),
            expression_bindings: Arc::new(expression_bindings),
        }
    }

    /// Return this environment with `identifier` bound to the data type
    /// `value`, replacing a binding it has.
    #[must_use]
    pub fn with_data_type_binding(&self, identifier: Identifier, value: DataType) -> Self {
        let mut next = self.clone();
        Arc::make_mut(&mut next.data_type_bindings).insert(identifier, value);
        next
    }

    /// Return this environment with `identifier` bound to the type `value`,
    /// replacing a binding it has.
    #[must_use]
    pub fn with_type_binding(&self, identifier: Identifier, value: Type) -> Self {
        let mut next = self.clone();
        Arc::make_mut(&mut next.type_bindings).insert(identifier, value);
        next
    }

    /// Return this environment with `identifier` bound to the expression
    /// `value`, replacing a binding it has.
    #[must_use]
    pub fn with_expression_binding(&self, identifier: Identifier, value: Expression) -> Self {
        let mut next = self.clone();
        Arc::make_mut(&mut next.expression_bindings).insert(identifier, value);
        next
    }

    /// Return the data type `identifier` is bound to.
    #[must_use]
    pub fn data_type_binding(&self, identifier: &Identifier) -> Option<&DataType> {
        self.data_type_bindings.get(identifier)
    }

    /// Return the type `identifier` is bound to.
    #[must_use]
    pub fn type_binding(&self, identifier: &Identifier) -> Option<&Type> {
        self.type_bindings.get(identifier)
    }

    /// Return the expression `identifier` is bound to.
    #[must_use]
    pub fn expression_binding(&self, identifier: &Identifier) -> Option<&Expression> {
        self.expression_bindings.get(identifier)
    }

    /// Return the data-type bindings, in no particular order.
    #[must_use]
    pub fn data_type_bindings(&self) -> impl ExactSizeIterator<Item = (&Identifier, &DataType)> {
        self.data_type_bindings.iter()
    }

    /// Return the type bindings, in no particular order.
    #[must_use]
    pub fn type_bindings(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Type)> {
        self.type_bindings.iter()
    }

    /// Return the expression bindings, in no particular order.
    #[must_use]
    pub fn expression_bindings(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Expression)> {
        self.expression_bindings.iter()
    }

    /// Return whether `other` binds the same identifiers in each table, to
    /// structurally equivalent values.
    ///
    /// # Errors
    ///
    /// Returns [`UnificationError::Extension`] for an extension that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, UnificationError> {
        Ok(is_table_equivalent(
            &self.data_type_bindings,
            &other.data_type_bindings,
            DataType::is_structurally_equivalent,
        )? && is_table_equivalent(
            &self.type_bindings,
            &other.type_bindings,
            Type::is_structurally_equivalent,
        )? && is_table_equivalent(
            &self.expression_bindings,
            &other.expression_bindings,
            |left, right| Ok(left == right),
        )?)
    }
}

/// Return whether two tables have the same keys and equivalent values.
fn is_table_equivalent<V>(
    left: &HashMap<Identifier, V>,
    right: &HashMap<Identifier, V>,
    is_equivalent: impl Fn(&V, &V) -> Result<bool, UnificationError>,
) -> Result<bool, UnificationError> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (identifier, value) in left {
        match right.get(identifier) {
            Some(other) if is_equivalent(value, other)? => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// Return the order-independent hash of a table's entries.
fn table_hash<V: Hash>(table: &HashMap<Identifier, V>) -> u64 {
    table.iter().fold(0_u64, |sum, entry| {
        let mut hasher = DefaultHasher::new();
        entry.hash(&mut hasher);
        sum.wrapping_add(hasher.finish())
    })
}

impl Hash for TypeUnificationEnvironment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        table_hash(&self.data_type_bindings).hash(state);
        table_hash(&self.type_bindings).hash(state);
        table_hash(&self.expression_bindings).hash(state);
    }
}
