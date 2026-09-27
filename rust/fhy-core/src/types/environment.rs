//! The binding environment of template binding, substitution and
//! unification.

use std::collections::HashMap;
use std::fmt;
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
/// cloning shares the tables. Each table is persistent, a stack of shared
/// layers that are merged as they grow, so a `with_*` copies no whole table:
/// it costs amortized time logarithmic in the table's size, and so does a
/// lookup. `==` compares the tables, and `Hash` does not depend on the
/// order of their entries.
#[expect(
    clippy::struct_field_names,
    reason = "the three tables are named after the three kinds of binding"
)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeUnificationEnvironment {
    data_type_bindings: Table<DataType>,
    type_bindings: Table<Type>,
    expression_bindings: Table<Expression>,
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
            data_type_bindings: Table::from_map(data_type_bindings),
            type_bindings: Table::from_map(type_bindings),
            expression_bindings: Table::from_map(expression_bindings),
        }
    }

    /// Return this environment with `identifier` bound to the data type
    /// `value`, replacing a binding it has.
    #[must_use]
    pub fn with_data_type_binding(&self, identifier: Identifier, value: DataType) -> Self {
        Self {
            data_type_bindings: self.data_type_bindings.insert(identifier, value),
            ..self.clone()
        }
    }

    /// Return this environment with `identifier` bound to the type `value`,
    /// replacing a binding it has.
    #[must_use]
    pub fn with_type_binding(&self, identifier: Identifier, value: Type) -> Self {
        Self {
            type_bindings: self.type_bindings.insert(identifier, value),
            ..self.clone()
        }
    }

    /// Return this environment with `identifier` bound to the expression
    /// `value`, replacing a binding it has.
    #[must_use]
    pub fn with_expression_binding(&self, identifier: Identifier, value: Expression) -> Self {
        Self {
            expression_bindings: self.expression_bindings.insert(identifier, value),
            ..self.clone()
        }
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

/// A persistent table: a stack of shared layers, the newest first, where an
/// entry of a newer layer shadows one of an older layer under its key.
///
/// A new entry is a layer of its own, merged into the layer below while
/// that one is no larger, as a binary counter carries, so the table holds
/// logarithmically many layers and each entry is copied logarithmically
/// often. The layers are shared between tables, never mutated.
#[derive(Clone)]
struct Table<V> {
    top: Option<Arc<Layer<V>>>,
}

/// One layer of a [`Table`].
struct Layer<V> {
    entries: HashMap<Identifier, V>,
    below: Option<Arc<Layer<V>>>,
    /// The number of keys of this layer and the layers below, each once.
    len: usize,
}

impl<V> Default for Table<V> {
    fn default() -> Self {
        Self { top: None }
    }
}

impl<V: Clone> Table<V> {
    /// Return the table of the entries of `map`.
    fn from_map(map: HashMap<Identifier, V>) -> Self {
        if map.is_empty() {
            return Self::default();
        }
        Self {
            top: Some(Arc::new(Layer {
                len: map.len(),
                entries: map,
                below: None,
            })),
        }
    }

    /// Return this table with `key` mapped to `value`.
    fn insert(&self, key: Identifier, value: V) -> Self {
        let len = self.len() + usize::from(self.get(&key).is_none());
        let mut entries = HashMap::from([(key, value)]);
        let mut below = self.top.clone();
        while let Some(layer) = below.take_if(|layer| layer.entries.len() <= entries.len()) {
            let mut merged = layer.entries.clone();
            merged.extend(entries);
            entries = merged;
            below.clone_from(&layer.below);
        }
        Self {
            top: Some(Arc::new(Layer {
                entries,
                below,
                len,
            })),
        }
    }
}

impl<V> Table<V> {
    /// Return the number of keys.
    fn len(&self) -> usize {
        self.top.as_ref().map_or(0, |layer| layer.len)
    }

    /// Return the layers, the newest first.
    fn layers(&self) -> impl Iterator<Item = &Layer<V>> {
        std::iter::successors(self.top.as_deref(), |layer| layer.below.as_deref())
    }

    /// Return the value of `key`.
    fn get(&self, key: &Identifier) -> Option<&V> {
        self.layers().find_map(|layer| layer.entries.get(key))
    }

    /// Return the entries, each key once with its newest value, in no
    /// particular order.
    fn iter(&self) -> impl ExactSizeIterator<Item = (&Identifier, &V)> {
        let layers: Vec<&Layer<V>> = self.layers().collect();
        let mut entries = Vec::with_capacity(self.len());
        for (depth, layer) in layers.iter().enumerate() {
            entries.extend(layer.entries.iter().filter(|(key, _)| {
                !layers[..depth]
                    .iter()
                    .any(|newer| newer.entries.contains_key(*key))
            }));
        }
        entries.into_iter()
    }
}

impl<V: PartialEq> PartialEq for Table<V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<V: Eq> Eq for Table<V> {}

impl<V: fmt::Debug> fmt::Debug for Table<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Return whether two tables have the same keys and equivalent values.
fn is_table_equivalent<V>(
    left: &Table<V>,
    right: &Table<V>,
    is_equivalent: impl Fn(&V, &V) -> Result<bool, UnificationError>,
) -> Result<bool, UnificationError> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (identifier, value) in left.iter() {
        match right.get(identifier) {
            Some(other) if is_equivalent(value, other)? => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// Return the order-independent hash of a table's entries.
fn table_hash<V: Hash>(table: &Table<V>) -> u64 {
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
