//! Bindings: the values a constraint is evaluated under.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::expression::Expression;
use crate::identifier::Identifier;

use super::value::Value;

/// A value bound to an identifier: an expression, or any other [`Value`].
///
/// A binding is not judged when it is made: only a constraint that reads
/// it decides whether it can use it.
#[expect(
    clippy::exhaustive_enums,
    reason = "a binding is an expression or a value, and nothing else"
)]
#[derive(Debug, Clone)]
pub enum Binding {
    /// An expression, symbolic or literal.
    Expression(Expression),
    /// A value that is not an expression.
    Value(Value),
}

impl From<Expression> for Binding {
    fn from(expression: Expression) -> Self {
        Self::Expression(expression)
    }
}

impl From<Value> for Binding {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

/// Bindings of identifiers, in the order they were made.
///
/// Binding an identifier again replaces its binding and keeps its place.
///
/// Bindings may also carry their caller's own form of them, its
/// [`source`](Self::source). It is the channel of a language binding's own
/// adapter: a binding that builds the bindings reads its source back in
/// its [`CustomConstraint`](super::CustomConstraint)s, which may decide from
/// values the core does not model. Core code never reads it.
#[derive(Clone, Default)]
pub struct Bindings {
    entries: Vec<(Identifier, Binding)>,
    positions: HashMap<Identifier, usize>,
    source: Option<Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for Bindings {
    /// Write the bindings, not their source.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bindings")
            .field("entries", &self.entries)
            .finish_non_exhaustive()
    }
}

impl Bindings {
    /// Return bindings of no identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind `identifier` to `binding`.
    pub fn insert(&mut self, identifier: Identifier, binding: impl Into<Binding>) {
        let binding = binding.into();
        if let Some(&position) = self.positions.get(&identifier) {
            self.entries[position].1 = binding;
        } else {
            self.positions
                .insert(identifier.clone(), self.entries.len());
            self.entries.push((identifier, binding));
        }
    }

    /// Return the binding of `identifier`, if any.
    #[must_use]
    pub fn get(&self, identifier: &Identifier) -> Option<&Binding> {
        self.positions
            .get(identifier)
            .map(|&position| &self.entries[position].1)
    }

    /// Return the bindings in the order they were made.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Binding)> + '_ {
        self.entries
            .iter()
            .map(|(identifier, binding)| (identifier, binding))
    }

    /// Return these bindings carrying `source`, their caller's own form.
    #[must_use]
    pub fn with_source(self, source: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            source: Some(source),
            ..self
        }
    }

    /// Return the caller's own form of the bindings, if it gave one.
    ///
    /// Only the binding that set it reads it (see the type's documentation);
    /// a Rust implementor decides from the bindings themselves.
    #[must_use]
    pub fn source(&self) -> Option<&(dyn Any + Send + Sync)> {
        self.source.as_deref()
    }

    /// Return the number of bound identifiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether no identifier is bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<B: Into<Binding>> FromIterator<(Identifier, B)> for Bindings {
    fn from_iter<I: IntoIterator<Item = (Identifier, B)>>(entries: I) -> Self {
        let mut bindings = Self::new();
        for (identifier, binding) in entries {
            bindings.insert(identifier, binding);
        }
        bindings
    }
}
