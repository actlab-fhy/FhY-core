//! An owned registry of user functions and constants, and inlining through
//! it.
//!
//! A [`FunctionRegistry`] holds the three kinds of entries a call or a
//! reference may resolve to beyond the built-in catalogue
//! ([`builtins`](super::builtins)):
//!
//! - a [`FunctionDefinition`], a function defined by an expression over its
//!   parameters;
//! - a [`NativeFunction`], a function computed outside the expression
//!   language, of which the registry knows only the signature;
//! - a [`NativeConstant`], a named literal, which the registry gives an
//!   identifier when it registers it.
//!
//! Every entry is keyed by a [`FunctionName`]: functions and constants share
//! one namespace, the one calls name functions in. A built-in function's
//! name is no [`FunctionName`] (D-9), and the registry refuses a built-in
//! constant's name too, so a name never means both a built-in and an entry.
//! Built-ins are never entries: [`inline`](FunctionRegistry::inline) and the
//! screens read them from the catalogue.
//!
//! A registry is an ordinary owned value, `Send + Sync`, and cloning one is
//! cheap: its entries are shared behind `Arc`s, and a clone is an
//! independent registry.
//!
//! # Examples
//!
//! ```
//! use fhy_core::expression::registry::{FunctionDefinition, FunctionRegistry};
//! use fhy_core::expression::{Expression, FunctionName, FunctionSort};
//! use fhy_core::identifier::Identifier;
//!
//! let mut registry = FunctionRegistry::new();
//! let x = Identifier::new("x");
//! registry.register_function(FunctionDefinition::try_new(
//!     FunctionName::try_new("double")?,
//!     [x.clone()],
//!     [FunctionSort::Real],
//!     FunctionSort::Real,
//!     Expression::from(x) * 2,
//! )?)?;
//!
//! let y = Identifier::new("y");
//! let call = Expression::call(FunctionName::try_new("double")?, [Expression::from(y.clone())]);
//! assert_eq!(registry.inline(&call)?, Expression::from(y) * 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod definition;
mod error;
mod inline;

use std::collections::HashMap;

use crate::identifier::Identifier;

use super::builtins::BuiltinConstant;
use super::callee::FunctionName;
use super::node::Expression;
use super::screen::SortLookup;
use super::sort::FunctionSort;

pub use definition::{FunctionDefinition, NativeConstant, NativeFunction};
pub use error::{ConstantValueError, FunctionDefinitionError, InlineError, RegistrationError};

/// One stored entry, in the registry's own representation.
#[derive(Debug, Clone)]
enum StoredEntry {
    Function(FunctionDefinition),
    Native(NativeFunction),
    Constant(NativeConstant, Identifier),
}

impl StoredEntry {
    /// Return the entry's name.
    fn name(&self) -> &FunctionName {
        match self {
            Self::Function(function) => function.name(),
            Self::Native(function) => function.name(),
            Self::Constant(constant, _) => constant.name(),
        }
    }

    /// Return the borrowed view of the entry.
    fn view(&self) -> RegistryEntry<'_> {
        match self {
            Self::Function(function) => RegistryEntry::Function(function),
            Self::Native(function) => RegistryEntry::Native(function),
            Self::Constant(constant, identifier) => RegistryEntry::Constant(constant, identifier),
        }
    }
}

/// An entry of a [`FunctionRegistry`], borrowed from it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RegistryEntry<'r> {
    /// A function defined by an expression.
    Function(&'r FunctionDefinition),
    /// A function computed natively.
    Native(&'r NativeFunction),
    /// A constant, with the identifier the registry minted for it.
    Constant(&'r NativeConstant, &'r Identifier),
}

impl<'r> RegistryEntry<'r> {
    /// Return the entry's name.
    #[must_use]
    pub fn name(&self) -> &'r FunctionName {
        match *self {
            Self::Function(function) => function.name(),
            Self::Native(function) => function.name(),
            Self::Constant(constant, _) => constant.name(),
        }
    }
}

/// The user functions and constants that calls and references resolve to,
/// in registration order.
///
/// See the [module documentation](self) for what it holds.
#[derive(Debug, Clone, Default)]
pub struct FunctionRegistry {
    entries: Vec<StoredEntry>,
    by_name: HashMap<String, usize>,
    constants: HashMap<Identifier, usize>,
}

impl FunctionRegistry {
    /// Create the registry holding no entry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the user function `function`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::NameTaken`] if an entry has the
    /// function's name, [`RegistrationError::BuiltinConstantName`] if the
    /// name is a built-in constant's, and
    /// [`RegistrationError::CapturedIdentifiers`] if the body refers to an
    /// identifier that is neither a parameter, nor the identifier of a
    /// constant registered so far, nor a built-in constant's
    /// ([`BuiltinConstant::identifier`]). The last check reads the registry
    /// as it is, so register a constant before the functions that refer to
    /// it.
    pub fn register_function(
        &mut self,
        function: FunctionDefinition,
    ) -> Result<(), RegistrationError> {
        self.check_name_is_free(function.name())?;
        let mut captured: Vec<Identifier> = function
            .body()
            .free_identifiers()
            .into_iter()
            .filter(|identifier| {
                !function.parameters().contains(identifier)
                    && !self.constants.contains_key(identifier)
                    && BuiltinConstant::of_identifier(identifier).is_none()
            })
            .collect();
        if !captured.is_empty() {
            captured.sort_by(|left, right| {
                left.name_hint()
                    .cmp(right.name_hint())
                    .then_with(|| left.id().cmp(&right.id()))
            });
            return Err(RegistrationError::CapturedIdentifiers {
                function: function.name().clone(),
                identifiers: captured,
            });
        }
        self.push(StoredEntry::Function(function));
        Ok(())
    }

    /// Register the native function `function`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::NameTaken`] if an entry has the
    /// function's name, and [`RegistrationError::BuiltinConstantName`] if
    /// the name is a built-in constant's.
    pub fn register_native_function(
        &mut self,
        function: NativeFunction,
    ) -> Result<(), RegistrationError> {
        self.check_name_is_free(function.name())?;
        self.push(StoredEntry::Native(function));
        Ok(())
    }

    /// Register the constant `constant`, minting the identifier an
    /// expression refers to it by, and return that identifier.
    ///
    /// The identifier's name hint is the constant's name. It is a new
    /// identifier, so an identifier created elsewhere with the same name
    /// hint never refers to the constant.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::NameTaken`] if an entry has the
    /// constant's name, and [`RegistrationError::BuiltinConstantName`] if
    /// the name is a built-in constant's.
    pub fn register_constant(
        &mut self,
        constant: NativeConstant,
    ) -> Result<Identifier, RegistrationError> {
        self.check_name_is_free(constant.name())?;
        let identifier = Identifier::new(constant.name().as_str());
        self.push(StoredEntry::Constant(constant, identifier.clone()));
        Ok(identifier)
    }

    /// Return the entry registered under `name`, or `None`.
    ///
    /// A built-in's name finds nothing: built-ins are no entries.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<RegistryEntry<'_>> {
        self.find(name).map(StoredEntry::view)
    }

    /// Return whether an entry is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Return the entries in registration order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = RegistryEntry<'_>> + '_ {
        self.entries.iter().map(StoredEntry::view)
    }

    /// Return the identifier the constant `name` was registered with, or
    /// `None` if no constant is registered under `name`.
    #[must_use]
    pub fn constant_identifier(&self, name: &str) -> Option<&Identifier> {
        match self.find(name)? {
            StoredEntry::Constant(_, identifier) => Some(identifier),
            StoredEntry::Function(_) | StoredEntry::Native(_) => None,
        }
    }

    /// Return the constant `identifier` refers to, or `None` if it is no
    /// registered constant's identifier.
    ///
    /// Only the identifier the registry minted refers to a constant; one
    /// that merely shares its name hint does not.
    #[must_use]
    pub fn constant(&self, identifier: &Identifier) -> Option<&NativeConstant> {
        match &self.entries[*self.constants.get(identifier)?] {
            StoredEntry::Constant(constant, _) => Some(constant),
            StoredEntry::Function(_) | StoredEntry::Native(_) => None,
        }
    }

    /// Return the result sort of the function registered under `name`, or
    /// `None` if `name` is unregistered or names a constant.
    #[must_use]
    pub fn result_sort(&self, name: &FunctionName) -> Option<FunctionSort> {
        match self.find(name.as_str())? {
            StoredEntry::Function(function) => Some(function.result_sort()),
            StoredEntry::Native(function) => Some(function.result_sort()),
            StoredEntry::Constant(..) => None,
        }
    }

    /// Return the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the registry holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keep only the entries for which `keep` returns `true`, in their
    /// order, each constant with its identifier.
    ///
    /// Nothing is checked again: a kept function may refer to a constant
    /// that is dropped, and its references then resolve to nothing.
    pub fn retain(&mut self, mut keep: impl FnMut(RegistryEntry<'_>) -> bool) {
        let entries = std::mem::take(&mut self.entries);
        self.by_name.clear();
        self.constants.clear();
        for entry in entries {
            if keep(entry.view()) {
                self.push(entry);
            }
        }
    }

    /// Replace every call of a composed built-in or of a user function in
    /// `expression` by the function's body over the call's arguments, and
    /// inline the result in turn.
    ///
    /// Arguments are inlined before they are substituted. A call of a
    /// native built-in or of a native user function is kept, with its
    /// arguments inlined, once its argument count is checked. A node that
    /// occurs in several places is inlined once and its result reused, so a
    /// shared argument substituted into a body that uses it twice stays one
    /// node, and the work is linear in the distinct nodes of the input and
    /// of the bodies it expands to. The walk keeps its pending nodes on the
    /// heap, so a tree of any depth inlines.
    ///
    /// Returns a handle to `expression` itself ([`Expression::ptr_eq`])
    /// when it calls no function to inline, and otherwise shares every
    /// subtree that has none.
    ///
    /// # Errors
    ///
    /// Returns [`InlineError::UnknownFunction`] for a call of an
    /// unregistered name, [`InlineError::NotCallable`] for a call of a
    /// registered or built-in constant's name, [`InlineError::ArityMismatch`] for a call passing a
    /// wrong number of arguments, [`InlineError::Recursive`] for a function
    /// reached again inside its own body, and [`InlineError::Piecewise`] if
    /// substituting an argument puts a literal other than a Boolean in a
    /// piecewise case condition. Arguments are checked before the call
    /// taking them.
    pub fn inline(&self, expression: &Expression) -> Result<Expression, InlineError> {
        inline::inline(self, expression)
    }

    /// Return the stored entry named `name`.
    fn find(&self, name: &str) -> Option<&StoredEntry> {
        self.by_name.get(name).map(|&index| &self.entries[index])
    }

    /// Refuse `name` if an entry or a built-in constant has it.
    fn check_name_is_free(&self, name: &FunctionName) -> Result<(), RegistrationError> {
        if let Some(constant) = find_builtin_constant(name) {
            return Err(RegistrationError::BuiltinConstantName(constant));
        }
        if self.contains(name.as_str()) {
            return Err(RegistrationError::NameTaken(name.clone()));
        }
        Ok(())
    }

    /// Append `entry`, whose name is free, and index it.
    fn push(&mut self, entry: StoredEntry) {
        let index = self.entries.len();
        self.by_name.insert(entry.name().as_str().to_owned(), index);
        if let StoredEntry::Constant(_, identifier) = &entry {
            self.constants.insert(identifier.clone(), index);
        }
        self.entries.push(entry);
    }
}

impl SortLookup for FunctionRegistry {
    /// Return the sort of the registered constant `identifier` refers to.
    fn native_constant_sort(&self, identifier: &Identifier) -> Option<FunctionSort> {
        self.constant(identifier).map(NativeConstant::sort)
    }

    /// Return the result sort of the function registered under `name`.
    fn call_result_sort(&self, name: &FunctionName) -> Option<FunctionSort> {
        self.result_sort(name)
    }
}

/// Return the built-in constant named `name`, if any.
fn find_builtin_constant(name: &FunctionName) -> Option<BuiltinConstant> {
    name.as_str().parse().ok()
}
