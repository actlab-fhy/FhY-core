//! Data types: the element types of numerical types.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::foreign::Part;
use crate::identifier::Identifier;

use super::core_data_type::CoreDataType;
use super::error::TemplateWidthError;
use super::extension::DataTypeExtension;

/// The element type of a [`NumericalType`](super::NumericalType).
///
/// `==` and `Hash` are structural for the built-in variants, and go
/// through [`DataTypeExtension::eq_part`] and
/// [`DataTypeExtension::hash_part`] for an extension. `Display` writes
/// a primitive type's name, a template's name hint, and an extension's own
/// text.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DataType {
    /// A core data type.
    Primitive(CoreDataType),
    /// A placeholder for a data type, bound during template binding.
    Template(TemplateDataType),
    /// A data type defined outside this crate.
    Extension(Part<dyn DataTypeExtension>),
}

/// A placeholder for a data type, identified by its [`Identifier`], with an
/// optional constraint on the bit widths it may be bound to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TemplateDataType {
    identifier: Identifier,
    widths: Option<Arc<[u32]>>,
}

impl TemplateDataType {
    /// Return the placeholder `identifier`, with no width constraint.
    #[must_use]
    pub fn new(identifier: Identifier) -> Self {
        Self {
            identifier,
            widths: None,
        }
    }

    /// Return the placeholder `identifier`, which only data types of one of
    /// the bit `widths` bind.
    ///
    /// The widths are a set: they are kept sorted and without repeats, so
    /// two templates whose widths differ only in order or repeats are
    /// equal.
    ///
    /// # Errors
    ///
    /// Returns [`TemplateWidthError`] if a width is zero, or if there is no
    /// width, which no data type could bind.
    pub fn with_widths(
        identifier: Identifier,
        widths: impl IntoIterator<Item = u32>,
    ) -> Result<Self, TemplateWidthError> {
        let mut widths: Vec<u32> = widths.into_iter().collect();
        if widths.is_empty() {
            return Err(TemplateWidthError::empty_list());
        }
        if widths.contains(&0) {
            return Err(TemplateWidthError::zero_width());
        }
        widths.sort_unstable();
        widths.dedup();
        Ok(Self {
            identifier,
            widths: Some(widths.into()),
        })
    }

    /// Return the placeholder's identifier.
    #[must_use]
    pub fn identifier(&self) -> &Identifier {
        &self.identifier
    }

    /// Return the width constraint, sorted and without repeats, or `None`
    /// if there is none.
    #[must_use]
    pub fn widths(&self) -> Option<&[u32]> {
        self.widths.as_deref()
    }
}

impl DataType {
    /// Return the name of the data type's kind, as a refusal names it:
    /// `PrimitiveDataType`, `TemplateDataType`, or the extension's name.
    #[must_use]
    pub fn kind_name(&self) -> String {
        match self {
            Self::Primitive(_) => "PrimitiveDataType".to_owned(),
            Self::Template(_) => "TemplateDataType".to_owned(),
            Self::Extension(extension) => extension.get().type_name().into_owned(),
        }
    }
}

impl From<CoreDataType> for DataType {
    fn from(core_data_type: CoreDataType) -> Self {
        Self::Primitive(core_data_type)
    }
}

impl From<TemplateDataType> for DataType {
    fn from(template: TemplateDataType) -> Self {
        Self::Template(template)
    }
}

impl PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Primitive(left), Self::Primitive(right)) => left == right,
            (Self::Template(left), Self::Template(right)) => left == right,
            (Self::Extension(left), Self::Extension(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for DataType {}

impl Hash for DataType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Primitive(core_data_type) => core_data_type.hash(state),
            Self::Template(template) => template.hash(state),
            Self::Extension(extension) => extension.hash(state),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primitive(core_data_type) => write!(f, "{core_data_type}"),
            Self::Template(template) => write!(f, "{}", template.identifier),
            Self::Extension(extension) => write!(f, "{}", extension.get()),
        }
    }
}
