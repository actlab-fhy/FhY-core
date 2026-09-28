//! The wire forms of the types, which hold extension parts as
//! [`Foreign`] parts until a resolver builds them.
//!
//! A [`Type`] serializes as `{"numerical": {"data_type", "shape"}}`,
//! `{"index": {"lower_bound", "upper_bound", "stride"}}` or `{"extension":
//! <foreign part>}`; a [`DataType`] as `{"primitive": "int32"}`,
//! `{"template": {"identifier", "widths"}}` (`widths` a list or `null`) or
//! `{"extension": <foreign part>}`; a [`Dimension`] as `{"expression":
//! <expression>}` or `"wildcard"`. Serializing an extension asks its
//! `to_foreign`. [`TypeData`] and [`DataTypeData`] read the same shapes,
//! and their `build` resolves the extension parts; the types' own
//! `Deserialize` builds with [`NoForeign`], which refuses them.
//!
//! # Examples
//!
//! ```
//! use fhy_core::expression::Expression;
//! use fhy_core::types::{CoreDataType, Dimension, NumericalType, Type};
//!
//! let ty = Type::from(NumericalType::new(CoreDataType::Int32, [Dimension::Wildcard]));
//! let text = serde_json::to_string(&ty)?;
//! assert_eq!(
//!     text,
//!     r#"{"numerical":{"data_type":{"primitive":"int32"},"shape":["wildcard"]}}"#
//! );
//! assert_eq!(serde_json::from_str::<Type>(&text)?, ty);
//! # Ok::<(), serde_json::Error>(())
//! ```

use serde::de::{self, Deserializer};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};

use crate::expression::Expression;
use crate::foreign::{BuildError, Foreign, ForeignError, NoForeign, Part, Resolve};
use crate::identifier::Identifier;

use super::core_data_type::CoreDataType;
use super::data_type::{DataType, TemplateDataType};
use super::extension::{DataTypeExtension, TypeExtension};
use super::ty::{Dimension, IndexType, NumericalType, Type};

/// The wire form of a [`Type`], its extension parts unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TypeData(TypeRepr);

/// The wire form of a [`DataType`], its extension part unresolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DataTypeData(DataTypeRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Type", rename_all = "snake_case")]
enum TypeRepr {
    Numerical(NumericalRepr),
    Index(IndexType),
    Extension(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "NumericalType", deny_unknown_fields)]
struct NumericalRepr {
    data_type: DataTypeRepr,
    shape: Vec<Dimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "DataType", rename_all = "snake_case")]
enum DataTypeRepr {
    Primitive(CoreDataType),
    Template(TemplateDataType),
    Extension(Foreign),
}

/// The resolvers a type's extension parts need.
pub trait TypeResolver:
    Resolve<Part<dyn TypeExtension>> + Resolve<Part<dyn DataTypeExtension>>
{
}

impl<R: Resolve<Part<dyn TypeExtension>> + Resolve<Part<dyn DataTypeExtension>> + ?Sized>
    TypeResolver for R
{
}

impl TypeData {
    /// Return the type, its extension parts resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses.
    pub fn build<R: TypeResolver + ?Sized>(self, resolver: &R) -> Result<Type, BuildError> {
        match self.0 {
            TypeRepr::Numerical(numerical) => Ok(Type::Numerical(numerical.build(resolver)?)),
            TypeRepr::Index(index) => Ok(Type::Index(index)),
            TypeRepr::Extension(foreign) => Ok(Type::Extension(resolver.resolve(&foreign)?)),
        }
    }

    /// Return the foreign part of an extension type, or `None` for a
    /// built-in one.
    #[must_use]
    pub const fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            TypeRepr::Extension(foreign) => Some(foreign),
            TypeRepr::Numerical(_) | TypeRepr::Index(_) => None,
        }
    }
}

impl DataTypeData {
    /// Return the data type, its extension part resolved by `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses.
    pub fn build<R: TypeResolver + ?Sized>(self, resolver: &R) -> Result<DataType, BuildError> {
        self.0.build(resolver)
    }

    /// Return the foreign part of an extension data type, or `None` for a
    /// built-in one.
    #[must_use]
    pub const fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            DataTypeRepr::Extension(foreign) => Some(foreign),
            DataTypeRepr::Primitive(_) | DataTypeRepr::Template(_) => None,
        }
    }
}

impl NumericalRepr {
    fn build<R: TypeResolver + ?Sized>(self, resolver: &R) -> Result<NumericalType, BuildError> {
        Ok(NumericalType::new(
            self.data_type.build(resolver)?,
            self.shape,
        ))
    }
}

impl DataTypeRepr {
    fn build<R: TypeResolver + ?Sized>(self, resolver: &R) -> Result<DataType, BuildError> {
        match self {
            Self::Primitive(core) => Ok(DataType::Primitive(core)),
            Self::Template(template) => Ok(DataType::Template(template)),
            Self::Extension(foreign) => Ok(DataType::Extension(resolver.resolve(&foreign)?)),
        }
    }

    fn of(data_type: &DataType) -> Result<Self, ForeignError> {
        Ok(match data_type {
            DataType::Primitive(core) => Self::Primitive(*core),
            DataType::Template(template) => Self::Template(template.clone()),
            DataType::Extension(extension) => Self::Extension(extension.get().to_foreign()?),
        })
    }
}

impl NumericalRepr {
    fn of(numerical: &NumericalType) -> Result<Self, ForeignError> {
        Ok(Self {
            data_type: DataTypeRepr::of(numerical.data_type())?,
            shape: numerical.shape().to_vec(),
        })
    }
}

impl Type {
    /// Return the wire form of the type, asking each extension for its
    /// foreign part.
    pub(crate) fn to_data(&self) -> Result<TypeData, ForeignError> {
        Ok(TypeData(match self {
            Self::Numerical(numerical) => TypeRepr::Numerical(NumericalRepr::of(numerical)?),
            Self::Index(index) => TypeRepr::Index(index.clone()),
            Self::Extension(extension) => TypeRepr::Extension(extension.get().to_foreign()?),
        }))
    }
}

/// Serializes the shape of the [module documentation](self); an extension
/// that cannot give its foreign part fails with its error.
impl Serialize for Type {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_data()
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing an
/// extension part.
impl<'de> Deserialize<'de> for Type {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        TypeData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// Serializes the shape of the [module documentation](self); an extension
/// that cannot give its foreign part fails with its error.
impl Serialize for DataType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        DataTypeRepr::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing an
/// extension part.
impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        DataTypeRepr::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// Serializes as `{"data_type", "shape"}`.
impl Serialize for NumericalType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        NumericalRepr::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes `{"data_type", "shape"}`, refusing an extension data type.
impl<'de> Deserialize<'de> for NumericalType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        NumericalRepr::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// The wire form of an [`IndexType`], borrowed for writing.
#[derive(Serialize)]
#[serde(rename = "IndexType")]
struct IndexRef<'a> {
    lower_bound: &'a Expression,
    upper_bound: &'a Expression,
    stride: &'a Expression,
}

/// The wire form of an [`IndexType`], read.
#[derive(Deserialize)]
#[serde(rename = "IndexType", deny_unknown_fields)]
struct IndexWire {
    lower_bound: Expression,
    upper_bound: Expression,
    stride: Expression,
}

/// Serializes as `{"lower_bound", "upper_bound", "stride"}`.
impl Serialize for IndexType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        IndexRef {
            lower_bound: self.lower_bound(),
            upper_bound: self.upper_bound(),
            stride: self.stride(),
        }
        .serialize(serializer)
    }
}

/// Deserializes `{"lower_bound", "upper_bound", "stride"}`.
impl<'de> Deserialize<'de> for IndexType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = IndexWire::deserialize(deserializer)?;
        Ok(Self::new(wire.lower_bound, wire.upper_bound, wire.stride))
    }
}

/// The wire form of a [`TemplateDataType`], borrowed for writing.
#[derive(Serialize)]
#[serde(rename = "TemplateDataType")]
struct TemplateRef<'a> {
    identifier: &'a Identifier,
    widths: Option<&'a [u32]>,
}

/// The wire form of a [`TemplateDataType`], read.
#[derive(Deserialize)]
#[serde(rename = "TemplateDataType", deny_unknown_fields)]
struct TemplateWire {
    identifier: Identifier,
    widths: Option<Vec<u32>>,
}

/// Serializes as `{"identifier", "widths"}`, `widths` a list or `null`.
impl Serialize for TemplateDataType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TemplateRef {
            identifier: self.identifier(),
            widths: self.widths(),
        }
        .serialize(serializer)
    }
}

/// Deserializes `{"identifier", "widths"}`, refusing a zero width and an
/// empty width list with [`TemplateWidthError`](super::TemplateWidthError)'s
/// text, and sorting and deduplicating the widths.
impl<'de> Deserialize<'de> for TemplateDataType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TemplateWire::deserialize(deserializer)?;
        match wire.widths {
            None => Ok(Self::new(wire.identifier)),
            Some(widths) => Self::with_widths(wire.identifier, widths).map_err(de::Error::custom),
        }
    }
}
