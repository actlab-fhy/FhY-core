//! The wire forms of the frames and the table.
//!
//! A [`SymbolFrame`] serializes as `{"import": {"name"}}`, `{"variable":
//! {"name", "type", "type_qualifier"}}` or `{"function": {"name",
//! "keyword", "signature": [{"type_qualifier", "type"}, ..]}}`, and a frame
//! another implementation defines as `{"custom": <foreign part>}`, which
//! [`SymbolFrameData`] reads and a `SymbolFrame` refuses. A
//! [`SymbolTable`] serializes as `{"namespaces": [{"namespace_name",
//! "parent_namespace_name", "symbols": [{"symbol_name", "frame"}, ..]},
//! ..]}`, in order, each frame in its own form; [`SymbolTableData`] reads it
//! and builds the table as [`SymbolTable::add_namespace`] and
//! [`SymbolTable::add_symbol`] would.
//!
//! # Examples
//!
//! ```
//! use fhy_core::identifier::Identifier;
//! use fhy_core::symbol_table::{ImportFrame, SymbolFrame, SymbolTable};
//!
//! let (ns, x) = (Identifier::try_restore(61_100, "ns")?, Identifier::try_restore(61_101, "x")?);
//! let mut table = SymbolTable::new();
//! table.add_namespace(ns.clone(), None)?;
//! table.add_symbol(&ns, x.clone(), SymbolFrame::Import(ImportFrame::new(x)))?;
//!
//! let text = serde_json::to_string(&table)?;
//! assert_eq!(
//!     text,
//!     concat!(
//!         r#"{"namespaces":[{"namespace_name":{"id":61100,"name_hint":"ns"},"#,
//!         r#""parent_namespace_name":null,"symbols":[{"symbol_name":{"id":61101,"name_hint":"x"},"#,
//!         r#""frame":{"import":{"name":{"id":61101,"name_hint":"x"}}}}]}]}"#,
//!     )
//! );
//! assert_eq!(serde_json::from_str::<SymbolTable<SymbolFrame>>(&text)?, table);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use serde::de::{self, Deserializer};
use serde::ser::{self, Serializer};
use serde::{Deserialize, Serialize};

use crate::foreign::{BuildError, Foreign, ForeignError, NoForeign};
use crate::identifier::Identifier;
use crate::types::TypeQualifier;
use crate::types::wire::{TypeData, TypeResolver};

use super::error::SymbolTableError;
use super::frame::{
    Frame, FunctionFrame, FunctionKeyword, ImportFrame, SymbolFrame, VariableFrame,
};
use super::table::SymbolTable;

/// The wire form of a frame: a built-in [`SymbolFrame`], its types'
/// extension parts unresolved, or a custom frame's [`Foreign`] part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SymbolFrameData(FrameRepr);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "SymbolFrame", rename_all = "snake_case")]
enum FrameRepr {
    Import(ImportFrame),
    Variable(VariableRepr),
    Function(FunctionRepr),
    Custom(Foreign),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "VariableFrame", deny_unknown_fields)]
struct VariableRepr {
    name: Identifier,
    #[serde(rename = "type")]
    ty: TypeData,
    type_qualifier: TypeQualifier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "FunctionFrame", deny_unknown_fields)]
struct FunctionRepr {
    name: Identifier,
    keyword: FunctionKeyword,
    signature: Vec<ParameterRepr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Parameter", deny_unknown_fields)]
struct ParameterRepr {
    type_qualifier: TypeQualifier,
    #[serde(rename = "type")]
    ty: TypeData,
}

impl SymbolFrameData {
    /// Return the wire form of a custom frame, whose part is `foreign`.
    #[must_use]
    pub fn custom(foreign: Foreign) -> Self {
        Self(FrameRepr::Custom(foreign))
    }

    /// Return the wire form of the built-in `frame`, asking each type
    /// extension for its foreign part.
    ///
    /// # Errors
    ///
    /// Returns the error of an extension that cannot give its part.
    pub fn of(frame: &SymbolFrame) -> Result<Self, ForeignError> {
        Ok(Self(match frame {
            SymbolFrame::Import(frame) => FrameRepr::Import(frame.clone()),
            SymbolFrame::Variable(frame) => FrameRepr::Variable(VariableRepr {
                name: frame.name().clone(),
                ty: frame.ty().to_data()?,
                type_qualifier: frame.qualifier(),
            }),
            SymbolFrame::Function(frame) => FrameRepr::Function(FunctionRepr {
                name: frame.name().clone(),
                keyword: frame.keyword(),
                signature: frame
                    .signature()
                    .iter()
                    .map(|(qualifier, ty)| {
                        Ok(ParameterRepr {
                            type_qualifier: *qualifier,
                            ty: ty.to_data()?,
                        })
                    })
                    .collect::<Result<_, ForeignError>>()?,
            }),
        }))
    }

    /// Return the foreign part of a custom frame, or `None` for a built-in
    /// one.
    #[must_use]
    pub fn foreign(&self) -> Option<&Foreign> {
        match &self.0 {
            FrameRepr::Custom(foreign) => Some(foreign),
            FrameRepr::Import(_) | FrameRepr::Variable(_) | FrameRepr::Function(_) => None,
        }
    }

    /// Return the built-in frame, its types' extension parts resolved by
    /// `resolver`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Foreign`] for a part `resolver` refuses, and
    /// for a custom frame, whose part a [`SymbolFrame`] cannot hold.
    pub fn build<R: TypeResolver + ?Sized>(self, resolver: &R) -> Result<SymbolFrame, BuildError> {
        Ok(match self.0 {
            FrameRepr::Import(frame) => SymbolFrame::Import(frame),
            FrameRepr::Variable(frame) => SymbolFrame::Variable(VariableFrame::new(
                frame.name,
                frame.ty.build(resolver)?,
                frame.type_qualifier,
            )),
            FrameRepr::Function(frame) => {
                let signature = frame
                    .signature
                    .into_iter()
                    .map(|parameter| Ok((parameter.type_qualifier, parameter.ty.build(resolver)?)))
                    .collect::<Result<Vec<_>, BuildError>>()?;
                SymbolFrame::Function(FunctionFrame::new(frame.name, frame.keyword, signature))
            }
            FrameRepr::Custom(foreign) => {
                return Err(BuildError::Foreign(ForeignError::Unresolved {
                    type_id: foreign.type_id().to_owned(),
                }));
            }
        })
    }
}

/// Serializes the shape of the [module documentation](self); a type
/// extension that cannot give its foreign part fails with its error.
impl Serialize for SymbolFrame {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SymbolFrameData::of(self)
            .map_err(ser::Error::custom)?
            .serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing a
/// custom frame and a type extension.
impl<'de> Deserialize<'de> for SymbolFrame {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SymbolFrameData::deserialize(deserializer)?
            .build(&NoForeign)
            .map_err(de::Error::custom)
    }
}

/// The wire form of a [`SymbolTable`] whose frames' wire form is `D`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "SymbolTable", deny_unknown_fields)]
pub struct SymbolTableData<D> {
    namespaces: Vec<NamespaceRepr<D>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Namespace", deny_unknown_fields)]
struct NamespaceRepr<D> {
    namespace_name: Identifier,
    parent_namespace_name: Option<Identifier>,
    symbols: Vec<SymbolRepr<D>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "Symbol", deny_unknown_fields)]
struct SymbolRepr<D> {
    symbol_name: Identifier,
    frame: D,
}

impl<D> SymbolTableData<D> {
    /// Return the wire form of `table`, each frame's the one `frame`
    /// returns.
    ///
    /// # Errors
    ///
    /// Returns the first error `frame` returns.
    pub fn of<'a, F, E>(
        table: &'a SymbolTable<F>,
        mut frame: impl FnMut(&'a F) -> Result<D, E>,
    ) -> Result<Self, E> {
        let mut namespaces = Vec::with_capacity(table.len());
        for namespace in table.namespaces() {
            let mut symbols = Vec::with_capacity(namespace.len());
            for (symbol, value) in namespace.iter() {
                symbols.push(SymbolRepr {
                    symbol_name: symbol.clone(),
                    frame: frame(value)?,
                });
            }
            namespaces.push(NamespaceRepr {
                namespace_name: namespace.name().clone(),
                parent_namespace_name: namespace.parent().cloned(),
                symbols,
            });
        }
        Ok(Self { namespaces })
    }

    /// Return the table, each frame the one `frame` builds from its wire
    /// form, adding every namespace, in order, and then the symbols of each,
    /// in order.
    ///
    /// Every namespace is in place before the first symbol is added, so the
    /// checks of [`SymbolTable::add_symbol`] see the whole chain of parents,
    /// and a table built through the checked API decodes whatever order its
    /// namespaces were added in.
    ///
    /// # Errors
    ///
    /// Returns the first error `frame` returns, and
    /// [`BuildError::Invalid`] with the [`SymbolTableError`] of a namespace
    /// defined twice or a symbol [`SymbolTable::add_symbol`] refuses.
    pub fn build<F>(
        self,
        mut frame: impl FnMut(D) -> Result<F, BuildError>,
    ) -> Result<SymbolTable<F>, BuildError> {
        let mut table = SymbolTable::new();
        let mut symbols = Vec::with_capacity(self.namespaces.len());
        for namespace in self.namespaces {
            table
                .add_namespace(
                    namespace.namespace_name.clone(),
                    namespace.parent_namespace_name,
                )
                .map_err(BuildError::invalid)?;
            symbols.push((namespace.namespace_name, namespace.symbols));
        }
        for (namespace, namespace_symbols) in symbols {
            for symbol in namespace_symbols {
                let value = frame(symbol.frame)?;
                table
                    .add_symbol(&namespace, symbol.symbol_name, value)
                    .map_err(|error: SymbolTableError| BuildError::invalid(error))?;
            }
        }
        Ok(table)
    }
}

/// Serializes the shape of the [module documentation](self), each frame in
/// its own form.
impl<F: Serialize> Serialize for SymbolTable<F> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let data: SymbolTableData<&F> =
            SymbolTableData::of(self, Ok::<_, std::convert::Infallible>)
                .unwrap_or_else(|never| match never {});
        data.serialize(serializer)
    }
}

/// Deserializes the shape of the [module documentation](self), refusing
/// what [`SymbolTableData::build`] refuses.
impl<'de, F: Deserialize<'de>> Deserialize<'de> for SymbolTable<F> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SymbolTableData::<F>::deserialize(deserializer)?
            .build(Ok)
            .map_err(de::Error::custom)
    }
}
