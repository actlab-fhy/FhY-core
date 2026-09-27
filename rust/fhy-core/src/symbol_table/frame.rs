//! The frames a symbol table holds: the [`Frame`] trait, and the built-in
//! [`SymbolFrame`]s.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::impl_name_text;
use crate::identifier::Identifier;
use crate::types::{Type, TypeQualifier};

/// A value a [`SymbolTable`](super::SymbolTable) holds for a symbol.
pub trait Frame {
    /// Return the name of the symbol the frame describes.
    fn name(&self) -> &Identifier;
}

/// The frame of an imported symbol.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportFrame {
    name: Identifier,
}

impl ImportFrame {
    /// Return the frame of the imported symbol `name`.
    #[must_use]
    pub fn new(name: Identifier) -> Self {
        Self { name }
    }
}

/// The frame of a variable: its type and qualifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariableFrame {
    name: Identifier,
    ty: Type,
    qualifier: TypeQualifier,
}

impl VariableFrame {
    /// Return the frame of the variable `name` of type `ty`, qualified
    /// `qualifier`.
    #[must_use]
    pub fn new(name: Identifier, ty: impl Into<Type>, qualifier: TypeQualifier) -> Self {
        Self {
            name,
            ty: ty.into(),
            qualifier,
        }
    }

    /// Return the variable's type.
    #[must_use]
    pub fn ty(&self) -> &Type {
        &self.ty
    }

    /// Return the variable's qualifier.
    #[must_use]
    pub fn qualifier(&self) -> TypeQualifier {
        self.qualifier
    }

    /// Return whether `other` names the same variable, with the same
    /// qualifier and a structurally equivalent type.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        self.name == other.name
            && self.qualifier == other.qualifier
            && self.ty.is_structurally_equivalent(&other.ty)
    }
}

/// The frame of a function: its keyword and signature, a qualified type
/// per parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionFrame {
    name: Identifier,
    keyword: FunctionKeyword,
    signature: Arc<[(TypeQualifier, Type)]>,
}

impl FunctionFrame {
    /// Return the frame of the function `name`, declared with `keyword`,
    /// whose parameters have the qualified types of `signature`, in order.
    #[must_use]
    pub fn new(
        name: Identifier,
        keyword: FunctionKeyword,
        signature: impl IntoIterator<Item = (TypeQualifier, Type)>,
    ) -> Self {
        Self {
            name,
            keyword,
            signature: signature.into_iter().collect(),
        }
    }

    /// Return the keyword the function is declared with.
    #[must_use]
    pub fn keyword(&self) -> FunctionKeyword {
        self.keyword
    }

    /// Return the qualified type of each parameter, in order.
    #[must_use]
    pub fn signature(&self) -> &[(TypeQualifier, Type)] {
        &self.signature
    }

    /// Return whether `other` names the same function, with the same
    /// keyword and a signature of the same qualifiers and structurally
    /// equivalent types.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        self.name == other.name
            && self.keyword == other.keyword
            && self.signature.len() == other.signature.len()
            && self.signature.iter().zip(other.signature.iter()).all(
                |((left_qualifier, left_type), (right_qualifier, right_type))| {
                    left_qualifier == right_qualifier
                        && left_type.is_structurally_equivalent(right_type)
                },
            )
    }
}

/// A built-in frame.
///
/// Equality and hashing compare every field, types through their own
/// equality; [`is_structurally_equivalent`](Self::is_structurally_equivalent)
/// compares types through theirs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SymbolFrame {
    /// An imported symbol.
    Import(ImportFrame),
    /// A variable.
    Variable(VariableFrame),
    /// A function.
    Function(FunctionFrame),
}

impl SymbolFrame {
    /// Return whether `other` is a frame of the same kind whose fields are
    /// equal, with types compared by
    /// [`Type::is_structurally_equivalent`].
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Import(left), Self::Import(right)) => left == right,
            (Self::Variable(left), Self::Variable(right)) => left.is_structurally_equivalent(right),
            (Self::Function(left), Self::Function(right)) => left.is_structurally_equivalent(right),
            _ => false,
        }
    }
}

impl From<ImportFrame> for SymbolFrame {
    fn from(frame: ImportFrame) -> Self {
        Self::Import(frame)
    }
}

impl From<VariableFrame> for SymbolFrame {
    fn from(frame: VariableFrame) -> Self {
        Self::Variable(frame)
    }
}

impl From<FunctionFrame> for SymbolFrame {
    fn from(frame: FunctionFrame) -> Self {
        Self::Function(frame)
    }
}

impl Frame for ImportFrame {
    fn name(&self) -> &Identifier {
        &self.name
    }
}

impl Frame for VariableFrame {
    fn name(&self) -> &Identifier {
        &self.name
    }
}

impl Frame for FunctionFrame {
    fn name(&self) -> &Identifier {
        &self.name
    }
}

impl Frame for SymbolFrame {
    fn name(&self) -> &Identifier {
        match self {
            Self::Import(frame) => frame.name(),
            Self::Variable(frame) => frame.name(),
            Self::Function(frame) => frame.name(),
        }
    }
}

/// The keyword a function is declared with.
///
/// Displays and parses as its short name (`proc`, `op` or `native`), which
/// is also its serde form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FunctionKeyword {
    /// A procedure, `proc`.
    #[serde(rename = "proc")]
    Procedure,
    /// An operation, `op`.
    #[serde(rename = "op")]
    Operation,
    /// A native function, `native`.
    #[serde(rename = "native")]
    Native,
}

impl FunctionKeyword {
    /// Return the short name, such as `proc`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Procedure => "proc",
            Self::Operation => "op",
            Self::Native => "native",
        }
    }
}

impl_name_text!(FunctionKeyword, "function keyword");
