//! Symbol tables: namespaces of symbols and the frames that describe them.
//!
//! - [`SymbolTable`] holds namespaces in insertion order. Each namespace may
//!   name a parent namespace, and maps its symbols, in insertion order, to
//!   frames. A lookup in a namespace walks up its parents, and a namespace
//!   never redefines a symbol one of its ancestors defines.
//! - [`Frame`] is what a table asks of the values it holds: the name of the
//!   symbol a frame describes. [`SymbolTable::violations`] reports each
//!   frame whose name is not its symbol, and each broken parent chain.
//! - [`SymbolFrame`] is the built-in frame: an [`ImportFrame`], a
//!   [`VariableFrame`] or a [`FunctionFrame`]. A table holds any frame type.
//!
//! # Examples
//!
//! ```
//! use fhy_core::identifier::Identifier;
//! use fhy_core::symbol_table::{SymbolFrame, SymbolTable, VariableFrame};
//! use fhy_core::types::{CoreDataType, NumericalType, TypeQualifier};
//!
//! let (module, function, x) = (Identifier::new("module"), Identifier::new("f"), Identifier::new("x"));
//! let int32 = NumericalType::scalar(CoreDataType::Int32);
//!
//! let mut table = SymbolTable::new();
//! table.add_namespace(module.clone(), None)?;
//! table.add_namespace(function.clone(), Some(module.clone()))?;
//! table.add_symbol(
//!     &module,
//!     x.clone(),
//!     SymbolFrame::from(VariableFrame::new(x.clone(), int32, TypeQualifier::State)),
//! )?;
//!
//! // A lookup from the function's namespace finds the module's variable.
//! assert!(table.lookup(&function, &x)?.is_some());
//! // The function's namespace cannot define it again.
//! assert!(table.add_symbol(&function, x.clone(), table.find(&x).unwrap().clone()).is_err());
//! assert!(table.violations().is_empty());
//! # Ok::<(), fhy_core::symbol_table::SymbolTableError>(())
//! ```

mod error;
mod frame;
mod ordered;
mod table;
pub mod wire;

pub use error::{SymbolTableError, Violation};
pub use frame::{Frame, FunctionFrame, FunctionKeyword, ImportFrame, SymbolFrame, VariableFrame};
pub use table::{Namespace, SymbolTable};
