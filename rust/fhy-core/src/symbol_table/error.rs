//! Errors of building and asking a symbol table, and the violations of its
//! invariants.

use std::error::Error;
use std::fmt;

use crate::identifier::Identifier;

/// A change or a lookup a [`SymbolTable`](super::SymbolTable) refuses.
///
/// Displays one lowercase line naming the identifiers with their `Debug`
/// form, such as `namespace ns::7 already defined in the symbol table`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SymbolTableError {
    /// The namespace is already defined.
    NamespaceAlreadyDefined {
        /// The namespace.
        namespace: Identifier,
    },
    /// The namespace is not defined.
    NamespaceNotFound {
        /// The namespace.
        namespace: Identifier,
    },
    /// The namespace cannot be removed: other namespaces name it as their
    /// parent.
    NamespaceHasChildren {
        /// The namespace.
        namespace: Identifier,
        /// The namespaces naming it as their parent, in namespace order.
        children: Vec<Identifier>,
    },
    /// The symbol is already defined in the namespace or in one of its
    /// ancestors.
    SymbolAlreadyDefined {
        /// The namespace the symbol was to be added to.
        namespace: Identifier,
        /// The symbol.
        symbol: Identifier,
        /// The namespace that defines it: `namespace` or an ancestor.
        defined_in: Identifier,
    },
    /// The symbol is already defined in a descendant of the namespace, a
    /// namespace whose chain of parents reaches it.
    SymbolDefinedInDescendant {
        /// The namespace the symbol was to be added to.
        namespace: Identifier,
        /// The symbol.
        symbol: Identifier,
        /// The descendant that defines it.
        defined_in: Identifier,
    },
    /// The symbol is not found: in the namespace, or anywhere when
    /// `namespace` is `None`.
    SymbolNotFound {
        /// The namespace searched, or `None` for the whole table.
        namespace: Option<Identifier>,
        /// The symbol.
        symbol: Identifier,
    },
    /// A lookup's walk up the parents came back to this namespace.
    CyclicNamespace {
        /// The first namespace the walk reached twice.
        namespace: Identifier,
    },
    /// A lookup's walk up the parents reached a parent that is not
    /// defined.
    ParentNotFound {
        /// The namespace whose parent is missing.
        namespace: Identifier,
        /// The missing parent.
        parent: Identifier,
    },
}

impl fmt::Display for SymbolTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NamespaceAlreadyDefined { namespace } => {
                write!(
                    f,
                    "namespace {namespace:?} already defined in the symbol table"
                )
            }
            Self::NamespaceNotFound { namespace } => {
                write!(f, "namespace {namespace:?} not found in the symbol table")
            }
            Self::NamespaceHasChildren {
                namespace,
                children,
            } => {
                write!(
                    f,
                    "namespace {namespace:?} cannot be removed because it is the parent of "
                )?;
                for (position, child) in children.iter().enumerate() {
                    if position > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{child:?}")?;
                }
                Ok(())
            }
            Self::SymbolAlreadyDefined {
                namespace,
                symbol,
                defined_in,
            } => {
                write!(
                    f,
                    "symbol {symbol:?} already defined in namespace {defined_in:?}"
                )?;
                if defined_in != namespace {
                    write!(f, ", an ancestor of namespace {namespace:?}")?;
                }
                Ok(())
            }
            Self::SymbolDefinedInDescendant {
                namespace,
                symbol,
                defined_in,
            } => write!(
                f,
                "symbol {symbol:?} already defined in namespace {defined_in:?}, a descendant of \
                 namespace {namespace:?}"
            ),
            Self::SymbolNotFound {
                namespace: Some(namespace),
                symbol,
            } => write!(f, "symbol {symbol:?} not found in namespace {namespace:?}"),
            Self::SymbolNotFound {
                namespace: None,
                symbol,
            } => write!(f, "symbol {symbol:?} not found in the symbol table"),
            Self::CyclicNamespace { namespace } => write!(
                f,
                "namespace {namespace:?} is cyclic: the walk up its parents returns to it"
            ),
            Self::ParentNotFound { namespace, parent } => write!(
                f,
                "namespace {namespace:?} references missing parent namespace {parent:?}"
            ),
        }
    }
}

impl Error for SymbolTableError {}

/// A broken invariant of a [`SymbolTable`](super::SymbolTable), which
/// [`SymbolTable::violations`](super::SymbolTable::violations) reports.
///
/// Displays one lowercase line naming the identifiers with their `Debug`
/// form, such as `namespace a::5 has a cyclic parent chain`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Violation {
    /// The namespace names a parent that is not defined.
    ParentNotFound {
        /// The namespace.
        namespace: Identifier,
        /// Its missing parent.
        parent: Identifier,
    },
    /// The namespace names itself as its parent.
    OwnParent {
        /// The namespace.
        namespace: Identifier,
    },
    /// The walk up the namespace's parents comes back to a namespace it
    /// passed.
    CyclicParentChain {
        /// The namespace the walk starts from.
        namespace: Identifier,
    },
    /// A symbol that an ancestor of its namespace also defines, which the
    /// ancestor's shadows for every lookup from the namespace down.
    ShadowedSymbol {
        /// The namespace holding the symbol.
        namespace: Identifier,
        /// The symbol.
        symbol: Identifier,
        /// The nearest ancestor that also defines it.
        ancestor: Identifier,
    },
    /// A symbol's frame names another symbol.
    FrameNameMismatch {
        /// The namespace holding the symbol.
        namespace: Identifier,
        /// The symbol.
        symbol: Identifier,
        /// The name its frame gives.
        frame_name: Identifier,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParentNotFound { namespace, parent } => write!(
                f,
                "namespace {namespace:?} references missing parent namespace {parent:?}"
            ),
            Self::OwnParent { namespace } => {
                write!(f, "namespace {namespace:?} cannot be its own parent")
            }
            Self::CyclicParentChain { namespace } => {
                write!(f, "namespace {namespace:?} has a cyclic parent chain")
            }
            Self::ShadowedSymbol {
                namespace,
                symbol,
                ancestor,
            } => write!(
                f,
                "namespace {namespace:?} has symbol {symbol:?}, which its ancestor namespace \
                 {ancestor:?} also defines"
            ),
            Self::FrameNameMismatch {
                namespace,
                symbol,
                frame_name,
            } => write!(
                f,
                "namespace {namespace:?} has symbol entry {symbol:?} whose frame name is \
                 {frame_name:?}"
            ),
        }
    }
}
