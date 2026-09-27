//! The text and the source of every variant of the `symbol_table` errors,
//! and the text of every violation, which is no error (F2-029).

use fhy_core::symbol_table::{SymbolTableError, Violation};
use rstest::rstest;

use crate::support::error_text::{Source, assert_error_text, fixed};

fn ns() -> fhy_core::identifier::Identifier {
    fixed(61_710, "ns")
}

fn child() -> fhy_core::identifier::Identifier {
    fixed(61_711, "child")
}

fn symbol() -> fhy_core::identifier::Identifier {
    fixed(61_712, "s")
}

#[rstest]
#[case::namespace_already_defined(
    SymbolTableError::NamespaceAlreadyDefined { namespace: ns() },
    "namespace ns::61710 already defined in the symbol table"
)]
#[case::namespace_not_found(
    SymbolTableError::NamespaceNotFound { namespace: ns() },
    "namespace ns::61710 not found in the symbol table"
)]
#[case::namespace_has_children(
    SymbolTableError::NamespaceHasChildren { namespace: ns(), children: vec![child(), symbol()] },
    "namespace ns::61710 cannot be removed because it is the parent of child::61711, s::61712"
)]
#[case::symbol_already_defined_here(
    SymbolTableError::SymbolAlreadyDefined { namespace: child(), symbol: symbol(), defined_in: child() },
    "symbol s::61712 already defined in namespace child::61711"
)]
#[case::symbol_already_defined_in_an_ancestor(
    SymbolTableError::SymbolAlreadyDefined { namespace: child(), symbol: symbol(), defined_in: ns() },
    "symbol s::61712 already defined in namespace ns::61710, an ancestor of namespace child::61711"
)]
#[case::symbol_defined_in_a_descendant(
    SymbolTableError::SymbolDefinedInDescendant { namespace: ns(), symbol: symbol(), defined_in: child() },
    "symbol s::61712 already defined in namespace child::61711, a descendant of namespace ns::61710"
)]
#[case::symbol_not_found_in_a_namespace(
    SymbolTableError::SymbolNotFound { namespace: Some(ns()), symbol: symbol() },
    "symbol s::61712 not found in namespace ns::61710"
)]
#[case::symbol_not_found_anywhere(
    SymbolTableError::SymbolNotFound { namespace: None, symbol: symbol() },
    "symbol s::61712 not found in the symbol table"
)]
#[case::cyclic_namespace(
    SymbolTableError::CyclicNamespace { namespace: ns() },
    "namespace ns::61710 is cyclic: the walk up its parents returns to it"
)]
#[case::parent_not_found(
    SymbolTableError::ParentNotFound { namespace: child(), parent: ns() },
    "namespace child::61711 references missing parent namespace ns::61710"
)]
fn symbol_table_error_text(#[case] error: SymbolTableError, #[case] text: &str) {
    assert_error_text(&error, text, Source::None);
}

#[rstest]
#[case::parent_not_found(
    Violation::ParentNotFound { namespace: child(), parent: ns() },
    "namespace child::61711 references missing parent namespace ns::61710"
)]
#[case::own_parent(
    Violation::OwnParent { namespace: ns() },
    "namespace ns::61710 cannot be its own parent"
)]
#[case::cyclic_parent_chain(
    Violation::CyclicParentChain { namespace: ns() },
    "namespace ns::61710 has a cyclic parent chain"
)]
#[case::shadowed_symbol(
    Violation::ShadowedSymbol { namespace: child(), symbol: symbol(), ancestor: ns() },
    "namespace child::61711 has symbol s::61712, which its ancestor namespace ns::61710 also \
     defines"
)]
#[case::frame_name_mismatch(
    Violation::FrameNameMismatch { namespace: ns(), symbol: symbol(), frame_name: child() },
    "namespace ns::61710 has symbol entry s::61712 whose frame name is child::61711"
)]
fn violation_text(#[case] violation: Violation, #[case] text: &str) {
    assert_eq!(violation.to_string(), text);
}
