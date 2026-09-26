//! Tests for `SymbolTable`: namespaces and their parents, symbols, the
//! lookups, merging, canonical order, violations and equivalence.
//!
//! Ported from `tests/test_symbol_table.py`; the traceability table is in
//! `docs/design/python-switch.md`, "S15: the symbol table".

use std::convert::Infallible;

use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};
use crate::support::types::scalar;

use fhy_core::identifier::Identifier;
use fhy_core::symbol_table::{
    Frame, ImportFrame, SymbolFrame, SymbolTable, SymbolTableError, VariableFrame, Violation,
};
use fhy_core::types::{CoreDataType, TypeQualifier};

/// Return a new identifier per name, in order.
fn identifiers<const N: usize>(names: [&str; N]) -> [Identifier; N] {
    names.map(Identifier::new)
}

/// Return the import frame of `symbol`.
fn import(symbol: &Identifier) -> ImportFrame {
    ImportFrame::new(symbol.clone())
}

/// Return the names of the table's namespaces, in order.
fn namespace_names<F>(table: &SymbolTable<F>) -> Vec<Identifier> {
    table
        .namespaces()
        .map(|namespace| namespace.name().clone())
        .collect()
}

/// Return the symbols of `namespace`, in order.
fn symbol_names<F>(table: &SymbolTable<F>, namespace: &Identifier) -> Vec<Identifier> {
    table
        .namespace(namespace)
        .expect("the namespace is defined")
        .iter()
        .map(|(symbol, _)| symbol.clone())
        .collect()
}

/// Return a table of `root` holding `symbol`, and `child` under `root`.
fn parent_and_child(
    root: &Identifier,
    child: &Identifier,
    symbol: &Identifier,
) -> SymbolTable<ImportFrame> {
    let mut table = SymbolTable::new();
    table.add_namespace(root.clone(), None).expect("new");
    table
        .add_namespace(child.clone(), Some(root.clone()))
        .expect("new");
    table
        .add_symbol(root, symbol.clone(), import(symbol))
        .expect("new");
    table
}

// ---------------------------------------------------------------------------
// Namespaces
// ---------------------------------------------------------------------------

#[test]
fn a_new_table_is_empty() {
    let table: SymbolTable<ImportFrame> = SymbolTable::new();

    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.namespaces().len(), 0);
    assert_eq!(SymbolTable::<ImportFrame>::default(), table);
}

#[test]
fn a_namespace_is_added_once() {
    let [namespace] = identifiers(["namespace"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();

    table.add_namespace(namespace.clone(), None).expect("new");

    assert!(table.contains_namespace(&namespace));
    assert_eq!(table.len(), 1);
    assert!(!table.is_empty());
    assert_eq!(
        table.add_namespace(namespace.clone(), None),
        Err(SymbolTableError::NamespaceAlreadyDefined {
            namespace: namespace.clone()
        })
    );
    assert_eq!(table.len(), 1);
}

#[test]
fn an_unknown_namespace_has_no_view() {
    let [namespace] = identifiers(["undefined"]);
    let table: SymbolTable<ImportFrame> = SymbolTable::new();

    assert!(table.namespace(&namespace).is_none());
    assert!(!table.contains_namespace(&namespace));
}

#[test]
fn a_parent_is_not_checked_when_a_namespace_is_added() {
    let [early, late, missing, own] = identifiers(["early", "late", "missing", "own"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();

    table
        .add_namespace(early.clone(), Some(late.clone()))
        .expect("a forward reference is accepted");
    table.add_namespace(late.clone(), None).expect("new");
    table
        .add_namespace(missing.clone(), Some(Identifier::new("nowhere")))
        .expect("a missing parent is accepted");
    table
        .add_namespace(own.clone(), Some(own.clone()))
        .expect("a namespace naming itself is accepted");

    assert_eq!(table.len(), 4);
    assert_eq!(
        table.namespace(&early).and_then(|view| view.parent()),
        Some(&late)
    );
    assert_eq!(table.namespace(&late).and_then(|view| view.parent()), None);
}

#[test]
fn namespaces_and_symbols_keep_insertion_order() {
    let [second, first, third] = identifiers(["second", "first", "third"]);
    let [late, early] = identifiers(["late", "early"]);
    let mut table = SymbolTable::new();
    for namespace in [&third, &first, &second] {
        table.add_namespace(namespace.clone(), None).expect("new");
    }
    table
        .add_symbol(&first, late.clone(), import(&late))
        .expect("new");
    table
        .add_symbol(&first, early.clone(), import(&early))
        .expect("new");

    assert_eq!(
        namespace_names(&table),
        [third.clone(), first.clone(), second.clone()]
    );
    assert_eq!(symbol_names(&table, &first), [late.clone(), early.clone()]);
    let view = table.namespace(&first).expect("defined");
    assert_eq!(view.name(), &first);
    assert_eq!(view.len(), 2);
    assert!(!view.is_empty());
    assert!(view.contains(&early));
    assert_eq!(view.get(&late), Some(&import(&late)));
    assert!(table.namespace(&second).expect("defined").is_empty());
}

// ---------------------------------------------------------------------------
// Symbols and lookups
// ---------------------------------------------------------------------------

#[test]
fn a_symbol_is_added_once_and_found() {
    let [namespace, symbol] = identifiers(["namespace", "symbol"]);
    let mut table = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");

    table
        .add_symbol(&namespace, symbol.clone(), import(&symbol))
        .expect("new");

    assert_eq!(table.find(&symbol), Some(&import(&symbol)));
    assert_eq!(
        table.lookup(&namespace, &symbol).expect("defined"),
        Some(&import(&symbol))
    );
    assert_eq!(
        table.add_symbol(&namespace, symbol.clone(), import(&symbol)),
        Err(SymbolTableError::SymbolAlreadyDefined {
            namespace: namespace.clone(),
            symbol: symbol.clone(),
            defined_in: namespace.clone(),
        })
    );
    assert_eq!(symbol_names(&table, &namespace), [symbol]);
}

#[test]
fn a_symbol_cannot_be_added_to_an_unknown_namespace() {
    let [namespace, symbol] = identifiers(["undefined", "symbol"]);
    let mut table = SymbolTable::new();

    assert_eq!(
        table.add_symbol(&namespace, symbol.clone(), import(&symbol)),
        Err(SymbolTableError::NamespaceNotFound { namespace })
    );
}

#[test]
fn a_missing_symbol_is_not_found() {
    let [namespace, symbol] = identifiers(["namespace", "undefined"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");

    assert_eq!(table.lookup(&namespace, &symbol), Ok(None));
    assert_eq!(table.find(&symbol), None);
}

#[test]
fn lookup_walks_up_the_parents() {
    let [root, middle, leaf, symbol] = identifiers(["root", "middle", "leaf", "symbol"]);
    let mut table = parent_and_child(&root, &middle, &symbol);
    table
        .add_namespace(leaf.clone(), Some(middle.clone()))
        .expect("new");

    assert_eq!(
        table.lookup(&leaf, &symbol).expect("the chain is sound"),
        Some(&import(&symbol))
    );
    assert_eq!(
        table.lookup(&middle, &symbol).expect("the chain is sound"),
        Some(&import(&symbol))
    );
    assert_eq!(table.lookup(&root, &Identifier::new("other")), Ok(None));
}

#[test]
fn the_nearest_namespace_holding_a_symbol_answers_a_lookup() {
    let [root, child, symbol] = identifiers(["root", "child", "symbol"]);
    let mut table = SymbolTable::new();
    table
        .add_namespace(child.clone(), Some(root.clone()))
        .expect("new");
    table.add_namespace(root.clone(), None).expect("new");
    let inner = ImportFrame::new(Identifier::new("inner"));
    table
        .add_symbol(&child, symbol.clone(), inner.clone())
        .expect("new");
    table
        .add_symbol(&root, symbol.clone(), import(&symbol))
        .expect("the root's own lookup does not see the child's symbol");

    assert_eq!(table.lookup(&child, &symbol), Ok(Some(&inner)));
    assert_eq!(table.lookup(&root, &symbol), Ok(Some(&import(&symbol))));
}

#[test]
fn an_inner_namespace_cannot_shadow_an_outer_symbol() {
    let [root, child, symbol] = identifiers(["root", "child", "symbol"]);
    let mut table = parent_and_child(&root, &child, &symbol);

    assert_eq!(
        table.add_symbol(&child, symbol.clone(), import(&symbol)),
        Err(SymbolTableError::SymbolAlreadyDefined {
            namespace: child.clone(),
            symbol: symbol.clone(),
            defined_in: root.clone(),
        })
    );
    assert!(table.namespace(&child).expect("defined").is_empty());
}

#[test]
fn a_lookup_of_an_unknown_namespace_fails() {
    let [namespace, symbol] = identifiers(["undefined", "symbol"]);
    let table: SymbolTable<ImportFrame> = SymbolTable::new();

    assert_eq!(
        table.lookup(&namespace, &symbol),
        Err(SymbolTableError::NamespaceNotFound { namespace })
    );
}

#[test]
fn a_cyclic_chain_fails_the_lookup() {
    let [a, b, c, symbol] = identifiers(["a", "b", "c", "symbol"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();
    table
        .add_namespace(a.clone(), Some(b.clone()))
        .expect("new");
    table
        .add_namespace(b.clone(), Some(c.clone()))
        .expect("new");
    table
        .add_namespace(c.clone(), Some(b.clone()))
        .expect("new");

    assert_eq!(
        table.lookup(&a, &symbol),
        Err(SymbolTableError::CyclicNamespace {
            namespace: b.clone()
        })
    );
    assert_eq!(
        table.add_symbol(&a, symbol.clone(), import(&symbol)),
        Err(SymbolTableError::CyclicNamespace { namespace: b })
    );
}

#[test]
fn a_symbol_found_before_the_cycle_answers_the_lookup() {
    let [a, b, symbol] = identifiers(["a", "b", "symbol"]);
    let mut table = SymbolTable::new();
    table
        .add_namespace(a.clone(), Some(b.clone()))
        .expect("new");
    table
        .add_namespace(b.clone(), Some(a.clone()))
        .expect("new");
    assert_eq!(
        table.add_symbol(&b, symbol.clone(), import(&symbol)),
        Err(SymbolTableError::CyclicNamespace {
            namespace: b.clone()
        }),
        "adding walks the whole cycle"
    );
    // A merge puts the symbol into b, whose parent it keeps.
    let mut source = SymbolTable::new();
    source.add_namespace(b.clone(), None).expect("new");
    source
        .add_symbol(&b, symbol.clone(), import(&symbol))
        .expect("new");
    table.update_namespaces(&source);

    assert_eq!(table.lookup(&a, &symbol), Ok(Some(&import(&symbol))));
    assert_eq!(table.lookup(&b, &symbol), Ok(Some(&import(&symbol))));
}

#[test]
fn a_missing_parent_fails_the_lookup() {
    let [child, missing, symbol] = identifiers(["child", "missing", "symbol"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();
    table
        .add_namespace(child.clone(), Some(missing.clone()))
        .expect("new");

    assert_eq!(
        table.lookup(&child, &symbol),
        Err(SymbolTableError::ParentNotFound {
            namespace: child,
            parent: missing,
        })
    );
}

#[test]
fn find_searches_every_namespace_in_order_ignoring_parents() {
    let [first, second, symbol] = identifiers(["first", "second", "symbol"]);
    let mut table = SymbolTable::new();
    table.add_namespace(first.clone(), None).expect("new");
    table.add_namespace(second.clone(), None).expect("new");
    let in_second = ImportFrame::new(Identifier::new("second_frame"));
    let in_first = ImportFrame::new(Identifier::new("first_frame"));
    table
        .add_symbol(&second, symbol.clone(), in_second)
        .expect("new");
    table
        .add_symbol(&first, symbol.clone(), in_first.clone())
        .expect("new");

    assert_eq!(table.find(&symbol), Some(&in_first));
}

// ---------------------------------------------------------------------------
// Removal
// ---------------------------------------------------------------------------

#[test]
fn remove_symbol_returns_its_frame() {
    let [namespace, symbol, other] = identifiers(["namespace", "symbol", "other"]);
    let mut table = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");
    table
        .add_symbol(&namespace, symbol.clone(), import(&symbol))
        .expect("new");
    table
        .add_symbol(&namespace, other.clone(), import(&other))
        .expect("new");

    assert_eq!(
        table.remove_symbol(&namespace, &symbol),
        Ok(import(&symbol))
    );

    assert_eq!(table.lookup(&namespace, &symbol), Ok(None));
    assert_eq!(
        symbol_names(&table, &namespace),
        std::slice::from_ref(&other)
    );
    assert_eq!(table.find(&other), Some(&import(&other)));
}

#[test]
fn remove_symbol_refuses_a_symbol_the_namespace_does_not_hold() {
    let [root, child, symbol] = identifiers(["root", "child", "symbol"]);
    let mut table = parent_and_child(&root, &child, &symbol);

    assert_eq!(
        table.remove_symbol(&child, &symbol),
        Err(SymbolTableError::SymbolNotFound {
            namespace: Some(child.clone()),
            symbol: symbol.clone(),
        })
    );
    assert_eq!(table.lookup(&root, &symbol), Ok(Some(&import(&symbol))));
}

#[test]
fn remove_symbol_refuses_an_unknown_namespace() {
    let [namespace, symbol] = identifiers(["undefined", "symbol"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();

    assert_eq!(
        table.remove_symbol(&namespace, &symbol),
        Err(SymbolTableError::NamespaceNotFound { namespace })
    );
}

#[test]
fn remove_namespace_drops_it_and_its_parent() {
    let [root, child, symbol] = identifiers(["root", "child", "symbol"]);
    let mut table = parent_and_child(&root, &child, &symbol);

    table.remove_namespace(&child).expect("a leaf is removed");

    assert!(!table.contains_namespace(&child));
    assert_eq!(namespace_names(&table), std::slice::from_ref(&root));
    assert!(table.violations().is_empty());
    table.remove_namespace(&root).expect("no child is left");
    assert!(table.is_empty());
    assert_eq!(table.find(&symbol), None);
}

#[test]
fn remove_namespace_refuses_a_parent_of_others() {
    let [root, first, second, symbol] = identifiers(["root", "first", "second", "symbol"]);
    let mut table = parent_and_child(&root, &first, &symbol);
    table
        .add_namespace(second.clone(), Some(root.clone()))
        .expect("new");

    assert_eq!(
        table.remove_namespace(&root),
        Err(SymbolTableError::NamespaceHasChildren {
            namespace: root.clone(),
            children: vec![first, second],
        })
    );
    assert!(table.contains_namespace(&root));
}

#[test]
fn remove_namespace_refuses_an_unknown_namespace() {
    let [namespace] = identifiers(["undefined"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();

    assert_eq!(
        table.remove_namespace(&namespace),
        Err(SymbolTableError::NamespaceNotFound { namespace })
    );
}

#[test]
fn remove_namespace_keeps_the_order_of_the_others() {
    let [a, b, c, symbol] = identifiers(["a", "b", "c", "symbol"]);
    let mut table = SymbolTable::new();
    for namespace in [&a, &b, &c] {
        table.add_namespace(namespace.clone(), None).expect("new");
    }
    table
        .add_symbol(&c, symbol.clone(), import(&symbol))
        .expect("new");

    table.remove_namespace(&a).expect("defined");

    assert_eq!(namespace_names(&table), [b, c.clone()]);
    assert_eq!(table.lookup(&c, &symbol), Ok(Some(&import(&symbol))));
}

// ---------------------------------------------------------------------------
// Merging
// ---------------------------------------------------------------------------

#[test]
fn update_namespaces_copies_the_other_tables_namespaces() {
    let [namespace, symbol] = identifiers(["shared", "symbol"]);
    let mut source = SymbolTable::new();
    source.add_namespace(namespace.clone(), None).expect("new");
    source
        .add_symbol(&namespace, symbol.clone(), import(&symbol))
        .expect("new");
    let mut destination = SymbolTable::new();

    destination.update_namespaces(&source);

    assert!(destination.contains_namespace(&namespace));
    assert_eq!(
        destination.lookup(&namespace, &symbol),
        Ok(Some(&import(&symbol)))
    );
}

#[test]
fn update_namespaces_leaves_the_tables_independent() {
    let [namespace, first, second] = identifiers(["ns", "first", "second"]);
    let mut source = SymbolTable::new();
    source.add_namespace(namespace.clone(), None).expect("new");
    source
        .add_symbol(&namespace, first.clone(), import(&first))
        .expect("new");
    let mut destination = SymbolTable::new();
    destination.update_namespaces(&source);

    source
        .add_symbol(&namespace, second.clone(), import(&second))
        .expect("new");
    destination
        .remove_symbol(&namespace, &first)
        .expect("copied");

    assert_eq!(destination.lookup(&namespace, &second), Ok(None));
    assert_eq!(source.lookup(&namespace, &first), Ok(Some(&import(&first))));
}

#[test]
fn update_namespaces_replaces_a_namespace_in_place() {
    let [a, b, c, old, new] = identifiers(["a", "b", "c", "old", "new"]);
    let mut destination = SymbolTable::new();
    destination.add_namespace(a.clone(), None).expect("new");
    destination.add_namespace(b.clone(), None).expect("new");
    destination
        .add_symbol(&a, old.clone(), import(&old))
        .expect("new");
    let mut source = SymbolTable::new();
    source.add_namespace(c.clone(), None).expect("new");
    source.add_namespace(a.clone(), None).expect("new");
    source
        .add_symbol(&a, new.clone(), import(&new))
        .expect("new");

    destination.update_namespaces(&source);

    assert_eq!(namespace_names(&destination), [a.clone(), b, c]);
    assert_eq!(symbol_names(&destination, &a), [new]);
}

#[test]
fn update_namespaces_sets_a_parent_only_where_the_other_table_names_one() {
    let [kept, replaced, added, parent, other_parent] =
        identifiers(["kept", "replaced", "added", "parent", "other_parent"]);
    let mut destination: SymbolTable<ImportFrame> = SymbolTable::new();
    destination
        .add_namespace(kept.clone(), Some(parent.clone()))
        .expect("new");
    destination
        .add_namespace(replaced.clone(), Some(parent.clone()))
        .expect("new");
    let mut source = SymbolTable::new();
    source.add_namespace(kept.clone(), None).expect("new");
    source
        .add_namespace(replaced.clone(), Some(other_parent.clone()))
        .expect("new");
    source.add_namespace(added.clone(), None).expect("new");

    destination.update_namespaces(&source);

    let parent_of = |namespace: &Identifier| {
        destination
            .namespace(namespace)
            .and_then(|view| view.parent().cloned())
    };
    assert_eq!(parent_of(&kept), Some(parent));
    assert_eq!(parent_of(&replaced), Some(other_parent));
    assert_eq!(parent_of(&added), None);
}

#[test]
fn insert_namespace_restores_any_state_without_checks() {
    let [a, b, symbol, other] = identifiers(["a", "b", "symbol", "other"]);
    let mut table = SymbolTable::new();
    table.insert_namespace(
        a.clone(),
        Some(b.clone()),
        [(symbol.clone(), import(&symbol))],
    );
    table.insert_namespace(
        b.clone(),
        Some(a.clone()),
        [
            (symbol.clone(), import(&other)),
            (other.clone(), import(&other)),
        ],
    );

    assert_eq!(namespace_names(&table), [a.clone(), b.clone()]);
    assert_eq!(symbol_names(&table, &b), [symbol.clone(), other.clone()]);
    assert_eq!(table.lookup(&a, &symbol), Ok(Some(&import(&symbol))));
    assert_eq!(table.violations().len(), 3, "{:?}", table.violations());

    table.insert_namespace(
        a.clone(),
        None,
        [
            (other.clone(), import(&symbol)),
            (other.clone(), import(&other)),
        ],
    );

    assert_eq!(namespace_names(&table), [a.clone(), b]);
    assert_eq!(table.namespace(&a).and_then(|view| view.parent()), None);
    assert_eq!(symbol_names(&table, &a), std::slice::from_ref(&other));
    assert_eq!(table.lookup(&a, &other), Ok(Some(&import(&other))));
}

// ---------------------------------------------------------------------------
// Canonical order
// ---------------------------------------------------------------------------

#[test]
fn canonicalize_sorts_namespaces_and_symbols_by_id() {
    let [low, high] = identifiers(["low", "high"]);
    let [first, second, third] = identifiers(["first", "second", "third"]);
    let mut table = SymbolTable::new();
    table
        .add_namespace(high.clone(), Some(low.clone()))
        .expect("new");
    table.add_namespace(low.clone(), None).expect("new");
    for symbol in [&third, &first, &second] {
        table
            .add_symbol(&high, symbol.clone(), import(symbol))
            .expect("new");
    }
    let before = table.clone();

    table.canonicalize();

    assert_eq!(namespace_names(&table), [low.clone(), high.clone()]);
    assert_eq!(symbol_names(&table, &high), [first, second, third]);
    assert_eq!(
        table.namespace(&high).and_then(|view| view.parent()),
        Some(&low)
    );
    assert_eq!(table, before);
}

#[test]
fn canonicalize_is_idempotent() {
    let [a, b, symbol] = identifiers(["a", "b", "symbol"]);
    let mut table = SymbolTable::new();
    table.add_namespace(b.clone(), None).expect("new");
    table.add_namespace(a.clone(), None).expect("new");
    table
        .add_symbol(&b, symbol.clone(), import(&symbol))
        .expect("new");
    table.canonicalize();
    let once = namespace_names(&table);

    table.canonicalize();

    assert_eq!(namespace_names(&table), once);
    assert_eq!(table.lookup(&b, &symbol), Ok(Some(&import(&symbol))));
}

// ---------------------------------------------------------------------------
// Violations
// ---------------------------------------------------------------------------

#[test]
fn violations_are_empty_for_a_well_formed_table() {
    let [root, child, symbol] = identifiers(["root", "child", "symbol"]);
    let table = parent_and_child(&root, &child, &symbol);

    assert!(table.violations().is_empty());
}

#[test]
fn violations_report_a_missing_parent() {
    let [child, missing] = identifiers(["child", "missing_parent"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();
    table
        .add_namespace(child.clone(), Some(missing.clone()))
        .expect("new");

    assert_eq!(
        table.violations(),
        [Violation::ParentNotFound {
            namespace: child,
            parent: missing,
        }]
    );
}

#[test]
fn violations_report_a_cyclic_chain_from_each_namespace_on_or_into_it() {
    let [a, b, c] = identifiers(["a", "b", "c"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();
    table
        .add_namespace(a.clone(), Some(b.clone()))
        .expect("new");
    table
        .add_namespace(b.clone(), Some(a.clone()))
        .expect("new");
    table
        .add_namespace(c.clone(), Some(a.clone()))
        .expect("new");

    assert_eq!(
        table.violations(),
        [
            Violation::CyclicParentChain { namespace: a },
            Violation::CyclicParentChain { namespace: b },
            Violation::CyclicParentChain { namespace: c },
        ]
    );
}

#[test]
fn violations_report_a_namespace_that_is_its_own_parent() {
    let [own] = identifiers(["own"]);
    let mut table: SymbolTable<ImportFrame> = SymbolTable::new();
    table
        .add_namespace(own.clone(), Some(own.clone()))
        .expect("new");

    assert_eq!(
        table.violations(),
        [
            Violation::OwnParent {
                namespace: own.clone()
            },
            Violation::CyclicParentChain { namespace: own },
        ]
    );
}

#[test]
fn violations_report_a_frame_naming_another_symbol() {
    let [namespace, symbol, frame_name] = identifiers(["namespace", "symbol", "other_symbol"]);
    let mut table = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");
    table
        .add_symbol(&namespace, symbol.clone(), import(&frame_name))
        .expect("the frame is not checked");

    assert_eq!(
        table.violations(),
        [Violation::FrameNameMismatch {
            namespace,
            symbol,
            frame_name,
        }]
    );
}

#[test]
fn violations_come_in_order_parents_then_cycles_then_frames() {
    let [bad_frame, cyclic, orphan, symbol] = identifiers(["frames", "cyclic", "orphan", "symbol"]);
    let missing = Identifier::new("missing");
    let mut table = SymbolTable::new();
    table.add_namespace(bad_frame.clone(), None).expect("new");
    table
        .add_symbol(&bad_frame, symbol.clone(), import(&missing))
        .expect("unchecked");
    table
        .add_namespace(cyclic.clone(), Some(cyclic.clone()))
        .expect("new");
    table
        .add_namespace(orphan.clone(), Some(missing.clone()))
        .expect("new");

    assert_eq!(
        table.violations(),
        [
            Violation::OwnParent {
                namespace: cyclic.clone()
            },
            Violation::ParentNotFound {
                namespace: orphan,
                parent: missing.clone(),
            },
            Violation::CyclicParentChain { namespace: cyclic },
            Violation::FrameNameMismatch {
                namespace: bad_frame,
                symbol,
                frame_name: missing,
            },
        ]
    );
}

// ---------------------------------------------------------------------------
// Equivalence
// ---------------------------------------------------------------------------

#[test]
fn equivalence_ignores_the_orders() {
    let [a, b, x, y] = identifiers(["a", "b", "x", "y"]);
    let mut left = SymbolTable::new();
    left.add_namespace(a.clone(), None).expect("new");
    left.add_namespace(b.clone(), Some(a.clone())).expect("new");
    left.add_symbol(&a, x.clone(), import(&x)).expect("new");
    left.add_symbol(&a, y.clone(), import(&y)).expect("new");
    let mut right = SymbolTable::new();
    right
        .add_namespace(b.clone(), Some(a.clone()))
        .expect("new");
    right.add_namespace(a.clone(), None).expect("new");
    right.add_symbol(&a, y.clone(), import(&y)).expect("new");
    right.add_symbol(&a, x.clone(), import(&x)).expect("new");

    assert_eq!(left, right);
}

#[test]
fn equivalence_tells_apart_namespaces_parents_symbols_and_frames() {
    let [root, child, symbol, other] = identifiers(["root", "child", "symbol", "other"]);
    let base = parent_and_child(&root, &child, &symbol);

    let mut orphan = SymbolTable::new();
    orphan.add_namespace(root.clone(), None).expect("new");
    orphan.add_namespace(child.clone(), None).expect("new");
    orphan
        .add_symbol(&root, symbol.clone(), import(&symbol))
        .expect("new");
    assert_ne!(base, orphan, "a parent differs");

    let mut extra_namespace = base.clone();
    extra_namespace
        .add_namespace(other.clone(), None)
        .expect("new");
    assert_ne!(base, extra_namespace, "a namespace differs");

    let mut extra_symbol = base.clone();
    extra_symbol
        .add_symbol(&child, other.clone(), import(&other))
        .expect("new");
    assert_ne!(base, extra_symbol, "a symbol differs");

    let mut other_frame = base.clone();
    other_frame.remove_symbol(&root, &symbol).expect("held");
    other_frame
        .add_symbol(&root, symbol.clone(), import(&other))
        .expect("new");
    assert_ne!(base, other_frame, "a frame differs");
}

#[test]
fn is_equivalent_by_stops_at_the_first_false_frame() {
    let [namespace, first, second] = identifiers(["namespace", "first", "second"]);
    let mut table = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");
    table
        .add_symbol(&namespace, first.clone(), import(&first))
        .expect("new");
    table
        .add_symbol(&namespace, second.clone(), import(&second))
        .expect("new");
    let mut asked = Vec::new();

    let answer = table.is_equivalent_by(&table, |left, _right| {
        asked.push(left.name().clone());
        Ok::<_, Infallible>(false)
    });

    assert_eq!(answer, Ok(false));
    assert_eq!(asked, [first]);
}

#[test]
fn is_equivalent_by_propagates_the_first_error() {
    let [namespace, first, second] = identifiers(["namespace", "first", "second"]);
    let mut table = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");
    table
        .add_symbol(&namespace, first.clone(), import(&first))
        .expect("new");
    table
        .add_symbol(&namespace, second.clone(), import(&second))
        .expect("new");

    let answer = table.is_equivalent_by(&table, |left, _right| Err(left.name().clone()));

    assert_eq!(answer, Err(first));
}

#[test]
fn is_equivalent_by_asks_no_frame_when_the_shapes_differ() {
    let [a, b, symbol] = identifiers(["a", "b", "symbol"]);
    let left = parent_and_child(&a, &b, &symbol);
    let right: SymbolTable<ImportFrame> = SymbolTable::new();

    let answer = left.is_equivalent_by(&right, |_, _| -> Result<bool, ()> {
        panic!("no frame is compared")
    });

    assert_eq!(answer, Ok(false));
}

#[test]
fn built_in_frames_compare_structurally() {
    let [namespace, x] = identifiers(["namespace", "x"]);
    let frame = |qualifier| {
        SymbolFrame::from(VariableFrame::new(
            x.clone(),
            scalar(CoreDataType::Int32),
            qualifier,
        ))
    };
    let build = |qualifier| {
        let mut table = SymbolTable::new();
        table.add_namespace(namespace.clone(), None).expect("new");
        table
            .add_symbol(&namespace, x.clone(), frame(qualifier))
            .expect("new");
        table
    };

    assert!(build(TypeQualifier::State).is_structurally_equivalent(&build(TypeQualifier::State)));
    assert!(!build(TypeQualifier::State).is_structurally_equivalent(&build(TypeQualifier::Input)));
}

// ---------------------------------------------------------------------------
// Scale, threads and text
// ---------------------------------------------------------------------------

#[test]
fn a_long_parent_chain_walks_on_a_small_stack() {
    run_on_small_stack(|| {
        let names: Vec<Identifier> = (0..SMALL_STACK_DEPTH / 10)
            .map(|_| Identifier::new("namespace"))
            .collect();
        let symbol = Identifier::new("symbol");
        let mut table = SymbolTable::new();
        table.add_namespace(names[0].clone(), None).expect("new");
        for pair in names.windows(2) {
            table
                .add_namespace(pair[1].clone(), Some(pair[0].clone()))
                .expect("new");
        }
        table
            .add_symbol(&names[0], symbol.clone(), import(&symbol))
            .expect("new");

        assert!(
            table
                .lookup(names.last().expect("some"), &symbol)
                .expect("sound")
                .is_some()
        );
        assert!(table.violations().is_empty());
    });
}

#[test]
fn a_table_of_built_in_frames_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SymbolTable<SymbolFrame>>();
}

#[test]
fn debug_renders_namespaces_and_frames() {
    let [namespace, symbol] = identifiers(["namespace", "symbol"]);
    let mut table = SymbolTable::new();
    table.add_namespace(namespace.clone(), None).expect("new");
    table
        .add_symbol(&namespace, symbol.clone(), import(&symbol))
        .expect("new");

    let text = format!("{table:?}");

    assert!(text.contains(&format!("{namespace:?}")), "{text}");
    assert!(text.contains("ImportFrame"), "{text}");
    let view = format!("{:?}", table.namespace(&namespace).expect("defined"));
    assert!(view.starts_with("Namespace"), "{view}");
}

#[test]
fn each_error_displays_one_lowercase_line() {
    let [ns, child, other, x, parent] = identifiers(["ns", "child", "other", "x", "parent"]);
    let cases = [
        (
            SymbolTableError::NamespaceAlreadyDefined {
                namespace: ns.clone(),
            },
            format!("namespace {ns:?} already defined in the symbol table"),
        ),
        (
            SymbolTableError::NamespaceNotFound {
                namespace: ns.clone(),
            },
            format!("namespace {ns:?} not found in the symbol table"),
        ),
        (
            SymbolTableError::NamespaceHasChildren {
                namespace: ns.clone(),
                children: vec![child.clone(), other.clone()],
            },
            format!(
                "namespace {ns:?} cannot be removed because it is the parent of {child:?}, {other:?}"
            ),
        ),
        (
            SymbolTableError::SymbolAlreadyDefined {
                namespace: ns.clone(),
                symbol: x.clone(),
                defined_in: ns.clone(),
            },
            format!("symbol {x:?} already defined in namespace {ns:?}"),
        ),
        (
            SymbolTableError::SymbolAlreadyDefined {
                namespace: child.clone(),
                symbol: x.clone(),
                defined_in: ns.clone(),
            },
            format!(
                "symbol {x:?} already defined in namespace {ns:?}, an ancestor of namespace {child:?}"
            ),
        ),
        (
            SymbolTableError::SymbolNotFound {
                namespace: Some(ns.clone()),
                symbol: x.clone(),
            },
            format!("symbol {x:?} not found in namespace {ns:?}"),
        ),
        (
            SymbolTableError::SymbolNotFound {
                namespace: None,
                symbol: x.clone(),
            },
            format!("symbol {x:?} not found in the symbol table"),
        ),
        (
            SymbolTableError::CyclicNamespace {
                namespace: ns.clone(),
            },
            format!("namespace {ns:?} is cyclic: the walk up its parents returns to it"),
        ),
        (
            SymbolTableError::ParentNotFound {
                namespace: child.clone(),
                parent: parent.clone(),
            },
            format!("namespace {child:?} references missing parent namespace {parent:?}"),
        ),
    ];
    for (error, expected) in cases {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn each_violation_displays_one_lowercase_line() {
    let [ns, parent, x, y] = identifiers(["ns", "parent", "x", "y"]);
    let cases = [
        (
            Violation::ParentNotFound {
                namespace: ns.clone(),
                parent: parent.clone(),
            },
            format!("namespace {ns:?} references missing parent namespace {parent:?}"),
        ),
        (
            Violation::OwnParent {
                namespace: ns.clone(),
            },
            format!("namespace {ns:?} cannot be its own parent"),
        ),
        (
            Violation::CyclicParentChain {
                namespace: ns.clone(),
            },
            format!("namespace {ns:?} has a cyclic parent chain"),
        ),
        (
            Violation::FrameNameMismatch {
                namespace: ns.clone(),
                symbol: x.clone(),
                frame_name: y.clone(),
            },
            format!("namespace {ns:?} has symbol entry {x:?} whose frame name is {y:?}"),
        ),
    ];
    for (violation, expected) in cases {
        assert_eq!(violation.to_string(), expected);
    }
}
