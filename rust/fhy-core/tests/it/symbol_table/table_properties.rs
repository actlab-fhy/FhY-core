//! Properties of `SymbolTable`.
//!
//! Over random well-formed tables (parents drawn from the namespaces added
//! before, frames named for their symbols, as
//! `tests/test_symbol_table_properties.py` draws them), `violations` is
//! empty, `canonicalize` is idempotent and keeps the table equal, and
//! `lookup` agrees with a walk up a reference model's parents. Over random
//! sequences of operations, the table agrees with a model of the Python
//! implementation's two dictionaries, which also refuses a symbol a
//! descendant defines.

use std::collections::HashMap;

use proptest::prelude::*;

use fhy_core::identifier::Identifier;
use fhy_core::symbol_table::{ImportFrame, SymbolTable, SymbolTableError};

/// A well-formed table's plan: per namespace, the index of an earlier
/// namespace as its parent, and its number of symbols.
fn plan_strategy() -> impl Strategy<Value = Vec<(Option<prop::sample::Index>, usize)>> {
    proptest::collection::vec(
        (
            proptest::option::of(any::<prop::sample::Index>()),
            0_usize..4,
        ),
        1..6,
    )
}

/// A built table, with the reference model of its parents and symbols.
struct Built {
    table: SymbolTable<ImportFrame>,
    namespaces: Vec<Identifier>,
    parents: HashMap<Identifier, Identifier>,
    symbols: Vec<(Identifier, Identifier)>,
}

/// Build the table a plan describes; the namespaces are added in reverse
/// identifier order when `reverse` is set, so `canonicalize` has work.
fn build(plan: &[(Option<prop::sample::Index>, usize)]) -> Built {
    let namespaces: Vec<Identifier> = plan.iter().map(|_| Identifier::new("ns")).collect();
    let mut table = SymbolTable::new();
    let mut parents = HashMap::new();
    let mut symbols = Vec::new();
    for (position, (parent, count)) in plan.iter().enumerate() {
        let parent = parent
            .filter(|_| position > 0)
            .map(|index| namespaces[index.index(position)].clone());
        if let Some(parent) = &parent {
            parents.insert(namespaces[position].clone(), parent.clone());
        }
        table
            .add_namespace(namespaces[position].clone(), parent)
            .expect("each namespace is new");
        for _ in 0..*count {
            let symbol = Identifier::new("symbol");
            table
                .add_symbol(
                    &namespaces[position],
                    symbol.clone(),
                    ImportFrame::new(symbol.clone()),
                )
                .expect("each symbol is new");
            symbols.push((namespaces[position].clone(), symbol));
        }
    }
    Built {
        table,
        namespaces,
        parents,
        symbols,
    }
}

/// Return the namespace of `symbols` that holds `symbol` nearest to
/// `namespace` along the model's parents.
fn model_lookup(built: &Built, namespace: &Identifier, symbol: &Identifier) -> Option<Identifier> {
    let mut current = Some(namespace.clone());
    while let Some(name) = current {
        if built
            .symbols
            .iter()
            .any(|(holder, held)| holder == &name && held == symbol)
        {
            return Some(name);
        }
        current = built.parents.get(&name).cloned();
    }
    None
}

/// An operation on a table over a small alphabet of identifiers.
#[derive(Debug, Clone)]
enum Operation {
    AddNamespace(usize, Option<usize>),
    RemoveNamespace(usize),
    AddSymbol(usize, usize),
    RemoveSymbol(usize, usize),
}

fn operation_strategy() -> impl Strategy<Value = Operation> {
    prop_oneof![
        (0_usize..4, proptest::option::of(0_usize..4))
            .prop_map(|(n, p)| Operation::AddNamespace(n, p)),
        (0_usize..4).prop_map(Operation::RemoveNamespace),
        (0_usize..4, 0_usize..4).prop_map(|(n, s)| Operation::AddSymbol(n, s)),
        (0_usize..4, 0_usize..4).prop_map(|(n, s)| Operation::RemoveSymbol(n, s)),
    ]
}

/// A model of the Python implementation: the namespaces with their symbols
/// in order, and the parent map.
#[derive(Default)]
struct Model {
    table: Vec<(usize, Vec<usize>)>,
    parents: HashMap<usize, usize>,
}

impl Model {
    fn position(&self, namespace: usize) -> Option<usize> {
        self.table.iter().position(|(name, _)| *name == namespace)
    }

    /// Python's `is_symbol_defined_in_namespace`, with `None` for an error.
    fn is_defined_in(&self, namespace: usize, symbol: usize) -> Option<bool> {
        self.position(namespace)?;
        let mut seen = Vec::new();
        let mut current = Some(namespace);
        while let Some(name) = current {
            if seen.contains(&name) {
                return None;
            }
            seen.push(name);
            let position = self.position(name)?;
            if self.table[position].1.contains(&symbol) {
                return Some(true);
            }
            current = self.parents.get(&name).copied();
        }
        Some(false)
    }

    /// Return whether a namespace other than `namespace`, whose chain of
    /// defined parents reaches it, holds `symbol`.
    fn is_defined_below(&self, namespace: usize, symbol: usize) -> bool {
        self.table.iter().any(|(holder, symbols)| {
            if *holder == namespace || !symbols.contains(&symbol) {
                return false;
            }
            let mut seen = vec![*holder];
            let mut current = *holder;
            while let Some(&parent) = self.parents.get(&current) {
                if self.position(parent).is_none() || seen.contains(&parent) {
                    return false;
                }
                if parent == namespace {
                    return true;
                }
                seen.push(parent);
                current = parent;
            }
            false
        })
    }

    /// Apply `operation`, returning whether the table accepts it: Python's
    /// rules, and the descendant check.
    fn apply(&mut self, operation: &Operation) -> bool {
        match *operation {
            Operation::AddNamespace(namespace, parent) => {
                if self.position(namespace).is_some() {
                    return false;
                }
                self.table.push((namespace, Vec::new()));
                if let Some(parent) = parent {
                    self.parents.insert(namespace, parent);
                }
                true
            }
            Operation::RemoveNamespace(namespace) => {
                let Some(position) = self.position(namespace) else {
                    return false;
                };
                if self.parents.values().any(|parent| *parent == namespace) {
                    return false;
                }
                self.table.remove(position);
                self.parents.remove(&namespace);
                true
            }
            Operation::AddSymbol(namespace, symbol) => {
                if self.is_defined_in(namespace, symbol) != Some(false)
                    || self.is_defined_below(namespace, symbol)
                {
                    return false;
                }
                let position = self.position(namespace).expect("defined");
                self.table[position].1.push(symbol);
                true
            }
            Operation::RemoveSymbol(namespace, symbol) => {
                let Some(position) = self.position(namespace) else {
                    return false;
                };
                let symbols = &mut self.table[position].1;
                let Some(index) = symbols.iter().position(|held| *held == symbol) else {
                    return false;
                };
                symbols.remove(index);
                true
            }
        }
    }
}

proptest! {
    #[test]
    fn a_well_formed_table_has_no_violations(plan in plan_strategy()) {
        let built = build(&plan);

        prop_assert!(built.table.violations().is_empty());
    }

    #[test]
    fn canonicalize_is_idempotent_and_keeps_the_table_equal(plan in plan_strategy()) {
        let built = build(&plan);
        let mut table = built.table.clone();

        table.canonicalize();
        let once: Vec<Identifier> = table.namespaces().map(|view| view.name().clone()).collect();
        table.canonicalize();
        let twice: Vec<Identifier> = table.namespaces().map(|view| view.name().clone()).collect();

        prop_assert_eq!(&once, &twice);
        prop_assert!(once.windows(2).all(|pair| pair[0].id() < pair[1].id()));
        prop_assert!(table == built.table);
    }

    #[test]
    fn lookup_agrees_with_a_walk_up_the_parents(plan in plan_strategy()) {
        let built = build(&plan);

        for namespace in &built.namespaces {
            for (_, symbol) in &built.symbols {
                let found = built.table.lookup(namespace, symbol).expect("the table is well formed");
                let expected = model_lookup(&built, namespace, symbol);
                prop_assert_eq!(found.is_some(), expected.is_some());
            }
        }
    }

    #[test]
    fn the_table_agrees_with_a_model_of_two_dictionaries(
        operations in proptest::collection::vec(operation_strategy(), 0..24),
    ) {
        let namespaces: Vec<Identifier> = (0..4).map(|_| Identifier::new("ns")).collect();
        let symbols: Vec<Identifier> = (0..4).map(|_| Identifier::new("symbol")).collect();
        let mut table = SymbolTable::new();
        let mut model = Model::default();

        for operation in &operations {
            let accepted = match *operation {
                Operation::AddNamespace(namespace, parent) => table
                    .add_namespace(namespaces[namespace].clone(), parent.map(|parent| namespaces[parent].clone()))
                    .is_ok(),
                Operation::RemoveNamespace(namespace) => table.remove_namespace(&namespaces[namespace]).is_ok(),
                Operation::AddSymbol(namespace, symbol) => table
                    .add_symbol(&namespaces[namespace], symbols[symbol].clone(), ImportFrame::new(symbols[symbol].clone()))
                    .is_ok(),
                Operation::RemoveSymbol(namespace, symbol) => {
                    table.remove_symbol(&namespaces[namespace], &symbols[symbol]).is_ok()
                }
            };
            prop_assert_eq!(accepted, model.apply(operation), "{:?}", operation);
        }

        let actual: Vec<(Identifier, Option<Identifier>, Vec<Identifier>)> = table
            .namespaces()
            .map(|view| (view.name().clone(), view.parent().cloned(), view.iter().map(|(symbol, _)| symbol.clone()).collect()))
            .collect();
        let expected: Vec<(Identifier, Option<Identifier>, Vec<Identifier>)> = model
            .table
            .iter()
            .map(|(name, held)| (
                namespaces[*name].clone(),
                model.parents.get(name).map(|parent| namespaces[*parent].clone()),
                held.iter().map(|symbol| symbols[*symbol].clone()).collect(),
            ))
            .collect();
        prop_assert_eq!(actual, expected);
        for (namespace, namespace_name) in namespaces.iter().enumerate() {
            for (symbol, symbol_name) in symbols.iter().enumerate() {
                let found = table.lookup(namespace_name, symbol_name);
                if let Some(defined) = model.is_defined_in(namespace, symbol) {
                    prop_assert_eq!(found.map(|frame| frame.is_some()), Ok(defined));
                } else {
                    let is_a_lookup_error = matches!(
                        found,
                        Err(SymbolTableError::NamespaceNotFound { .. }
                            | SymbolTableError::CyclicNamespace { .. }
                            | SymbolTableError::ParentNotFound { .. })
                    );
                    prop_assert!(is_a_lookup_error, "{:?}", found);
                }
            }
        }
    }
}
