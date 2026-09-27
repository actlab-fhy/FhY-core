//! The symbol table: namespaces, their parents, and their symbols' frames.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::fmt;

use crate::identifier::Identifier;
use crate::types::UnificationError;

use super::error::{SymbolTableError, Violation};
use super::frame::{Frame, SymbolFrame};
use super::ordered::OrderedMap;

/// A table of namespaces, each mapping its symbols to frames of type `F`.
///
/// Namespaces keep the order they were added in, and so do the symbols of
/// each namespace. A namespace may name a parent namespace:
/// [`lookup`](Self::lookup) walks from a namespace up its parents, and
/// [`add_symbol`](Self::add_symbol) refuses a symbol the namespace or one of
/// its ancestors or one of its descendants already defines, so an inner
/// namespace never shadows an outer one, whatever order they are filled
/// in.
///
/// A parent is not checked when a namespace is added: it may be added
/// later, and a namespace may even name itself. [`violations`](Self::violations)
/// reports the parents that are missing or that close a cycle, the symbols
/// an ancestor also defines, and the frames whose names are not their
/// symbols; a lookup that meets a missing parent or a cycle fails.
///
/// Every walk is a loop, so a long chain of parents needs no stack.
#[derive(Clone)]
pub struct SymbolTable<F> {
    namespaces: OrderedMap<NamespaceData<F>>,
}

/// A namespace's parent and its symbols' frames.
#[derive(Debug, Clone)]
struct NamespaceData<F> {
    parent: Option<Identifier>,
    symbols: OrderedMap<F>,
}

/// A view of one namespace of a [`SymbolTable`]: its name, its parent and
/// its symbols in order.
pub struct Namespace<'a, F> {
    name: &'a Identifier,
    data: &'a NamespaceData<F>,
}

impl<F> Clone for Namespace<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for Namespace<'_, F> {}

impl<F: fmt::Debug> fmt::Debug for Namespace<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Namespace")
            .field("name", self.name)
            .field("parent", &self.data.parent)
            .field("symbols", &self.data.symbols)
            .finish()
    }
}

impl<'a, F> Namespace<'a, F> {
    /// Return the namespace's name.
    #[must_use]
    pub fn name(&self) -> &'a Identifier {
        self.name
    }

    /// Return the parent the namespace names, which may not be defined.
    #[must_use]
    pub fn parent(&self) -> Option<&'a Identifier> {
        self.data.parent.as_ref()
    }

    /// Return the number of symbols the namespace itself holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.symbols.len()
    }

    /// Return whether the namespace itself holds no symbol.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.symbols.len() == 0
    }

    /// Return the frame of `symbol`, if the namespace itself holds it.
    #[must_use]
    pub fn get(&self, symbol: &Identifier) -> Option<&'a F> {
        self.data.symbols.get(symbol)
    }

    /// Return whether the namespace itself holds `symbol`.
    #[must_use]
    pub fn contains(&self, symbol: &Identifier) -> bool {
        self.data.symbols.contains_key(symbol)
    }

    /// Return the namespace's symbols and their frames, in insertion order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&'a Identifier, &'a F)> + 'a {
        self.data.symbols.iter()
    }
}

/// What the walk from one namespace up its parents found, as
/// [`SymbolTable::violations`] classifies each chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chain {
    /// The walk ends: at a namespace without a parent, or at a parent that
    /// is not defined.
    Ends,
    /// The walk comes back to a namespace it passed.
    Cycles,
}

impl<F> SymbolTable<F> {
    /// Return an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            namespaces: OrderedMap::new(),
        }
    }

    /// Return the number of namespaces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.namespaces.len()
    }

    /// Return whether the table has no namespace.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.namespaces.len() == 0
    }

    /// Return whether `namespace` is defined.
    #[must_use]
    pub fn contains_namespace(&self, namespace: &Identifier) -> bool {
        self.namespaces.contains_key(namespace)
    }

    /// Return the view of `namespace`, if it is defined.
    #[must_use]
    pub fn namespace(&self, namespace: &Identifier) -> Option<Namespace<'_, F>> {
        let (name, data) = self.namespaces.get_key_value(namespace)?;
        Some(Namespace { name, data })
    }

    /// Return the views of the namespaces, in insertion order.
    #[must_use]
    pub fn namespaces(&self) -> impl ExactSizeIterator<Item = Namespace<'_, F>> + '_ {
        self.namespaces
            .iter()
            .map(|(name, data)| Namespace { name, data })
    }

    /// Add the empty namespace `namespace`, whose parent is `parent`.
    ///
    /// The parent is not checked: it may be added later.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolTableError::NamespaceAlreadyDefined`] if `namespace`
    /// is defined, and leaves the table unchanged.
    pub fn add_namespace(
        &mut self,
        namespace: Identifier,
        parent: Option<Identifier>,
    ) -> Result<(), SymbolTableError> {
        if self.namespaces.contains_key(&namespace) {
            return Err(SymbolTableError::NamespaceAlreadyDefined { namespace });
        }
        self.namespaces.insert(
            namespace,
            NamespaceData {
                parent,
                symbols: OrderedMap::new(),
            },
        );
        Ok(())
    }

    /// Remove `namespace` and its symbols.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolTableError::NamespaceNotFound`] if `namespace` is not
    /// defined, and [`SymbolTableError::NamespaceHasChildren`] if other
    /// namespaces name it as their parent.
    pub fn remove_namespace(&mut self, namespace: &Identifier) -> Result<(), SymbolTableError> {
        if !self.namespaces.contains_key(namespace) {
            return Err(SymbolTableError::NamespaceNotFound {
                namespace: namespace.clone(),
            });
        }
        let children: Vec<Identifier> = self
            .namespaces
            .iter()
            .filter(|(_, data)| data.parent.as_ref() == Some(namespace))
            .map(|(child, _)| child.clone())
            .collect();
        if !children.is_empty() {
            return Err(SymbolTableError::NamespaceHasChildren {
                namespace: namespace.clone(),
                children,
            });
        }
        self.namespaces.remove(namespace);
        Ok(())
    }

    /// Add `symbol`, described by `frame`, to `namespace`.
    ///
    /// The frame is not checked against the symbol;
    /// [`violations`](Self::violations) reports a frame naming another
    /// symbol.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolTableError::SymbolAlreadyDefined`] if
    /// [`lookup`](Self::lookup) finds `symbol` from `namespace`, and the
    /// errors of that lookup; then
    /// [`SymbolTableError::SymbolDefinedInDescendant`] if a namespace whose
    /// chain of parents reaches `namespace` holds `symbol`. The table is
    /// unchanged on an error.
    pub fn add_symbol(
        &mut self,
        namespace: &Identifier,
        symbol: Identifier,
        frame: F,
    ) -> Result<(), SymbolTableError> {
        if let Some((defined_in, _)) = self.resolve(namespace, &symbol)? {
            return Err(SymbolTableError::SymbolAlreadyDefined {
                namespace: namespace.clone(),
                symbol,
                defined_in: defined_in.clone(),
            });
        }
        if let Some(defined_in) = self.descendant_holding(namespace, &symbol) {
            return Err(SymbolTableError::SymbolDefinedInDescendant {
                namespace: namespace.clone(),
                symbol,
                defined_in: defined_in.clone(),
            });
        }
        let data = self
            .namespaces
            .get_mut(namespace)
            .unwrap_or_else(|| unreachable!("the lookup found the namespace"));
        data.symbols.insert(symbol, frame);
        Ok(())
    }

    /// Remove `symbol` from `namespace`, which must hold it itself, and
    /// return its frame.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolTableError::NamespaceNotFound`] if `namespace` is not
    /// defined, and [`SymbolTableError::SymbolNotFound`] if it does not
    /// itself hold `symbol`.
    pub fn remove_symbol(
        &mut self,
        namespace: &Identifier,
        symbol: &Identifier,
    ) -> Result<F, SymbolTableError> {
        let Some(data) = self.namespaces.get_mut(namespace) else {
            return Err(SymbolTableError::NamespaceNotFound {
                namespace: namespace.clone(),
            });
        };
        data.symbols
            .remove(symbol)
            .ok_or_else(|| SymbolTableError::SymbolNotFound {
                namespace: Some(namespace.clone()),
                symbol: symbol.clone(),
            })
    }

    /// Return the frame of `symbol` in `namespace` or its nearest ancestor
    /// that holds it, or `None` if none does.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolTableError::NamespaceNotFound`] if `namespace` is not
    /// defined, [`SymbolTableError::CyclicNamespace`] if the walk up the
    /// parents comes back to a namespace before it finds `symbol`, and
    /// [`SymbolTableError::ParentNotFound`] if it reaches a parent that is
    /// not defined.
    pub fn lookup(
        &self,
        namespace: &Identifier,
        symbol: &Identifier,
    ) -> Result<Option<&F>, SymbolTableError> {
        Ok(self.resolve(namespace, symbol)?.map(|(_, frame)| frame))
    }

    /// Return the namespace, `namespace` or its nearest ancestor, that
    /// holds `symbol`, with the frame; the errors are [`lookup`](Self::lookup)'s.
    fn resolve(
        &self,
        namespace: &Identifier,
        symbol: &Identifier,
    ) -> Result<Option<(&Identifier, &F)>, SymbolTableError> {
        let Some((mut name, mut data)) = self.namespaces.get_key_value(namespace) else {
            return Err(SymbolTableError::NamespaceNotFound {
                namespace: namespace.clone(),
            });
        };
        let mut passed = HashSet::new();
        loop {
            if !passed.insert(name) {
                return Err(SymbolTableError::CyclicNamespace {
                    namespace: name.clone(),
                });
            }
            if let Some(frame) = data.symbols.get(symbol) {
                return Ok(Some((name, frame)));
            }
            let Some(parent) = &data.parent else {
                return Ok(None);
            };
            let Some((parent_name, parent_data)) = self.namespaces.get_key_value(parent) else {
                return Err(SymbolTableError::ParentNotFound {
                    namespace: name.clone(),
                    parent: parent.clone(),
                });
            };
            (name, data) = (parent_name, parent_data);
        }
    }

    /// Return the first namespace, in insertion order, that holds `symbol`
    /// and whose chain of parents reaches `namespace`.
    ///
    /// Only the namespaces holding `symbol` walk up their parents, each walk
    /// stopping at a namespace it passed, so a cycle ends it.
    fn descendant_holding(
        &self,
        namespace: &Identifier,
        symbol: &Identifier,
    ) -> Option<&Identifier> {
        self.namespaces
            .iter()
            .filter(|(name, data)| *name != namespace && data.symbols.contains_key(symbol))
            .find(|(name, _)| self.ancestors(name).any(|ancestor| ancestor == namespace))
            .map(|(name, _)| name)
    }

    /// Return the ancestors of `namespace`, nearest first: each defined
    /// parent up the chain, stopping at a missing parent or at a namespace
    /// already passed, `namespace` itself included.
    fn ancestors<'t>(
        &'t self,
        namespace: &'t Identifier,
    ) -> impl Iterator<Item = &'t Identifier> + 't {
        let mut passed: HashSet<&Identifier> = HashSet::from([namespace]);
        let mut current = namespace;
        std::iter::from_fn(move || {
            let (parent, _) = self
                .namespaces
                .get(current)
                .and_then(|data| data.parent.as_ref())
                .and_then(|parent| self.namespaces.get_key_value(parent))?;
            if !passed.insert(parent) {
                return None;
            }
            current = parent;
            Some(parent)
        })
    }

    /// Return the frame of `symbol` in the first namespace, in insertion
    /// order, that holds it, ignoring parents.
    #[must_use]
    pub fn find(&self, symbol: &Identifier) -> Option<&F> {
        self.namespaces
            .iter()
            .find_map(|(_, data)| data.symbols.get(symbol))
    }

    /// Copy every namespace of `other` into the table.
    ///
    /// A namespace the table defines keeps its position and has its symbols
    /// replaced by `other`'s; any other is added at the end. A namespace
    /// takes `other`'s parent when `other` names one, and otherwise keeps
    /// its own. The two tables share nothing afterwards but the frames'
    /// clones.
    pub fn update_namespaces(&mut self, other: &Self)
    where
        F: Clone,
    {
        for (name, incoming) in other.namespaces.iter() {
            match self.namespaces.get_mut(name) {
                Some(existing) => {
                    existing.symbols = incoming.symbols.clone();
                    if incoming.parent.is_some() {
                        existing.parent.clone_from(&incoming.parent);
                    }
                }
                None => {
                    self.namespaces.insert(name.clone(), incoming.clone());
                }
            }
        }
    }

    /// Set `namespace`'s parent and symbols, replacing the namespace in
    /// place if it is defined and adding it at the end otherwise, without
    /// checking anything.
    ///
    /// This restores a table from its parts, such as a saved table's
    /// namespaces in order, whatever state it was in: a symbol an ancestor
    /// also defines, or a parent chain that cycles, which
    /// [`add_symbol`](Self::add_symbol) refuses to build. A later symbol of
    /// `symbols` replaces an earlier one of the same identifier.
    pub fn insert_namespace(
        &mut self,
        namespace: Identifier,
        parent: Option<Identifier>,
        symbols: impl IntoIterator<Item = (Identifier, F)>,
    ) {
        let mut ordered = OrderedMap::new();
        for (symbol, frame) in symbols {
            ordered.insert(symbol, frame);
        }
        self.namespaces.insert(
            namespace,
            NamespaceData {
                parent,
                symbols: ordered,
            },
        );
    }

    /// Sort the namespaces, and each namespace's symbols, by identifier id.
    pub fn canonicalize(&mut self) {
        self.namespaces.sort_by_id();
        for data in self.namespaces.values_mut() {
            data.symbols.sort_by_id();
        }
    }

    /// Return whether `other` has the same namespaces, the same parents, and
    /// in each namespace the same symbols, with `frames` holding for each
    /// pair of frames of one symbol; the orders do not count.
    ///
    /// The frames are compared in the table's order, and the comparison
    /// stops at the first pair `frames` answers `false` for.
    ///
    /// # Errors
    ///
    /// Returns the first error `frames` returns.
    pub fn is_equivalent_by<G, E>(
        &self,
        other: &SymbolTable<G>,
        mut frames: impl FnMut(&F, &G) -> Result<bool, E>,
    ) -> Result<bool, E> {
        if self.namespaces.len() != other.namespaces.len() {
            return Ok(false);
        }
        let mut pairs = Vec::with_capacity(self.namespaces.len());
        for (name, data) in self.namespaces.iter() {
            let Some(other_data) = other.namespaces.get(name) else {
                return Ok(false);
            };
            if data.parent != other_data.parent {
                return Ok(false);
            }
            pairs.push((data, other_data));
        }
        for (data, other_data) in pairs {
            if data.symbols.len() != other_data.symbols.len() {
                return Ok(false);
            }
            if data
                .symbols
                .iter()
                .any(|(symbol, _)| !other_data.symbols.contains_key(symbol))
            {
                return Ok(false);
            }
            for (symbol, frame) in data.symbols.iter() {
                let other_frame = other_data
                    .symbols
                    .get(symbol)
                    .unwrap_or_else(|| unreachable!("the symbol sets are equal"));
                if !frames(frame, other_frame)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Return the table's broken invariants, in this order: for each
    /// namespace, a parent that is not defined or is itself; then each
    /// namespace whose chain of parents cycles; then each frame whose name
    /// is not its symbol; then each symbol that an ancestor of its
    /// namespace also defines, outside a cycle. The list is empty for a
    /// well-formed table.
    #[must_use]
    pub fn violations(&self) -> Vec<Violation>
    where
        F: Frame,
    {
        let mut violations = Vec::new();
        for (name, data) in self.namespaces.iter() {
            let Some(parent) = &data.parent else {
                continue;
            };
            if !self.namespaces.contains_key(parent) {
                violations.push(Violation::ParentNotFound {
                    namespace: name.clone(),
                    parent: parent.clone(),
                });
            }
            if parent == name {
                violations.push(Violation::OwnParent {
                    namespace: name.clone(),
                });
            }
        }
        let chains = self.classify_chains();
        for (name, _) in self.namespaces.iter() {
            if chains.get(name) == Some(&Chain::Cycles) {
                violations.push(Violation::CyclicParentChain {
                    namespace: name.clone(),
                });
            }
        }
        for (name, data) in self.namespaces.iter() {
            for (symbol, frame) in data.symbols.iter() {
                if frame.name() != symbol {
                    violations.push(Violation::FrameNameMismatch {
                        namespace: name.clone(),
                        symbol: symbol.clone(),
                        frame_name: frame.name().clone(),
                    });
                }
            }
        }
        for (name, data) in self.namespaces.iter() {
            if chains.get(name) == Some(&Chain::Cycles) {
                continue;
            }
            for (symbol, _) in data.symbols.iter() {
                let shadowing = self.ancestors(name).find(|ancestor| {
                    self.namespaces
                        .get(ancestor)
                        .is_some_and(|ancestor| ancestor.symbols.contains_key(symbol))
                });
                if let Some(ancestor) = shadowing {
                    violations.push(Violation::ShadowedSymbol {
                        namespace: name.clone(),
                        symbol: symbol.clone(),
                        ancestor: ancestor.clone(),
                    });
                }
            }
        }
        violations
    }

    /// Return, for each namespace, whether the walk up its parents ends or
    /// cycles, in time linear in the namespaces: each walk stops at the
    /// first namespace already classified, and its whole path shares the
    /// answer.
    fn classify_chains(&self) -> HashMap<&Identifier, Chain> {
        let mut chains: HashMap<&Identifier, Chain> = HashMap::new();
        for (start, _) in self.namespaces.iter() {
            let mut path: Vec<&Identifier> = Vec::new();
            let mut on_path: HashSet<&Identifier> = HashSet::new();
            let mut current = Some(start);
            let answer = loop {
                let Some(name) = current else {
                    break Chain::Ends;
                };
                if let Some(&known) = chains.get(name) {
                    break known;
                }
                if !on_path.insert(name) {
                    break Chain::Cycles;
                }
                path.push(name);
                current = self
                    .namespaces
                    .get(name)
                    .and_then(|data| data.parent.as_ref());
            };
            for name in path {
                chains.insert(name, answer);
            }
        }
        chains
    }
}

impl SymbolTable<SymbolFrame> {
    /// Return whether `other` is equivalent, comparing frames with
    /// [`SymbolFrame::is_structurally_equivalent`].
    ///
    /// # Errors
    ///
    /// Returns [`UnificationError::Extension`] for a type extension that
    /// fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, UnificationError> {
        self.is_equivalent_by(other, SymbolFrame::is_structurally_equivalent)
    }
}

impl<F> Default for SymbolTable<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PartialEq> PartialEq for SymbolTable<F> {
    /// Return whether the two tables are equivalent, comparing frames with
    /// `==`; the orders do not count.
    fn eq(&self, other: &Self) -> bool {
        self.is_equivalent_by(other, |left, right| Ok::<_, Infallible>(left == right))
            .unwrap_or_else(|never| match never {})
    }
}

impl<F: Eq> Eq for SymbolTable<F> {}

impl<F: fmt::Debug> fmt::Debug for SymbolTable<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.namespaces.iter()).finish()
    }
}
