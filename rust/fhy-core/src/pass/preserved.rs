//! Analysis identities and the sets of analyses a pass run preserves.

use std::any::{TypeId, type_name};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use super::analysis::Analysis;
use crate::identifier::Identifier;

/// The identity of an analysis, keying cached results and preservation sets.
///
/// An analysis is named either by its Rust type, through
/// [`of`](Self::of), or, for analyses no Rust type names, such as ones a
/// language binding defines at run time, by an [`Identifier`], through
/// [`of_identifier`](Self::of_identifier). Two ids are equal exactly when
/// they name the same type, or the same identifier; an id of a type never
/// equals an id of an identifier. Cloning an id is cheap.
///
/// Ids of types order first, by the type's name, so a preservation set
/// lists them in a stable order. Ids of identifiers follow, ordered by the
/// identifier's id, which is the order the identifiers were issued in.
#[derive(Clone)]
pub struct AnalysisId(AnalysisName);

/// What an [`AnalysisId`] names.
#[derive(Clone)]
enum AnalysisName {
    /// A Rust analysis type.
    Type {
        type_id: TypeId,
        type_name: &'static str,
    },
    /// An analysis named by an identifier.
    Identifier(Identifier),
}

impl AnalysisId {
    /// Return the id of the analysis type `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::pass::{Analysis, AnalysisId};
    ///
    /// struct Liveness;
    /// struct Dominance;
    ///
    /// # impl Analysis for Liveness {
    /// #     type Ir = ();
    /// #     type Output = ();
    /// #     fn run(&self, _ir: &()) {}
    /// # }
    /// # impl Analysis for Dominance {
    /// #     type Ir = ();
    /// #     type Output = ();
    /// #     fn run(&self, _ir: &()) {}
    /// # }
    /// assert_eq!(AnalysisId::of::<Liveness>(), AnalysisId::of::<Liveness>());
    /// assert_ne!(AnalysisId::of::<Liveness>(), AnalysisId::of::<Dominance>());
    /// ```
    #[must_use]
    pub fn of<A: Analysis>() -> Self {
        Self(AnalysisName::Type {
            type_id: TypeId::of::<A>(),
            type_name: type_name::<A>(),
        })
    }

    /// Return the id of the analysis named `name`, for an analysis no Rust
    /// type names.
    ///
    /// The id equals the id of every identifier equal to `name`, and
    /// displays as its name hint. Its results are requested with
    /// [`PassContext::analysis_by_id`](super::PassContext::analysis_by_id).
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::pass::{AnalysisId, PreservedAnalyses};
    ///
    /// let liveness = Identifier::new("liveness");
    /// let id = AnalysisId::of_identifier(&liveness);
    ///
    /// assert_eq!(id, AnalysisId::of_identifier(&liveness.clone()));
    /// assert_ne!(id, AnalysisId::of_identifier(&Identifier::new("liveness")));
    /// assert_eq!(id.to_string(), "liveness");
    /// assert!(PreservedAnalyses::none().preserve_id(id.clone()).is_id_preserved(&id));
    /// ```
    #[must_use]
    pub fn of_identifier(name: &Identifier) -> Self {
        Self(AnalysisName::Identifier(name.clone()))
    }

    /// Return the identifier the id was built from, or `None` for the id of
    /// an analysis type.
    #[must_use]
    pub const fn identifier(&self) -> Option<&Identifier> {
        match &self.0 {
            AnalysisName::Type { .. } => None,
            AnalysisName::Identifier(name) => Some(name),
        }
    }
}

impl PartialEq for AnalysisId {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (
                AnalysisName::Type { type_id, .. },
                AnalysisName::Type {
                    type_id: other_type_id,
                    ..
                },
            ) => type_id == other_type_id,
            (AnalysisName::Identifier(name), AnalysisName::Identifier(other_name)) => {
                name == other_name
            }
            _ => false,
        }
    }
}

impl Eq for AnalysisId {}

impl Hash for AnalysisId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(&self.0).hash(state);
        match &self.0 {
            AnalysisName::Type { type_id, .. } => type_id.hash(state),
            AnalysisName::Identifier(name) => name.hash(state),
        }
    }
}

/// Order ids of types by the type's name, then by the type itself, and
/// after them ids of identifiers by the identifier's id.
///
/// Identifiers order by id because identifiers are equal exactly when their
/// ids are: two equal identifiers may carry different name hints.
impl Ord for AnalysisId {
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self.0, &other.0) {
            (
                AnalysisName::Type { type_id, type_name },
                AnalysisName::Type {
                    type_id: other_type_id,
                    type_name: other_type_name,
                },
            ) => type_name
                .cmp(other_type_name)
                .then_with(|| type_id.cmp(other_type_id)),
            (AnalysisName::Type { .. }, AnalysisName::Identifier(_)) => Ordering::Less,
            (AnalysisName::Identifier(_), AnalysisName::Type { .. }) => Ordering::Greater,
            (AnalysisName::Identifier(name), AnalysisName::Identifier(other_name)) => {
                name.id().cmp(&other_name.id())
            }
        }
    }
}

impl PartialOrd for AnalysisId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Render the analysis type's name, for example `my_crate::Liveness`, or
/// the identifier's name hint.
impl fmt::Display for AnalysisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            AnalysisName::Type { type_name, .. } => f.write_str(type_name),
            AnalysisName::Identifier(name) => fmt::Display::fmt(name, f),
        }
    }
}

impl fmt::Debug for AnalysisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tuple = f.debug_tuple("AnalysisId");
        match &self.0 {
            AnalysisName::Type { type_name, .. } => tuple.field(type_name),
            AnalysisName::Identifier(name) => tuple.field(name),
        }
        .finish()
    }
}

/// Which analyses a set preserves.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Preservation {
    All,
    Only(BTreeSet<AnalysisId>),
}

/// The analyses a pass run leaves valid for its output.
///
/// A set either preserves every analysis or exactly the listed ones. A pass
/// manager carries the cached results of the preserved analyses from a
/// pass's input over to its output; the output does not inherit the rest.
///
/// # Examples
///
/// ```
/// use fhy_core::pass::{Analysis, PreservedAnalyses};
///
/// struct Liveness;
/// struct Dominance;
///
/// # impl Analysis for Liveness {
/// #     type Ir = ();
/// #     type Output = ();
/// #     fn run(&self, _ir: &()) {}
/// # }
/// # impl Analysis for Dominance {
/// #     type Ir = ();
/// #     type Output = ();
/// #     fn run(&self, _ir: &()) {}
/// # }
/// let preserved = PreservedAnalyses::none().preserve::<Liveness>();
///
/// assert!(preserved.is_preserved::<Liveness>());
/// assert!(!preserved.is_preserved::<Dominance>());
/// assert!(PreservedAnalyses::all().is_preserved::<Dominance>());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreservedAnalyses {
    preservation: Preservation,
}

impl PreservedAnalyses {
    /// Return the set that preserves every analysis.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            preservation: Preservation::All,
        }
    }

    /// Return the set that preserves no analysis.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            preservation: Preservation::Only(BTreeSet::new()),
        }
    }

    /// Return this set with the analysis `A` preserved as well.
    ///
    /// A set that preserves every analysis comes back unchanged. Only an
    /// [`Analysis`] can be preserved:
    ///
    /// ```compile_fail
    /// use fhy_core::pass::PreservedAnalyses;
    ///
    /// let preserved = PreservedAnalyses::none().preserve::<String>();
    /// ```
    #[must_use]
    pub fn preserve<A: Analysis>(self) -> Self {
        self.preserve_id(AnalysisId::of::<A>())
    }

    /// Return this set with the analysis `id` preserved as well.
    ///
    /// A set that preserves every analysis comes back unchanged.
    #[must_use]
    pub fn preserve_id(mut self, id: AnalysisId) -> Self {
        if let Preservation::Only(ids) = &mut self.preservation {
            ids.insert(id);
        }
        self
    }

    /// Return whether the set preserves the analysis `A`.
    #[must_use]
    pub fn is_preserved<A: Analysis>(&self) -> bool {
        self.is_id_preserved(&AnalysisId::of::<A>())
    }

    /// Return whether the set preserves the analysis `id`.
    #[must_use]
    pub fn is_id_preserved(&self, id: &AnalysisId) -> bool {
        match &self.preservation {
            Preservation::All => true,
            Preservation::Only(ids) => ids.contains(id),
        }
    }

    /// Return whether the set preserves every analysis.
    #[must_use]
    pub const fn preserves_all(&self) -> bool {
        matches!(self.preservation, Preservation::All)
    }

    /// Return the ids listed in a set that preserves only some analyses, in
    /// the order of [`AnalysisId`]: ids of types by type name, then ids of
    /// identifiers.
    ///
    /// A set that preserves every analysis lists no ids.
    pub fn preserved_ids(&self) -> impl Iterator<Item = &AnalysisId> + '_ {
        let ids = match &self.preservation {
            Preservation::All => None,
            Preservation::Only(ids) => Some(ids),
        };
        ids.into_iter().flatten()
    }
}
