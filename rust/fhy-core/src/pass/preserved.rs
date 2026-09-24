//! Analysis identities and the sets of analyses a pass run preserves.

use std::any::{TypeId, type_name};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::hash::{Hash, Hasher};

/// The identity of an analysis type, keying cached results and preservation
/// sets.
///
/// Two ids are equal exactly when they name the same type. Ids order by the
/// type's name, so a preservation set lists its ids in a stable order.
#[derive(Clone, Copy)]
pub struct AnalysisId {
    type_id: TypeId,
    type_name: &'static str,
}

impl AnalysisId {
    /// Return the id of the analysis type `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::pass::AnalysisId;
    ///
    /// struct Liveness;
    /// struct Dominance;
    ///
    /// assert_eq!(AnalysisId::of::<Liveness>(), AnalysisId::of::<Liveness>());
    /// assert_ne!(AnalysisId::of::<Liveness>(), AnalysisId::of::<Dominance>());
    /// ```
    #[must_use]
    pub fn of<A: ?Sized + 'static>() -> Self {
        Self {
            type_id: TypeId::of::<A>(),
            type_name: type_name::<A>(),
        }
    }
}

impl PartialEq for AnalysisId {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}

impl Eq for AnalysisId {}

impl Hash for AnalysisId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
    }
}

/// Order by the type's name, then by the type itself.
impl Ord for AnalysisId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.type_name
            .cmp(other.type_name)
            .then_with(|| self.type_id.cmp(&other.type_id))
    }
}

impl PartialOrd for AnalysisId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Render the analysis type's name, for example `my_crate::Liveness`.
impl fmt::Display for AnalysisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.type_name)
    }
}

impl fmt::Debug for AnalysisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AnalysisId").field(&self.type_name).finish()
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
/// manager keeps the cached results of preserved analyses when it moves from
/// a pass's input to its output, and drops the rest.
///
/// # Examples
///
/// ```
/// use fhy_core::pass::PreservedAnalyses;
///
/// struct Liveness;
/// struct Dominance;
///
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
    pub fn all() -> Self {
        Self {
            preservation: Preservation::All,
        }
    }

    /// Return the set that preserves no analysis.
    #[must_use]
    pub fn none() -> Self {
        Self {
            preservation: Preservation::Only(BTreeSet::new()),
        }
    }

    /// Return this set with the analysis `A` preserved as well.
    ///
    /// A set that preserves every analysis comes back unchanged.
    #[must_use]
    pub fn preserve<A: ?Sized + 'static>(self) -> Self {
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
    pub fn is_preserved<A: ?Sized + 'static>(&self) -> bool {
        self.is_id_preserved(AnalysisId::of::<A>())
    }

    /// Return whether the set preserves the analysis `id`.
    #[must_use]
    pub fn is_id_preserved(&self, id: AnalysisId) -> bool {
        match &self.preservation {
            Preservation::All => true,
            Preservation::Only(ids) => ids.contains(&id),
        }
    }

    /// Return whether the set preserves every analysis.
    #[must_use]
    pub fn preserves_all(&self) -> bool {
        matches!(self.preservation, Preservation::All)
    }

    /// Return the ids listed in a set that preserves only some analyses,
    /// ordered by type name.
    ///
    /// A set that preserves every analysis lists no ids.
    pub fn preserved_ids(&self) -> impl Iterator<Item = AnalysisId> + '_ {
        let ids = match &self.preservation {
            Preservation::All => None,
            Preservation::Only(ids) => Some(ids),
        };
        ids.into_iter().flatten().copied()
    }
}
