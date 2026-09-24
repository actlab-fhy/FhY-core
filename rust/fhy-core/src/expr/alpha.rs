//! The free-identifier renaming that alpha equivalence of expressions is
//! checked under.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::identifier::Identifier;

use super::error::NonInjectiveRenamingError;

/// An injective renaming of free identifiers, the correspondence
/// [`Expression::is_alpha_equivalent_under`](super::Expression::is_alpha_equivalent_under)
/// checks two trees against.
///
/// Each mapped identifier on one side corresponds to its image on the
/// other; an unmapped identifier corresponds only to itself, and only when
/// it is not an image. [`try_new`](Self::try_new) refuses a map sending two
/// identifiers to one image, which would relate `a + b` to `c + c` and make
/// the relation one-directional. The default renaming maps nothing, so
/// every identifier corresponds only to itself.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{AlphaRenaming, Expression};
///
/// let (a, b, c) = (Identifier::new("a"), Identifier::new("b"), Identifier::new("c"));
/// let renaming = AlphaRenaming::try_new(HashMap::from([(a.clone(), c.clone())]))
///     .expect("one pair is injective");
/// let sum = Expression::from(a.clone()) + 1;
/// assert!(sum.is_alpha_equivalent_under(&(Expression::from(c.clone()) + 1), &renaming));
///
/// let colliding = HashMap::from([(a, c.clone()), (b, c)]);
/// assert!(AlphaRenaming::try_new(colliding).is_err());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlphaRenaming {
    free_renaming: HashMap<Identifier, Identifier>,
    images: HashSet<Identifier>,
}

impl AlphaRenaming {
    /// Construct the renaming sending each key of `free_renaming` to its
    /// value.
    ///
    /// # Errors
    ///
    /// Returns [`NonInjectiveRenamingError`] naming a shared image if two
    /// identifiers map to the same image.
    pub fn try_new<S: BuildHasher>(
        free_renaming: HashMap<Identifier, Identifier, S>,
    ) -> Result<Self, NonInjectiveRenamingError> {
        let mut images = HashSet::with_capacity(free_renaming.len());
        for image in free_renaming.values() {
            if !images.insert(image.clone()) {
                return Err(NonInjectiveRenamingError::new(image.clone()));
            }
        }
        Ok(Self {
            free_renaming: free_renaming.into_iter().collect(),
            images,
        })
    }

    /// Return whether `left` on one side corresponds to `right` on the
    /// other: `right` is the image of a mapped `left`, or an unmapped
    /// `left` itself when it is not an image.
    #[must_use]
    pub fn are_identifiers_alpha_equivalent(&self, left: &Identifier, right: &Identifier) -> bool {
        match self.free_renaming.get(left) {
            Some(image) => image == right,
            None => !self.images.contains(right) && left == right,
        }
    }

    /// Return whether the renaming maps no identifier.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.free_renaming.is_empty()
    }
}
