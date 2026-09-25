//! The renaming that alpha equivalence of expressions is checked under: a
//! stack of binder frames over a free-identifier renaming.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::identifier::Identifier;

use super::error::{NonInjectiveRenamingError, RenamingPart};

/// The correspondence between identifiers on two sides of a comparison that
/// [`Expression::is_alpha_equivalent_under`](super::Expression::is_alpha_equivalent_under)
/// checks two trees against: a stack of binder frames over a free-identifier
/// renaming.
///
/// - **Binder frames.** A term that binds identifiers, such as a parameter
///   list over a body, compares its body with the other term's under one
///   more frame, pairing each of its bound identifiers with the other term's
///   ([`enter_binder`](Self::enter_binder)), and drops the frame after
///   ([`leave_binder`](Self::leave_binder)). An inner frame shadows outer
///   ones, and two frames may share an image, as two nested binders may
///   bind one name.
/// - **The free renaming** ([`try_new`](Self::try_new)) pairs identifiers
///   that no frame binds, for comparing two terms drawn from different
///   scopes. An identifier it does not map corresponds only to itself, and
///   only when it is not an image.
///
/// [`is_corresponding`](Self::is_corresponding) decides whether two
/// identifiers correspond. Each frame and the free renaming must be
/// injective: a map sending two identifiers to one image would relate
/// `a + b` to `c + c` and make the relation one-directional. The default
/// renaming has no frame and maps nothing, so every identifier corresponds
/// only to itself.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expression::{AlphaRenaming, Expression};
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
///
/// Comparing the bodies of `\x. x + z` and `\y. y + z` under the binders'
/// frame:
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expression::{AlphaRenaming, Expression};
///
/// let (x, y, z) = (Identifier::new("x"), Identifier::new("y"), Identifier::new("z"));
/// let mut renaming = AlphaRenaming::default();
///
/// renaming
///     .enter_binder(HashMap::from([(x.clone(), y.clone())]))
///     .expect("one pair is injective");
/// let left = Expression::from(x) + Expression::from(z.clone());
/// let right = Expression::from(y.clone()) + Expression::from(z);
/// assert!(left.is_alpha_equivalent_under(&right, &renaming));
///
/// // Under the frame, `y` is bound on the right only, so no free `y` on the
/// // left corresponds to it: `\x. y` is not `\y. y`.
/// assert!(!renaming.is_corresponding(&y, &y));
/// assert!(renaming.leave_binder());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlphaRenaming {
    /// The binder frames, outermost first.
    frames: Vec<Bijection>,
    free_renaming: Bijection,
}

impl AlphaRenaming {
    /// Construct the renaming with no binder frame that sends each key of
    /// `free_renaming` to its value.
    ///
    /// # Errors
    ///
    /// Returns [`NonInjectiveRenamingError`] naming a shared image and
    /// [`RenamingPart::FreeRenaming`] if two identifiers map to the same
    /// image.
    pub fn try_new<S: BuildHasher>(
        free_renaming: HashMap<Identifier, Identifier, S>,
    ) -> Result<Self, NonInjectiveRenamingError> {
        Ok(Self {
            frames: Vec::new(),
            free_renaming: Bijection::try_new(free_renaming, RenamingPart::FreeRenaming)?,
        })
    }

    /// Push an innermost binder frame pairing each key of `bindings`, a
    /// bound identifier on this side, with its value, the bound identifier
    /// on the other side.
    ///
    /// Call it before comparing a binder's body and
    /// [`leave_binder`](Self::leave_binder) after. The frame shadows every
    /// outer frame and the free renaming. It may share images with outer
    /// frames, but a map cannot bind one identifier twice, so a caller
    /// pairing two parameter lists should refuse a list that repeats one.
    ///
    /// # Errors
    ///
    /// Returns [`NonInjectiveRenamingError`] naming a shared image and
    /// [`RenamingPart::BinderFrame`] if two identifiers map to the same
    /// image. The renaming is then unchanged.
    pub fn enter_binder<S: BuildHasher>(
        &mut self,
        bindings: HashMap<Identifier, Identifier, S>,
    ) -> Result<(), NonInjectiveRenamingError> {
        self.frames
            .push(Bijection::try_new(bindings, RenamingPart::BinderFrame)?);
        Ok(())
    }

    /// Pop the innermost binder frame, returning whether there was one.
    pub fn leave_binder(&mut self) -> bool {
        self.frames.pop().is_some()
    }

    /// Return the number of binder frames entered and not left.
    #[must_use]
    pub fn binder_depth(&self) -> usize {
        self.frames.len()
    }

    /// Return the identifier on the other side that `identifier` resolves
    /// to: its image in the innermost frame binding it, else its image in
    /// the free renaming, else itself.
    ///
    /// Resolution looks at this side only. Where the result is bound on the
    /// other side by a frame that does not bind `identifier`,
    /// [`is_corresponding`](Self::is_corresponding) still refuses the pair.
    #[must_use]
    pub fn resolve<'a>(&'a self, identifier: &'a Identifier) -> &'a Identifier {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.image_of(identifier))
            .or_else(|| self.free_renaming.image_of(identifier))
            .unwrap_or(identifier)
    }

    /// Return whether `left` on one side corresponds to `right` on the
    /// other.
    ///
    /// The innermost frame that binds `left` or has `right` as an image
    /// decides: the two correspond when that frame pairs them. So an
    /// identifier bound on one side only corresponds to nothing on the
    /// other, which keeps a binder from capturing a free identifier. When
    /// no frame decides, the free renaming does: `right` must be the image
    /// of a mapped `left`, or an unmapped `left` itself when it is not an
    /// image.
    #[must_use]
    pub fn is_corresponding(&self, left: &Identifier, right: &Identifier) -> bool {
        for frame in self.frames.iter().rev() {
            if let Some(image) = frame.image_of(left) {
                return image == right;
            }
            if frame.has_image(right) {
                return false;
            }
        }
        match self.free_renaming.image_of(left) {
            Some(image) => image == right,
            None => !self.free_renaming.has_image(right) && left == right,
        }
    }

    /// Return whether the renaming maps no identifier, in any frame or in
    /// the free renaming, so every identifier corresponds exactly to
    /// itself.
    ///
    /// A renaming holding only empty frames is empty, although it differs
    /// from the default renaming.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.free_renaming.is_empty() && self.frames.iter().all(Bijection::is_empty)
    }
}

/// An injective map of identifiers, with its set of images for the reverse
/// lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Bijection {
    images_by_identifier: HashMap<Identifier, Identifier>,
    images: HashSet<Identifier>,
}

impl Bijection {
    /// Construct the bijection of `map`, refusing a shared image as `part`.
    fn try_new<S: BuildHasher>(
        map: HashMap<Identifier, Identifier, S>,
        part: RenamingPart,
    ) -> Result<Self, NonInjectiveRenamingError> {
        let mut images = HashSet::with_capacity(map.len());
        for image in map.values() {
            if !images.insert(image.clone()) {
                return Err(NonInjectiveRenamingError::new(image.clone(), part));
            }
        }
        Ok(Self {
            images_by_identifier: map.into_iter().collect(),
            images,
        })
    }

    fn image_of(&self, identifier: &Identifier) -> Option<&Identifier> {
        self.images_by_identifier.get(identifier)
    }

    fn has_image(&self, identifier: &Identifier) -> bool {
        self.images.contains(identifier)
    }

    fn is_empty(&self) -> bool {
        self.images_by_identifier.is_empty()
    }
}
