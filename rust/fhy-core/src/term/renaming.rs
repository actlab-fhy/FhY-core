//! The renaming that alpha equivalence is checked under: a stack of binder
//! frames over a free-identifier renaming.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

use crate::identifier::Identifier;

use super::error::{BinderPairingError, NonInjectiveRenamingError, RenamingPart};

/// The correspondence between identifiers on two sides of a comparison that
/// [`AlphaEquivalence::is_alpha_equivalent_under`](super::AlphaEquivalence::is_alpha_equivalent_under)
/// checks two terms against: a stack of binder frames over a free-identifier
/// renaming.
///
/// - **Binder frames.** A term that binds identifiers, such as a parameter
///   list over a body, compares its body with the other term's under one
///   more frame, pairing each of its bound identifiers with the other term's
///   ([`enter_binders`](Self::enter_binders) or
///   [`enter_binder`](Self::enter_binder)), and drops the frame after
///   ([`leave_binder`](Self::leave_binder)); [`extended`](Self::extended)
///   returns a renaming with one more frame instead. An inner frame shadows
///   outer ones, and two frames may share an image, as two nested binders
///   may bind one name.
/// - **The free renaming** ([`new`](Self::new)) pairs identifiers
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
/// The frames and the free renaming are shared between clones, so cloning a
/// renaming, or extending one by a frame, copies no map. Two renamings are
/// equal when they have equal free renamings and equal frames in the same
/// order, an empty frame included, and [`Hash`] agrees with that equality.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::expression::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::term::AlphaRenaming;
///
/// let (a, b, c) = (Identifier::new("a"), Identifier::new("b"), Identifier::new("c"));
/// let renaming = AlphaRenaming::new(HashMap::from([(a.clone(), c.clone())]))
///     .expect("one pair is injective");
/// let sum = Expression::from(a.clone()) + 1;
/// assert!(sum.is_alpha_equivalent_under(&(Expression::from(c.clone()) + 1), &renaming));
///
/// let colliding = HashMap::from([(a, c.clone()), (b, c)]);
/// assert!(AlphaRenaming::new(colliding).is_err());
/// ```
///
/// Comparing the bodies of `\x. x + z` and `\y. y + z` under the binders'
/// frame:
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::term::AlphaRenaming;
///
/// let (x, y, z) = (Identifier::new("x"), Identifier::new("y"), Identifier::new("z"));
/// let mut renaming = AlphaRenaming::default();
///
/// renaming
///     .enter_binders(&[x.clone()], &[y.clone()])
///     .expect("one identifier on each side pairs");
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
    frames: Vec<Arc<Bijection>>,
    free_renaming: Arc<Bijection>,
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
    pub fn new<S: BuildHasher>(
        free_renaming: HashMap<Identifier, Identifier, S>,
    ) -> Result<Self, NonInjectiveRenamingError> {
        Ok(Self {
            frames: Vec::new(),
            free_renaming: Arc::new(Bijection::try_new(
                free_renaming,
                RenamingPart::FreeRenaming,
            )?),
        })
    }

    /// Push an innermost binder frame pairing each key of `bindings`, a
    /// bound identifier on this side, with its value, the bound identifier
    /// on the other side.
    ///
    /// Call it before comparing a binder's body and
    /// [`leave_binder`](Self::leave_binder) after. The frame shadows every
    /// outer frame and the free renaming. It may share images with outer
    /// frames. A map cannot bind one identifier twice, so a caller pairing
    /// two lists of bound identifiers uses
    /// [`enter_binders`](Self::enter_binders), which refuses a list that
    /// repeats one.
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
        self.frames.push(Arc::new(Bijection::try_new(
            bindings,
            RenamingPart::BinderFrame,
        )?));
        Ok(())
    }

    /// Push an innermost binder frame pairing `left[i]`, the `i`-th
    /// identifier a binder on this side binds, with `right[i]`, the one the
    /// binder on the other side binds at the same position.
    ///
    /// Two empty lists push an empty frame. The frame is otherwise the one
    /// [`enter_binder`](Self::enter_binder) pushes for the pairs.
    ///
    /// # Errors
    ///
    /// Returns [`BinderPairingError::ArityMismatch`] if the lists have
    /// different lengths, and [`BinderPairingError::RepeatedIdentifier`] if
    /// either list holds an identifier more than once, checked on the left
    /// first. So a binder that binds one identifier twice pairs with no
    /// binder, itself included. The renaming is then unchanged.
    pub fn enter_binders(
        &mut self,
        left: &[Identifier],
        right: &[Identifier],
    ) -> Result<(), BinderPairingError> {
        if left.len() != right.len() {
            return Err(BinderPairingError::ArityMismatch {
                left: left.len(),
                right: right.len(),
            });
        }
        let mut keys = HashSet::with_capacity(left.len());
        for identifier in left {
            if !keys.insert(identifier) {
                return Err(BinderPairingError::RepeatedIdentifier(identifier.clone()));
            }
        }
        let mut images = HashSet::with_capacity(right.len());
        for identifier in right {
            if !images.insert(identifier.clone()) {
                return Err(BinderPairingError::RepeatedIdentifier(identifier.clone()));
            }
        }
        let images_by_identifier = left.iter().cloned().zip(right.iter().cloned()).collect();
        self.frames.push(Arc::new(Bijection {
            images_by_identifier,
            images,
        }));
        Ok(())
    }

    /// Return this renaming with one more innermost binder frame, the one
    /// [`enter_binder`](Self::enter_binder) would push for `bindings`,
    /// leaving `self` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`NonInjectiveRenamingError`] as
    /// [`enter_binder`](Self::enter_binder) does.
    pub fn extended<S: BuildHasher>(
        &self,
        bindings: HashMap<Identifier, Identifier, S>,
    ) -> Result<Self, NonInjectiveRenamingError> {
        let mut extended = self.clone();
        extended.enter_binder(bindings)?;
        Ok(extended)
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
        self.free_renaming.is_empty() && self.frames.iter().all(|frame| frame.is_empty())
    }

    /// Return a view of the free renaming.
    #[must_use]
    pub fn free_renaming(&self) -> RenamingMap<'_> {
        RenamingMap {
            bijection: &self.free_renaming,
        }
    }

    /// Return a view of each binder frame, outermost first.
    #[must_use]
    pub fn frames(&self) -> impl ExactSizeIterator<Item = RenamingMap<'_>> + DoubleEndedIterator {
        self.frames
            .iter()
            .map(|frame| RenamingMap { bijection: frame })
    }
}

/// Hashes the free renaming, then each frame in order, each map by its pairs
/// in the order of their left identifiers' ids, so equal renamings hash
/// alike whatever order their maps were built in.
impl Hash for AlphaRenaming {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.free_renaming.hash_pairs(state);
        state.write_usize(self.frames.len());
        for frame in &self.frames {
            frame.hash_pairs(state);
        }
    }
}

/// A read-only view of one injective map of an [`AlphaRenaming`]: a binder
/// frame or the free renaming.
#[derive(Debug, Clone, Copy)]
pub struct RenamingMap<'a> {
    bijection: &'a Bijection,
}

impl<'a> RenamingMap<'a> {
    /// Return the image of `identifier`, if the map maps it.
    #[must_use]
    pub fn get(&self, identifier: &Identifier) -> Option<&'a Identifier> {
        self.bijection.image_of(identifier)
    }

    /// Return the pairs of the map, each identifier with its image, in no
    /// particular order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&'a Identifier, &'a Identifier)> + 'a {
        self.bijection.images_by_identifier.iter()
    }

    /// Return the number of identifiers the map maps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bijection.images_by_identifier.len()
    }

    /// Return whether the map maps no identifier.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bijection.is_empty()
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

    /// Feed the number of pairs, then the pairs in the order of their keys'
    /// ids, to `state`, so the hash does not depend on the map's order.
    fn hash_pairs<H: Hasher>(&self, state: &mut H) {
        let mut pairs: Vec<(&Identifier, &Identifier)> = self.images_by_identifier.iter().collect();
        pairs.sort_unstable_by_key(|(key, _)| key.id());
        state.write_usize(pairs.len());
        for (key, image) in pairs {
            key.hash(state);
            image.hash(state);
        }
    }
}
