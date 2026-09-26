//! Errors of the renamings that alpha equivalence is checked under.
//!
//! [`NonInjectiveRenamingError`] refuses a free renaming or a binder frame
//! that is not injective, and [`BinderPairingError`] two binder lists that
//! cannot be paired into a frame.

use std::error::Error;
use std::fmt;

use crate::identifier::Identifier;

/// A free-identifier renaming or a binder frame that sends two identifiers
/// to one image.
///
/// Returned by [`AlphaRenaming::try_new`](super::AlphaRenaming::try_new) for
/// the free renaming and by
/// [`AlphaRenaming::enter_binder`](super::AlphaRenaming::enter_binder) for a
/// frame. Displays as `a free-identifier renaming must be injective, but
/// more than one identifier maps to {name}::{id}`, or `a binder frame must
/// be injective, but ...` for a frame, with the shared image's name hint and
/// id.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::term::{AlphaRenaming, RenamingPart};
/// use fhy_core::identifier::Identifier;
///
/// let (a, b, c) = (Identifier::new("a"), Identifier::new("b"), Identifier::new("c"));
/// let mut renaming = AlphaRenaming::default();
///
/// let error = renaming
///     .enter_binder(HashMap::from([(a, c.clone()), (b, c.clone())]))
///     .expect_err("a and b share the image c");
///
/// assert_eq!(error.image(), &c);
/// assert_eq!(error.part(), RenamingPart::BinderFrame);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NonInjectiveRenamingError {
    image: Identifier,
    part: RenamingPart,
}

impl NonInjectiveRenamingError {
    pub(super) fn new(image: Identifier, part: RenamingPart) -> Self {
        Self { image, part }
    }

    /// Return an image that more than one identifier maps to.
    #[must_use]
    pub fn image(&self) -> &Identifier {
        &self.image
    }

    /// Return the part of the renaming that is not injective.
    #[must_use]
    pub fn part(&self) -> RenamingPart {
        self.part
    }
}

impl fmt::Display for NonInjectiveRenamingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let part = match self.part {
            RenamingPart::FreeRenaming => "a free-identifier renaming",
            RenamingPart::BinderFrame => "a binder frame",
        };
        write!(
            f,
            "{part} must be injective, but more than one identifier maps to {}::{}",
            self.image.name_hint(),
            self.image.id()
        )
    }
}

/// The part of an [`AlphaRenaming`](super::AlphaRenaming) a
/// [`NonInjectiveRenamingError`] refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RenamingPart {
    /// The free-identifier renaming given to
    /// [`AlphaRenaming::try_new`](super::AlphaRenaming::try_new).
    FreeRenaming,
    /// A binder frame given to
    /// [`AlphaRenaming::enter_binder`](super::AlphaRenaming::enter_binder).
    BinderFrame,
}

impl Error for NonInjectiveRenamingError {}

/// Two lists of bound identifiers that
/// [`AlphaRenaming::enter_binders`](super::AlphaRenaming::enter_binders)
/// cannot pair into a binder frame.
///
/// Lists of different lengths bind different numbers of identifiers, and a
/// list that repeats an identifier binds it twice, so neither pairs with any
/// list. Displays as `binder lists of 1 and 2 identifiers cannot be
/// paired`, or `a binder list repeats the identifier {name}::{id}`.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::term::{AlphaRenaming, BinderPairingError};
///
/// let (x, a, b) = (Identifier::new("x"), Identifier::new("a"), Identifier::new("b"));
/// let mut renaming = AlphaRenaming::default();
///
/// let error = renaming
///     .enter_binders(&[x.clone(), x.clone()], &[a, b])
///     .expect_err("the left list repeats x");
///
/// assert_eq!(error, BinderPairingError::RepeatedIdentifier(x));
/// assert_eq!(renaming.binder_depth(), 0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinderPairingError {
    /// The two lists have different lengths.
    ArityMismatch {
        /// The length of the list on this side.
        left: usize,
        /// The length of the list on the other side.
        right: usize,
    },
    /// A list, on either side, holds this identifier more than once.
    RepeatedIdentifier(Identifier),
}

impl fmt::Display for BinderPairingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArityMismatch { left, right } => write!(
                f,
                "binder lists of {left} and {right} identifiers cannot be paired"
            ),
            Self::RepeatedIdentifier(identifier) => write!(
                f,
                "a binder list repeats the identifier {}::{}",
                identifier.name_hint(),
                identifier.id()
            ),
        }
    }
}

impl Error for BinderPairingError {}
