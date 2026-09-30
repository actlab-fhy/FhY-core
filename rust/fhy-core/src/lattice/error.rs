//! Errors of building and asking a partial order.

use std::error::Error;
use std::fmt;

/// An element or an order a [`PartiallyOrderedSet`](super::PartiallyOrderedSet)
/// or a [`Lattice`](super::Lattice) refuses.
///
/// Displays one lowercase line naming the elements with their `Debug`
/// form, such as `3 is not a member of the partially ordered set` or
/// `ordering 2 below 1 would close a cycle`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OrderError<T> {
    /// The element is already a member.
    AlreadyAMember(T),
    /// The element is not a member.
    NotAMember(T),
    /// Ordering `lower` below `upper` would close a cycle: `upper` is
    /// already at most `lower`, which includes `lower == upper`.
    WouldCycle {
        /// The element that was to be ordered below.
        lower: T,
        /// The element that was to be ordered above.
        upper: T,
    },
}

impl<T: fmt::Debug> fmt::Display for OrderError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyAMember(element) => {
                write!(
                    f,
                    "{element:?} is already a member of the partially ordered set"
                )
            }
            Self::NotAMember(element) => {
                write!(
                    f,
                    "{element:?} is not a member of the partially ordered set"
                )
            }
            Self::WouldCycle { lower, upper } => {
                write!(f, "ordering {lower:?} below {upper:?} would close a cycle")
            }
        }
    }
}

impl<T: fmt::Debug> Error for OrderError<T> {}
