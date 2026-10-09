//! Partial orders and lattices over any hashable elements.
//!
//! - [`PartiallyOrderedSet`] holds elements and the order relations added
//!   between them. Its order is reflexive: every element is at most itself.
//!   It keeps each element's up-set, so asking whether one element is at
//!   most another is a bit test, and it iterates its elements in a
//!   topological order in which insertion order breaks ties.
//! - [`Lattice`] asks a partially ordered set for greatest lower bounds
//!   (meets) and least upper bounds (joins), and reports the pairs that
//!   lack one.
//!
//! # Examples
//!
//! ```
//! use fhy_core::lattice::Lattice;
//!
//! let mut lattice = Lattice::new();
//! for element in ["bottom", "left", "right", "top"] {
//!     lattice.add_element(element)?;
//! }
//! lattice.add_order(&"bottom", &"left")?;
//! lattice.add_order(&"bottom", &"right")?;
//! lattice.add_order(&"left", &"top")?;
//! lattice.add_order(&"right", &"top")?;
//!
//! assert_eq!(lattice.join(&"left", &"right")?, Some(&"top"));
//! assert_eq!(lattice.meet(&"left", &"right")?, Some(&"bottom"));
//! assert!(lattice.is_lattice());
//! # Ok::<(), fhy_core::lattice::OrderError<&str>>(())
//! ```

mod bits;
mod bounds;
mod error;
mod poset;

pub use bounds::{Lattice, MissingBound};
pub use error::OrderError;
pub use poset::PartiallyOrderedSet;
