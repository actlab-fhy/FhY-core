//! The equivalence walk over spaces and their parts: the frame pairing two
//! spaces' names, the correspondence of params, domains and constraint
//! systems under it, and the structural comparison with names compared by
//! `==`.
//!
//! An identifier member of a domain or of a set constraint is a reference:
//! it corresponds to the other side's member only as
//! [`AlphaRenaming::is_corresponding`](crate::term::AlphaRenaming::is_corresponding)
//! says, a categorical domain's and a set constraint's members as a
//! bijection. A constraint system's members pair up in any order, since its
//! canonical order follows identifiers' ids, which a renaming changes.
