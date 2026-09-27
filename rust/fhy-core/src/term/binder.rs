//! The traits of terms: alpha equivalence, free identifiers, substitution,
//! and the binders that introduce a scope.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::hash::BuildHasher;

use crate::identifier::Identifier;

use super::renaming::AlphaRenaming;

/// Comparison of two terms up to a consistent renaming of the identifiers
/// they bind, and of their free identifiers by an [`AlphaRenaming`].
///
/// Implementations are reflexive, symmetric and transitive on terms whose
/// binders bind each identifier once. A term that binds one identifier
/// twice in one binder list is alpha-equivalent to no term, itself included,
/// since [`AlphaRenaming::enter_binders`] pairs its list with none.
///
/// A comparison that runs code another implementation defines can fail,
/// with the term's [`Error`](Self::Error). A term whose comparison cannot
/// fail, such as an [`Expression`](crate::expression::Expression), uses
/// [`Infallible`](std::convert::Infallible), and its callers write
/// `let Ok(is_equivalent) = ...;`.
pub trait AlphaEquivalence {
    /// The error a comparison fails with.
    type Error: Error + Send + Sync + 'static;

    /// Return whether `self` and `other` are alpha-equivalent when their
    /// identifiers correspond by `renaming`.
    ///
    /// A binder compares its scoped children under `renaming` with one more
    /// frame pairing its bound identifiers with `other`'s; a reference asks
    /// [`AlphaRenaming::is_corresponding`]; any other term passes `renaming`
    /// on to its children unchanged.
    ///
    /// # Errors
    ///
    /// Returns the term's error when a comparison it runs fails.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, Self::Error>;

    /// Return whether `self` and `other` are alpha-equivalent with no binder
    /// in scope, so their free identifiers correspond only to themselves.
    ///
    /// # Errors
    ///
    /// Returns what [`is_alpha_equivalent_under`](Self::is_alpha_equivalent_under)
    /// returns.
    fn is_alpha_equivalent(&self, other: &Self) -> Result<bool, Self::Error> {
        self.is_alpha_equivalent_under(other, &AlphaRenaming::default())
    }
}

/// A term that reports the identifiers occurring free in it.
///
/// Reporting them can fail, with the term's [`Error`](Self::Error), when it
/// runs code another implementation defines; an
/// [`Expression`](crate::expression::Expression) uses
/// [`Infallible`](std::convert::Infallible).
pub trait FreeIdentifiers {
    /// The error reporting the identifiers fails with.
    type Error: Error + Send + Sync + 'static;

    /// Return the identifiers that occur in this term outside every binder
    /// of them.
    ///
    /// # Errors
    ///
    /// Returns the term's error when code it runs fails.
    fn free_identifiers(&self) -> Result<HashSet<Identifier>, Self::Error>;
}

/// A term a [`Binder`] scopes over: it compares by alpha equivalence,
/// reports its free identifiers, and substitutes terms for them.
pub trait Term: AlphaEquivalence + FreeIdentifiers + Clone {
    /// The error a substitution that would build an invalid term returns.
    type SubstituteError;

    /// Return this term with every free occurrence of a key of
    /// `replacements` replaced by its value, simultaneously and without
    /// capturing a free identifier of a value.
    ///
    /// # Errors
    ///
    /// Returns [`SubstituteError`](Self::SubstituteError) if the result
    /// would not be a valid term.
    fn substitute<S: BuildHasher>(
        &self,
        replacements: &HashMap<Identifier, Self, S>,
    ) -> Result<Self, Self::SubstituteError>;
}

/// A node that binds identifiers over its scoped children, such as a lambda
/// over its body or a function over its parameters.
///
/// An implementation gives the bound identifiers and the scoped children,
/// and two ways to rebuild the node. The provided methods derive alpha
/// equivalence, free identifiers and capture-avoiding substitution from
/// them; a term type with a binder variant calls them from its
/// [`AlphaEquivalence`], [`FreeIdentifiers`] and [`Term`] implementations.
///
/// # Examples
///
/// A lambda over one body, in a term language of variables and lambdas:
///
/// ```
/// use std::collections::{HashMap, HashSet};
/// use std::convert::Infallible;
/// use std::hash::BuildHasher;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::term::{AlphaEquivalence, AlphaRenaming, Binder, FreeIdentifiers, Term};
///
/// #[derive(Debug, Clone)]
/// enum Lambda {
///     Var(Identifier),
///     Lam(Lam),
/// }
///
/// #[derive(Debug, Clone)]
/// struct Lam {
///     parameters: Vec<Identifier>,
///     body: Vec<Lambda>,
/// }
///
/// impl Binder for Lam {
///     type Child = Lambda;
///     type RebuildError = Infallible;
///
///     fn bound_identifiers(&self) -> &[Identifier] {
///         &self.parameters
///     }
///
///     fn scoped_children(&self) -> &[Lambda] {
///         &self.body
///     }
///
///     fn rename_bound_identifier(&self, old: &Identifier, new: Identifier) -> Result<Self, Infallible> {
///         let parameters = self.parameters.iter()
///             .map(|parameter| if parameter == old { new.clone() } else { parameter.clone() })
///             .collect();
///         let renamed = HashMap::from([(old.clone(), Lambda::Var(new))]);
///         let body = self.body.iter().map(|child| child.substitute(&renamed)).collect::<Result<_, _>>()?;
///         Ok(Lam { parameters, body })
///     }
///
///     fn rebuild_with_scoped_children(&self, body: Vec<Lambda>) -> Result<Self, Infallible> {
///         Ok(Lam { parameters: self.parameters.clone(), body })
///     }
/// }
///
/// impl AlphaEquivalence for Lambda {
///     type Error = Infallible;
///
///     fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> Result<bool, Infallible> {
///         match (self, other) {
///             (Lambda::Var(left), Lambda::Var(right)) => Ok(renaming.is_corresponding(left, right)),
///             (Lambda::Lam(left), Lambda::Lam(right)) => left.is_binder_alpha_equivalent_under(right, renaming),
///             _ => Ok(false),
///         }
///     }
/// }
///
/// impl FreeIdentifiers for Lambda {
///     type Error = Infallible;
///
///     fn free_identifiers(&self) -> Result<HashSet<Identifier>, Infallible> {
///         match self {
///             Lambda::Var(identifier) => Ok(HashSet::from([identifier.clone()])),
///             Lambda::Lam(lam) => lam.binder_free_identifiers(),
///         }
///     }
/// }
///
/// impl Term for Lambda {
///     type SubstituteError = Infallible;
///
///     fn substitute<S: BuildHasher>(&self, replacements: &HashMap<Identifier, Self, S>) -> Result<Self, Infallible> {
///         match self {
///             Lambda::Var(identifier) => Ok(replacements.get(identifier).cloned().unwrap_or_else(|| self.clone())),
///             Lambda::Lam(lam) => Ok(Lambda::Lam(lam.substitute_avoiding_capture(replacements)?)),
///         }
///     }
/// }
///
/// let (x, y) = (Identifier::new("x"), Identifier::new("y"));
/// let identity_x = Lambda::Lam(Lam { parameters: vec![x.clone()], body: vec![Lambda::Var(x.clone())] });
/// let identity_y = Lambda::Lam(Lam { parameters: vec![y.clone()], body: vec![Lambda::Var(y.clone())] });
/// let Ok(is_equivalent) = identity_x.is_alpha_equivalent(&identity_y);
/// assert!(is_equivalent);
///
/// // Substituting `x` for `y` in `\x. y` renames the binder, so `x` stays free.
/// let constant = Lambda::Lam(Lam { parameters: vec![x.clone()], body: vec![Lambda::Var(y.clone())] });
/// let substituted = constant.substitute(&HashMap::from([(y, Lambda::Var(x.clone()))]))?;
/// assert_eq!(substituted.free_identifiers()?, HashSet::from([x]));
/// # Ok::<(), Infallible>(())
/// ```
pub trait Binder: Clone {
    /// The type of the scoped children.
    type Child: Term;
    /// The error rebuilding the node, substituting into its children,
    /// comparing them or reporting their free identifiers returns.
    type RebuildError: From<<Self::Child as Term>::SubstituteError>
        + From<<Self::Child as AlphaEquivalence>::Error>
        + From<<Self::Child as FreeIdentifiers>::Error>;

    /// Return the identifiers this node binds over its scoped children, in
    /// order.
    fn bound_identifiers(&self) -> &[Identifier];

    /// Return the children the bound identifiers are in scope for.
    fn scoped_children(&self) -> &[Self::Child];

    /// Return this node with its bound identifier `old` renamed to `new`,
    /// in the bound identifiers and in every scoped child.
    ///
    /// `new` is fresh: it occurs nowhere in the node.
    ///
    /// # Errors
    ///
    /// Returns [`RebuildError`](Self::RebuildError) if the node cannot be
    /// rebuilt.
    fn rename_bound_identifier(
        &self,
        old: &Identifier,
        new: Identifier,
    ) -> Result<Self, Self::RebuildError>;

    /// Return this node with its scoped children replaced by `children`,
    /// positionally, and its bound identifiers unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`RebuildError`](Self::RebuildError) if the node cannot be
    /// rebuilt from `children`.
    fn rebuild_with_scoped_children(
        &self,
        children: Vec<Self::Child>,
    ) -> Result<Self, Self::RebuildError>;

    /// Return whether `other` is this node up to renaming its bound
    /// identifiers, with free identifiers corresponding by `renaming`.
    ///
    /// The two must bind as many identifiers and have as many scoped
    /// children. Then each pair of children is compared under `renaming`
    /// with one more frame pairing the bound identifiers by position
    /// ([`AlphaRenaming::enter_binders`]); a pairing it refuses, such as a
    /// list that repeats an identifier, is not equivalent.
    ///
    /// # Errors
    ///
    /// Returns the first child comparison's error.
    fn is_binder_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, Self::RebuildError> {
        let (bound, other_bound) = (self.bound_identifiers(), other.bound_identifiers());
        if bound.len() != other_bound.len() {
            return Ok(false);
        }
        let (children, other_children) = (self.scoped_children(), other.scoped_children());
        if children.len() != other_children.len() {
            return Ok(false);
        }
        let mut extended = renaming.clone();
        if extended.enter_binders(bound, other_bound).is_err() {
            return Ok(false);
        }
        for (child, other_child) in children.iter().zip(other_children) {
            if !child.is_alpha_equivalent_under(other_child, &extended)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return the free identifiers of the scoped children, minus the bound
    /// identifiers.
    ///
    /// # Errors
    ///
    /// Returns the first child's error.
    fn binder_free_identifiers(&self) -> Result<HashSet<Identifier>, Self::RebuildError> {
        let mut free = HashSet::new();
        for child in self.scoped_children() {
            free.extend(child.free_identifiers()?);
        }
        for bound in self.bound_identifiers() {
            free.remove(bound);
        }
        Ok(free)
    }

    /// Return this node with every free occurrence of a key of
    /// `replacements` in its scoped children replaced by its value.
    ///
    /// Only a key free in this node applies: a key it binds is shadowed,
    /// and a key its children do not mention changes nothing. When no key
    /// applies, the result is a clone of `self`, and nothing is renamed.
    /// Otherwise each distinct bound identifier that is free in an applying
    /// value is first renamed, once, to a fresh identifier with the same
    /// name hint ([`rename_bound_identifier`](Self::rename_bound_identifier)),
    /// so the binder captures none of them; the identifiers renamed are
    /// read from the node as renamed so far, so the hook is only asked
    /// about identifiers the node binds. Then each scoped child is
    /// substituted, and the node is rebuilt from the results.
    ///
    /// # Errors
    ///
    /// Returns [`RebuildError`](Self::RebuildError) if renaming, a child's
    /// substitution, or rebuilding fails.
    fn substitute_avoiding_capture<S: BuildHasher>(
        &self,
        replacements: &HashMap<Identifier, Self::Child, S>,
    ) -> Result<Self, Self::RebuildError> {
        if replacements.is_empty() {
            return Ok(self.clone());
        }
        let bound: HashSet<&Identifier> = self.bound_identifiers().iter().collect();
        let unshadowed: Vec<&Identifier> = replacements
            .keys()
            .filter(|identifier| !bound.contains(identifier))
            .collect();
        if unshadowed.is_empty() {
            return Ok(self.clone());
        }
        let mut free = HashSet::new();
        for child in self.scoped_children() {
            free.extend(child.free_identifiers()?);
        }
        let active: HashMap<Identifier, Self::Child> = unshadowed
            .into_iter()
            .filter(|identifier| free.contains(*identifier))
            .map(|identifier| (identifier.clone(), replacements[identifier].clone()))
            .collect();
        if active.is_empty() {
            return Ok(self.clone());
        }
        let mut capturable = HashSet::new();
        for term in active.values() {
            capturable.extend(term.free_identifiers()?);
        }
        let mut safe = self.clone();
        let mut renamed = HashSet::new();
        let mut position = 0;
        while let Some(bound_identifier) = safe.bound_identifiers().get(position) {
            position += 1;
            if !capturable.contains(bound_identifier) || renamed.contains(bound_identifier) {
                continue;
            }
            let bound_identifier = bound_identifier.clone();
            let fresh = Identifier::new(bound_identifier.name_hint());
            safe = safe.rename_bound_identifier(&bound_identifier, fresh)?;
            renamed.insert(bound_identifier);
        }
        let children = safe
            .scoped_children()
            .iter()
            .map(|child| child.substitute(&active))
            .collect::<Result<Vec<_>, _>>()?;
        safe.rebuild_with_scoped_children(children)
    }
}
