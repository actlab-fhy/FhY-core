//! A small lambda calculus over the term traits: variables, applications,
//! and lambdas that bind a list of parameters over a list of body terms.
//! `Lam` implements `Binder`, and `Lambda` calls its provided methods from
//! its `AlphaEquivalence`, `FreeIdentifiers` and `Term` implementations, as
//! a term type with a binder variant does.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::hash::BuildHasher;
use std::sync::Arc;

use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming, Binder, FreeIdentifiers, Term};

/// A term of the calculus. Equality is structural, with identifiers
/// compared by id.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Lambda {
    /// A reference to an identifier.
    Var(Identifier),
    /// The application of one term to another.
    App(Arc<Self>, Arc<Self>),
    /// A lambda.
    Lam(Lam),
}

/// A lambda binding `parameters` over the terms of its body, behind one
/// handle, so a substitution that changes nothing returns the same handle.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Lam(Arc<LamNode>);

#[derive(Debug, PartialEq)]
pub(crate) struct LamNode {
    parameters: Vec<Identifier>,
    body: Vec<Lambda>,
}

impl Lam {
    pub(crate) fn new(parameters: Vec<Identifier>, body: Vec<Lambda>) -> Self {
        Self(Arc::new(LamNode { parameters, body }))
    }

    /// Return whether `self` and `other` are the same handle.
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Return `names` as freshly created identifiers.
pub(crate) fn build_identifiers<const N: usize>(names: [&str; N]) -> [Identifier; N] {
    names.map(Identifier::new)
}

/// Return the variable `identifier`.
pub(crate) fn var(identifier: &Identifier) -> Lambda {
    Lambda::Var(identifier.clone())
}

/// Return the application of `function` to `argument`.
pub(crate) fn app(function: Lambda, argument: Lambda) -> Lambda {
    Lambda::App(Arc::new(function), Arc::new(argument))
}

/// Return the lambda binding `parameters` over the one term `body`.
pub(crate) fn lam<const N: usize>(parameters: [&Identifier; N], body: Lambda) -> Lambda {
    block(parameters, vec![body])
}

/// Return the lambda binding `parameters` over the terms `body`.
pub(crate) fn block<const N: usize>(parameters: [&Identifier; N], body: Vec<Lambda>) -> Lambda {
    Lambda::Lam(Lam::new(parameters.into_iter().cloned().collect(), body))
}

/// Return the lambda of `term`, which must be one.
pub(crate) fn expect_lam(term: &Lambda) -> &Lam {
    match term {
        Lambda::Lam(lam) => lam,
        other => panic!("expected a lambda, got {other:?}"),
    }
}

impl Lam {
    pub(crate) fn parameters(&self) -> &[Identifier] {
        &self.0.parameters
    }

    pub(crate) fn body(&self) -> &[Lambda] {
        &self.0.body
    }
}

impl Binder for Lam {
    type Child = Lambda;
    type RebuildError = Infallible;

    fn bound_identifiers(&self) -> &[Identifier] {
        &self.0.parameters
    }

    fn scoped_children(&self) -> &[Lambda] {
        &self.0.body
    }

    fn rename_bound_identifier(
        &self,
        old: &Identifier,
        new: Identifier,
    ) -> Result<Self, Infallible> {
        let parameters = self
            .0
            .parameters
            .iter()
            .map(|parameter| {
                if parameter == old {
                    new.clone()
                } else {
                    parameter.clone()
                }
            })
            .collect();
        let renamed = HashMap::from([(old.clone(), Lambda::Var(new))]);
        let body = self
            .0
            .body
            .iter()
            .map(|child| child.substitute(&renamed))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(parameters, body))
    }

    fn rebuild_with_scoped_children(&self, children: Vec<Lambda>) -> Result<Self, Infallible> {
        Ok(Self::new(self.0.parameters.clone(), children))
    }
}

impl AlphaEquivalence for Lambda {
    type Error = Infallible;

    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, Infallible> {
        match (self, other) {
            (Self::Var(left), Self::Var(right)) => Ok(renaming.is_corresponding(left, right)),
            (
                Self::App(left_function, left_argument),
                Self::App(right_function, right_argument),
            ) => Ok(
                left_function.is_alpha_equivalent_under(right_function, renaming)?
                    && left_argument.is_alpha_equivalent_under(right_argument, renaming)?,
            ),
            (Self::Lam(left), Self::Lam(right)) => {
                left.is_binder_alpha_equivalent_under(right, renaming)
            }
            _ => Ok(false),
        }
    }
}

impl FreeIdentifiers for Lambda {
    type Error = Infallible;

    fn free_identifiers(&self) -> Result<HashSet<Identifier>, Infallible> {
        match self {
            Self::Var(identifier) => Ok(HashSet::from([identifier.clone()])),
            Self::App(function, argument) => {
                let mut free = function.free_identifiers()?;
                free.extend(argument.free_identifiers()?);
                Ok(free)
            }
            Self::Lam(lam) => lam.binder_free_identifiers(),
        }
    }
}

/// The answers of [`AlphaEquivalence`] for a comparison that does not fail
/// in a test.
pub(crate) trait Alpha: AlphaEquivalence {
    /// Return whether `self` and `other` are alpha-equivalent.
    fn alpha_equivalent(&self, other: &Self) -> bool;

    /// Return whether `self` and `other` are alpha-equivalent under
    /// `renaming`.
    fn alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool;
}

impl<T: AlphaEquivalence> Alpha for T {
    fn alpha_equivalent(&self, other: &Self) -> bool {
        self.is_alpha_equivalent(other)
            .expect("the comparison does not fail")
    }

    fn alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        self.is_alpha_equivalent_under(other, renaming)
            .expect("the comparison does not fail")
    }
}

/// The answer of [`FreeIdentifiers`] for a term whose scope does not fail
/// in a test.
pub(crate) trait Free: FreeIdentifiers {
    /// Return the free identifiers.
    fn free(&self) -> HashSet<Identifier>;
}

impl<T: FreeIdentifiers> Free for T {
    fn free(&self) -> HashSet<Identifier> {
        self.free_identifiers().expect("the scope does not fail")
    }
}

impl Term for Lambda {
    type SubstituteError = Infallible;

    fn substitute<S: BuildHasher>(
        &self,
        replacements: &HashMap<Identifier, Self, S>,
    ) -> Result<Self, Infallible> {
        match self {
            Self::Var(identifier) => Ok(replacements
                .get(identifier)
                .cloned()
                .unwrap_or_else(|| self.clone())),
            Self::App(function, argument) => Ok(app(
                function.substitute(replacements)?,
                argument.substitute(replacements)?,
            )),
            Self::Lam(lam) => Ok(Self::Lam(lam.substitute_avoiding_capture(replacements)?)),
        }
    }
}
