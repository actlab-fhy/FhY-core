//! A registry of the validators that verify IR, by the kind of IR they
//! apply to.

use std::any::{TypeId, type_name};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

use super::validation::{ValidationManager, Validator, ValidatorRecord};
use crate::diagnostic::ValidationReport;
use crate::identifier::Identifier;

/// The name of the validation pipeline [`VerificationRegistry::verifier`]
/// builds.
const VERIFIER_NAME: &str = "verification";

/// The identity of a registration in a [`VerificationRegistry`].
///
/// A registration is named either by its validator's Rust type, through
/// [`of`](Self::of), or, for validators no Rust type tells apart, such as
/// ones a language binding defines at run time, by the address of an object
/// that stands for it, through [`of_ptr`](Self::of_ptr). Two ids are equal
/// exactly when they name the same type, or the same address; an id of a
/// type never equals an id of an address.
#[derive(Clone, Copy)]
pub struct VerifierId(VerifierName);

/// What a [`VerifierId`] names.
#[derive(Clone, Copy)]
enum VerifierName {
    /// A Rust validator type.
    Type {
        type_id: TypeId,
        type_name: &'static str,
    },
    /// The address of an object that stands for a validator.
    Address(usize),
}

impl VerifierId {
    /// Return the id of the validator type `V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::pass::VerifierId;
    ///
    /// struct BoundsCheck;
    /// struct TypeCheck;
    ///
    /// assert_eq!(VerifierId::of::<BoundsCheck>(), VerifierId::of::<BoundsCheck>());
    /// assert_ne!(VerifierId::of::<BoundsCheck>(), VerifierId::of::<TypeCheck>());
    /// ```
    #[must_use]
    pub fn of<V: ?Sized + 'static>() -> Self {
        Self(VerifierName::Type {
            type_id: TypeId::of::<V>(),
            type_name: type_name::<V>(),
        })
    }

    /// Return the id of the object `pointer` points to, for a validator no
    /// Rust type tells apart.
    ///
    /// The id is the pointer's address, without its metadata, and the
    /// pointer is never dereferenced. It is unique only while the object is
    /// alive, so whoever registers under it keeps the object alive for as
    /// long as the registration exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::pass::VerifierId;
    ///
    /// let first = Box::new(1);
    /// let second = Box::new(2);
    ///
    /// assert_eq!(VerifierId::of_ptr(&raw const *first), VerifierId::of_ptr(&raw const *first));
    /// assert_ne!(VerifierId::of_ptr(&raw const *first), VerifierId::of_ptr(&raw const *second));
    /// ```
    #[must_use]
    pub fn of_ptr<T: ?Sized>(pointer: *const T) -> Self {
        Self(VerifierName::Address(pointer.cast::<()>().addr()))
    }
}

impl PartialEq for VerifierId {
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (
                VerifierName::Type { type_id, .. },
                VerifierName::Type {
                    type_id: other_type_id,
                    ..
                },
            ) => type_id == other_type_id,
            (VerifierName::Address(address), VerifierName::Address(other_address)) => {
                address == other_address
            }
            _ => false,
        }
    }
}

impl Eq for VerifierId {}

impl Hash for VerifierId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(&self.0).hash(state);
        match self.0 {
            VerifierName::Type { type_id, .. } => type_id.hash(state),
            VerifierName::Address(address) => address.hash(state),
        }
    }
}

/// Render the validator type's name, or the address.
impl fmt::Debug for VerifierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            VerifierName::Type { type_name, .. } => write!(f, "VerifierId({type_name})"),
            VerifierName::Address(address) => write!(f, "VerifierId({address:#x})"),
        }
    }
}

/// Builds a new validator of `I`.
type ValidatorFactory<I> = Arc<dyn Fn() -> Box<dyn Validator<I> + Send> + Send + Sync>;

/// One registration under a kind: its identity and its factory.
struct Registration<I> {
    id: VerifierId,
    factory: ValidatorFactory<I>,
}

impl<I> Clone for Registration<I> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            factory: Arc::clone(&self.factory),
        }
    }
}

/// An owned registry of the validators that verify IR of type `I`, by the
/// kind `K` of IR each applies to.
///
/// A registration binds a validator factory to a kind under a
/// [`VerifierId`]. A lookup takes a *lineage*: the kinds an IR belongs to,
/// in the order their validators run, such as a general kind before a
/// specific one. It finds every registration under those kinds, in lineage
/// order and then registration order, each id once, at its first position,
/// and builds a new validator of each. [`verifier`](Self::verifier) turns a
/// lookup into a [`ValidationManager`] for a pipeline's
/// [`set_verifier`](super::PassManager::set_verifier), and
/// [`verify`](Self::verify) runs one.
///
/// # Examples
///
/// ```
/// use fhy_core::diagnostic::DiagnosticLevel;
/// use fhy_core::pass::{PassContext, Validator, VerificationRegistry};
/// use fhy_core::foreign::BoxError;
///
/// #[derive(PartialEq, Eq, Hash)]
/// enum Kind {
///     Any,
///     Index,
/// }
///
/// struct Finite;
/// struct NonNegative;
///
/// impl Validator<f64> for Finite {
///     fn validate(&mut self, ir: &f64, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
///         if !ir.is_finite() {
///             cx.report_text(DiagnosticLevel::Error, "not finite", None);
///         }
///         Ok(())
///     }
/// }
///
/// impl Validator<f64> for NonNegative {
///     fn validate(&mut self, ir: &f64, cx: &mut PassContext<'_>) -> Result<(), BoxError> {
///         if *ir < 0.0 {
///             cx.report_text(DiagnosticLevel::Error, "negative", None);
///         }
///         Ok(())
///     }
/// }
///
/// let mut registry = VerificationRegistry::new();
/// registry.register(Kind::Any, || Finite);
/// registry.register(Kind::Index, || NonNegative);
///
/// let report = registry.verify([&Kind::Any, &Kind::Index], &-1.0);
///
/// assert_eq!(report.errors().count(), 1);
/// assert_eq!(report.records()[0].validator_name(), "Finite");
/// assert_eq!(report.records()[1].validator_name(), "NonNegative");
/// ```
pub struct VerificationRegistry<K, I> {
    registrations: HashMap<K, Vec<Registration<I>>>,
}

impl<K, I> VerificationRegistry<K, I> {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registrations: HashMap::new(),
        }
    }

    /// Return the number of registrations, counting a validator registered
    /// under two kinds twice.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registrations.values().map(Vec::len).sum()
    }

    /// Return whether the registry has no registrations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }
}

impl<K: Eq + Hash, I: 'static> VerificationRegistry<K, I> {
    /// Register the validators `factory` builds for the kind `kind`, under
    /// the id of their type, [`VerifierId::of::<V>()`](VerifierId::of).
    ///
    /// Return whether the registration is new. Registering the same type
    /// under the same kind again changes nothing, and `factory` is dropped.
    pub fn register<V>(&mut self, kind: K, factory: impl Fn() -> V + Send + Sync + 'static) -> bool
    where
        V: Validator<I> + Send + 'static,
    {
        self.register_with_id(kind, VerifierId::of::<V>(), factory)
    }

    /// Register the validators `factory` builds for the kind `kind`, under
    /// the id `id`.
    ///
    /// Return whether the registration is new. Registering the same id under
    /// the same kind again changes nothing, and `factory` is dropped.
    pub fn register_with_id<V>(
        &mut self,
        kind: K,
        id: VerifierId,
        factory: impl Fn() -> V + Send + Sync + 'static,
    ) -> bool
    where
        V: Validator<I> + Send + 'static,
    {
        let registrations = self.registrations.entry(kind).or_default();
        if registrations
            .iter()
            .any(|registration| registration.id == id)
        {
            return false;
        }
        registrations.push(Registration {
            id,
            factory: Arc::new(move || Box::new(factory()) as Box<dyn Validator<I> + Send>),
        });
        true
    }

    /// Return the registrations found along `lineage`, each id once, at its
    /// first position.
    fn found<'k>(&self, lineage: impl IntoIterator<Item = &'k K>) -> Vec<&Registration<I>>
    where
        K: 'k,
    {
        let mut seen = HashSet::new();
        lineage
            .into_iter()
            .filter_map(|kind| self.registrations.get(kind))
            .flatten()
            .filter(|registration| seen.insert(registration.id))
            .collect()
    }

    /// Return the ids of the registrations found along `lineage`: those of
    /// each kind in lineage order, each in registration order, and each id
    /// once, at its first position. A kind without registrations adds none.
    pub fn ids_for<'k>(&self, lineage: impl IntoIterator<Item = &'k K>) -> Vec<VerifierId>
    where
        K: 'k,
    {
        self.found(lineage)
            .into_iter()
            .map(|registration| registration.id)
            .collect()
    }

    /// Return a new validator of each registration found along `lineage`, in
    /// the order of [`ids_for`](Self::ids_for).
    pub fn validators_for<'k>(
        &self,
        lineage: impl IntoIterator<Item = &'k K>,
    ) -> Vec<Box<dyn Validator<I> + Send>>
    where
        K: 'k,
    {
        self.found(lineage)
            .into_iter()
            .map(|registration| (registration.factory)())
            .collect()
    }

    /// Return the validation pipeline `verification` of the validators of
    /// [`validators_for(lineage)`](Self::validators_for).
    #[must_use]
    pub fn verifier<'k>(
        &self,
        lineage: impl IntoIterator<Item = &'k K>,
    ) -> ValidationManager<'static, I>
    where
        K: 'k,
    {
        let mut verifier = ValidationManager::new(Identifier::new(VERIFIER_NAME));
        for validator in self.validators_for(lineage) {
            verifier.add(validator);
        }
        verifier
    }

    /// Verify `ir` with the validators of
    /// [`validators_for(lineage)`](Self::validators_for), and return the
    /// report, with one record per validator.
    #[must_use]
    pub fn verify<'k>(
        &self,
        lineage: impl IntoIterator<Item = &'k K>,
        ir: &I,
    ) -> ValidationReport<ValidatorRecord>
    where
        K: 'k,
    {
        self.verifier(lineage).validate(ir)
    }
}

impl<K, I> Default for VerificationRegistry<K, I> {
    /// Create an empty registry.
    fn default() -> Self {
        Self::new()
    }
}

/// Clone the registrations; the clones share their factories.
impl<K: Clone, I> Clone for VerificationRegistry<K, I> {
    fn clone(&self) -> Self {
        Self {
            registrations: self.registrations.clone(),
        }
    }
}

/// Render the number of kinds and of registrations.
impl<K, I> fmt::Debug for VerificationRegistry<K, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerificationRegistry")
            .field("kinds", &self.registrations.len())
            .field("registrations", &self.len())
            .finish()
    }
}
