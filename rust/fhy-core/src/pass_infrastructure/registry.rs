//! The process-wide pass registry and run counters.
//!
//! Registration binds a name and a description to a pass type together with
//! a factory that builds it, so [`create_pass`] can build a pass from its
//! name. The registered name becomes the type's default
//! [`CompilerPass::name`], which also keys its run counter. Every pass run
//! that is not skipped by [`CompilerPass::should_run`] counts, whether or not
//! the pass is registered.

use std::any::{Any, TypeId, type_name};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use super::error::PassRegistrationError;
use super::pass::CompilerPass;

/// Metadata of a registered pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassInfo {
    name: String,
    description: String,
    type_name: &'static str,
}

impl PassInfo {
    /// Return the name the pass is registered under.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the registered description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Return the full name of the registered pass type.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}

/// Builds a boxed pass from `I` to `O`.
type BoxedPassFactory<I, O> = Box<dyn Fn() -> Box<dyn CompilerPass<I, O>> + Send + Sync>;

/// A factory stored type-erased, recovered by downcasting to
/// [`BoxedPassFactory`] for the requested IR types.
type ErasedPassFactory = Arc<dyn Any + Send + Sync>;

/// One registration.
struct Registration {
    info: PassInfo,
    /// Identifies the pass type together with its input and output types.
    pass_type: (TypeId, TypeId, TypeId),
    input_type_name: &'static str,
    output_type_name: &'static str,
    factory: ErasedPassFactory,
}

/// The registrations and run counters of the process.
#[derive(Default)]
struct Registry {
    registrations: BTreeMap<String, Registration>,
    /// The latest name registered for each pass type, by type name.
    names_by_type: HashMap<&'static str, String>,
    run_counts: HashMap<String, u64>,
    total_run_count: u64,
}

/// The registry of the process.
static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(Mutex::default);

/// Return whether Python's `str.isspace` holds for `character`, which
/// also counts the four information separators U+001C to U+001F.
fn is_python_whitespace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// Return whether `text` is empty or only whitespace, as Python's
/// `not text.strip()` decides.
fn is_blank(text: &str) -> bool {
    text.chars().all(is_python_whitespace)
}

/// Return the last path segment of `type_name` without generic arguments,
/// for example `Fold` for `my_crate::passes::Fold<i64>`.
fn strip_type_path(type_name: &str) -> &str {
    let without_generics = type_name.split('<').next().unwrap_or(type_name);
    without_generics
        .rsplit("::")
        .next()
        .unwrap_or(without_generics)
}

/// Take the registry's lock.
///
/// No code runs under the lock that can panic between two updates of one
/// operation, so a poisoned lock holds a consistent registry and is
/// recovered.
fn lock_registry() -> MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Return the default name of the pass type `P`: its registered name, else
/// the last segment of its type name without generic arguments.
pub(super) fn find_default_pass_name<P: ?Sized>() -> String {
    let type_name = type_name::<P>();
    let registered = lock_registry().names_by_type.get(type_name).cloned();
    registered.unwrap_or_else(|| strip_type_path(type_name).to_owned())
}

/// Return the description registered for the pass type `P`, if it is
/// registered.
pub(super) fn find_registered_description<P: ?Sized>() -> Option<String> {
    let registry = lock_registry();
    let name = registry.names_by_type.get(type_name::<P>())?;
    registry
        .registrations
        .get(name)
        .map(|registration| registration.info.description.clone())
}

/// Count one run of the pass named `pass_name`.
pub(super) fn record_run(pass_name: &str) {
    let mut registry = lock_registry();
    registry.total_run_count += 1;
    if let Some(count) = registry.run_counts.get_mut(pass_name) {
        *count += 1;
    } else {
        registry.run_counts.insert(pass_name.to_owned(), 1);
    }
}

/// Register the pass type `P` from `I` to `O` under `name` with
/// `description`, built by `factory`.
///
/// Registering the same pass type under the same name and description again
/// changes nothing. After registration `name` is the type's default
/// [`CompilerPass::name`] and `description` its default
/// [`CompilerPass::description`]; a type registered under several names
/// takes the latest.
///
/// # Errors
///
/// Returns an error, and leaves the registry unchanged, if `name` or
/// `description` is empty or whitespace, if `name` is registered to a
/// different pass type, or if `name` is registered to this pass type with a
/// different description.
///
/// # Examples
///
/// ```
/// use fhy_core::pass_infrastructure::{
///     CompilerPass, PassContext, PassFailure, create_pass, register_pass,
/// };
///
/// struct Negate;
///
/// impl CompilerPass<i64> for Negate {
///     fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
///         Ok(-ir)
///     }
///
///     fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
///         Ok(input != output)
///     }
/// }
///
/// register_pass::<Negate, i64, i64>("docs.negate", "Negate an integer.", || Negate)?;
/// let pass = create_pass::<i64, i64>("docs.negate")?;
///
/// assert_eq!(pass.name(), "docs.negate");
/// assert_eq!(pass.description(), "Negate an integer.");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn register_pass<P, I, O>(
    name: &str,
    description: &str,
    factory: impl Fn() -> P + Send + Sync + 'static,
) -> Result<(), PassRegistrationError>
where
    P: CompilerPass<I, O> + 'static,
    I: 'static,
    O: 'static,
{
    if is_blank(name) {
        return Err(PassRegistrationError::new(
            "Pass name cannot be empty.".to_owned(),
        ));
    }
    if is_blank(description) {
        return Err(PassRegistrationError::new(
            "Pass description cannot be empty.".to_owned(),
        ));
    }
    let pass_type = (TypeId::of::<P>(), TypeId::of::<I>(), TypeId::of::<O>());
    let mut registry = lock_registry();
    if let Some(existing) = registry.registrations.get(name) {
        let info = &existing.info;
        let message = if existing.pass_type != pass_type {
            format!(
                "Pass name \"{name}\" is already registered by {} with description {:?}.",
                info.type_name, info.description
            )
        } else if info.description != description {
            format!(
                "Pass name \"{name}\" is already registered by {} with description {:?}; \
                 refusing to overwrite with new description {description:?}.",
                type_name::<P>(),
                info.description
            )
        } else {
            return Ok(());
        };
        return Err(PassRegistrationError::new(message));
    }
    let build: BoxedPassFactory<I, O> =
        Box::new(move || Box::new(factory()) as Box<dyn CompilerPass<I, O>>);
    let registration = Registration {
        info: PassInfo {
            name: name.to_owned(),
            description: description.to_owned(),
            type_name: type_name::<P>(),
        },
        pass_type,
        input_type_name: type_name::<I>(),
        output_type_name: type_name::<O>(),
        factory: Arc::new(build),
    };
    registry.registrations.insert(name.to_owned(), registration);
    registry
        .names_by_type
        .insert(type_name::<P>(), name.to_owned());
    Ok(())
}

/// Build a new instance of the pass registered under `name`.
///
/// # Errors
///
/// Returns an error if no pass is registered under `name`, or if the pass
/// registered under it does not go from `I` to `O`.
pub fn create_pass<I: 'static, O: 'static>(
    name: &str,
) -> Result<Box<dyn CompilerPass<I, O>>, PassRegistrationError> {
    let (factory, input_type_name, output_type_name) = {
        let registry = lock_registry();
        let Some(registration) = registry.registrations.get(name) else {
            return Err(PassRegistrationError::new(format!(
                "Unknown pass \"{name}\"."
            )));
        };
        (
            Arc::clone(&registration.factory),
            registration.input_type_name,
            registration.output_type_name,
        )
    };
    match factory.downcast_ref::<BoxedPassFactory<I, O>>() {
        Some(build) => Ok(build()),
        None => Err(PassRegistrationError::new(format!(
            "Pass \"{name}\" takes {input_type_name} to {output_type_name}, not {} to {}.",
            type_name::<I>(),
            type_name::<O>()
        ))),
    }
}

/// Return the metadata of every registered pass, by name.
#[must_use]
pub fn registered_passes() -> BTreeMap<String, PassInfo> {
    lock_registry()
        .registrations
        .iter()
        .map(|(name, registration)| (name.clone(), registration.info.clone()))
        .collect()
}

/// Return how many runs of the pass type `P` were counted, under its default
/// name.
///
/// A pass whose [`CompilerPass::name`] is overridden counts under that name;
/// read it with [`run_count_of`].
#[must_use]
pub fn run_count<P: ?Sized>() -> u64 {
    run_count_of(&find_default_pass_name::<P>())
}

/// Return how many runs of passes named `name` were counted.
#[must_use]
pub fn run_count_of(name: &str) -> u64 {
    lock_registry().run_counts.get(name).copied().unwrap_or(0)
}

/// Return how many pass runs were counted in the process.
#[must_use]
pub fn total_run_count() -> u64 {
    lock_registry().total_run_count
}
