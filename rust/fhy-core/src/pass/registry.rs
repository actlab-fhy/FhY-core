//! An owned registry of pass factories, looked up by pass name.

use std::any::{Any, TypeId, type_name};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use super::compiler_pass::CompilerPass;

/// Metadata of a registered pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassInfo {
    name: Cow<'static, str>,
    description: Cow<'static, str>,
    pass_type_id: TypeId,
    input_type_id: TypeId,
    output_type_id: TypeId,
    pass_type_name: &'static str,
    input_type_name: &'static str,
    output_type_name: &'static str,
}

impl PassInfo {
    /// Return the name the pass is registered under: its
    /// [`CompilerPass::name`].
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the registered description: the pass's
    /// [`CompilerPass::description`].
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Return the type id of the registered pass type.
    #[must_use]
    pub fn pass_type_id(&self) -> TypeId {
        self.pass_type_id
    }

    /// Return the type id of the IR the registered pass takes.
    #[must_use]
    pub fn input_type_id(&self) -> TypeId {
        self.input_type_id
    }

    /// Return the type id of the IR the registered pass produces.
    #[must_use]
    pub fn output_type_id(&self) -> TypeId {
        self.output_type_id
    }

    /// Return the full name of the registered pass type, as
    /// [`std::any::type_name`] writes it.
    ///
    /// The text is for messages only: `type_name` does not guarantee it
    /// stays the same across compiler versions.
    #[must_use]
    pub fn pass_type_name(&self) -> &'static str {
        self.pass_type_name
    }

    /// Return whether this registration is of the pass type, input type and
    /// output type `other` is.
    fn has_identity_of(&self, other: &Self) -> bool {
        (self.pass_type_id, self.input_type_id, self.output_type_id)
            == (
                other.pass_type_id,
                other.input_type_id,
                other.output_type_id,
            )
    }
}

/// Builds a boxed pass from `I` to `O`.
type BoxedPassFactory<I, O> = Box<dyn Fn() -> Box<dyn CompilerPass<I, O> + Send> + Send + Sync>;

/// One registration: its metadata and its factory, stored type-erased and
/// recovered by downcasting to [`BoxedPassFactory`] for the requested IR
/// types.
struct Registration {
    info: PassInfo,
    factory: Box<dyn Any + Send + Sync>,
}

/// Return whether `text` is empty or only whitespace, as
/// [`char::is_whitespace`] decides.
fn is_blank(text: &str) -> bool {
    text.chars().all(char::is_whitespace)
}

/// An owned registry of pass factories, looked up by name.
///
/// Registering a pass type binds its factory to the pass's own
/// [`CompilerPass::name`] and [`CompilerPass::description`], read from one
/// instance the factory builds, so [`create`](Self::create) can build a pass
/// from its name.
///
/// # Examples
///
/// ```
/// use fhy_core::pass::{CompilerPass, PassContext, PassFailure, PassRegistry};
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
/// let mut registry = PassRegistry::new();
/// registry.register::<Negate, i64, i64>(|| Negate)?;
/// let pass = registry.create::<i64, i64>("Negate")?;
///
/// assert_eq!(pass.name(), "Negate");
/// assert_eq!(registry.len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Default)]
pub struct PassRegistry {
    registrations: BTreeMap<Cow<'static, str>, Registration>,
}

impl PassRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the pass type `P` from `I` to `O`, built by `factory`.
    ///
    /// The factory is called once here, and the instance it builds names the
    /// registration: its [`CompilerPass::name`] is the key and its
    /// [`CompilerPass::description`] the description. The factory should
    /// build instances that share that name and description. A
    /// registration's identity is its pass type together with `I` and `O`.
    /// Registering the same identity under the same name and description
    /// again changes nothing, and the new factory is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error, and leaves the registry unchanged:
    ///
    /// - [`PassRegistrationError::EmptyName`] if the name is empty or only
    ///   whitespace;
    /// - [`PassRegistrationError::EmptyDescription`] if the description is;
    /// - [`PassRegistrationError::NameTaken`] if the name is registered to a
    ///   different identity, such as another pass type of the same name or
    ///   the same pass over other IR types;
    /// - [`PassRegistrationError::DescriptionConflict`] if the name is
    ///   registered to this identity with a different description.
    pub fn register<P, I, O>(
        &mut self,
        factory: impl Fn() -> P + Send + Sync + 'static,
    ) -> Result<(), PassRegistrationError>
    where
        P: CompilerPass<I, O> + Send + 'static,
        I: 'static,
        O: 'static,
    {
        let (name, description) = {
            let pass = factory();
            (pass.name(), pass.description())
        };
        if is_blank(&name) {
            return Err(PassRegistrationError::EmptyName);
        }
        if is_blank(&description) {
            return Err(PassRegistrationError::EmptyDescription { name });
        }
        let info = PassInfo {
            name,
            description,
            pass_type_id: TypeId::of::<P>(),
            input_type_id: TypeId::of::<I>(),
            output_type_id: TypeId::of::<O>(),
            pass_type_name: type_name::<P>(),
            input_type_name: type_name::<I>(),
            output_type_name: type_name::<O>(),
        };
        if let Some(existing) = self.registrations.get(info.name()) {
            let existing = &existing.info;
            if !existing.has_identity_of(&info) {
                return Err(PassRegistrationError::NameTaken {
                    name: info.name,
                    registered_pass_type_name: existing.pass_type_name,
                });
            }
            if existing.description != info.description {
                return Err(PassRegistrationError::DescriptionConflict {
                    name: info.name,
                    registered: existing.description.clone(),
                    requested: info.description,
                });
            }
            return Ok(());
        }
        let build: BoxedPassFactory<I, O> =
            Box::new(move || Box::new(factory()) as Box<dyn CompilerPass<I, O> + Send>);
        self.registrations.insert(
            info.name.clone(),
            Registration {
                info,
                factory: Box::new(build),
            },
        );
        Ok(())
    }

    /// Build a new instance of the pass registered under `name`.
    ///
    /// # Errors
    ///
    /// Returns [`CreatePassError::UnknownPass`] if no pass is registered
    /// under `name`, and [`CreatePassError::IrTypeMismatch`] if the pass
    /// registered under it does not go from `I` to `O`.
    pub fn create<I: 'static, O: 'static>(
        &self,
        name: &str,
    ) -> Result<Box<dyn CompilerPass<I, O> + Send>, CreatePassError> {
        let Some(registration) = self.registrations.get(name) else {
            return Err(CreatePassError::UnknownPass {
                name: name.to_owned(),
            });
        };
        let build = registration
            .factory
            .downcast_ref::<BoxedPassFactory<I, O>>()
            .ok_or_else(|| CreatePassError::IrTypeMismatch {
                name: name.to_owned(),
                registered_input: registration.info.input_type_name,
                registered_output: registration.info.output_type_name,
                requested_input: type_name::<I>(),
                requested_output: type_name::<O>(),
            })?;
        Ok(build())
    }

    /// Return the metadata of the pass registered under `name`, if any.
    #[must_use]
    pub fn info(&self, name: &str) -> Option<&PassInfo> {
        self.registrations
            .get(name)
            .map(|registration| &registration.info)
    }

    /// Return the metadata of every registration, ordered by name.
    pub fn iter(&self) -> impl Iterator<Item = &PassInfo> + '_ {
        self.registrations
            .values()
            .map(|registration| &registration.info)
    }

    /// Return the number of registrations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registrations.len()
    }

    /// Return whether nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registrations.is_empty()
    }
}

/// Render the metadata of every registration, ordered by name.
impl fmt::Debug for PassRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PassRegistry>();
};

/// A pass registration that [`PassRegistry::register`] refused.
///
/// More variants may be added, so a `match` on one needs a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PassRegistrationError {
    /// The pass's name is empty or only whitespace, as
    /// [`char::is_whitespace`] decides.
    EmptyName,
    /// The pass's description is empty or only whitespace.
    #[non_exhaustive]
    EmptyDescription {
        /// The pass's name.
        name: Cow<'static, str>,
    },
    /// The name is registered to a different pass type or to the same pass
    /// type over other IR types.
    #[non_exhaustive]
    NameTaken {
        /// The name.
        name: Cow<'static, str>,
        /// The full name of the pass type registered under it, for messages
        /// only.
        registered_pass_type_name: &'static str,
    },
    /// The name is registered to this pass type with a different
    /// description.
    #[non_exhaustive]
    DescriptionConflict {
        /// The name.
        name: Cow<'static, str>,
        /// The description it is registered with.
        registered: Cow<'static, str>,
        /// The description the refused registration has.
        requested: Cow<'static, str>,
    },
}

/// Render the refusal in one line, for example
/// `pass name "fold" is already registered to another pass`.
impl fmt::Display for PassRegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => f.write_str("pass name is blank"),
            Self::EmptyDescription { name } => {
                write!(f, "pass {name:?} has a blank description")
            }
            Self::NameTaken { name, .. } => {
                write!(
                    f,
                    "pass name {name:?} is already registered to another pass"
                )
            }
            Self::DescriptionConflict { name, .. } => write!(
                f,
                "pass {name:?} is already registered with a different description"
            ),
        }
    }
}

impl Error for PassRegistrationError {}

/// A pass that [`PassRegistry::create`] cannot build.
///
/// More variants may be added, so a `match` on one needs a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CreatePassError {
    /// No pass is registered under the name.
    #[non_exhaustive]
    UnknownPass {
        /// The name.
        name: String,
    },
    /// The pass registered under the name goes between other IR types than
    /// the requested ones.
    ///
    /// The type names are [`std::any::type_name`] output, for messages only.
    #[non_exhaustive]
    IrTypeMismatch {
        /// The name.
        name: String,
        /// The IR type the registered pass takes.
        registered_input: &'static str,
        /// The IR type the registered pass produces.
        registered_output: &'static str,
        /// The requested input IR type.
        requested_input: &'static str,
        /// The requested output IR type.
        requested_output: &'static str,
    },
}

/// Render the failure in one line, for example
/// `no pass is registered as "fold"`.
impl fmt::Display for CreatePassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPass { name } => write!(f, "no pass is registered as {name:?}"),
            Self::IrTypeMismatch {
                name,
                registered_input,
                registered_output,
                requested_input,
                requested_output,
            } => write!(
                f,
                "pass {name:?} takes {registered_input} to {registered_output}, \
                 not {requested_input} to {requested_output}"
            ),
        }
    }
}

impl Error for CreatePassError {}
