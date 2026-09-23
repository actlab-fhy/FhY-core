//! The compiler-pass trait, its guarded lifecycle, and the outcome of a run.

use std::error::Error;
use std::fmt;

use super::context::PassContext;
use super::error::{PassError, PassErrorClass, PassHook};
use super::preserved::PreservedAnalyses;
use super::registry;
use crate::diagnostic::{Diagnostic, DiagnosticLevel};

/// The error a pass hook returns.
///
/// Boxed so that [`CompilerPass`] stays object-safe and one pipeline can hold
/// passes whose own error types differ. A hook that returns a [`PassError`]
/// of its own class (validation for the validation hooks, execution for the
/// others, either for [`CompilerPass::run`]) hands it through unchanged; any
/// other error is wrapped in a [`PassError`] naming the pass and the hook.
pub type PassFailure = Box<dyn Error + Send + Sync + 'static>;

/// The error [`CompilerPass::noop_output`] returns by default.
#[derive(Debug)]
struct MissingNoopOutput;

impl fmt::Display for MissingNoopOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the pass has no no-op output")
    }
}

impl Error for MissingNoopOutput {}

/// A compiler pass from IR of type `I` to IR of type `O`.
///
/// A run goes through a fixed lifecycle, which [`ExecutePass::execute`] and
/// the [`PassManager`](super::PassManager) drive:
///
/// 1. [`validate_input`](Self::validate_input);
/// 2. [`should_run`](Self::should_run); when it returns `false`, the run
///    ends with [`noop_output`](Self::noop_output), unchanged, and the
///    analyses from [`preserved_analyses`](Self::preserved_analyses);
/// 3. [`run`](Self::run), counted in the run counters before it is called;
/// 4. [`validate_output`](Self::validate_output);
/// 5. [`did_change`](Self::did_change);
/// 6. [`preserved_analyses`](Self::preserved_analyses).
///
/// Every hook receives the run's [`PassContext`] for diagnostics and
/// analyses, and takes `&mut self`, so a pass may keep results in its own
/// fields for its caller to read after the run. The trait is object-safe.
///
/// # Examples
///
/// ```
/// use fhy_core::pass_infrastructure::{CompilerPass, ExecutePass, PassContext, PassFailure};
///
/// struct Increment;
///
/// impl CompilerPass<i64> for Increment {
///     fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
///         Ok(ir + 1)
///     }
///
///     fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
///         Ok(input != output)
///     }
/// }
///
/// let outcome = Increment.execute(&41)?;
///
/// assert_eq!(*outcome.output(), 42);
/// assert!(outcome.is_changed());
/// assert_eq!(Increment.name(), "Increment");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait CompilerPass<I, O = I> {
    /// Return the pass's name, the source of its diagnostics and the key of
    /// its run counter.
    ///
    /// By default: the name the pass's type was registered under with
    /// [`register_pass`](super::register_pass), else the type's name without
    /// its module path and generic arguments.
    fn name(&self) -> String {
        registry::find_default_pass_name::<Self>()
    }

    /// Return a human-readable description of the pass.
    ///
    /// By default: the description the pass's type was registered with, else
    /// [`name`](Self::name).
    fn description(&self) -> String {
        registry::find_registered_description::<Self>().unwrap_or_else(|| self.name())
    }

    /// Check the input before the pass runs.
    ///
    /// # Errors
    ///
    /// Returns an error to reject `ir`. By default, accepts every input.
    fn validate_input(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        let _ = (ir, cx);
        Ok(())
    }

    /// Return whether the pass runs on `ir`.
    ///
    /// # Errors
    ///
    /// Returns an error if the decision cannot be made. By default, the pass
    /// always runs.
    fn should_run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        let _ = (ir, cx);
        Ok(true)
    }

    /// Return the output of a run that [`should_run`](Self::should_run)
    /// skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if the pass has no output for a skipped run, which is
    /// the default.
    fn noop_output(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure> {
        let _ = (ir, cx);
        Err(Box::new(MissingNoopOutput))
    }

    /// Transform `ir`.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure>;

    /// Check the output after the pass ran.
    ///
    /// # Errors
    ///
    /// Returns an error to reject `output`. By default, accepts every output.
    fn validate_output(
        &mut self,
        input: &I,
        output: &O,
        cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        let _ = (input, output, cx);
        Ok(())
    }

    /// Return whether `output` differs from `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if the comparison fails.
    fn did_change(&mut self, input: &I, output: &O) -> Result<bool, PassFailure>;

    /// Return the analyses the run leaves valid for `output`.
    ///
    /// By default: none when the run `changed` the IR, all otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the set cannot be determined.
    fn preserved_analyses(
        &mut self,
        input: &I,
        output: &O,
        changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        let _ = (input, output);
        Ok(if changed {
            PreservedAnalyses::none()
        } else {
            PreservedAnalyses::all()
        })
    }
}

/// Forward every hook to the borrowed pass, so a pipeline can run a pass its
/// caller keeps and reads afterwards.
impl<I, O, P: CompilerPass<I, O> + ?Sized> CompilerPass<I, O> for &mut P {
    fn name(&self) -> String {
        (**self).name()
    }

    fn description(&self) -> String {
        (**self).description()
    }

    fn validate_input(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        (**self).validate_input(ir, cx)
    }

    fn should_run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        (**self).should_run(ir, cx)
    }

    fn noop_output(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure> {
        (**self).noop_output(ir, cx)
    }

    fn run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure> {
        (**self).run(ir, cx)
    }

    fn validate_output(
        &mut self,
        input: &I,
        output: &O,
        cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        (**self).validate_output(input, output, cx)
    }

    fn did_change(&mut self, input: &I, output: &O) -> Result<bool, PassFailure> {
        (**self).did_change(input, output)
    }

    fn preserved_analyses(
        &mut self,
        input: &I,
        output: &O,
        changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        (**self).preserved_analyses(input, output, changed)
    }
}

/// Forward every hook to the boxed pass.
impl<I, O, P: CompilerPass<I, O> + ?Sized> CompilerPass<I, O> for Box<P> {
    fn name(&self) -> String {
        (**self).name()
    }

    fn description(&self) -> String {
        (**self).description()
    }

    fn validate_input(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        (**self).validate_input(ir, cx)
    }

    fn should_run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        (**self).should_run(ir, cx)
    }

    fn noop_output(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure> {
        (**self).noop_output(ir, cx)
    }

    fn run(&mut self, ir: &I, cx: &mut PassContext<'_>) -> Result<O, PassFailure> {
        (**self).run(ir, cx)
    }

    fn validate_output(
        &mut self,
        input: &I,
        output: &O,
        cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        (**self).validate_output(input, output, cx)
    }

    fn did_change(&mut self, input: &I, output: &O) -> Result<bool, PassFailure> {
        (**self).did_change(input, output)
    }

    fn preserved_analyses(
        &mut self,
        input: &I,
        output: &O,
        changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        (**self).preserved_analyses(input, output, changed)
    }
}

/// The result of a pass run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassOutcome<O> {
    output: O,
    changed: bool,
    diagnostics: Vec<Diagnostic>,
    preserved: PreservedAnalyses,
}

impl<O> PassOutcome<O> {
    /// Create the outcome of a run that produced `output`.
    fn new(
        output: O,
        changed: bool,
        diagnostics: Vec<Diagnostic>,
        preserved: PreservedAnalyses,
    ) -> Self {
        Self {
            output,
            changed,
            diagnostics,
            preserved,
        }
    }

    /// Return the output IR.
    #[must_use]
    pub fn output(&self) -> &O {
        &self.output
    }

    /// Return the output IR, consuming the outcome.
    #[must_use]
    pub fn into_output(self) -> O {
        self.output
    }

    /// Return whether the run changed the IR.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        self.changed
    }

    /// Return the diagnostics the run emitted, in emission order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Return the analyses the run left valid for its output.
    #[must_use]
    pub fn preserved_analyses(&self) -> &PreservedAnalyses {
        &self.preserved
    }
}

/// The part of a run's result the lifecycle produces; the diagnostics stay
/// in the run's context.
#[derive(Debug)]
pub(super) struct LifecycleResult<O> {
    pub(super) output: O,
    pub(super) changed: bool,
    pub(super) preserved: PreservedAnalyses,
}

/// Return the class of the failure `hook` reports.
fn classify_hook(hook: PassHook) -> PassErrorClass {
    match hook {
        PassHook::ValidateInput | PassHook::ValidateOutput => PassErrorClass::Validation,
        PassHook::ShouldRun
        | PassHook::NoopOutput
        | PassHook::Run
        | PassHook::DidChange
        | PassHook::PreservedAnalyses => PassErrorClass::Execution,
    }
}

/// Return whether a [`PassError`] of `class` may leave `hook` unchanged:
/// [`CompilerPass::run`] hands both classes through, every other hook only
/// its own.
fn is_passed_through(hook: PassHook, class: PassErrorClass) -> bool {
    hook == PassHook::Run || classify_hook(hook) == class
}

/// Turn the error `failure` of `hook` into a [`PassError`], recording an
/// error diagnostic in `cx` unless `failure` is handed through.
fn wrap_hook_failure(failure: PassFailure, hook: PassHook, cx: &mut PassContext<'_>) -> PassError {
    let failure = match failure.downcast::<PassError>() {
        Ok(error) if is_passed_through(hook, error.class()) => return *error,
        Ok(error) => error as PassFailure,
        Err(other) => other,
    };
    let pass_name = cx.pass_name().to_owned();
    let message = format!("Pass \"{pass_name}\" failed {hook} with {failure}");
    cx.report_text(DiagnosticLevel::Error, message.clone(), None);
    PassError::new_hook_failure(
        classify_hook(hook),
        pass_name,
        hook,
        message,
        failure,
        cx.diagnostics().to_vec(),
    )
}

/// Return the value of a hook's `result`, or its error as a [`PassError`].
fn guard_hook<T>(
    result: Result<T, PassFailure>,
    hook: PassHook,
    cx: &mut PassContext<'_>,
) -> Result<T, PassError> {
    result.map_err(|failure| wrap_hook_failure(failure, hook, cx))
}

/// Run `pass` over `ir` through the guarded lifecycle, reporting into `cx`.
///
/// A hook error becomes a [`PassError`] after an error diagnostic records
/// it, unless it already is a [`PassError`] the hook may hand through.
pub(super) fn run_lifecycle<I, O, P>(
    pass: &mut P,
    ir: &I,
    cx: &mut PassContext<'_>,
) -> Result<LifecycleResult<O>, PassError>
where
    P: CompilerPass<I, O> + ?Sized,
{
    guard_hook(pass.validate_input(ir, cx), PassHook::ValidateInput, cx)?;
    if !guard_hook(pass.should_run(ir, cx), PassHook::ShouldRun, cx)? {
        let output = guard_hook(pass.noop_output(ir, cx), PassHook::NoopOutput, cx)?;
        let preserved = guard_hook(
            pass.preserved_analyses(ir, &output, false),
            PassHook::PreservedAnalyses,
            cx,
        )?;
        return Ok(LifecycleResult {
            output,
            changed: false,
            preserved,
        });
    }
    registry::record_run(cx.pass_name());
    let output = guard_hook(pass.run(ir, cx), PassHook::Run, cx)?;
    guard_hook(
        pass.validate_output(ir, &output, cx),
        PassHook::ValidateOutput,
        cx,
    )?;
    let changed = guard_hook(pass.did_change(ir, &output), PassHook::DidChange, cx)?;
    let preserved = guard_hook(
        pass.preserved_analyses(ir, &output, changed),
        PassHook::PreservedAnalyses,
        cx,
    )?;
    Ok(LifecycleResult {
        output,
        changed,
        preserved,
    })
}

/// Standalone execution of a pass through its guarded lifecycle.
///
/// Implemented for every [`CompilerPass`]. A standalone run computes each
/// requested analysis afresh.
pub trait ExecutePass<I, O>: CompilerPass<I, O> {
    /// Run the pass over `ir`.
    ///
    /// # Errors
    ///
    /// Returns a validation failure if [`CompilerPass::validate_input`] or
    /// [`CompilerPass::validate_output`] fails, and an execution failure if
    /// any other hook fails. A hook's own [`PassError`] of the matching class
    /// comes back unchanged; [`CompilerPass::run`] hands back a
    /// [`PassError`] of either class unchanged.
    fn execute(&mut self, ir: &I) -> Result<PassOutcome<O>, PassError>;
}

impl<I, O, P: CompilerPass<I, O> + ?Sized> ExecutePass<I, O> for P {
    fn execute(&mut self, ir: &I) -> Result<PassOutcome<O>, PassError> {
        let mut cx = PassContext::new(self.name(), None);
        let result = run_lifecycle(self, ir, &mut cx)?;
        let (_, diagnostics) = cx.into_parts();
        Ok(PassOutcome::new(
            result.output,
            result.changed,
            diagnostics,
            result.preserved,
        ))
    }
}
