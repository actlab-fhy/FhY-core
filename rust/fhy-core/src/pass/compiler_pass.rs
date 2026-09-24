//! The compiler-pass trait, its guarded lifecycle, and the outcome of a run.

use std::any::type_name;
use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use super::context::PassContext;
use super::error::{PassError, PassErrorClass, PassHook};
use super::preserved::PreservedAnalyses;
use crate::diagnostic::{Diagnostic, DiagnosticLevel};

/// The error a pass hook returns.
///
/// Boxed so that [`CompilerPass`] stays object-safe and one pipeline can hold
/// passes whose own error types differ. A hook that returns a [`PassError`]
/// of its own class (validation for the validation hooks, execution for the
/// others, either for [`CompilerPass::run`]) hands it through unchanged; any
/// other error is wrapped in a [`PassError`] naming the pass and the hook.
pub type PassFailure = Box<dyn Error + Send + Sync + 'static>;

/// Return whether `character` may appear in an identifier or a number.
fn is_identifier_char(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// Return the segment of `path` a short name keeps: the last `::` segment,
/// or the last two when the last is a `{{...}}` segment such as
/// `{{closure}}`.
fn keep_last_segment(path: &str) -> &str {
    let mut segments = path.rmatch_indices("::");
    let Some((last_separator, _)) = segments.next() else {
        return path;
    };
    let last = &path[last_separator + 2..];
    if !last.starts_with("{{") {
        return last;
    }
    match segments.next() {
        Some((separator, _)) => &path[separator + 2..],
        None => path,
    }
}

/// Return whether `name` is a nominal type: a path of identifier, `::` and
/// `{{...}}` pieces, followed by at most one generic argument list that ends
/// the name.
fn is_nominal(name: &str) -> bool {
    let (path, arguments) = name.split_at(name.find('<').unwrap_or(name.len()));
    let is_path = !path.is_empty()
        && !path.starts_with(':')
        && path
            .chars()
            .all(|character| is_identifier_char(character) || "{}:".contains(character));
    if !is_path {
        return false;
    }
    let mut depth = 0_usize;
    let mut previous = ' ';
    for (position, character) in arguments.char_indices() {
        match character {
            '<' => depth += 1,
            '>' if previous != '-' => {
                depth = depth.saturating_sub(1);
                if depth == 0 && position + 1 != arguments.len() {
                    return false;
                }
            }
            _ => {}
        }
        previous = character;
    }
    depth == 0
}

/// Return the characters of `name` outside every generic argument list,
/// with their byte positions. A `>` that is part of `->` closes no list, and
/// an unbalanced `<` drops the rest of the name.
fn strip_generic_arguments(name: &str) -> Vec<(usize, char)> {
    let mut kept = Vec::with_capacity(name.len());
    let mut depth = 0_usize;
    let mut previous = ' ';
    for (position, character) in name.char_indices() {
        match character {
            '<' => depth += 1,
            '>' if previous != '-' && depth > 0 => depth -= 1,
            _ if depth == 0 => kept.push((position, character)),
            _ => {}
        }
        previous = character;
    }
    kept
}

/// Return the length of the path starting at `kept[start]`: identifier,
/// `{{...}}` and `::` pieces, or 0 if no path starts there.
fn measure_path(kept: &[(usize, char)], start: usize) -> usize {
    let at = |index: usize| kept.get(index).map(|&(_, character)| character);
    let mut end = start;
    loop {
        if at(end) == Some('{') && at(end + 1) == Some('{') {
            end += 2;
            while end < kept.len() && !(at(end) == Some('}') && at(end + 1) == Some('}')) {
                end += 1;
            }
            end = (end + 2).min(kept.len());
        } else {
            while at(end).is_some_and(is_identifier_char) {
                end += 1;
            }
        }
        if at(end) == Some(':') && at(end + 1) == Some(':') {
            end += 2;
        } else {
            return end - start;
        }
    }
}

/// Return `name` shortened as [`short_type_name`] describes, borrowed
/// whenever the result is a contiguous slice of `name`.
fn shorten(name: &'static str) -> Cow<'static, str> {
    if is_nominal(name) {
        let path = &name[..name.find('<').unwrap_or(name.len())];
        return Cow::Borrowed(keep_last_segment(path));
    }
    let kept = strip_generic_arguments(name);
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    let mut index = 0;
    while index < kept.len() {
        let length = measure_path(&kept, index).max(1);
        let path: String = kept[index..index + length]
            .iter()
            .map(|&(_, character)| character)
            .collect();
        let skipped = path.chars().count() - keep_last_segment(&path).chars().count();
        for &(position, character) in &kept[index + skipped..index + length] {
            let end = position + character.len_utf8();
            match pieces.last_mut() {
                Some((_, last_end)) if *last_end == position => *last_end = end,
                _ => pieces.push((position, end)),
            }
        }
        index += length;
    }
    match pieces.as_slice() {
        [] => Cow::Borrowed(""),
        &[(start, end)] => Cow::Borrowed(&name[start..end]),
        _ => Cow::Owned(
            pieces
                .iter()
                .map(|&(start, end)| &name[start..end])
                .collect(),
        ),
    }
}

/// Return the default pass name of the type `T`: its type name without
/// module paths and generic arguments.
///
/// The name is [`std::any::type_name`] with every generic argument list
/// `<...>` removed and every path `a::b::C` replaced by its last segment,
/// keeping the last two segments when the last is a `{{...}}` segment such
/// as `{{closure}}`. Everything else is kept, so `&mut my_crate::Fold` is
/// `&mut Fold` and `(a::B, a::Fold<a::B>)` is `(B, Fold)`. The result
/// borrows from the type name whenever it is one contiguous piece of it, so
/// a nominal type, generic or not, is named without allocating.
///
/// `type_name` does not guarantee its output stays the same across compiler
/// versions, so a pass whose name is part of a contract, such as a registry
/// key or a diagnostic source something matches, overrides
/// [`CompilerPass::name`] instead.
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
///
/// use fhy_core::pass::short_type_name;
///
/// struct Fold<T>(T);
///
/// assert!(matches!(short_type_name::<Fold<Vec<u8>>>(), Cow::Borrowed("Fold")));
/// assert_eq!(short_type_name::<(Fold<u8>, &mut u8)>(), "(Fold, &mut u8)");
/// ```
#[must_use]
pub fn short_type_name<T: ?Sized>() -> Cow<'static, str> {
    shorten(type_name::<T>())
}

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
/// 3. [`run`](Self::run);
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
/// use fhy_core::pass::{CompilerPass, ExecutePass, PassContext, PassFailure};
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
    /// Return the pass's name, the source of its diagnostics and its key in
    /// a [`PassRegistry`](super::PassRegistry).
    ///
    /// By default: [`short_type_name::<Self>()`](short_type_name), the
    /// type's name without module paths and generic arguments, borrowed
    /// from the type name. Registering the pass never changes it.
    fn name(&self) -> Cow<'static, str> {
        short_type_name::<Self>()
    }

    /// Return a human-readable description of the pass.
    ///
    /// By default: [`name`](Self::name).
    fn description(&self) -> Cow<'static, str> {
        self.name()
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
    fn name(&self) -> Cow<'static, str> {
        (**self).name()
    }

    fn description(&self) -> Cow<'static, str> {
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
    fn name(&self) -> Cow<'static, str> {
        (**self).name()
    }

    fn description(&self) -> Cow<'static, str> {
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
    skipped: bool,
    diagnostics: Vec<Diagnostic>,
    preserved: PreservedAnalyses,
}

impl<O> PassOutcome<O> {
    /// Create the outcome of a run that produced `result`, emitting
    /// `diagnostics`.
    fn new(result: LifecycleResult<O>, diagnostics: Vec<Diagnostic>) -> Self {
        Self {
            output: result.output,
            changed: result.changed,
            skipped: result.skipped,
            diagnostics,
            preserved: result.preserved,
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

    /// Return whether the pass skipped the run: its output came from
    /// [`CompilerPass::noop_output`], and [`CompilerPass::run`] was not
    /// called.
    #[must_use]
    pub fn is_skipped(&self) -> bool {
        self.skipped
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
    pub(super) skipped: bool,
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
    let pass_name = cx.shared_pass_name();
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
            skipped: true,
            preserved,
        });
    }
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
        skipped: false,
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
        Ok(PassOutcome::new(result, diagnostics))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Test a type name shortens to the documented default pass name, and
    /// borrows exactly when the result is one piece of the type name.
    #[rstest]
    #[case::generic_nominal("my_crate::passes::Fold<i64>", "Fold", true)]
    #[case::nominal("my_crate::Fold", "Fold", true)]
    #[case::bare("Fold", "Fold", true)]
    #[case::nested_generics("a::Fold<b::Pair<c::X, d::Y<e::Z>>>", "Fold", true)]
    #[case::tuple("(main::a::B, main::a::Fold<main::a::B>)", "(B, Fold)", false)]
    #[case::closure("main::main::{{closure}}", "main::{{closure}}", true)]
    #[case::lone_closure("{{closure}}", "{{closure}}", true)]
    #[case::reference("&mut main::a::B", "&mut B", false)]
    #[case::boxed_closure("alloc::boxed::Box<dyn core::ops::function::Fn()>", "Box", true)]
    #[case::boxed_function_returning(
        "alloc::boxed::Box<dyn core::ops::function::Fn(i32) -> i32>",
        "Box",
        true
    )]
    #[case::array("[main::a::B; 2]", "[B; 2]", false)]
    #[case::function_pointer("fn(i32) -> i32", "fn(i32) -> i32", true)]
    #[case::qualified_path("<a::B as c::Tr>::X", "X", true)]
    #[case::trait_object("dyn a::Tr", "dyn Tr", false)]
    #[case::unit("()", "()", true)]
    #[case::unicode("a::Ünïcode<b::Ç>", "Ünïcode", true)]
    fn short_type_name_shortens_a_type_name(
        #[case] name: &'static str,
        #[case] expected: &str,
        #[case] is_borrowed: bool,
    ) {
        let short = shorten(name);

        assert_eq!(short, expected);
        assert_eq!(matches!(short, Cow::Borrowed(_)), is_borrowed, "{short:?}");
    }

    /// Test unbalanced or odd type names shorten to a best-effort name
    /// without panicking.
    #[rstest]
    #[case::unclosed("a::Fold<b::X", "Fold")]
    #[case::extra_close("a::Fold>", "Fold>")]
    #[case::dangling_separator("a::", "")]
    #[case::only_separators("::::", "")]
    #[case::unclosed_closure("a::{{clos", "a::{{clos")]
    #[case::empty("", "")]
    #[case::arrow_only("->", "->")]
    fn short_type_name_handles_odd_names(#[case] name: &'static str, #[case] expected: &str) {
        assert_eq!(shorten(name), expected);
    }

    /// Test the public function shortens the name of a real type.
    #[test]
    fn short_type_name_shortens_the_name_of_a_type() {
        struct Fold<T>(T);

        assert!(matches!(
            short_type_name::<Fold<Vec<String>>>(),
            Cow::Borrowed("Fold")
        ));
        assert_eq!(short_type_name::<&mut Fold<u8>>(), "&mut Fold");
    }
}
