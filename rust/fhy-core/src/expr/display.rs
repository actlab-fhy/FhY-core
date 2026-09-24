//! Text rendering of expressions in symbolic or functional notation.
//!
//! [`Expression::display`] renders an [`Expression`] as text under
//! [`FormatOptions`]: a [`Notation`] choosing between infix operator symbols
//! and prefix operation names, and an [`IdentifierStyle`] choosing whether an
//! identifier reference shows its id. The [`ExpressionDisplay`] it returns
//! implements [`Display`](fmt::Display), and `Expression`'s own `Display`
//! uses the default options. [`ExpressionPrettyFormatter`] renders an
//! expression as a compiler pass. Every unary and binary node is
//! parenthesized, so the text shows the tree's shape exactly and needs no
//! precedence rules. The text is meant for people: it is not parsed back,
//! and distinct trees may print alike (the integer `1` and the float `1.0`,
//! or two identifiers with the same name hint when ids are hidden).

use std::fmt;
use std::iter;

use crate::identifier::Identifier;
use crate::pass::{CompilerPass, PassContext, PassFailure};

use super::node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, PiecewiseExpression,
    UnaryExpression,
};

/// One pending piece of output: a node still to print, or text to write.
enum Step<'a> {
    Print(&'a Expression),
    Write(&'a str),
}

/// Push `steps` onto the `pending` stack so that they pop in the order given.
fn schedule<'a>(pending: &mut Vec<Step<'a>>, steps: impl IntoIterator<Item = Step<'a>>) {
    let start = pending.len();
    pending.extend(steps);
    pending[start..].reverse();
}

/// Write `identifier` to `f` in `style`.
fn write_identifier(
    f: &mut fmt::Formatter<'_>,
    identifier: &Identifier,
    style: IdentifierStyle,
) -> fmt::Result {
    match style {
        IdentifierStyle::NameHint => f.write_str(identifier.name_hint()),
        IdentifierStyle::NameHintWithId => {
            write!(f, "{}::{}", identifier.name_hint(), identifier.id())
        }
    }
}

/// Schedule the pieces of a unary node.
fn schedule_unary<'a>(pending: &mut Vec<Step<'a>>, node: &'a UnaryExpression, notation: Notation) {
    let operation = node.operation();
    match notation {
        Notation::Symbolic => schedule(
            pending,
            [
                Step::Write("("),
                Step::Write(operation.symbol()),
                Step::Print(node.operand()),
                Step::Write(")"),
            ],
        ),
        Notation::Functional => schedule(
            pending,
            [
                Step::Write("("),
                Step::Write(operation.as_str()),
                Step::Write(" "),
                Step::Print(node.operand()),
                Step::Write(")"),
            ],
        ),
    }
}

/// Schedule the pieces of a binary node.
fn schedule_binary<'a>(
    pending: &mut Vec<Step<'a>>,
    node: &'a BinaryExpression,
    notation: Notation,
) {
    let operation = node.operation();
    match notation {
        Notation::Symbolic => schedule(
            pending,
            [
                Step::Write("("),
                Step::Print(node.left()),
                Step::Write(" "),
                Step::Write(operation.symbol()),
                Step::Write(" "),
                Step::Print(node.right()),
                Step::Write(")"),
            ],
        ),
        Notation::Functional => schedule(
            pending,
            [
                Step::Write("("),
                Step::Write(operation.as_str()),
                Step::Write(" "),
                Step::Print(node.left()),
                Step::Write(" "),
                Step::Print(node.right()),
                Step::Write(")"),
            ],
        ),
    }
}

/// Schedule the pieces of a piecewise node: `{v0 if c0; ...; o otherwise}`
/// or `(piecewise c0 v0 ... o)`.
fn schedule_piecewise<'a>(
    pending: &mut Vec<Step<'a>>,
    node: &'a PiecewiseExpression,
    notation: Notation,
) {
    let cases = node.cases();
    match notation {
        Notation::Symbolic => {
            let clauses = cases
                .iter()
                .enumerate()
                .flat_map(|(index, (condition, value))| {
                    [
                        Step::Write(if index == 0 { "" } else { "; " }),
                        Step::Print(value),
                        Step::Write(" if "),
                        Step::Print(condition),
                    ]
                });
            schedule(
                pending,
                iter::once(Step::Write("{")).chain(clauses).chain([
                    Step::Write("; "),
                    Step::Print(node.otherwise()),
                    Step::Write(" otherwise}"),
                ]),
            );
        }
        Notation::Functional => {
            let parts = cases.iter().flat_map(|(condition, value)| {
                [
                    Step::Write(" "),
                    Step::Print(condition),
                    Step::Write(" "),
                    Step::Print(value),
                ]
            });
            schedule(
                pending,
                iter::once(Step::Write("(piecewise")).chain(parts).chain([
                    Step::Write(" "),
                    Step::Print(node.otherwise()),
                    Step::Write(")"),
                ]),
            );
        }
    }
}

/// Schedule the pieces of a call: `f(a, b)` or `(f a b)`.
fn schedule_call<'a>(pending: &mut Vec<Step<'a>>, node: &'a CallExpression, notation: Notation) {
    let arguments = node.arguments();
    match notation {
        Notation::Symbolic => {
            let listed = arguments.iter().enumerate().flat_map(|(index, argument)| {
                [
                    Step::Write(if index == 0 { "" } else { ", " }),
                    Step::Print(argument),
                ]
            });
            schedule(
                pending,
                [Step::Write(node.function_name()), Step::Write("(")]
                    .into_iter()
                    .chain(listed)
                    .chain(iter::once(Step::Write(")"))),
            );
        }
        Notation::Functional => {
            let listed = arguments
                .iter()
                .flat_map(|argument| [Step::Write(" "), Step::Print(argument)]);
            schedule(
                pending,
                [Step::Write("("), Step::Write(node.function_name())]
                    .into_iter()
                    .chain(listed)
                    .chain(iter::once(Step::Write(")"))),
            );
        }
    }
}

/// How operations, piecewise nodes, and calls are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Notation {
    /// Infix operator symbols: `(-x)`, `(x + 1)`, `{1 if (x > 0); 0 otherwise}`,
    /// `f(x, 1)`.
    #[default]
    Symbolic,
    /// Prefix operation names in parentheses: `(negate x)`, `(add x 1)`,
    /// `(piecewise (greater x 0) 1 0)`, `(f x 1)`.
    Functional,
}

/// How an identifier reference is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IdentifierStyle {
    /// The identifier's name hint alone: `x`.
    #[default]
    NameHint,
    /// The name hint, two colons, and the id: `x::41`.
    NameHintWithId,
}

/// The options of [`Expression::display`].
///
/// The default is [`Notation::Symbolic`] with [`IdentifierStyle::NameHint`];
/// [`with_notation`](Self::with_notation) and
/// [`with_identifier_style`](Self::with_identifier_style) change one option
/// at a time.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{FormatOptions, IdentifierStyle, Notation};
///
/// let options = FormatOptions::default()
///     .with_notation(Notation::Functional)
///     .with_identifier_style(IdentifierStyle::NameHintWithId);
///
/// assert_eq!(options.notation(), Notation::Functional);
/// assert_eq!(options.identifier_style(), IdentifierStyle::NameHintWithId);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct FormatOptions {
    notation: Notation,
    identifier_style: IdentifierStyle,
}

impl FormatOptions {
    /// Return these options writing nodes in `notation`.
    #[must_use]
    pub fn with_notation(self, notation: Notation) -> Self {
        Self { notation, ..self }
    }

    /// Return these options writing identifier references in `style`.
    #[must_use]
    pub fn with_identifier_style(self, style: IdentifierStyle) -> Self {
        Self {
            identifier_style: style,
            ..self
        }
    }

    /// Return the notation nodes are written in.
    #[must_use]
    pub fn notation(&self) -> Notation {
        self.notation
    }

    /// Return the style identifier references are written in.
    #[must_use]
    pub fn identifier_style(&self) -> IdentifierStyle {
        self.identifier_style
    }
}

/// A compiler pass formatting an expression as text under
/// [`FormatOptions`], as [`Expression::display`] does.
///
/// Every run counts as a change, since its text output is never its input
/// expression. The default formatter uses the default options.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::ExecutePass;
/// use fhy_core::expr::{
///     Expression, ExpressionPrettyFormatter, FormatOptions, Notation,
/// };
///
/// let x = Expression::from(Identifier::new("x"));
/// let options = FormatOptions::default().with_notation(Notation::Functional);
/// let mut formatter = ExpressionPrettyFormatter::new(options);
///
/// let outcome = formatter.execute(&(&x + 1))?;
///
/// assert_eq!(outcome.output(), "(add x 1)");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExpressionPrettyFormatter {
    options: FormatOptions,
}

impl ExpressionPrettyFormatter {
    /// Create the pass formatting under `options`.
    #[must_use]
    pub fn new(options: FormatOptions) -> Self {
        Self { options }
    }

    /// Return the options the pass formats under.
    #[must_use]
    pub fn options(&self) -> FormatOptions {
        self.options
    }
}

impl CompilerPass<Expression, String> for ExpressionPrettyFormatter {
    fn run(&mut self, ir: &Expression, _cx: &mut PassContext<'_>) -> Result<String, PassFailure> {
        Ok(ir.display(self.options).to_string())
    }

    fn did_change(&mut self, input: &Expression, output: &String) -> Result<bool, PassFailure> {
        let _ = (input, output);
        Ok(true)
    }
}

/// Write a leaf `node` to `f`, or schedule the pieces of an inner `node`
/// on `pending`.
fn print_node<'a>(
    node: &'a Expression,
    options: FormatOptions,
    f: &mut fmt::Formatter<'_>,
    pending: &mut Vec<Step<'a>>,
) -> fmt::Result {
    let notation = options.notation;
    match node.kind() {
        ExpressionKind::Identifier(identifier) => {
            write_identifier(f, identifier, options.identifier_style)?;
        }
        ExpressionKind::Literal(value) => write!(f, "{value}")?,
        ExpressionKind::Unary(unary) => schedule_unary(pending, unary, notation),
        ExpressionKind::Binary(binary) => schedule_binary(pending, binary, notation),
        ExpressionKind::Piecewise(piecewise) => schedule_piecewise(pending, piecewise, notation),
        ExpressionKind::Call(call) => schedule_call(pending, call, notation),
    }
    Ok(())
}

/// Write `expression` to `f` under `options`, from an explicit work stack.
fn write_expression(
    expression: &Expression,
    options: FormatOptions,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let mut pending = vec![Step::Print(expression)];
    while let Some(step) = pending.pop() {
        match step {
            Step::Write(piece) => f.write_str(piece)?,
            Step::Print(node) => print_node(node, options, f, &mut pending)?,
        }
    }
    Ok(())
}

/// An expression rendered as text under [`FormatOptions`]; what
/// [`Expression::display`] returns.
///
/// Its [`Display`](fmt::Display) writes the text straight into the
/// formatter, with no intermediate string per node.
#[derive(Debug, Clone, Copy)]
pub struct ExpressionDisplay<'a> {
    expression: &'a Expression,
    options: FormatOptions,
}

impl fmt::Display for ExpressionDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_expression(self.expression, self.options, f)
    }
}

impl Expression {
    /// Return the expression rendered as text under `options`, as a value
    /// implementing [`Display`](fmt::Display).
    ///
    /// Each node is written as follows, in [`Notation::Symbolic`] and then
    /// in [`Notation::Functional`]:
    ///
    /// | Node | Symbolic | Functional |
    /// |---|---|---|
    /// | unary | `(` symbol operand `)`: `(-x)`, `(+x)`, `(!p)` | `(negate x)`, `(positive x)`, `(logical_not p)` |
    /// | binary | `(left symbol right)`: `(x // 2)` | `(floor_divide x 2)` |
    /// | piecewise | `{v0 if c0; v1 if c1; o otherwise}` | `(piecewise c0 v0 c1 v1 o)` |
    /// | call | `f(a, b)`, `f()` | `(f a b)`, `(f)` |
    ///
    /// The symbols and names are the operations'
    /// [`symbol`](super::BinaryOperation::symbol) and
    /// [`as_str`](super::BinaryOperation::as_str) texts. Cases are written
    /// in order, then the otherwise branch; arguments in order.
    ///
    /// A literal is written as its [`Display`](fmt::Display) text in both
    /// notations (`true`, `-1`, `10000000000000000`, `NaN`, `1.5`), so a
    /// negative literal operand stays bare: `(-1 ** 2)` is the literal `-1`
    /// raised to `2`, and `((-1) ** 2)` a negation raised to `2`, and a unary
    /// node over a negative literal reads `(--1)`. An identifier reference is
    /// written as its name hint, or as `name::id` under
    /// [`IdentifierStyle::NameHintWithId`], in both notations and with no
    /// quoting or escaping of the name hint.
    ///
    /// Writing does not recurse, so a tree of any depth displays without
    /// exhausting the thread's stack. A subtree occurring in several places
    /// is written at every occurrence, so the text of a DAG can be
    /// exponential in its depth.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::{Expression, FormatOptions, Notation};
    ///
    /// let x = Expression::from(Identifier::new("x"));
    /// let tree = (&x + 1) * 2;
    ///
    /// assert_eq!(tree.display(FormatOptions::default()).to_string(), "((x + 1) * 2)");
    /// let functional = FormatOptions::default().with_notation(Notation::Functional);
    /// assert_eq!(tree.display(functional).to_string(), "(multiply (add x 1) 2)");
    /// assert_eq!(tree.to_string(), "((x + 1) * 2)");
    /// ```
    #[must_use]
    pub fn display(&self, options: FormatOptions) -> ExpressionDisplay<'_> {
        ExpressionDisplay {
            expression: self,
            options,
        }
    }
}

/// The full text of the expression under the default [`FormatOptions`], as
/// [`Expression::display`] writes it.
///
/// A subtree occurring in several places is written at every occurrence, so
/// the text of a DAG can be exponential in its depth.
impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_expression(self, FormatOptions::default(), f)
    }
}
