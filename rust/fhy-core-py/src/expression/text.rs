//! The printed text of expressions (D-S4-1): `str` is the core's `Display`,
//! `repr` its bounded `Debug` text under the node's class name, and
//! `pformat_expression` the core's `display` under the matching options.

use fhy_core::expression::{Expression, ExpressionKind, FormatOptions, IdentifierStyle, Notation};

/// Return the name of the public class of `expression`'s node kind.
pub(super) fn kind_class_name(expression: &Expression) -> &'static str {
    match expression.kind() {
        ExpressionKind::Unary(_) => "UnaryExpression",
        ExpressionKind::Binary(_) => "BinaryExpression",
        ExpressionKind::Logical(_) => "LogicalExpression",
        ExpressionKind::Identifier(_) => "IdentifierExpression",
        ExpressionKind::Literal(_) => "LiteralExpression",
        ExpressionKind::Piecewise(_) => "PiecewiseExpression",
        ExpressionKind::Call(_) => "CallExpression",
    }
}

/// Return the `repr` of `expression` under `class_name`: the class name and,
/// in parentheses, the core's bounded diagnostic text (the functional
/// notation with identifier ids, eliding every node after the first 1,000).
///
/// For example, `BinaryExpression((add x::7 1))` and `LiteralExpression(1.5)`.
pub(super) fn render_repr(class_name: &str, expression: &Expression) -> String {
    let debug = format!("{expression:?}");
    let body = debug.strip_prefix("Expression").unwrap_or(&debug);
    let mut text = String::with_capacity(class_name.len() + body.len());
    text.push_str(class_name);
    text.push_str(body);
    text
}

/// Return the `repr` of `expression` under its node kind's class name.
pub(super) fn render_kind_repr(expression: &Expression) -> String {
    render_repr(kind_class_name(expression), expression)
}

/// Return the text `pformat_expression` renders for `expression`: the
/// core's `display` in symbolic or functional notation, with or without
/// identifier ids.
pub(super) fn render_formatted(expression: &Expression, show_id: bool, functional: bool) -> String {
    let notation = if functional {
        Notation::Functional
    } else {
        Notation::Symbolic
    };
    let identifier_style = if show_id {
        IdentifierStyle::NameHintWithId
    } else {
        IdentifierStyle::NameHint
    };
    let options = FormatOptions::default()
        .with_notation(notation)
        .with_identifier_style(identifier_style);
    expression.display(options).to_string()
}
