"""Canonical ordering keys for expressions and constraints.

``build_constraint_ordering_key`` is constant on structural-equivalence
classes: two structurally equivalent constraints always key alike,
independent of construction order or the per-process hash seed that
would otherwise leak into a naive ``repr``-based sort.
``ConstraintSystem`` (``fhy_core.symbolic.constraint.system``) sorts its
members by this key, and the param layer orders each parameter's
constraint tuple by it, so the two layers agree on canonical order. The
private helpers here build a textual key for an ``Expression`` subtree
(``_build_expression_ordering_key``/``_render_expression_node_ordering_data``)
or a bare literal (``_build_literal_ordering_key``); the member-set key
used for a set constraint (``_build_member_ordering_key``) lives in
``fhy_core.symbolic.constraint.members``.
"""

__all__ = [
    "build_constraint_ordering_key",
]

from decimal import Decimal

from fhy_core.symbolic.expression import (
    BinaryExpression,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    UnaryExpression,
)

from .core import Constraint, EquationConstraint, _SetConstraint
from .members import _build_member_ordering_key


def _build_literal_ordering_key(value: LiteralType) -> str:
    """Return an ordering key constant on ``LiteralExpression`` equivalence.

    ``LiteralExpression`` compares literals by bucket and canonical form
    rather than by stored Python type, so the key has to collapse the same
    forms: the integer-grammar strings ``"5"`` and ``"05"`` key alike with
    the integer ``5``, the float-grammar strings ``"1.5"`` and ``"1.50"``
    key alike as one exact decimal, and ``-0.0`` keys alike with ``0.0``.
    A ``bool`` keys apart from every integer, and an exact-decimal string
    apart from the binary ``float`` carrying the same digits.

    Args:
        value: Stored value of a ``LiteralExpression``.

    Returns:
        Bucket-prefixed textual key.

    """
    if isinstance(value, bool):
        return f"bool:{value}"
    elif isinstance(value, int):
        return f"int:{value}"
    elif isinstance(value, float):
        # Adding zero maps -0.0 to 0.0; the two are equal and so must key alike.
        return f"float-binary:{value + 0.0!r}"
    # A string-form literal matches the integer grammar or the float grammar,
    # and only the latter carries a decimal point.
    elif "." in value:
        return f"float-decimal:{Decimal(value).normalize()}"
    return f"int:{int(value)}"


def _build_expression_ordering_key(expression: Expression) -> str:
    """Return an ordering key constant on expression structural equivalence.

    Renders the tree as ``NodeType[node data](child keys)``. Node data is
    whatever the node compares by beyond its children: a literal's bucket
    and canonical form, an identifier's ``id``, an operation's name, or a
    call's function name. A ``PiecewiseExpression`` needs none, since its
    children already encode the cases and the fallback.

    Args:
        expression: Expression to key.

    Returns:
        Textual key for the whole subtree.

    """
    children = ",".join(
        _build_expression_ordering_key(child)
        for child in expression.get_visit_children()
    )
    node_data = _render_expression_node_ordering_data(expression)
    return f"{type(expression).__name__}[{node_data}]({children})"


def _render_expression_node_ordering_data(expression: Expression) -> str:
    """Return one node's own ordering data, excluding its children."""
    if isinstance(expression, LiteralExpression):
        return _build_literal_ordering_key(expression.value)
    elif isinstance(expression, IdentifierExpression):
        return f"id:{expression.identifier.id}"
    elif isinstance(expression, (BinaryExpression, UnaryExpression)):
        return expression.operation.value
    elif isinstance(expression, CallExpression):
        return f"call:{expression.function_name}"
    return ""


def build_constraint_ordering_key(constraint: Constraint) -> str:
    """Return the canonical ordering key for a constraint.

    Constant on structural-equivalence classes: two structurally
    equivalent constraints always key alike, so a system's member order
    does not depend on construction order. It is keyed on the same
    things equivalence compares -- the concrete kind, and either the
    expression tree or the variable's ``Identifier.id`` and the
    type-strict member set -- rather than on ``repr``, which neither
    separates every distinct constraint nor agrees on every equivalent
    pair. ``ConstraintSystem`` orders its members by this key, and the
    param layer orders each parameter's constraint tuple by it, so the
    two layers agree on canonical order.

    A ``Constraint`` subclass declared outside this module falls back to
    its ``repr``, which the subclassing contract requires to identify the
    kind and the scope.

    Args:
        constraint: Member to key.

    Returns:
        Textual key ordering the member within its system.

    """
    kind = type(constraint).__name__
    if isinstance(constraint, EquationConstraint):
        expression_key = _build_expression_ordering_key(constraint.expression)
        return f"{kind}|{expression_key}"
    elif isinstance(constraint, _SetConstraint):
        members = ",".join(
            sorted(_build_member_ordering_key(member) for member in constraint.members)
        )
        return f"{kind}|{constraint.variable.id}|{{{members}}}"
    return f"{kind}|{constraint!r}"
