"""Canonical ordering keys for expression trees.

``_build_expression_ordering_key`` is constant on structural-equivalence
classes: two structurally equivalent expressions always key alike,
independent of construction order or the per-process hash seed that
would otherwise leak into a naive ``repr``-based sort. A literal leaf
keys through ``build_literal_equivalence_key``, which renders the same
bucket and canonical form ``LiteralExpression`` compares by, so for
literals the guarantee holds by construction.
``EquationConstraint.build_ordering_key``
(``fhy_core.symbolic.constraint.core``) keys its wrapped expression with
it; the member-set key a set constraint uses
(``_build_member_ordering_key``) lives in
``fhy_core.symbolic.constraint.members`` instead. Nothing here knows
about the constraint family, so this module sits below it.
"""

from fhy_core.symbolic.expression import (
    BinaryExpression,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    build_literal_equivalence_key,
)


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
        return build_literal_equivalence_key(expression.value)
    elif isinstance(expression, IdentifierExpression):
        return f"id:{expression.identifier.id}"
    elif isinstance(expression, (BinaryExpression, UnaryExpression)):
        return expression.operation.value
    elif isinstance(expression, CallExpression):
        return f"call:{expression.function_name}"
    return ""
