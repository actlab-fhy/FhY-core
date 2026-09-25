"""Tests `Expression` participating in the `HasFreeIdentifiers` / `Term` traits.

The expression IR has no internal binders, so every referenced identifier is
free and substitution is always capture-free; these tests pin that behavior on
the trait methods exposed directly on expression nodes.
"""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    logical_and,
    piecewise,
)
from fhy_core.term import AlphaRenaming, HasFreeIdentifiers, Term


def test_expression_is_a_term() -> None:
    """Test expression nodes satisfy the `Term` and `HasFreeIdentifiers` protocols."""
    node = IdentifierExpression(Identifier("x"))
    assert isinstance(node, Term)
    assert isinstance(node, HasFreeIdentifiers)
    assert isinstance(LiteralExpression(1), Term)


def test_identifier_expression_free_identifier_is_itself() -> None:
    """Test an identifier reference reports itself as free."""
    x = Identifier("x")
    assert IdentifierExpression(x).get_free_identifiers() == frozenset({x})


def test_literal_expression_has_no_free_identifiers() -> None:
    """Test a literal has no free identifiers."""
    assert LiteralExpression(7).get_free_identifiers() == frozenset()


def test_composite_free_identifiers_union_over_children() -> None:
    """Test free identifiers of a composite union over its sub-expressions."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(5),
            IdentifierExpression(y),
        ),
    )
    assert expression.get_free_identifiers() == frozenset({x, y})


def test_unary_expression_free_identifiers_walk_into_operand() -> None:
    """Test free identifiers reach into a unary operand."""
    x = Identifier("x")
    node = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))
    assert node.get_free_identifiers() == frozenset({x})


def test_call_expression_free_identifiers_union_over_arguments() -> None:
    """Test free identifiers of a call union over its argument expressions."""
    x = Identifier("x")
    y = Identifier("y")
    call = CallExpression("f", (IdentifierExpression(x), IdentifierExpression(y)))
    assert call.get_free_identifiers() == frozenset({x, y})


def test_substitute_replaces_identifier_with_term() -> None:
    """Test substituting a mapped identifier yields the replacement expression."""
    x = Identifier("x")
    result = IdentifierExpression(x).substitute({x: LiteralExpression(5)})
    assert result.is_structurally_equivalent(LiteralExpression(5))


def test_substitute_leaves_unmapped_identifier_untouched() -> None:
    """Test an identifier absent from the map is returned unchanged."""
    x = Identifier("x")
    y = Identifier("y")
    node = IdentifierExpression(y)
    assert node.substitute({x: LiteralExpression(5)}) is node


def test_substitute_returns_literal_unchanged() -> None:
    """Test substituting into a literal returns the same instance."""
    literal = LiteralExpression(3)
    assert literal.substitute({}) is literal


def test_substitute_recurses_into_composite() -> None:
    """Test substitution rewrites mapped identifiers throughout a composite."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = expression.substitute({x: LiteralExpression(1)})

    expected = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), IdentifierExpression(y)
    )
    assert result.is_structurally_equivalent(expected)


def test_substitute_recurses_into_unary_expression() -> None:
    """Test substitution traverses into a unary operand."""
    x = Identifier("x")
    node = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))

    result = node.substitute({x: LiteralExpression(7)})

    expected = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(7))
    assert result.is_structurally_equivalent(expected)


def test_substitute_preserves_unchanged_nested_literal_identity() -> None:
    """Test unchanged literal children keep their instance through substitution."""
    literal = LiteralExpression(42)
    tree = BinaryExpression(BinaryOperation.ADD, literal, literal)

    result = tree.substitute({})

    assert isinstance(result, BinaryExpression)
    assert result.left is literal
    assert result.right is literal


@pytest.mark.parametrize(
    "replacements",
    [
        pytest.param({}, id="empty"),
        pytest.param({Identifier("absent"): LiteralExpression(1)}, id="absent_key"),
        pytest.param({"x": LiteralExpression(1)}, id="non_identifier_key"),
    ],
)
def test_substitute_that_replaces_nothing_returns_the_expression_itself(
    replacements: dict[object, LiteralExpression],
) -> None:
    """Test a substitution replacing no reference returns the same object.

    The core returns a handle to every subtree it leaves unchanged, so the
    whole expression comes back itself when nothing is replaced.
    """
    x = Identifier("x")
    tree = IdentifierExpression(x) + LiteralExpression(1)

    assert tree.substitute(replacements) is tree  # type: ignore[arg-type]


def test_substitute_keeps_the_objects_of_unchanged_subtrees() -> None:
    """Test only the nodes above a replaced reference are new objects."""
    x = Identifier("x")
    y = Identifier("y")
    unchanged = IdentifierExpression(y) * LiteralExpression(2)
    tree = (IdentifierExpression(x) + LiteralExpression(1)) - unchanged

    result = tree.substitute({x: LiteralExpression(5)})

    assert result == (LiteralExpression(5) + LiteralExpression(1)) - unchanged
    assert isinstance(result, BinaryExpression)
    assert isinstance(result.left, BinaryExpression)
    assert isinstance(tree.left, BinaryExpression)
    assert result.right is unchanged
    assert result.left.right is tree.left.right


def test_substitute_places_the_replacement_object_at_every_occurrence() -> None:
    """Test each replaced reference becomes the replacement object itself."""
    x = Identifier("x")
    replacement = IdentifierExpression(Identifier("y")) + LiteralExpression(1)
    tree = IdentifierExpression(x) * IdentifierExpression(x)

    result = tree.substitute({x: replacement})

    assert isinstance(result, BinaryExpression)
    assert result.left is replacement
    assert result.right is replacement


def test_substitute_is_simultaneous() -> None:
    """Test replacements are not substituted into in turn."""
    x = Identifier("x")
    y = Identifier("y")
    tree = IdentifierExpression(x) + IdentifierExpression(y)

    result = tree.substitute({x: IdentifierExpression(y), y: IdentifierExpression(x)})

    assert result == IdentifierExpression(y) + IdentifierExpression(x)


def test_substitute_reaches_every_node_kind() -> None:
    """Test substitution reaches under a connective, a piecewise and a call."""
    x = Identifier("x")
    reference = IdentifierExpression(x)
    tree = piecewise(
        (logical_and(reference > 0, reference < 9), CallExpression("f", (reference,))),
        otherwise=-reference,
    )

    result = tree.substitute({x: LiteralExpression(3)})

    three = LiteralExpression(3)
    assert result == piecewise(
        (logical_and(three > 0, three < 9), CallExpression("f", (three,))),
        otherwise=-three,
    )
    assert isinstance(result, PiecewiseExpression)
    assert isinstance(result.conditions[0], LogicalExpression)


def test_substitute_keeps_a_shared_subtree_shared() -> None:
    """Test a subtree occurring twice is substituted into once and shared."""
    x = Identifier("x")
    shared = IdentifierExpression(x) + LiteralExpression(1)
    tree = shared * shared

    result = tree.substitute({x: LiteralExpression(2)})

    assert isinstance(result, BinaryExpression)
    assert result.left is result.right
    assert result == (LiteralExpression(2) + 1) * (LiteralExpression(2) + 1)


def test_substitute_ignores_a_non_expression_value_of_an_absent_identifier() -> None:
    """Test a non-expression value is refused only for a referenced identifier."""
    x = Identifier("x")
    node = IdentifierExpression(x) + 1

    assert node.substitute({Identifier("other"): object()}) is node  # type: ignore[dict-item]


# =============================================================================
# Alpha equivalence under a renaming's binder frames (D-S4-3)
# =============================================================================


def test_alpha_equivalence_under_a_free_renaming_maps_each_identifier() -> None:
    """Test a free renaming maps identifiers of one side onto the other's."""
    x, y, a, b = (Identifier(name) for name in ("x", "y", "a", "b"))
    left = IdentifierExpression(x) + IdentifierExpression(y)
    right = IdentifierExpression(a) + IdentifierExpression(b)

    renaming = AlphaRenaming.with_free_renaming({x: a, y: b})

    assert left.is_alpha_equivalent_under(right, renaming)
    assert not left.is_alpha_equivalent_under(
        right, AlphaRenaming.with_free_renaming({x: b, y: a})
    )
    assert not left.is_alpha_equivalent(right)


def test_alpha_equivalence_follows_the_innermost_binder_frame() -> None:
    """Test an inner frame shadows an outer one binding the same identifier."""
    x, a, b = Identifier("x"), Identifier("a"), Identifier("b")
    renaming = AlphaRenaming.empty().extend({x: a}).extend({x: b})

    assert IdentifierExpression(x).is_alpha_equivalent_under(
        IdentifierExpression(b), renaming
    )
    assert not IdentifierExpression(x).is_alpha_equivalent_under(
        IdentifierExpression(a), renaming
    )


def test_alpha_equivalence_refuses_to_capture_a_free_identifier() -> None:
    """Test a free identifier on one side never matches a bound one on the other."""
    x, y = Identifier("x"), Identifier("y")
    renaming = AlphaRenaming.empty().extend({x: y})

    assert not IdentifierExpression(y).is_alpha_equivalent_under(
        IdentifierExpression(y), renaming
    )


def test_alpha_equivalence_under_the_empty_renaming_is_structural_equality() -> None:
    """Test comparing under the empty renaming is ``==``."""
    x = Identifier("x")
    left = IdentifierExpression(x) + 1
    right = IdentifierExpression(x) + 1

    assert left.is_alpha_equivalent_under(right, AlphaRenaming.empty())
    assert left.is_alpha_equivalent(right)


def test_alpha_equivalence_refuses_a_renaming_of_another_type() -> None:
    """Test the renaming must be an ``AlphaRenaming``."""
    x = Identifier("x")
    with pytest.raises(TypeError, match="AlphaRenaming"):
        IdentifierExpression(x).is_alpha_equivalent_under(
            IdentifierExpression(x),
            {x: x},  # type: ignore[arg-type]
        )


def test_substitute_renames_identifier_via_identifier_expression() -> None:
    """Test substituting an identifier with an identifier reference renames it."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(5)
    )

    result = expression.substitute({x: IdentifierExpression(y)})

    expected = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(5)
    )
    assert result.is_structurally_equivalent(expected)


def test_substitute_rejects_non_expression_replacement() -> None:
    """Test substituting an identifier with a non-Expression term raises `TypeError`."""
    x = Identifier("x")
    node = IdentifierExpression(x)
    with pytest.raises(TypeError):
        node.substitute({x: object()})  # type: ignore[dict-item]
