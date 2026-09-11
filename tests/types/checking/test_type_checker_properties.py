"""Hypothesis property tests for the expression type checker (P36).

Checks that `check_expression_type(e, t, lookup)` succeeds and reproduces
`t` when `t` is exactly what `synthesize_expression_type(e, lookup)` just
returned, for numeric and boolean gate-grammar trees over a pool of
identifiers all bound to a concrete `int32` scalar. Numeric gate trees are
narrowed to always include a concrete-typed identifier (see the composite
strategy's docstring for why); the excluded, weak-literal-only shape is
covered separately by the strict xfail at the bottom.
"""

import pytest

pytest.importorskip("hypothesis")

from collections.abc import Callable

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
    is_structurally_equivalent,
)
from fhy_core.types.checking.type_checker import (
    check_expression_type,
    synthesize_expression_type,
)

from ...strategies.expressions import (
    build_boolean_expression_strategy,
    build_numeric_expression_strategy,
)
from ...strategies.identifiers import build_identifier_pool, build_identifier_strategy

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(4)
_MAX_LEAVES = 8
_INT32_SCALAR: Type = NumericalType(PrimitiveDataType(CoreDataType.INT32), [])


def build_pool_lookup(
    bindings: dict[Identifier, tuple[Type, TypeQualifier]],
) -> Callable[[Identifier], tuple[Type, TypeQualifier]]:
    """Build a `get_identifier_type` callable bound to `bindings`.

    Mirrors the lookup `make_identifier_checker`
    (`tests/types/checking/conftest.py`) builds internally: an unknown
    identifier raises `KeyError`, which `ExpressionTypeChecker` catches
    and frames as a type error.
    """

    def lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
        if identifier in bindings:
            return bindings[identifier]
        raise KeyError(identifier.name_hint)

    return lookup


_LOOKUP = build_pool_lookup(dict.fromkeys(_POOL, (_INT32_SCALAR, TypeQualifier.PARAM)))


@st.composite
def draw_numeric_gate_tree_with_a_concrete_anchor(draw: st.DrawFn) -> Expression:
    """Draw a numeric gate tree summed with a pool identifier.

    A numeric gate tree built entirely from literals (no identifier and
    no call to a fixed-result native anywhere) synthesizes a *weak*,
    unresolved-width `NumericalType` (e.g. plain `uint`, not `uint8`);
    `check_expression_type` then re-infers the same literal-only
    expression *with* that weak type as context, which resolves it to a
    concrete width and rejects the mismatch against the weak type it was
    just handed back. Summing with a pool identifier (bound to a concrete
    `int32`) forces the whole tree's synthesized type to be concrete by
    construction, sidestepping the gap; the excluded weak-literal shape
    is covered on its own by the strict xfail at the bottom of this file.
    """
    tree = draw(
        build_numeric_expression_strategy(_POOL, _MAX_LEAVES, include_calls=True)
    )
    anchor = draw(build_identifier_strategy(_POOL))
    return make_binary_expression(
        BinaryOperation.ADD, tree, IdentifierExpression(anchor)
    )


@given(draw_numeric_gate_tree_with_a_concrete_anchor())
def test_synthesize_then_check_round_trips_for_numeric_gate_trees(
    expression: Expression,
) -> None:
    """Test checking a numeric gate tree against its own synthesized type succeeds.

    Oracle: `synthesize_expression_type` and `check_expression_type`
    should agree once the expected type given to `check` is exactly what
    `synthesize` already produced.
    """
    synthesized_type, _ = synthesize_expression_type(expression, _LOOKUP)
    assert isinstance(synthesized_type, NumericalType)
    assert isinstance(synthesized_type.data_type, PrimitiveDataType)
    assert synthesized_type.data_type.core_data_type != CoreDataType.BOOL

    checked_type, _ = check_expression_type(expression, synthesized_type, _LOOKUP)
    assert is_structurally_equivalent(checked_type, synthesized_type)


@given(build_boolean_expression_strategy(_POOL, _MAX_LEAVES))
def test_synthesize_then_check_round_trips_for_boolean_gate_trees(
    expression: Expression,
) -> None:
    """Test checking a boolean gate tree against its own synthesized type succeeds.

    Oracle: same round trip as the numeric case; boolean gate trees need
    no anchor identifier because `BOOL` is always concrete (there is no
    weak boolean type to fall back to).
    """
    synthesized_type, _ = synthesize_expression_type(expression, _LOOKUP)
    assert isinstance(synthesized_type, NumericalType)
    assert isinstance(synthesized_type.data_type, PrimitiveDataType)
    assert synthesized_type.data_type.core_data_type == CoreDataType.BOOL

    checked_type, _ = check_expression_type(expression, synthesized_type, _LOOKUP)
    assert is_structurally_equivalent(checked_type, synthesized_type)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "synthesize_expression_type on a numeric gate tree with no "
        "identifier and no native call anywhere (so nothing forces a "
        "concrete width) returns a weak core data type (e.g. plain "
        "CoreDataType.UINT for a bare non-negative integer literal). "
        "check_expression_type then re-infers the same expression *with* "
        "that weak type as the expected type, which resolves the literal "
        "concretely (e.g. to UINT8) and rejects it as 'wider than the "
        "expected type' against the weak type it was just given. "
        "src/fhy_core/types/checking/type_checker.py: bare-literal "
        "inference and expected-type-directed inference disagree on how "
        "concrete a literal's type should be. Minimal example: "
        "e = LiteralExpression(0); synthesize_expression_type(e, lookup) "
        "returns a weak uint[] NumericalType, and "
        "check_expression_type(e, that_type, lookup) raises "
        "FhYCoreTypeError instead of reproducing it."
    ),
)
def test_synthesize_then_check_round_trips_for_a_bare_weak_literal() -> None:
    """Test (expected to fail) the round trip for a tree with no concrete anchor.

    Documents the shape excluded from
    `test_synthesize_then_check_round_trips_for_numeric_gate_trees` above
    by `draw_numeric_gate_tree_with_a_concrete_anchor`'s forced anchor.
    """
    expression: Expression = LiteralExpression(0)
    synthesized_type, _ = synthesize_expression_type(expression, _LOOKUP)
    checked_type, _ = check_expression_type(expression, synthesized_type, _LOOKUP)
    assert is_structurally_equivalent(checked_type, synthesized_type)
