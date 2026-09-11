"""Hypothesis property tests for the type-unification dispatchers.

Covers `unify`: reflexivity of unifying a type with itself against the
empty environment, symmetry of unifying two distinct types, and that
binding a template pattern and substituting the result reproduces the
concrete actual it was bound against. A hand-built numerical-type and a
hand-built index-type pattern/actual pair are pinned as examples on the
bind-then-substitute property below.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.traits import VerificationError
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeUnificationEnvironment,
    bind_template,
    is_structurally_equivalent,
    substitute_template,
    unify,
)

from ..strategies.identifiers import MOCK_IDENTIFIER_ID_BASE, build_identifier_pool
from ..strategies.types import (
    build_primitive_data_type_strategy,
    draw_template_free_type,
)
from .conftest import mock_identifier

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(8)

_MAX_BIND_RANK = 3
_MIN_BIND_LITERAL = 1
_MAX_BIND_LITERAL = 8


# =============================================================================
# (a) unify(t, t, empty) is reflexive
# =============================================================================


@given(draw_template_free_type(_POOL))
def test_unify_of_a_type_with_itself_is_reflexive(type_: Type) -> None:
    """Test `unify(t, t, empty)` returns `t` and leaves the environment empty.

    Oracle: `TypeUnificationEnvironment.empty()`, an independently
    constructed environment compared structurally against the one
    `unify` returns.
    """
    empty_environment = TypeUnificationEnvironment.empty()
    unified_type, resulting_environment = unify(type_, type_, empty_environment)
    assert is_structurally_equivalent(unified_type, type_)
    assert resulting_environment.is_structurally_equivalent(empty_environment)


# =============================================================================
# (b) unify is symmetric on identifier-free (non-placeholder-shape) types
# =============================================================================


def unify_or_capture(
    expected: Type, actual: Type, environment: TypeUnificationEnvironment
) -> tuple[Type, TypeUnificationEnvironment] | None:
    """Return `unify`'s result, or `None` if it raised `VerificationError`."""
    try:
        return unify(expected, actual, environment)
    except VerificationError:
        return None


@given(draw_template_free_type(()), draw_template_free_type(()))
def test_unify_is_symmetric_for_identifier_free_shapes(left: Type, right: Type) -> None:
    """Test `unify(a, b, env)` and `unify(b, a, env)` agree.

    Either both raise `VerificationError`, or both succeed with
    structurally equivalent unified types. Limited to shapes with no free
    shape identifiers (`draw_template_free_type` given an empty identifier
    pool draws only literal bounds and dimensions); with free identifiers
    only the outcome is symmetric, see
    `test_unify_succeeds_or_fails_symmetrically_for_any_shapes`.
    """
    empty_environment = TypeUnificationEnvironment.empty()
    forward = unify_or_capture(left, right, empty_environment)
    backward = unify_or_capture(right, left, empty_environment)
    assert (forward is None) == (backward is None)
    if forward is not None and backward is not None:
        forward_type, _ = forward
        backward_type, _ = backward
        assert is_structurally_equivalent(forward_type, backward_type)


@given(draw_template_free_type(_POOL), draw_template_free_type(_POOL))
def test_unify_succeeds_or_fails_symmetrically_for_any_shapes(
    left: Type, right: Type
) -> None:
    """Test `unify(a, b, env)` and `unify(b, a, env)` both succeed or both raise.

    Over shapes that may carry free identifiers only the outcome is
    symmetric: two distinct free identifiers in the same position unify by
    binding one to the other, so which identifier survives depends on
    argument order and the two unified types need not be structurally
    equivalent. That is the unifier picking a representative, not a fault.
    """
    empty_environment = TypeUnificationEnvironment.empty()

    forward = unify_or_capture(left, right, empty_environment)
    backward = unify_or_capture(right, left, empty_environment)

    assert (forward is None) == (backward is None)


# =============================================================================
# (c) bind_template then substitute_template reproduces the actual type
# =============================================================================


@st.composite
def draw_numerical_bind_case(draw: st.DrawFn) -> tuple[Type, Type]:
    """Draw a `NumericalType` template pattern and a matching concrete actual.

    The pattern's data type is always a fresh `TemplateDataType`
    placeholder; each shape dimension is either a fresh placeholder
    identifier (which the actual's corresponding literal dimension binds
    freely) or a literal value that the actual repeats exactly, so
    `bind_template` always succeeds. Mirrors the shape of the numerical-type
    pattern/actual pair pinned as an example below.
    """
    permutation = draw(st.permutations(_POOL))
    rank = draw(st.integers(min_value=0, max_value=_MAX_BIND_RANK))
    template_identifier = permutation[0]
    shape_identifiers = permutation[1 : 1 + rank]
    data_type = draw(build_primitive_data_type_strategy())

    pattern_dimensions: list[Expression] = []
    actual_dimensions: list[Expression] = []
    for index in range(rank):
        if draw(st.booleans()):
            pattern_dimensions.append(IdentifierExpression(shape_identifiers[index]))
            actual_dimensions.append(
                LiteralExpression(
                    draw(st.integers(_MIN_BIND_LITERAL, _MAX_BIND_LITERAL))
                )
            )
        else:
            value = draw(st.integers(_MIN_BIND_LITERAL, _MAX_BIND_LITERAL))
            pattern_dimensions.append(LiteralExpression(value))
            actual_dimensions.append(LiteralExpression(value))

    pattern: Type = NumericalType(
        TemplateDataType(template_identifier), pattern_dimensions
    )
    actual: Type = NumericalType(data_type, actual_dimensions)
    return pattern, actual


@st.composite
def draw_index_bind_case(draw: st.DrawFn) -> tuple[Type, Type]:
    """Draw an `IndexType` pattern and a matching concrete actual.

    Each of the lower bound, upper bound, and stride is either a fresh
    placeholder identifier or a literal the actual repeats exactly, so
    `bind_template` always succeeds. Mirrors the shape of the index-type
    pattern/actual pair pinned as an example below.
    """
    permutation = draw(st.permutations(_POOL))
    part_identifiers = permutation[:3]

    pattern_parts: list[Expression] = []
    actual_parts: list[Expression] = []
    for index in range(3):
        if draw(st.booleans()):
            pattern_parts.append(IdentifierExpression(part_identifiers[index]))
            actual_parts.append(
                LiteralExpression(
                    draw(st.integers(_MIN_BIND_LITERAL, _MAX_BIND_LITERAL))
                )
            )
        else:
            value = draw(st.integers(_MIN_BIND_LITERAL, _MAX_BIND_LITERAL))
            pattern_parts.append(LiteralExpression(value))
            actual_parts.append(LiteralExpression(value))

    lower, upper, stride = pattern_parts
    actual_lower, actual_upper, actual_stride = actual_parts
    pattern: Type = IndexType(lower, upper, stride)
    actual: Type = IndexType(actual_lower, actual_upper, actual_stride)
    return pattern, actual


def draw_bind_and_substitute_case() -> st.SearchStrategy[tuple[Type, Type]]:
    """Return a strategy over `NumericalType` and `IndexType` bind/substitute cases."""
    return st.one_of(draw_numerical_bind_case(), draw_index_bind_case())


_PINNED_NUMERICAL_TEMPLATE_IDENTIFIER = mock_identifier(
    "T", MOCK_IDENTIFIER_ID_BASE + 910
)
_PINNED_NUMERICAL_N_IDENTIFIER = mock_identifier("N", MOCK_IDENTIFIER_ID_BASE + 911)
_PINNED_NUMERICAL_M_IDENTIFIER = mock_identifier("M", MOCK_IDENTIFIER_ID_BASE + 912)
_PINNED_NUMERICAL_PATTERN: Type = NumericalType(
    TemplateDataType(_PINNED_NUMERICAL_TEMPLATE_IDENTIFIER),
    [
        IdentifierExpression(_PINNED_NUMERICAL_N_IDENTIFIER),
        IdentifierExpression(_PINNED_NUMERICAL_M_IDENTIFIER),
    ],
)
_PINNED_NUMERICAL_ACTUAL: Type = NumericalType(
    PrimitiveDataType(CoreDataType.FLOAT32),
    [LiteralExpression(10), LiteralExpression(20)],
)
"""A hand-built numerical-type pattern/actual pair, pinned as an example."""

_PINNED_INDEX_N_IDENTIFIER = mock_identifier("N", MOCK_IDENTIFIER_ID_BASE + 913)
_PINNED_INDEX_PATTERN: Type = IndexType(
    LiteralExpression(0),
    IdentifierExpression(_PINNED_INDEX_N_IDENTIFIER),
    LiteralExpression(1),
)
_PINNED_INDEX_ACTUAL: Type = IndexType(
    LiteralExpression(0), LiteralExpression(64), LiteralExpression(1)
)
"""A hand-built index-type pattern/actual pair, pinned as an example."""


@example(case=(_PINNED_NUMERICAL_PATTERN, _PINNED_NUMERICAL_ACTUAL))
@example(case=(_PINNED_INDEX_PATTERN, _PINNED_INDEX_ACTUAL))
@given(draw_bind_and_substitute_case())
def test_bind_template_then_substitute_reproduces_the_actual_type(
    case: tuple[Type, Type],
) -> None:
    """Test bind-then-substitute reproduces the actual type the pattern was bound to.

    Oracle: self, by construction. `pattern` and `actual` are built so
    every non-placeholder dimension already matches exactly and every
    placeholder is free to bind, so
    `substitute_template(pattern, bind_template(pattern, actual, empty))`
    must be structurally equivalent to `actual`.
    """
    pattern, actual = case
    environment = bind_template(pattern, actual, TypeUnificationEnvironment.empty())
    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)
