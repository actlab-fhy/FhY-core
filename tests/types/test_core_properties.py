"""Hypothesis property tests for `fhy_core.types.core` (P33, P35).

P33 checks that `promote_core_data_types`, `promote_primitive_data_types`,
and `promote_type_qualifiers` are commutative, associative, and (mostly)
idempotent; where a promotion raises, the law relaxes to "both orders (or
both associations) raise the same exception type". P35 checks that
`is_structurally_equivalent` is reflexive and symmetric on drawn `Type`
trees, and that a DICT serialization round trip preserves it (entry
points read from `tests/types/test_serialization.py`).
"""

import pytest

pytest.importorskip("hypothesis")

from collections.abc import Callable
from typing import TypeVar

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    PrimitiveDataType,
    Type,
    TypeQualifier,
    is_structurally_equivalent,
    promote_core_data_types,
    promote_primitive_data_types,
    promote_type_qualifiers,
)

from ..strategies.identifiers import build_identifier_pool
from ..strategies.types import (
    build_core_data_type_strategy,
    build_primitive_data_type_strategy,
    build_type_qualifier_strategy,
    draw_template_free_type,
)

pytestmark = pytest.mark.property

_T = TypeVar("_T")

_POOL = build_identifier_pool(4)

_IDEMPOTENT_TYPE_QUALIFIERS: tuple[TypeQualifier, ...] = (
    TypeQualifier.PARAM,
    TypeQualifier.TEMP,
)


def evaluate_or_capture_type_error(
    operation: Callable[[_T, _T], _T], left: _T, right: _T
) -> tuple[_T, None] | tuple[None, type[FhYCoreTypeError]]:
    """Call `operation(left, right)`, capturing a raised `FhYCoreTypeError`.

    Returns:
        `(result, None)` on success, or `(None, exception_type)` if
        `operation` raised `FhYCoreTypeError`.
    """
    try:
        return operation(left, right), None
    except FhYCoreTypeError as exc:
        return None, type(exc)


def assert_commutative_or_consistent_raise(
    operation: Callable[[_T, _T], _T],
    left: _T,
    right: _T,
    equals: Callable[[_T, _T], bool],
) -> None:
    """Test `operation(left, right)` and `operation(right, left)` agree.

    Either both raise `FhYCoreTypeError`, or both succeed with results
    `equals` considers equal.
    """
    forward_result, forward_exception = evaluate_or_capture_type_error(
        operation, left, right
    )
    backward_result, backward_exception = evaluate_or_capture_type_error(
        operation, right, left
    )
    assert (forward_exception is None) == (backward_exception is None)
    if forward_exception is None:
        assert forward_result is not None and backward_result is not None
        assert equals(forward_result, backward_result)
    else:
        assert forward_exception == backward_exception


def _evaluate_left_associated(
    operation: Callable[[_T, _T], _T], first: _T, second: _T, third: _T
) -> tuple[_T, None] | tuple[None, type[FhYCoreTypeError]]:
    """Evaluate `operation(operation(first, second), third)`, capturing any raise."""
    inner_result, inner_exception = evaluate_or_capture_type_error(
        operation, first, second
    )
    if inner_exception is not None:
        return None, inner_exception
    assert inner_result is not None
    return evaluate_or_capture_type_error(operation, inner_result, third)


def _evaluate_right_associated(
    operation: Callable[[_T, _T], _T], first: _T, second: _T, third: _T
) -> tuple[_T, None] | tuple[None, type[FhYCoreTypeError]]:
    """Evaluate `operation(first, operation(second, third))`, capturing any raise."""
    inner_result, inner_exception = evaluate_or_capture_type_error(
        operation, second, third
    )
    if inner_exception is not None:
        return None, inner_exception
    assert inner_result is not None
    return evaluate_or_capture_type_error(operation, first, inner_result)


def assert_associative_or_consistent_raise(
    operation: Callable[[_T, _T], _T],
    first: _T,
    second: _T,
    third: _T,
    equals: Callable[[_T, _T], bool],
) -> None:
    """Test left- and right-associating `operation` over three values agree.

    Either both associations raise `FhYCoreTypeError` (from either the
    inner or the outer application), or both succeed with results
    `equals` considers equal.
    """
    left_result, left_exception = _evaluate_left_associated(
        operation, first, second, third
    )
    right_result, right_exception = _evaluate_right_associated(
        operation, first, second, third
    )
    assert (left_exception is None) == (right_exception is None)
    if left_exception is None:
        assert left_result is not None and right_result is not None
        assert equals(left_result, right_result)
    else:
        assert left_exception == right_exception


def _primitive_data_types_are_equivalent(
    left: PrimitiveDataType, right: PrimitiveDataType
) -> bool:
    """Return whether two `PrimitiveDataType` values are structurally equivalent.

    `PrimitiveDataType` has no value-based `__eq__` (each promotion builds
    a fresh instance), so `==` cannot compare two promotion results.
    """
    return left.is_structurally_equivalent(right)


# =============================================================================
# P33: `promote_core_data_types`
# =============================================================================


@given(build_core_data_type_strategy(), build_core_data_type_strategy())
def test_promote_core_data_types_is_commutative_or_raises_consistently(
    left: CoreDataType, right: CoreDataType
) -> None:
    """Test `promote_core_data_types` agrees under argument order or raises both ways.

    Oracle: swapping the argument order is a second, independent call to
    the same total-or-raising operation.
    """
    assert_commutative_or_consistent_raise(
        promote_core_data_types, left, right, lambda a, b: a == b
    )


@given(
    build_core_data_type_strategy(),
    build_core_data_type_strategy(),
    build_core_data_type_strategy(),
)
def test_promote_core_data_types_is_associative_or_raises_consistently(
    first: CoreDataType, second: CoreDataType, third: CoreDataType
) -> None:
    """Test left- and right-associated `promote_core_data_types` calls agree."""
    assert_associative_or_consistent_raise(
        promote_core_data_types, first, second, third, lambda a, b: a == b
    )


@given(build_core_data_type_strategy())
def test_promote_core_data_types_is_idempotent(core_data_type: CoreDataType) -> None:
    """Test `promote_core_data_types(x, x)` equals `x` for every core data type.

    `BOOL` promotes only with itself and every other core data type
    belongs to exactly one of the integer or float/complex promotion
    lattices, so no value is ever excluded from this law.
    """
    assert promote_core_data_types(core_data_type, core_data_type) == core_data_type


# =============================================================================
# P33: `promote_primitive_data_types`
# =============================================================================


@given(build_primitive_data_type_strategy(), build_primitive_data_type_strategy())
def test_promote_primitive_data_types_is_commutative_or_raises_consistently(
    left: PrimitiveDataType, right: PrimitiveDataType
) -> None:
    """Test `promote_primitive_data_types` agrees under either argument order.

    Either both orders raise `FhYCoreTypeError`, or both succeed with
    structurally equivalent results.
    """
    assert_commutative_or_consistent_raise(
        promote_primitive_data_types,
        left,
        right,
        _primitive_data_types_are_equivalent,
    )


@given(
    build_primitive_data_type_strategy(),
    build_primitive_data_type_strategy(),
    build_primitive_data_type_strategy(),
)
def test_promote_primitive_data_types_is_associative_or_raises_consistently(
    first: PrimitiveDataType, second: PrimitiveDataType, third: PrimitiveDataType
) -> None:
    """Test left- and right-associated `promote_primitive_data_types` calls agree."""
    assert_associative_or_consistent_raise(
        promote_primitive_data_types,
        first,
        second,
        third,
        _primitive_data_types_are_equivalent,
    )


@given(build_primitive_data_type_strategy())
def test_promote_primitive_data_types_is_idempotent(
    primitive_data_type: PrimitiveDataType,
) -> None:
    """Test `promote_primitive_data_types(x, x)` is structurally equivalent to `x`."""
    result = promote_primitive_data_types(primitive_data_type, primitive_data_type)
    assert result.is_structurally_equivalent(primitive_data_type)


# =============================================================================
# P33: `promote_type_qualifiers`
# =============================================================================


@given(build_type_qualifier_strategy(), build_type_qualifier_strategy())
def test_promote_type_qualifiers_is_commutative(
    left: TypeQualifier, right: TypeQualifier
) -> None:
    """Test `promote_type_qualifiers` never raises and is commutative."""
    assert promote_type_qualifiers(left, right) == promote_type_qualifiers(right, left)


@given(
    build_type_qualifier_strategy(),
    build_type_qualifier_strategy(),
    build_type_qualifier_strategy(),
)
def test_promote_type_qualifiers_is_associative(
    first: TypeQualifier, second: TypeQualifier, third: TypeQualifier
) -> None:
    """Test `promote_type_qualifiers` associates the same both ways."""
    left_associated = promote_type_qualifiers(
        promote_type_qualifiers(first, second), third
    )
    right_associated = promote_type_qualifiers(
        first, promote_type_qualifiers(second, third)
    )
    assert left_associated == right_associated


@given(st.sampled_from(_IDEMPOTENT_TYPE_QUALIFIERS))
def test_promote_type_qualifiers_is_idempotent_for_param_and_temp(
    type_qualifier: TypeQualifier,
) -> None:
    """Test `promote_type_qualifiers(x, x)` equals `x` for `PARAM` and `TEMP`.

    These are the only two qualifiers a promotion can ever return: the
    documented rule yields `PARAM` only when both operands are `PARAM` and
    `TEMP` otherwise (pinned by the example table in `test_core.py`), so
    self-promotion of any other qualifier is `TEMP` by design rather than
    the identity.
    """
    assert promote_type_qualifiers(type_qualifier, type_qualifier) == type_qualifier


# =============================================================================
# P35: `is_structurally_equivalent` on `Type` trees
# =============================================================================


@given(draw_template_free_type(_POOL))
def test_is_structurally_equivalent_is_reflexive(type_: Type) -> None:
    """Test a drawn `Type` is structurally equivalent to itself.

    Oracle: the reflexivity law every equivalence relation must satisfy.
    """
    assert is_structurally_equivalent(type_, type_)
    assert type_.is_structurally_equivalent(type_)


@given(draw_template_free_type(_POOL), draw_template_free_type(_POOL))
def test_is_structurally_equivalent_is_symmetric(left: Type, right: Type) -> None:
    """Test `is_structurally_equivalent(a, b)` equals the swapped-argument call.

    Oracle: the symmetry law every equivalence relation must satisfy.
    """
    assert is_structurally_equivalent(left, right) == is_structurally_equivalent(
        right, left
    )


@given(draw_template_free_type(_POOL))
def test_dict_round_trip_preserves_structural_equivalence(type_: Type) -> None:
    """Test a DICT `serialize_to_dict`/`deserialize_from_dict` round trip is equivalent.

    Oracle: the serialize/deserialize inverse pair (entry points also
    exercised in `tests/types/test_serialization.py`).
    """
    serialized = type_.serialize_to_dict()
    deserialized = type(type_).deserialize_from_dict(serialized)
    assert is_structurally_equivalent(type_, deserialized)
