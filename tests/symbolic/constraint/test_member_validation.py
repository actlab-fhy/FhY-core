"""Member validation for the set-constraint family.

Validation is exercised through the public `InSetConstraint` and
`NotInSetConstraint` constructors. Tests are parametrized across both
kinds.
"""

from collections.abc import Callable
from typing import Any

import pytest

import fhy_core.symbolic.constraint as constraint_package
from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintError,
    InSetConstraint,
)

from .conftest import (
    SET_KINDS,
    HashableNotSerializable,
    SerializableHashRaises,
    UnhashableTuple,
    mock_identifier,
)

SetConstraintFactory = Callable[[Identifier, Any], Constraint]


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param({None}, id="bare_none"),
        pytest.param([(1, None)], id="none_in_tuple"),
        pytest.param([(1, (2, None))], id="none_doubly_nested"),
        pytest.param([(1, frozenset({"ok"}), None)], id="none_alongside_frozenset"),
    ],
)
def test_set_constraint_rejects_none_member(
    factory: SetConstraintFactory, values: Any
) -> None:
    """Test ``None``, bare or nested, is rejected by member validation."""
    with pytest.raises(ConstraintError, match=r"cannot be `None`"):
        factory(mock_identifier("x", 0), values)


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "values, expected_match",
    [
        pytest.param(
            {HashableNotSerializable(1)},
            r"primitive literal",
            id="hashable_but_not_serializable",
        ),
        pytest.param([{"a": 1}], r"primitive literal", id="unhashable_dict"),
        pytest.param(
            [UnhashableTuple((1, 2))],
            r"containers must be hashable",
            id="tuple_subclass_with_disabled_hash",
        ),
        pytest.param(
            [SerializableHashRaises()],
            r"unhashable after validation",
            id="serializable_with_hash_that_raises",
        ),
    ],
)
def test_set_constraint_rejects_unsupported_member(
    factory: SetConstraintFactory, values: Any, expected_match: str
) -> None:
    """Test member must be a primitive, hashable serializable, or container."""
    with pytest.raises(ConstraintError, match=expected_match):
        factory(mock_identifier("x", 0), values)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_rejects_unhashable_outer_container_before_nested(
    factory: SetConstraintFactory,
) -> None:
    """Test outer-container hashability is checked before nested validation."""
    # The outer container is an UnhashableTuple AND contains a None.
    # The error message should mention the outer-container hashability
    # failure, not the None.
    outer = UnhashableTuple((None,))

    with pytest.raises(ConstraintError, match=r"(?i)hashable"):
        factory(mock_identifier("x", 0), [outer])


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_supports_deeply_nested_collection_members(
    factory: SetConstraintFactory,
) -> None:
    """Test the recursive validator accepts deeply nested tuple/frozenset members."""
    x = mock_identifier("x", 0)
    nested_member = (1, (2, 3), frozenset({4, 5}))
    constraint = factory(x, [nested_member])

    outcome = constraint.is_satisfied_with_bindings(
        {x: nested_member}  # type: ignore[dict-item]  # test: nested tuple/frozenset member off-union
    )

    assert outcome is (factory is InSetConstraint)


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1 + 2j, id="complex"),
        pytest.param(b"abc", id="bytes"),
        pytest.param(bytearray(b"abc"), id="bytearray"),
        pytest.param(range(3), id="range"),
    ],
)
def test_set_constraint_rejects_non_primitive_builtin_types(
    factory: SetConstraintFactory, value: Any
) -> None:
    """Test non-allow-listed builtin types are rejected as members."""
    with pytest.raises(ConstraintError, match=r"primitive literal"):
        factory(mock_identifier("x", 0), [value])


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_unhashable_after_validation_error_names_offending_value(
    factory: SetConstraintFactory,
) -> None:
    """Test the post-validation hash error embeds the offending value."""
    bad = SerializableHashRaises()

    with pytest.raises(ConstraintError, match="SerializableHashRaises"):
        factory(mock_identifier("x", 0), [bad])


# =============================================================================
# `MemberCollection` rejects `Mapping` (a one-character typo away from a set)
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param({1: "a", 2: "b"}, id="dict"),
        pytest.param({}, id="empty_dict"),
    ],
)
def test_set_constraint_rejects_mapping_as_member_collection(
    factory: SetConstraintFactory, values: Any
) -> None:
    """Test a `Mapping` is rejected instead of silently keeping only its keys.

    ``dict`` structurally satisfies ``MemberCollection`` (``__iter__``,
    ``__len__``, ``__contains__``), so ``{1: 2}`` and ``{1, 2}`` both
    type-check for the same constructor argument despite meaning
    something entirely different: iterating a dict yields only its keys,
    silently discarding its values.
    """
    with pytest.raises(ConstraintError, match=r"(?i)discard"):
        factory(mock_identifier("x", 0), values)


# =============================================================================
# `MemberCollection` public export
# =============================================================================


def test_member_collection_is_exported_from_the_package() -> None:
    """Test `MemberCollection` is part of the constraint package's public API."""
    assert "MemberCollection" in constraint_package.__all__
