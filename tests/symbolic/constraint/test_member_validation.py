"""Member validation for the set-constraint family.

Validation is exercised through the public `InSetConstraint` and
`NotInSetConstraint` constructors. Tests are parametrized across both
kinds.
"""

from collections.abc import Callable
from typing import Any

import pytest

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
    with pytest.raises(ConstraintError):
        factory(mock_identifier("x", 0), values)


@pytest.mark.parametrize("factory", SET_KINDS)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param({HashableNotSerializable(1)}, id="hashable_but_not_serializable"),
        pytest.param([{"a": 1}], id="unhashable_dict"),
        pytest.param([UnhashableTuple((1, 2))], id="tuple_subclass_with_disabled_hash"),
        pytest.param(
            [SerializableHashRaises()], id="serializable_with_hash_that_raises"
        ),
    ],
)
def test_set_constraint_rejects_unsupported_member(
    factory: SetConstraintFactory, values: Any
) -> None:
    """Test member must be a primitive, hashable serializable, or container."""
    with pytest.raises(ConstraintError):
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

    outcome = constraint.is_satisfied_with_bindings({x: nested_member})

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
    with pytest.raises(ConstraintError):
        factory(mock_identifier("x", 0), [value])


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_unhashable_after_validation_error_names_offending_value(
    factory: SetConstraintFactory,
) -> None:
    """Test the post-validation hash error embeds the offending value."""
    bad = SerializableHashRaises()

    with pytest.raises(ConstraintError) as exc_info:
        factory(mock_identifier("x", 0), [bad])

    assert "SerializableHashRaises" in str(exc_info.value) or repr(bad) in str(
        exc_info.value
    )
