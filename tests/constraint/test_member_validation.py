"""Member validation for the set-constraint family.

Validation is exercised through the public `InSetConstraint` and
`NotInSetConstraint` constructors. Tests are parametrized across both
kinds and across each rejection scenario.
"""

from typing import Any, Callable

import pytest

from fhy_core.constraint import (
    Constraint,
    ConstraintError,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.identifier import Identifier

from .conftest import (
    HashableNotSerializable,
    SerializableHashRaises,
    UnhashableTuple,
    mock_identifier,
)

SetConstraintFactory = Callable[[Identifier, Any], Constraint]

_KINDS = [
    pytest.param(InSetConstraint, id="in_set"),
    pytest.param(NotInSetConstraint, id="not_in_set"),
]


@pytest.mark.parametrize("factory", _KINDS)
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


@pytest.mark.parametrize("factory", _KINDS)
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
    """Test members must be primitive, ``Serializable+Hashable``, or a valid container.

    Anything else raises ``ConstraintError`` during construction.
    """
    with pytest.raises(ConstraintError):
        factory(mock_identifier("x", 0), values)


@pytest.mark.parametrize("factory", _KINDS)
def test_set_constraint_supports_deeply_nested_collection_members(
    factory: SetConstraintFactory,
) -> None:
    """Test the recursive validator accepts deeply nested tuple/frozenset members."""
    nested_member = (1, (2, 3), frozenset({4, 5}))
    constraint = factory(mock_identifier("x", 0), [nested_member])
    assert constraint.is_satisfied(nested_member) is (factory is InSetConstraint)


@pytest.mark.parametrize("factory", _KINDS)
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
    """Test non-allow-listed builtin types are rejected as members.

    The primitive literal allow-list is ``{str, int, float, bool}``;
    other builtin scalars/sequences (``complex``, ``bytes``,
    ``bytearray``, ``range``) are not allow-listed and are not
    ``Serializable``.
    """
    with pytest.raises(ConstraintError):
        factory(mock_identifier("x", 0), [value])
