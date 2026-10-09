"""Tests covering wrapped-leaf value-semantics admissibility paths.

`create_ordinal_param`, `create_categorical_param`, and `create_permutation_param`
accept either primitive values or `Serializable` values that satisfy the
appropriate value-semantics contract. These tests exercise the wrapped-leaf
branches in `_is_categorical_value`, `_is_ordinal_value`, and
`_is_permutation_member_value` through the public constructors only.
"""

import pytest

from fhy_core.symbolic.param import (
    create_categorical_param,
    create_ordinal_param,
    create_permutation_param,
)

from .conftest import (
    SerializableEqualHashable,
    SerializableNonComparable,
    SerializableOrderableInherited,
    SerializableOrderableSelf,
    SerializableOrderableTrait,
)

# =============================================================================
# Equal-value semantics through `create_categorical_param`
# =============================================================================


def test_categorical_param_accepts_serializable_with_equality_and_hash() -> None:
    """Test `create_categorical_param` accepts a `Serializable` with eq and hash.

    The value type overrides ``__eq__`` and keeps a non-``None`` ``__hash__``,
    which together satisfy the equal-value-semantics contract.
    """
    a = SerializableEqualHashable(1)
    b = SerializableEqualHashable(2)

    param = create_categorical_param([a, b])  # type: ignore[type-var]  # test: bespoke `Serializable` value

    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_categorical_param_admissibility_does_not_raise_on_serializable_value() -> None:
    """Test `create_categorical_param` admissibility evaluates without ``TypeError``.

    Admissibility of a `Serializable` categorical value is computed without
    raising, exercising the equal-semantics path end to end.
    """
    value = SerializableEqualHashable(1)

    param = create_categorical_param([value])  # type: ignore[type-var]  # test: bespoke `Serializable` value

    assert param.is_value_admissible(value)


# =============================================================================
# Orderable-value semantics through `create_ordinal_param`
# =============================================================================


def test_ordinal_param_accepts_serializable_with_lt_inherited_from_parent() -> None:
    """Test `create_ordinal_param` accepts a `Serializable` with inherited ``__lt__``.

    The value type inherits ``__lt__`` from a parent class rather than defining
    it on the leaf, so ordering semantics are discovered up the MRO.
    """
    a = SerializableOrderableInherited(1)
    b = SerializableOrderableInherited(2)

    param = create_ordinal_param([a, b])  # type: ignore[type-var]  # test: bespoke `Serializable` value

    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_ordinal_param_accepts_serializable_with_lt_on_self_class() -> None:
    """Test `create_ordinal_param` accepts `Serializable` with ``__lt__`` on the class.

    The value type defines ``__lt__`` directly on the leaf class, so ordering
    semantics are discovered without walking to a parent.
    """
    a = SerializableOrderableSelf(1)
    b = SerializableOrderableSelf(2)

    param = create_ordinal_param([a, b])  # type: ignore[type-var]  # test: bespoke `Serializable` value

    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_ordinal_param_accepts_value_satisfying_orderable_protocol() -> None:
    """Test `create_ordinal_param` accepts values satisfying the `Orderable` protocol.

    A value implementing the `Orderable` runtime protocol is admitted via its
    ``supports_ordering`` flag rather than through MRO inspection of ``__lt__``.
    """
    a = SerializableOrderableTrait(1)
    b = SerializableOrderableTrait(2)

    param = create_ordinal_param([a, b])

    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_ordinal_param_rejects_serializable_without_orderable_semantics() -> None:
    """Test `create_ordinal_param` rejects a `Serializable` that lacks ordering.

    A `Serializable` value with neither ``__lt__`` nor `Orderable` support is
    rejected at construction time.
    """
    with pytest.raises(TypeError, match="orderable semantics"):
        create_ordinal_param(  # type: ignore[type-var]  # test: invalid input
            [SerializableNonComparable(1), SerializableNonComparable(2)]
        )


# =============================================================================
# Equal-value semantics through `create_permutation_param`
# =============================================================================


def test_perm_param_rejects_serializable_without_equal_semantics() -> None:
    """Test `create_permutation_param` rejects a `Serializable` lacking equal semantics.

    A `Serializable` value with identity-based equality (no overridden
    ``__eq__``) is rejected at construction time.
    """
    with pytest.raises(TypeError, match="equal semantics"):
        create_permutation_param(  # type: ignore[type-var]  # test: invalid input
            [SerializableNonComparable(1), SerializableNonComparable(2)]
        )
