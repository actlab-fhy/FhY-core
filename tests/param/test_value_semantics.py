"""Tests covering wrapped-leaf value-semantics admissibility paths.

`OrdinalParam`, `CategoricalParam`, and `PermParam` accept either primitive
values or `Serializable` values that satisfy the appropriate value-semantics
contract. These tests exercise the wrapped-leaf branches in
`_is_categorical_value`, `_is_ordinal_value`, and `_is_permutation_member_value`
through the public constructors only.
"""

import pytest

from fhy_core.param import CategoricalParam, OrdinalParam, PermParam

from .conftest import (
    SerializableEqualHashable,
    SerializableNonComparable,
    SerializableOrderableInherited,
    SerializableOrderableSelf,
    SerializableOrderableTrait,
)

# =============================================================================
# Equal-value semantics through `CategoricalParam`
# =============================================================================


def test_categorical_param_accepts_serializable_with_equality_and_hash() -> None:
    """Test `CategoricalParam` accepts a `Serializable` with eq and hash dunders.

    Pins down both halves of the equal-semantics check: the value type must
    have an overridden ``__eq__`` *and* a non-``None`` ``__hash__``. Mutating
    either probe weakens admissibility for this kind of value.
    """
    a = SerializableEqualHashable(1)
    b = SerializableEqualHashable(2)
    param: CategoricalParam[SerializableEqualHashable] = CategoricalParam([a, b])
    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_categorical_param_admissibility_does_not_raise_on_serializable_value() -> None:
    """Test `CategoricalParam` admissibility evaluates without ``TypeError``.

    The wrapped-leaf equal-semantics check probes ``__eq__`` via ``is not``;
    swapping that for an order operator (``<`` / ``<=`` / ``>`` / ``>=``)
    would raise ``TypeError`` when comparing slot-wrapper objects. This test
    pins the current admissibility path as exception-free.
    """
    value = SerializableEqualHashable(1)
    param: CategoricalParam[SerializableEqualHashable] = CategoricalParam([value])
    assert param.is_value_admissible(value)


# =============================================================================
# Orderable-value semantics through `OrdinalParam`
# =============================================================================


def test_ordinal_param_accepts_serializable_with_lt_inherited_from_parent() -> None:
    """Test `OrdinalParam` accepts a `Serializable` whose ``__lt__`` is inherited.

    The MRO walk in `_supports_orderable_value_semantics` must look past
    ``mro[0]`` to discover ordering. This subclass leaves ``__lt__`` to its
    parent, so any mutation that restricts the walk to the leaf class fails.
    """
    a = SerializableOrderableInherited(1)
    b = SerializableOrderableInherited(2)
    param: OrdinalParam[SerializableOrderableInherited] = OrdinalParam([a, b])
    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_ordinal_param_accepts_serializable_with_lt_on_self_class() -> None:
    """Test `OrdinalParam` accepts a `Serializable` with ``__lt__`` on the class itself.

    Pins down a non-empty MRO slice in `_supports_orderable_value_semantics`:
    mutations that empty the slice (e.g., ``[:0]`` / ``[:-0]`` / ``[:not 1]``)
    would always return ``False`` and reject this value at construction time.
    """
    a = SerializableOrderableSelf(1)
    b = SerializableOrderableSelf(2)
    param: OrdinalParam[SerializableOrderableSelf] = OrdinalParam([a, b])
    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_ordinal_param_accepts_value_satisfying_orderable_protocol() -> None:
    """Test `OrdinalParam` accepts a value that satisfies the `Orderable` protocol.

    Pins down the ``isinstance(value, Orderable)`` early-return in
    `_supports_orderable_value_semantics`, which short-circuits to
    ``value.supports_ordering`` without walking the MRO for ``__lt__``.
    """
    a = SerializableOrderableTrait(1)
    b = SerializableOrderableTrait(2)
    param: OrdinalParam[SerializableOrderableTrait] = OrdinalParam([a, b])
    assert param.is_value_admissible(a)
    assert param.is_value_admissible(b)


def test_ordinal_param_rejects_serializable_without_orderable_semantics() -> None:
    """Test `OrdinalParam` rejects a `Serializable` that lacks ordering.

    The wrapped-leaf branch combines ``isinstance(value, Serializable)`` with
    `_supports_orderable_value_semantics(value)` via ``and``; mutating that
    junction to ``or`` would admit a non-orderable `Serializable`.
    """
    with pytest.raises(TypeError, match="orderable semantics"):
        OrdinalParam(  # type: ignore[type-var]  # test: invalid input
            [SerializableNonComparable(1), SerializableNonComparable(2)]
        )


# =============================================================================
# Equal-value semantics through `PermParam`
# =============================================================================


def test_perm_param_rejects_serializable_without_equal_semantics() -> None:
    """Test `PermParam` rejects a `Serializable` that lacks equal semantics.

    The wrapped-leaf branch combines ``isinstance(value, Serializable)`` with
    `_supports_equal_value_semantics(value)` via ``and``; mutating that
    junction to ``or`` would admit a non-equal `Serializable`.
    """
    with pytest.raises(TypeError, match="equal semantics"):
        PermParam(  # type: ignore[type-var]  # test: invalid input
            [SerializableNonComparable(1), SerializableNonComparable(2)]
        )
