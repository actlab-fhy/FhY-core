"""Property tests for the basic trait laws: equality, ordering, and freezing.

Covers three algebraic-law properties (P29): hash-equality consistency and
the reflexive/symmetric laws for an ``EqualMixin``-shaped value
(``SerializableEqualHashable``); the total-order laws (trichotomy,
transitivity, sort consistency) for a hand-rolled ``OrderableMixin``
implementor; and the ``FrozenMixin`` contract that every attribute write on a
frozen instance raises ``FrozenMutationError``, checked over every
serializable family that mixes in ``FrozenMixin``.
"""

import dataclasses

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.traits import FrozenMixin, FrozenMutationError, OrderableMixin
from fhy_core.utils.override import override

from .conftest import SerializableEqualHashable
from .strategies.serializables import SerializableCase, draw_serializable_case

pytestmark = pytest.mark.property

_SMALL_INT: st.SearchStrategy[int] = st.integers(min_value=-1000, max_value=1000)


# =============================================================================
# Equality: hash-consistency, reflexivity, symmetry
# =============================================================================


@given(left_value=_SMALL_INT, right_value=_SMALL_INT)
def test_equal_mixin_value_satisfies_hash_consistency_and_relational_laws(
    left_value: int, right_value: int
) -> None:
    """Test SerializableEqualHashable satisfies == laws and hash-equality consistency.

    Oracle: the Python data-model equality contract -- ``==`` is reflexive
    and symmetric, and ``a == b`` implies ``hash(a) == hash(b)`` -- checked
    against ``tests/conftest.py::SerializableEqualHashable``, a plain
    integer-valued equality double.
    """
    left = SerializableEqualHashable(left_value)
    right = SerializableEqualHashable(right_value)

    assert left == left  # noqa: PLR0124 - reflexivity, not a typo
    assert (left == right) == (right == left)
    if left == right:
        assert hash(left) == hash(right)


# =============================================================================
# Ordering: trichotomy, transitivity, sort consistency
# =============================================================================


class _OrderableInt(OrderableMixin):
    """Minimal hand-rolled total-ordering value, backed by a plain ``int``.

    Mirrors ``tests/test_basic_traits.py::_ManualOrderableValue``, plus a
    value-based ``__eq__`` (that fixture relies on identity equality, which
    would break trichotomy for two distinct instances holding the same int).
    """

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _OrderableInt):
            return NotImplemented
        return self.value < other.value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OrderableInt) and self.value == other.value

    @override
    def __hash__(self) -> int:
        return hash(self.value)


@given(first=_SMALL_INT, second=_SMALL_INT, third=_SMALL_INT)
def test_orderable_mixin_value_satisfies_total_order_laws(
    first: int, second: int, third: int
) -> None:
    """Test an OrderableMixin implementor satisfies trichotomy, transitivity, sorting.

    Oracle: the total-order contract -- for any pair exactly one of ``<``,
    ``==``, and the reverse ``<`` holds; ``<`` is transitive; and sorting by
    ``<`` agrees with sorting the underlying ints -- checked against
    ``_OrderableInt``, whose ``__lt__``/``__eq__`` mirror plain ``int``
    comparison.
    """
    left, middle, right = (
        _OrderableInt(first),
        _OrderableInt(second),
        _OrderableInt(third),
    )

    for one, other in ((left, middle), (middle, right), (left, right)):
        assert sum((one < other, one == other, other < one)) == 1

    if left < middle < right:
        assert left < right

    sorted_values = [item.value for item in sorted((left, middle, right))]
    assert sorted_values == sorted((first, second, third))


# =============================================================================
# Frozen: every attribute write on a frozen instance raises
# =============================================================================

_NEW_ATTRIBUTE_NAME = "_a_brand_new_attribute_this_class_never_declared"


@given(case=draw_serializable_case())
def test_frozen_mixin_instances_reject_every_attribute_write(
    case: SerializableCase,
) -> None:
    """Test setattr on a FrozenMixin instance always raises FrozenMutationError.

    Oracle: the ``FrozenMixin`` contract -- every attribute write on a
    frozen instance raises, whether the attribute is a declared dataclass
    field (re-set to its own current value) or an entirely new name.
    Exercised over every generated serializable family that mixes in
    ``FrozenMixin``; not every family is a ``@dataclass`` (the ``types``
    family hand-writes its own ``__init__``), so the field-name loop runs
    only when ``dataclasses.fields`` applies.
    """
    instance = case.instance
    if not isinstance(instance, FrozenMixin):
        return

    if dataclasses.is_dataclass(instance):
        for field_definition in dataclasses.fields(instance):
            with pytest.raises(FrozenMutationError):
                setattr(
                    instance,
                    field_definition.name,
                    getattr(instance, field_definition.name),
                )

    with pytest.raises(FrozenMutationError):
        setattr(instance, _NEW_ATTRIBUTE_NAME, object())
