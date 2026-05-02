"""Frozen contract for the constraint family.

Every constraint instance must be frozen on construction, advertise
``is_frozen``, and reject attribute mutation. The ``deep`` argument to
``freeze`` has no observable effect on the public interface for the
current attribute shapes (``Expression`` is itself a frozen dataclass;
the value-set attributes are already ``frozenset`` instances of
immutable atoms), so no kill test exists for the
``deep=True -> deep=False`` mutations.
"""

from typing import Callable

import pytest

from fhy_core.constraint import Constraint
from fhy_core.identifier import Identifier
from fhy_core.trait import Frozen, FrozenMutationError

from .conftest import (
    build_equation_constraint,
    build_in_set_constraint,
    build_not_in_set_constraint,
    mock_identifier,
)

ConstraintFactory = Callable[[Identifier], Constraint]


_KIND_FACTORIES = [
    pytest.param(build_equation_constraint, id="equation"),
    pytest.param(build_in_set_constraint, id="in_set"),
    pytest.param(build_not_in_set_constraint, id="not_in_set"),
]


@pytest.mark.parametrize("factory", _KIND_FACTORIES)
def test_constraint_implements_frozen_protocol_and_is_frozen(
    factory: ConstraintFactory,
) -> None:
    """Test every constraint kind satisfies `Frozen` and reports ``is_frozen``."""
    constraint = factory(mock_identifier("x", 0))
    assert isinstance(constraint, Frozen)
    assert constraint.is_frozen


@pytest.mark.parametrize("factory", _KIND_FACTORIES)
def test_constraint_rejects_arbitrary_attribute_assignment(
    factory: ConstraintFactory,
) -> None:
    """Test setattr on a frozen constraint raises ``FrozenMutationError``."""
    constraint = factory(mock_identifier("x", 0))
    with pytest.raises(FrozenMutationError):
        constraint.arbitrary_probe = "mutation"
