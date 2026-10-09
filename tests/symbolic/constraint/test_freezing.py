"""Frozen contract for the constraint family."""

from collections.abc import Callable

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import Constraint
from fhy_core.traits import Frozen, FrozenMutationError

from .conftest import ALL_KINDS, mock_identifier

ConstraintFactory = Callable[[Identifier], Constraint]


@pytest.mark.parametrize("factory", ALL_KINDS)
def test_constraint_implements_frozen_protocol_and_is_frozen(
    factory: ConstraintFactory,
) -> None:
    """Test every constraint kind satisfies `Frozen` and reports ``is_frozen``."""
    constraint = factory(mock_identifier("x", 0))

    assert isinstance(constraint, Frozen)
    assert constraint.is_frozen


@pytest.mark.parametrize("factory", ALL_KINDS)
def test_constraint_rejects_arbitrary_attribute_assignment(
    factory: ConstraintFactory,
) -> None:
    """Test setattr on a frozen constraint raises ``FrozenMutationError``."""
    constraint = factory(mock_identifier("x", 0))

    with pytest.raises(FrozenMutationError):
        constraint.arbitrary_probe = "mutation"
