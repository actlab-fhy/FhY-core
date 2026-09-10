"""Tests for the tri-state `ConstraintOutcome` enum."""

import pytest

from fhy_core.symbolic.constraint import ConstraintOutcome


@pytest.mark.parametrize("outcome", list(ConstraintOutcome), ids=lambda o: o.name)
def test_outcome_member_has_no_truth_value(outcome: ConstraintOutcome) -> None:
    """Test ``bool()`` on every member raises ``TypeError`` naming the tri-state."""
    with pytest.raises(TypeError, match="tri-state"):
        bool(outcome)
