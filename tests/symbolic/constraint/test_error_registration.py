"""Tests covering `ConstraintError` identity."""

from fhy_core.symbolic.constraint import ConstraintError


def test_constraint_error_is_value_error_subclass() -> None:
    """Test `ConstraintError` is a `ValueError` subclass."""
    assert issubclass(ConstraintError, ValueError)
