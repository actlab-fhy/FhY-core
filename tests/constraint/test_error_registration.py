"""Tests covering `ConstraintError` identity."""

from fhy_core.constraint import ConstraintError


def test_constraint_error_is_value_error_subclass() -> None:
    """Test `ConstraintError` remains a `ValueError` for backward compatibility."""
    assert issubclass(ConstraintError, ValueError)
