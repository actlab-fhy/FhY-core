"""Tests covering `ConstraintError` identity.

The ``@register_error`` decorator records `ConstraintError` in the
private ``_COMPILER_ERRORS`` registry, but that registry has no public
reader; the registration mutation is therefore equivalent through the
public interface alone and is documented as such, not tested here.
"""

from fhy_core.constraint import ConstraintError


def test_constraint_error_is_value_error_subclass() -> None:
    """Test `ConstraintError` remains a `ValueError` for backward compatibility."""
    assert issubclass(ConstraintError, ValueError)
