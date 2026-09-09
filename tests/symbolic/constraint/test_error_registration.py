"""Tests covering `ConstraintError` and `MissingSymbolTypeError` identity."""

from fhy_core.symbolic.constraint import ConstraintError, MissingSymbolTypeError


def test_constraint_error_is_value_error_subclass() -> None:
    """Test `ConstraintError` is a `ValueError` subclass."""
    assert issubclass(ConstraintError, ValueError)


def test_missing_symbol_type_error_is_value_error_subclass() -> None:
    """Test `MissingSymbolTypeError` is a `ValueError` subclass, not `KeyError`."""
    assert issubclass(MissingSymbolTypeError, ValueError)
    assert not issubclass(MissingSymbolTypeError, KeyError)
