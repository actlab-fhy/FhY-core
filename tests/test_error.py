"""Tests for `fhy_core.error`."""

import pytest

from fhy_core.error import get_registered_errors, register_error
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.constraint import ConstraintError
from fhy_core.symbolic.param import ParamError
from fhy_core.types import FhYCoreTypeError

# =============================================================================
# get_registered_errors
# =============================================================================


def test_get_registered_errors_returns_a_mapping_proxy() -> None:
    """Test ``get_registered_errors`` returns a read-only ``MappingProxyType``.

    The view must reject mutation so callers cannot accidentally corrupt
    the registry through it.
    """
    registry = get_registered_errors()
    with pytest.raises(TypeError):
        registry[ValueError] = "I should not be allowed"  # type: ignore[index]


def test_known_registered_errors_appear_in_the_registry() -> None:
    """Test a sample of known ``@register_error``-decorated classes are present.

    Pins the integration between ``register_error`` and the read API: any
    refactor that quietly stops adding entries to the underlying dict
    will fail this assertion.
    """
    registry = get_registered_errors()
    for exception_class in (
        ConstraintError,
        FhYCoreTypeError,
        ParamError,
        PassExecutionError,
    ):
        assert exception_class in registry


def test_register_error_adds_a_fresh_class_to_the_registry() -> None:
    """Test ``@register_error`` makes a freshly-decorated exception discoverable.

    Pins the registration side effect; the docstring of the class is the
    stored value (or its name when the docstring is empty).
    """

    @register_error
    class _FreshlyRegisteredError(Exception):
        """A docstring used as the registry value."""

    registry = get_registered_errors()
    assert _FreshlyRegisteredError in registry
    assert (
        registry[_FreshlyRegisteredError] == "A docstring used as the registry value."
    )


def test_register_error_falls_back_to_class_name_for_undocumented_class() -> None:
    """Test ``@register_error`` stores the class name when no docstring is set.

    Pins the ``error.__doc__ or error.__name__`` fallback so future
    decorator refactors can't silently drop the class-name path.
    """

    @register_error
    class _UndocumentedError(Exception):
        pass

    registry = get_registered_errors()
    assert registry[_UndocumentedError] == "_UndocumentedError"
