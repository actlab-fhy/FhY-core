"""Errors shared by the expression registry and its body-validation pass.

Defining the errors in a dependency-free module lets both
:mod:`fhy_core.expression.registry` and
:mod:`fhy_core.expression.passes.body_type_checker` raise the same
typed exceptions without forming an import cycle. The registry
re-exports both names so external callers continue to import them from
``fhy_core.expression.registry`` (and from ``fhy_core.expression``).
"""

__all__ = [
    "FunctionLookupError",
    "FunctionRegistrationError",
]

from fhy_core.error import register_error


@register_error
class FunctionRegistrationError(RuntimeError):
    """Raised when a registry entry cannot be registered.

    Possible causes:

    - A name is already in use by any registered entry kind.
    - A function body references free identifiers that are not declared
      as parameters and do not match a registered constant.
    - ``parameter_sorts`` length does not equal ``parameters`` length.
    - A function body synthesizes a type whose core data type is not
      compatible with the declared ``result_sort``.
    - A constant value is not compatible with its declared sort per
      :func:`is_python_value_compatible_with_sort`.
    """


@register_error
class FunctionLookupError(KeyError):
    """Raised when a name is requested but not registered."""
