"""Errors shared by the expression registry, body-validation, and runtime passes.

Defining the errors in a dependency-free module lets the registry, the
body-validation pass, and the evaluator/z3 helpers raise the same typed
exceptions without forming import cycles. The registry and pass modules
re-export the names that they raise so external callers can import them
from ``fhy_core.expression.registry`` and ``fhy_core.expression``.
"""

__all__ = [
    "EntryLookupError",
    "EntryRegistrationError",
    "NativeResultSortError",
    "StringLiteralPrecisionError",
    "UnboundVariableError",
    "UndecidableError",
    "UnsupportedNumpyLoweringError",
]

from fhy_core.error import register_error


@register_error
class EntryRegistrationError(RuntimeError):
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
    - The implementation of a ``NativeFunction`` has an arity that
      cannot accept the declared ``parameter_sorts`` length.
    """


@register_error
class EntryLookupError(KeyError):
    """Raised when a name is requested but not registered."""


@register_error
class NativeResultSortError(RuntimeError):
    """Raised when a native function returns a value of an incompatible sort.

    The evaluator validates the implementation's actual return value
    against the declared ``result_sort`` after invocation. A mismatch
    indicates the native implementation's contract is broken: the
    declared sort promised one runtime type family, but the
    implementation produced another.
    """


@register_error
class StringLiteralPrecisionError(ValueError):
    """Raised when a float-grammar string literal cannot be coerced losslessly.

    ``LiteralExpression`` preserves float-grammar string literals to keep
    their exact decimal value. Coercing such a literal to a binary
    ``float`` -- as the native folding evaluator and the NumPy evaluator
    must to hand off to Python or NumPy -- would discard that precision,
    so both refuse the coercion and raise this error. Use a ``float``
    literal when binary-float semantics are intended.
    """


@register_error
class UndecidableError(RuntimeError):
    """Raised by strict Z3 companions when the solver returns ``unknown``.

    The lenient ``holds_for_all_free_assignments`` / ``does_expression_imply``
    functions return ``None`` in this case so callers can choose their
    own conservative interpretation; ``assert_holds_for_all_free_assignments``
    and ``assert_expression_implies`` raise this error instead.
    """


@register_error
class UnboundVariableError(ValueError):
    """Raised when a free identifier reaches NumPy evaluation with no value.

    :func:`~fhy_core.expression.evaluate_expression_with_numpy` requires
    every free identifier to be either bound in the caller's environment
    or to match a registered native constant. An identifier that is
    neither raises this error, since the NumPy evaluator cannot produce
    a value for an unbound variable.
    """


@register_error
class UnsupportedNumpyLoweringError(TypeError):
    """Raised when an expression node has no NumPy lowering.

    Surfaced by :func:`~fhy_core.expression.evaluate_expression_with_numpy`
    when a node cannot be evaluated with NumPy. Current cases:

    - The ``erf`` native function (and therefore ``gelu``, whose body
      calls ``erf``): NumPy has no vectorized ``erf`` ufunc.
    - Any registered :class:`NativeFunction` outside the built-in math
      set that the evaluator maps to NumPy ufuncs.
    """
