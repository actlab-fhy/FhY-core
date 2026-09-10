"""Errors shared by the expression registry, body-validation, and runtime passes.

Defining the errors in a dependency-free module lets the registry, the
body-validation pass, and the evaluator/z3 helpers raise the same typed
exceptions without forming import cycles. The registry and pass modules
re-export the names that they raise so external callers can import them
from ``fhy_core.symbolic.expression.registry`` and ``fhy_core.symbolic.expression``.
"""

__all__ = [
    "EntryLookupError",
    "EntryRegistrationError",
    "NativeResultSortError",
    "NonBooleanLogicalOperandError",
    "NonFiniteCastError",
    "PartialPiecewiseError",
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
class NonBooleanLogicalOperandError(TypeError):
    """Raised when a logical connective is applied to a non-Boolean operand.

    ``LOGICAL_AND``, ``LOGICAL_OR``, and ``LOGICAL_NOT`` denote Boolean
    connectives, so an operand that provably denotes a number -- a
    non-``bool`` literal, an arithmetic node, or a piecewise whose every
    branch value is numeric -- has no meaning under them. Neither
    symbolic backend refuses such an operand on its own terms: SymPy's
    ``&``/``|`` are *bitwise* on ``sympy.Integer`` and its ``Not``
    coerces by truthiness, while Z3 reports the sort mismatch as a
    backend ``z3.z3types.Z3Exception``. Both bridges screen the
    expression before lowering and raise this error instead, so one
    ill-typed expression is refused the same way whichever bridge a
    caller reaches.

    A ``TypeError`` because such an expression is ill-typed rather than
    merely hard to settle. Contrast :class:`UndecidableError`, which
    reports a well-typed query the solver declined to decide and which a
    different solver configuration might decide; no configuration gives
    ``logical_and(2, 4)`` a meaning.
    """


@register_error
class NonFiniteCastError(ValueError):
    """Raised when a non-finite NumPy result reaches an integer-sorted cast.

    The NumPy evaluator casts a native call's result to the dtype of its
    declared result sort. A ``nan``/``inf`` value has no faithful
    ``BOOL``-, ``NAT``-, or ``INT``-sorted representation, so casting it
    would silently produce a platform-defined sentinel; the evaluator
    checks for non-finite values before the cast and raises this error
    instead.
    """


@register_error
class PartialPiecewiseError(ValueError):
    """Raised when a lifted ``sympy.Piecewise`` does not cover its domain.

    ``PiecewiseExpression`` is a total function: it always denotes
    ``otherwise`` when no case condition holds. Lifting a
    ``sympy.Piecewise`` whose final branch condition is not
    ``sympy.true`` -- including the single-branch shape, whose only
    surviving construction has a non-``True`` condition -- has no
    faithful representation in the IR, since treating the final
    branch's value as ``otherwise`` would silently cover the region the
    original condition excluded.
    """


@register_error
class StringLiteralPrecisionError(ValueError):
    """Raised when a float-grammar string literal cannot be coerced losslessly.

    ``LiteralExpression`` preserves float-grammar string literals to keep
    their exact decimal value. Coercing such a literal to a binary
    ``float`` -- which handing off to Python or NumPy would require --
    discards that precision, so the native folding evaluator and the
    NumPy evaluator both refuse the coercion and raise this error. Use a
    ``float`` literal when binary-float semantics are intended.
    """


@register_error
class UndecidableError(RuntimeError):
    """Raised by strict Z3 companions when a query cannot be decided.

    The lenient ``holds_for_all_free_assignments`` / ``does_expression_imply``
    functions return ``None`` in this case so callers can choose their
    own conservative interpretation; ``assert_holds_for_all_free_assignments``
    and ``assert_expression_implies`` raise this error instead.

    Carries a machine-readable ``reason`` alongside the human-readable
    message: Z3's own ``reason_unknown()`` text (for example
    ``"timeout"``) when the solver ran and gave up, or a fixed marker
    when the expression was refused by the solver seam's hazard screen
    before Z3 was ever consulted. A caller can use ``reason`` to tell a
    retryable timeout apart from a query that is undecidable in
    principle, which a larger ``timeout_milliseconds`` will not change.
    """

    _reason: str

    def __init__(self, message: str = "", *, reason: str = "") -> None:
        super().__init__(message)
        self._reason = reason

    @property
    def reason(self) -> str:
        """Machine-readable reason the query could not be decided."""
        return self._reason


@register_error
class UnboundVariableError(ValueError):
    """Raised when a free identifier reaches NumPy evaluation with no value.

    :func:`~fhy_core.symbolic.expression.evaluate_expression_with_numpy` requires
    every free identifier to be either bound in the caller's environment
    or to match a registered native constant. An identifier that is
    neither raises this error, since the NumPy evaluator cannot produce
    a value for an unbound variable.

    This is a ``ValueError`` (a caller precondition violation), not a
    ``KeyError``: the identifier-not-bound message must render cleanly in
    a traceback, and a bare ``except KeyError`` must not silently swallow
    this programming error. Contrast :class:`EntryLookupError`, which does
    subclass ``KeyError`` because it models a literal registry-dict miss.
    """


@register_error
class UnsupportedNumpyLoweringError(RuntimeError):
    """Raised when an expression node has no NumPy lowering.

    Surfaced by :func:`~fhy_core.symbolic.expression.evaluate_expression_with_numpy`
    when a node cannot be evaluated with NumPy. Current cases:

    - The ``erf`` native function (and therefore ``gelu``, whose body
      calls ``erf``): NumPy has no vectorized ``erf`` ufunc.
    - Any registered :class:`NativeFunction` the evaluator has no ufunc
      mapping for (for example, a caller-registered native).
    """
