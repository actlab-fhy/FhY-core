"""The SymPy classes and hooks of fhy-core's SymPy backend.

SymPy is extended by subclassing and by hook functions, which only Python
code can define, so this module holds exactly those: the parity-opaque
piecewise class, the ``round`` function, and the helpers their methods call.
The Rust backend (the binding's ``solver::sympy::SympySimplifier``) runs
this source once per interpreter and publishes it as the module
``_fhy_core_sympy_<version>_<hash>``, whose ``__fhy_core_prelude__`` is the
hash of this source; every backend of the same version and source uses the
one published module, so every lowered piecewise has one class.
"""

from typing import Any

import sympy  # type: ignore[import-untyped]

__all__ = [
    "ROUND",
    "AbortWalk",
    "ParityOpaquePiecewise",
    "hide_piecewise_parity",
    "holds_partial_piecewise",
]


class AbortWalk(BaseException):
    """Raised by a Rust hook of a SymPy walk to stop the walk.

    The hook keeps the reason in Rust. It is a ``BaseException``, so no
    ``except Exception`` inside SymPy catches it.
    """


def is_partial_piecewise(piecewise: Any) -> bool:
    """Return whether ``piecewise`` has branches but no final ``True`` condition.

    Such a piecewise has no value where every condition fails, so no total
    expression represents it.
    """
    return bool(piecewise.args) and piecewise.args[-1][1] is not sympy.true


def holds_partial_piecewise(expression: Any) -> bool:
    """Return whether ``expression`` is or contains a partial ``sympy.Piecewise``."""
    return isinstance(expression, sympy.Basic) and any(
        is_partial_piecewise(piecewise)
        for piecewise in expression.atoms(sympy.Piecewise)
    )


class ParityOpaquePiecewise(sympy.Piecewise):  # type: ignore[misc]
    """A ``sympy.Piecewise`` that makes no claim about its value's parity.

    SymPy 1.14's ``Mul._eval_is_integer`` counts a factor known to be even
    as exactly one factor of two and ignores the odd part of the
    denominator, so it calls ``n / 6`` an integer and ``n / 3`` a
    non-integer for any ``n`` it knows is even. A ``Piecewise`` whose
    branch values are all even is known even, so ``Mod(Piecewise((2, b),
    (0, True)), -6)`` evaluates to ``0``. Leaving the parity unknown keeps
    every such consumer from reasoning about it; every other assumption,
    integrality and sign included, is kept.

    SymPy rebuilds a piecewise through its own class wherever it can, and
    ``piecewise_simplify`` builds a plain ``Piecewise``, so simplifying one
    of these returns the result with every plain ``Piecewise`` in it
    rebuilt as this class.

    Evaluating a piecewise also prunes a piecewise branch value by the
    branch's own condition, which can leave a partial piecewise that no
    expression represents. An evaluation that leaves such a partial
    piecewise is skipped, and the piecewise keeps its own total branches.
    """

    @classmethod
    def eval(cls, *args: Any) -> Any:  # noqa: D102
        evaluated = super().eval(*args)
        if holds_partial_piecewise(evaluated) and not any(
            holds_partial_piecewise(arg) for arg in args
        ):
            return None
        return evaluated

    def _eval_is_even(self) -> bool | None:
        return None

    def _eval_is_odd(self) -> bool | None:
        return None

    def _eval_simplify(self, **kwargs: Any) -> Any:
        return hide_piecewise_parity(super()._eval_simplify(**kwargs))


def hide_piecewise_parity(expression: Any) -> Any:
    """Return ``expression`` with each plain ``Piecewise`` made parity-opaque."""
    if not isinstance(expression, sympy.Basic):
        return expression
    return expression.replace(
        lambda node: type(node) is sympy.Piecewise,
        lambda piecewise: ParityOpaquePiecewise(*piecewise.args, evaluate=False),
        simultaneous=False,
    )


def fold_round_over_an_integer(value: Any) -> Any:
    """Return ``value`` when it is a sympy integer, and ``None`` otherwise.

    SymPy calls this to decide whether a ``round`` application evaluates,
    and a ``None`` result keeps the application unevaluated. Rounding an
    integer is the identity under every rounding rule, so that case folds.
    SymPy reads the ``eval`` hook off the class, which hands a plain
    function the argument alone; a plain function, unlike a
    ``classmethod``, also keeps a lowered ``round`` node picklable.
    """
    if value.is_Integer:
        return value
    return None


# SymPy has no rounding operator, so ``round`` lowers to a function of its
# own that folds only over an integer argument.
ROUND: Any = sympy.Function("round", eval=fold_round_over_an_integer)
