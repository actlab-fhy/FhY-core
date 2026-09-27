"""Expression passes that interface with SymPy, and the SymPy simplifier.

The lowering to SymPy, the simplification with its workarounds, and the
lifting back run in the Rust core (``fhy_core::solver::SympySimplifier``,
S12 of ``docs/design/python-switch.md``), so there is one copy of the
mapping. :class:`SympySimplifier` is that backend, the
:class:`~fhy_core.symbolic.solver.Simplifier` that ``SolverBackend.SYMPY``
names; a :class:`~fhy_core.symbolic.solver.Solver` holding it simplifies in
Rust. The passes and functions below lower, substitute and lift on their own
through it.

The lowering is exact: a ``bool`` becomes ``sympy.true`` or ``sympy.false``,
an ``int`` an ``Integer``, a ``float`` the ``Float`` of its binary value, and
a ``Decimal`` the ``Rational`` it denotes; an identifier becomes the symbol
``<name_hint>_<id>``, and a native constant its value. The lifting is its
inverse: a ``Rational`` becomes the decimal-string literal of its value when
some binary ``float`` equals it, and a ``DIVIDE`` of its numerator and
denominator otherwise; ``oo``, ``-oo`` and ``nan`` become the ``inf`` and
``nan`` constants; ``zoo`` raises :class:`ComplexInfinityLiftError`, and a
piecewise without a final ``True`` branch :class:`PartialPiecewiseError`.
Simplification is best-effort: where ``sympy.simplify`` raises
``PrecisionExhausted``, drops a piecewise's otherwise branch, or cannot
compare a piecewise that has a Boolean identifier in a case condition and a
branch that is not a real number, the expression is kept as it stands.

Importing this module imports ``sympy``; without the ``sympy`` package it
raises :class:`~fhy_core.symbolic.solver.SolverBackendUnavailableError`,
naming the extra to install.
"""

from fhy_core.symbolic.solver import SolverBackendUnavailableError

__all__ = [
    "SympySimplifier",
    "convert_expression_to_sympy_expression",
    "convert_sympy_expression_to_expression",
    "substitute_sympy_expression_variables",
]

from collections.abc import Mapping
from typing import Any

from immutabledict import immutabledict

try:
    import sympy  # type: ignore
except ImportError as error:
    raise SolverBackendUnavailableError(
        "The sympy solver backend needs the sympy package, which is not "
        "installed; install it with `pip install fhy_core[sympy]`, or "
        "`pip install fhy_core[solvers]` for every solver backend."
    ) from error

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassExecutionError,
    register_pass,
)
from fhy_core.utils.override import override

from ..core import Expression, validate_logical_operands

SympySimplifier = _rs.SympySimplifier

# The backend the passes and functions below run on. SymPy is imported
# already, so loading it here publishes the backend's prelude module,
# ``_fhy_core_sympy_<version>_<hash>``, whose classes a pickled lowered
# expression names.
_BACKEND = SympySimplifier()
_BACKEND.load()


@register_pass(
    "fhy_core.symbolic.expression.to_sympy",
    "Lower expression IR into an equivalent SymPy expression.",
)
class ExpressionToSympyConverter(CompilerPass[Expression, Any]):
    """Lowers an expression to a SymPy expression, in the Rust core."""

    @override
    def run_pass(self, ir: Expression) -> Any:
        return _BACKEND.lower(ir)

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        """Return the name of the SymPy symbol of ``identifier``."""
        return f"{identifier.name_hint}_{identifier.id}"

    @override
    def get_noop_output(self, ir: Expression) -> Any:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def convert_expression_to_sympy_expression(expression: Expression) -> Any:
    """Convert an expression to a SymPy expression.

    Screens the expression before lowering: a provably numeric operand of
    a logical connective, or a provably numeric piecewise case condition,
    is refused here rather than handed to SymPy.

    Args:
        expression: Expression to convert.

    Returns:
        SymPy expression.

    Raises:
        NonBooleanLogicalOperandError: If an operand of a ``LogicalExpression``
            or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in ``expression`` provably denotes a number.
        PassExecutionError: Wrapping a ``TypeError`` as ``__cause__`` for a
            call SymPy has no function for.

    """
    validate_logical_operands(expression)
    return ExpressionToSympyConverter()(expression)


@register_pass(
    "fhy_core.symbolic.expression.substitute_sympy_variables",
    "Replace bound symbols in a SymPy expression via xreplace.",
)
class SympyVariableSubstitutionPass(CompilerPass[Any, Any]):
    """Applies a symbol-to-value replacement, so a SymPy failure is wrapped.

    The replacement is simultaneous and rebuilds every substituted node
    bottom-up, which can make SymPy auto-evaluate a relational it cannot
    represent, for example a comparison against ``zoo`` or NaN, and raise;
    running it as a pass wraps that failure as ``PassExecutionError``. A
    piecewise reaching a Boolean position is rewritten as a Boolean first.
    """

    def __init__(self, replacements: Mapping[Any, Any]) -> None:
        super().__init__()
        self._replacements: immutabledict[Any, Any] = immutabledict(replacements)

    @override
    def run_pass(self, ir: Any) -> Any:
        return _BACKEND.substitute_symbols(ir, dict(self._replacements))

    @override
    def get_noop_output(self, ir: Any) -> Any:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def substitute_sympy_expression_variables(
    sympy_expression: Any,
    environment: Mapping[Identifier, Expression],
) -> Any:
    """Substitute variables in a SymPy expression.

    Every binding in ``environment`` is applied simultaneously: a
    replacement value is never itself re-substituted by another binding in
    the same call, matching ``Expression.substitute``. For example,
    substituting ``{x: y, y: 5}`` into ``x < 5`` yields ``y < 5``.

    Args:
        sympy_expression: SymPy expression to substitute variables in.
        environment: Environment to substitute variables from.

    Returns:
        SymPy expression with substituted variables.

    Raises:
        NativeConstantBindingError: If ``environment`` binds a native
            constant's canonical identifier that is free in
            ``sympy_expression`` as a symbol.
        NonBooleanLogicalOperandError: If a replacement value in
            ``environment`` contains a ``LogicalExpression`` or
            ``LOGICAL_NOT`` node whose operand, or a piecewise whose case
            condition, provably denotes a number.
        PassExecutionError: Wrapping the originating ``TypeError`` as
            ``__cause__`` if applying a replacement makes SymPy
            auto-evaluate a relational it cannot represent, for example a
            comparison against ``zoo`` or against NaN.

    """
    # SymPy can fold a Boolean to a plain Python ``bool``, which has nothing
    # to substitute.
    if isinstance(sympy_expression, bool):
        return sympy_expression
    return _BACKEND.substitute(sympy_expression, environment)


@register_pass(
    "fhy_core.symbolic.expression.from_sympy",
    "Lift a SymPy expression into the FhY expression IR.",
)
class SymPyToExpressionConverter(CompilerPass[Any, Expression]):
    """Converts a SymPy expression to an expression tree, in the Rust core."""

    @override
    def run_pass(self, ir: Any) -> Expression:
        return self.convert(ir)

    @override
    def get_noop_output(self, ir: Any) -> Expression:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )

    def convert(self, node: Any) -> Expression:
        """Convert a SymPy node.

        Args:
            node: SymPy node to convert.

        Returns:
            Expression tree.

        """
        return _BACKEND.lift(node)

    def convert_expr(self, expr: Any) -> Expression:
        """Convert a SymPy ``Expr``, refusing any other node."""
        if not isinstance(expr, sympy.Expr):
            raise TypeError(f"unsupported expression type: {type(expr)}")
        return _BACKEND.lift(expr)

    def convert_bool(self, boolean_expression: Any) -> Expression:
        """Convert a SymPy Boolean that is no ``Expr``, refusing any other node."""
        if isinstance(boolean_expression, sympy.Expr) or not isinstance(
            boolean_expression, sympy.logic.boolalg.Boolean
        ):
            raise TypeError(
                f"unsupported boolean expression type: {type(boolean_expression)}"
            )
        return _BACKEND.lift(boolean_expression)

    def convert_relational(self, relational: Any) -> Expression:
        """Convert a SymPy relational, refusing any other node."""
        if not isinstance(relational, sympy.core.relational.Relational):
            raise TypeError(f"unsupported relational type: {type(relational)}")
        return _BACKEND.lift(relational)


def convert_sympy_expression_to_expression(sympy_expression: Any) -> Expression:
    """Convert a SymPy expression to an expression.

    Args:
        sympy_expression: SymPy expression to convert.

    Returns:
        Expression.

    Raises:
        PassExecutionError: Wrapping :class:`PartialPiecewiseError` as
            ``__cause__`` if ``sympy_expression`` contains a
            ``sympy.Piecewise`` whose final branch condition is not
            ``sympy.true``, or :class:`ComplexInfinityLiftError` if it
            contains ``sympy.zoo``.

    """
    return SymPyToExpressionConverter()(sympy_expression)
