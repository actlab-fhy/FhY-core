"""Pretty-printer for expressions.

:func:`pformat_expression` prints the Rust core's text: a literal as the
core writes it (``true``, ``1``, ``NaN``, ``1.5``), a conjunction or
disjunction as one n-ary node (``(a && b && c)``, functionally
``(and a b c)``), and each operation's functional name as its Rust name,
which is its member's value (``(floor_mod x 3)``).
:class:`ExpressionPrettyFormatter` is the same rendering as a compiler pass.
"""

__all__ = ["pformat_expression"]

from typing import Any

from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.utils.override import override

from .core import Expression


class ExpressionPrettyFormatter(CompilerPass[Expression, str]):
    """Pass formatting an expression as :func:`pformat_expression` does.

    The core renders the text, so the formatter has no per-node hook: a
    subclass defining a ``visit_*`` method is refused when it is created,
    rather than having its override ignored.
    """

    _is_id_shown: bool
    _is_printed_functional: bool

    def __init__(
        self, is_id_shown: bool = False, is_printed_functional: bool = False
    ) -> None:
        super().__init__()
        self._is_id_shown = is_id_shown
        self._is_printed_functional = is_printed_functional

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        overrides = sorted(name for name in vars(cls) if name.startswith("visit_"))
        if overrides:
            raise TypeError(
                f"{cls.__name__} defines {', '.join(overrides)}, but "
                "ExpressionPrettyFormatter renders through the Rust core and has "
                "no per-node hooks."
            )
        super().__init_subclass__(**kwargs)

    @override
    def run_pass(self, ir: Expression) -> str:
        return pformat_expression(
            ir, show_id=self._is_id_shown, functional=self._is_printed_functional
        )

    @override
    def did_change(self, input_ir: Expression, output: str) -> bool:
        _ = (input_ir, output)
        return True

    @override
    def get_noop_output(self, ir: Expression) -> str:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def pformat_expression(
    expression: Expression, show_id: bool = False, functional: bool = False
) -> str:
    """Pretty-format an expression.

    Args:
        expression: Expression to pretty-format.
        show_id: Whether to show the identifier ID.
        functional: Whether to use functional notation.

    Returns:
        Pretty-formatted expression.

    Raises:
        TypeError: If ``expression`` is not an ``Expression``.

    """
    if not isinstance(expression, Expression):
        raise TypeError(
            f"pformat_expression takes an Expression, got {type(expression).__name__}."
        )
    return expression._format(show_id, functional)
