"""Symbolic expression, constraint, and parameter subsystem.

Aggregates the ``expression -> constraint -> param`` vertical stack (plus
the family-owned ``symbol_type`` sort vocabulary) behind one namespace.
Import what you need from the relevant submodule
(``fhy_core.symbolic.param``, ``fhy_core.symbolic.expression``,
``fhy_core.symbolic.constraint``, ``fhy_core.symbolic.solver``,
``fhy_core.symbolic.symbol_type``), or reach it as a namespace after
``import fhy_core.symbolic`` (e.g.
``fhy_core.symbolic.param.create_integer_param``).

``fhy_core.symbolic.solver`` is the entry point for solver queries:
simplification, satisfiability, implication, and universal validity all
route through it rather than through the bridge passes they delegate to.
"""

__all__ = [
    "SymbolType",
    "constraint",
    "expression",
    "param",
    "solver",
    "symbol_type",
]

from . import constraint, expression, param, solver, symbol_type
from .symbol_type import SymbolType
