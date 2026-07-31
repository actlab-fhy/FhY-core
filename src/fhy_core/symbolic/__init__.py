"""Symbolic expression, constraint, and parameter subsystem.

Aggregates the ``expression -> constraint -> param`` vertical stack (plus
the family-owned ``symbol_type`` sort vocabulary) behind one namespace,
and re-exports the ``solver`` module's full public surface -- the single
documented entry point for solver queries -- so that
``fhy_core.symbolic.simplify_expression`` works directly.

Import order matters: ``expression`` is imported first because
``fhy_core.types`` has a mutual edge with ``expression.core``/
``expression.pprint``, resolved only by submodule load order (see
``expression/__init__.py``, whose own internal import order is
load-bearing for the same reason). Import what you need from the
relevant submodule (``fhy_core.symbolic.param``,
``fhy_core.symbolic.expression``, ``fhy_core.symbolic.constraint``,
``fhy_core.symbolic.solver``, ``fhy_core.symbolic.symbol_type``), or
reach it as a namespace after ``import fhy_core.symbolic`` (e.g.
``fhy_core.symbolic.param.create_integer_param``).
"""

__all__ = [
    "SolverBackend",
    "SolverCapabilityError",
    "SolverQueryKind",
    "SymbolType",
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
    "check_expression_satisfiability",
    "constraint",
    "does_expression_imply",
    "expression",
    "get_backend_capabilities",
    "holds_for_all_free_assignments",
    "param",
    "simplify_expression",
    "solver",
    "symbol_type",
]

# `expression` must be the first submodule imported here: `fhy_core.types`
# has a mutual edge with `expression.core`/`expression.pprint`, resolved
# only by this load order (see the module docstring above). `solver`,
# `constraint`, and `param` follow in that order (each of `constraint`
# and `param` depends on both `expression` and `solver`).
from . import expression

# isort: split

from . import solver
from .solver import (
    SolverBackend,
    SolverCapabilityError,
    SolverQueryKind,
    assert_expression_implies,
    assert_holds_for_all_free_assignments,
    check_expression_satisfiability,
    does_expression_imply,
    get_backend_capabilities,
    holds_for_all_free_assignments,
    simplify_expression,
)

# isort: split

from . import constraint

# isort: split

from . import param

# isort: split

from . import symbol_type
from .symbol_type import SymbolType
