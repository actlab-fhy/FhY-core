"""Symbolic expression, constraint, and parameter subsystem.

Aggregates the ``expression -> constraint -> param`` vertical stack (plus
the family-owned ``symbol_type`` sort vocabulary) behind one namespace.

Import order matters: ``expression`` is imported first because
``fhy_core.types`` has a mutual edge with ``expression.core``/
``expression.pprint``, resolved only by submodule load order (see
``expression/__init__.py``, whose own internal import order is
load-bearing for the same reason). Import what you need from the
relevant submodule (``fhy_core.symbolic.param``,
``fhy_core.symbolic.expression``, ``fhy_core.symbolic.constraint``,
``fhy_core.symbolic.symbol_type``), or reach it as a namespace after
``import fhy_core.symbolic`` (e.g.
``fhy_core.symbolic.param.create_integer_param``).
"""

__all__ = [
    "SymbolType",
    "constraint",
    "expression",
    "param",
    "symbol_type",
]

# `expression` must be the first submodule imported here: `fhy_core.types`
# has a mutual edge with `expression.core`/`expression.pprint`, resolved
# only by this load order (see the module docstring above). `constraint`
# and `param` follow in that order (`param` depends on `constraint`).
from . import expression

# isort: split

from . import constraint

# isort: split

from . import param

# isort: split

from . import symbol_type
from .symbol_type import SymbolType
