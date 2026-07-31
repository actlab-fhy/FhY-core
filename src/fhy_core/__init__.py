"""FhY compiler core utilities.

The public API is organized by subsystem. Import what you need from the relevant
submodule (``fhy_core.symbolic.param``, ``fhy_core.symbolic.expression``,
``fhy_core.types``, ``fhy_core.symbolic.constraint``, ``fhy_core.traits``,
``fhy_core.pass_infrastructure``, ``fhy_core.serialization``, and so on), or
reach it as a namespace after ``import fhy_core`` (e.g.
``fhy_core.symbolic.param.create_integer_param``). Only the ownerless
primitives used across every subsystem are re-exported at the top level.
"""

from importlib.metadata import version

# Two submodules must bootstrap before the rest, both for load-bearing
# circular-import reasons that are otherwise invisible from this file:
#
# - `traits` before anything that touches `identifier` (e.g. `diagnostic`):
#   `identifier.py` imports `.traits.equality`/`.traits.frozen` directly,
#   and `traits.alpha_equivalence` imports `fhy_core.identifier` back.
#   Importing `traits` first here resolves that reverse edge against a
#   fresh (not partially-initialized) `identifier` module; touching
#   `identifier` first would instead re-enter `traits` mid-import and fail.
# - `symbolic` before anything that touches `types` (e.g. `symbol_table`,
#   which -- because of the underscore in its name -- sorts before
#   `symbolic` and would otherwise reach `types` first): `types/core.py`
#   imports `..symbolic.expression.core`/`.pprint`, while
#   `symbolic/expression/passes/{type_checker,numpy,body_type_checker}.py`
#   import `fhy_core.types`. `symbolic/expression/__init__.py` fully loads
#   `.core` (via its `.builtins` import) before any pass imports
#   `fhy_core.types`, so importing `symbolic` first here means `types` is
#   still untouched when those passes trigger its first (fresh) import,
#   and `expression.core` is already complete by the time `types/core.py`
#   reaches back for it.
from . import traits

# isort: split

from . import symbolic

# isort: split

from . import (
    diagnostic,
    error,
    identifier,
    lattice,
    logger,
    op_attribute,
    pass_infrastructure,
    provenance,
    serialization,
    symbol_table,
    testing_patches,
    types,
    utils,
    value_domain,
)
from .identifier import Identifier

__version__ = version("fhy_core")

__all__ = [
    "Identifier",
    "diagnostic",
    "error",
    "identifier",
    "lattice",
    "logger",
    "op_attribute",
    "pass_infrastructure",
    "provenance",
    "serialization",
    "symbol_table",
    "symbolic",
    "testing_patches",
    "traits",
    "types",
    "utils",
    "value_domain",
]
