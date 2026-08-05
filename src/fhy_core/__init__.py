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

# `traits` and `symbolic` must each import before anything that touches
# their circular-import partner, or the partial-init reentry fails:
#
# - `traits` before `identifier`: `identifier.py` imports
#   `.traits.equality`/`.traits.frozen` directly, and
#   `traits.alpha_equivalence` imports `fhy_core.identifier` back.
#   Importing `traits` first resolves that reverse edge against a fresh
#   `identifier` module.
# - `symbolic` before `types`: `types/core.py` imports
#   `..symbolic.expression.core`/`.pprint`, while
#   `symbolic/expression/passes/{type_checker,numpy,body_type_checker}.py`
#   import `fhy_core.types`. `symbolic/expression/__init__.py` fully
#   loads `.core` before any pass imports `fhy_core.types`, so importing
#   `symbolic` first means `expression.core` is already complete when
#   `types/core.py` reaches back for it. `symbol_table` sorts ahead of
#   `symbolic` alphabetically, so this order is not isort's default.
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
