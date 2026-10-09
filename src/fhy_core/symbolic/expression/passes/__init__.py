"""Analysis and transformation passes over the expression IR.

This package intentionally does not eagerly import its submodules, so
that importing one pass does not drag in the others (and their optional
third-party backends: SymPy, Z3, NumPy). Import from the specific
submodule (for example, ``from
fhy_core.symbolic.expression.passes.evaluate import evaluate_expression``)
or from the top-level :mod:`fhy_core.symbolic.expression` package, which
exposes the public API.
"""
