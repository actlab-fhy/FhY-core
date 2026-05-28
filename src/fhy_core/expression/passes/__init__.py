"""Analysis and transformation passes over the expression IR.

This package intentionally does not eagerly import its submodules.
Several passes import from :mod:`fhy_core.expression.registry`, which in
turn imports the registry-internal :mod:`body_type_checker` at module
load time; an eager re-export here would re-enter the registry mid-load
and break that chain. Import from the specific submodule (for example,
``from fhy_core.expression.passes.evaluate import evaluate_expression``)
or from the top-level :mod:`fhy_core.expression` package, which exposes
the public API.
"""
