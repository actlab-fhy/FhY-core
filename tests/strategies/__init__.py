"""Shared Hypothesis strategies for the FhY-core property-based test suites.

One module per domain: ``identifiers``, ``literals``, ``expressions``,
``structural_expressions``, ``params``, ``constraints``, ``types``,
``orders``, and ``serializables``; ``settings`` holds the shared
Hypothesis settings helper. Nothing is
re-exported here; import each strategy from its owning submodule, for
example ``from .strategies.expressions import build_numeric_expression_strategy``.

This package is imported only by ``*_properties.py`` test files, which
guard their own ``hypothesis`` import with ``pytest.importorskip``. No
``conftest.py`` may import this package, since ``hypothesis`` is an
optional test dependency and ``conftest.py`` must import cleanly without
it.
"""
