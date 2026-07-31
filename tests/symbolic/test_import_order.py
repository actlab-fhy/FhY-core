"""Pins the load-bearing import order behind the `types <-> symbolic.expression`
circular import.

`types/core.py` and `types/dispatch.py` import `..symbolic.expression.core`
/ `.pprint`, while `symbolic/expression/passes/{type_checker,numpy,
body_type_checker}.py` import `fhy_core.types`. The cycle resolves only
because `symbolic/expression/__init__.py` fully loads `.core` (via its
`.builtins` import) before any pass module imports `fhy_core.types`, and
`symbolic/__init__.py` imports `expression` before anything else. Each
test starts a fresh interpreter (module state from earlier imports in
this process must not mask a reordering regression) and imports a
different entry point first.
"""

import subprocess
import sys

import pytest


def _run_fresh_import(entry_point: str) -> None:
    subprocess.run(
        [sys.executable, "-c", entry_point],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.import_order
@pytest.mark.parametrize(
    "entry_point",
    [
        "import fhy_core.types",
        "import fhy_core.symbolic.param",
        "import fhy_core.symbolic.expression.core",
    ],
    ids=["types", "symbolic.param", "symbolic.expression.core"],
)
def test_entry_point_imports_successfully_in_a_fresh_interpreter(
    entry_point: str,
) -> None:
    """Test importing this entry point first does not deadlock the expression cycle."""
    _run_fresh_import(entry_point + "\n")


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.import_order
def test_importing_symbolic_loads_expression_and_solver() -> None:
    """Test `fhy_core.symbolic` exposes `expression`, `solver`, and the solver surface.

    A fresh interpreter that only does `import fhy_core.symbolic` reaching
    all three attributes without raising confirms the package's internal
    import order completed without a partial-init failure.
    """
    _run_fresh_import(
        "import fhy_core.symbolic\n"
        "assert hasattr(fhy_core.symbolic, 'expression')\n"
        "assert hasattr(fhy_core.symbolic, 'solver')\n"
        "assert hasattr(fhy_core.symbolic, 'simplify_expression')\n"
    )
