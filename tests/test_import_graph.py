"""Pins the `fhy_core` package import graph as a DAG with no ordering rule.

Nothing in the package imports itself back at module load time: the
executed module-level imports form a directed acyclic graph. Two
consequences are pinned here, because both would break the moment a
module-level cycle reappeared.

First, every public entry point imports standalone. A fresh interpreter
that does nothing but ``import <entry point>`` succeeds for each of them,
in any order, so no `__init__.py` has to sequence its submodule imports
and no downstream package has to import one subsystem before another.

Second, the graph really is acyclic rather than merely resolvable: a
cycle whose modules happen to load in a lucky order also imports
successfully, so import success alone cannot distinguish a DAG from a
cycle that got away with it. The graph check reads the source directly
and fails on any cycle.

Only imports Python executes while loading a module count as edges:
imports inside a function body run on call, and imports under
``if TYPE_CHECKING`` never run at all. The package uses both to keep two
otherwise-mutual pairs off the load path, so counting them as edges would
report cycles that no interpreter can hit.
"""

import ast
import graphlib
import pathlib
import subprocess
import sys
from collections.abc import Iterable, Iterator
from typing import TypeGuard

import pytest

import fhy_core

_SOURCE_ROOT = pathlib.Path(fhy_core.__file__).parent

# Every module the package publishes as an entry point: the top-level
# package, each subsystem, and each family namespace under `symbolic`.
_ENTRY_POINTS = [
    "fhy_core",
    "fhy_core.traits",
    "fhy_core.identifier",
    "fhy_core.term",
    "fhy_core.provenance",
    "fhy_core.types",
    "fhy_core.types.checking",
    "fhy_core.symbol_table",
    "fhy_core.pass_infrastructure",
    "fhy_core.symbolic",
    "fhy_core.symbolic.expression",
    "fhy_core.symbolic.constraint",
    "fhy_core.symbolic.param",
    "fhy_core.symbolic.solver",
]


def _find_module_name(path: pathlib.Path) -> str:
    """Return the dotted module name of a source file under the package root."""
    parts = list(path.relative_to(_SOURCE_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_type_checking_guard(node: ast.AST) -> TypeGuard[ast.If]:
    """Return whether the node is a plain ``if TYPE_CHECKING:`` statement."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )


def _iter_executed_imports(
    nodes: Iterable[ast.AST],
) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yield the import statements that run while the module is being loaded."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif _is_type_checking_guard(node):
            # Only the `else` branch of a TYPE_CHECKING guard is executed.
            yield from _iter_executed_imports(node.orelse)
        else:
            yield from _iter_executed_imports(ast.iter_child_nodes(node))


def _resolve_to_module(dotted: str, modules: frozenset[str]) -> str | None:
    """Return the longest prefix of a dotted name that is a package module."""
    while dotted:
        if dotted in modules:
            return dotted
        dotted = dotted.rpartition(".")[0]
    return None


def _find_imported_names(node: ast.Import | ast.ImportFrom, package: str) -> list[str]:
    """Return the dotted names one import statement refers to."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    prefix = node.module or ""
    if node.level:
        # `from . import x` is relative to the containing package; each extra
        # dot strips one more trailing component off it.
        base = package.rsplit(".", node.level - 1)[0]
        prefix = f"{base}.{prefix}" if prefix else base
    return [f"{prefix}.{alias.name}" for alias in node.names]


def _build_load_time_import_graph() -> dict[str, set[str]]:
    """Return each package module mapped to the modules it imports at load time."""
    paths = sorted(
        path for path in _SOURCE_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    )
    # A compiled extension module is declared by a type stub with no Python
    # source beside it. It is a leaf here: its load-time imports are not
    # visible, and without it an import of the extension would resolve to the
    # package that contains it.
    extension_modules = frozenset(
        _find_module_name(path)
        for path in _SOURCE_ROOT.rglob("*.pyi")
        if not path.with_suffix(".py").exists()
    )
    modules = frozenset(_find_module_name(path) for path in paths) | extension_modules
    graph: dict[str, set[str]] = {module: set() for module in extension_modules}
    for path in paths:
        module = _find_module_name(path)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in _iter_executed_imports(ast.iter_child_nodes(tree)):
            for name in _find_imported_names(node, package):
                target = _resolve_to_module(name, modules)
                if target is not None and target != module:
                    imported.add(target)
        graph[module] = imported
    return graph


def _run_fresh_program(source: str) -> None:
    """Run a program in a fresh interpreter, raising if it exits non-zero."""
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.import_graph
def test_the_load_time_import_graph_is_acyclic() -> None:
    """Test no package module imports itself back at module load time."""
    graph = _build_load_time_import_graph()

    try:
        graphlib.TopologicalSorter(graph).prepare()
    except graphlib.CycleError as error:
        cycle = " -> ".join(error.args[1])
        pytest.fail(f"module-level import cycle: {cycle}")


@pytest.mark.import_graph
def test_the_import_graph_covers_every_entry_point() -> None:
    """Test the graph is built over real modules, including every entry point."""
    graph = _build_load_time_import_graph()

    assert set(_ENTRY_POINTS) <= set(graph)


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.import_graph
@pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINTS)
def test_entry_point_imports_standalone_in_a_fresh_interpreter(
    entry_point: str,
) -> None:
    """Test this entry point is the only import a fresh interpreter needs."""
    _run_fresh_program(f"import {entry_point}\n")


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.import_graph
def test_importing_symbolic_exposes_its_family_submodules() -> None:
    """Test `import fhy_core.symbolic` alone reaches every family submodule.

    Each submodule is an attribute only once its own module body has run
    to completion, so reaching all five in a fresh interpreter that
    imported nothing else shows the family loaded fully rather than
    leaving a partially initialized module behind.
    """
    _run_fresh_program(
        "import fhy_core.symbolic\n"
        "for name in ('constraint', 'expression', 'param', 'solver', "
        "'symbol_type'):\n"
        "    getattr(fhy_core.symbolic, name)\n"
    )
