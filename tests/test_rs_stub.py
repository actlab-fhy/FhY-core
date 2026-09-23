"""Tests checking the `_rs.pyi` stub against the built extension.

The stub describes the public surface of the compiled extension
`fhy_core._rs` for type checkers. This suite parses the stub with `ast` and
compares its top-level public names against the names the built extension
actually exports, in both directions, so the two cannot drift apart
silently.
"""

import ast
import importlib
from pathlib import Path

import pytest

import fhy_core


def _read_public_stub_names() -> set[str]:
    """Return the public top-level names declared in `_rs.pyi`.

    Returns:
        The name of every top-level function, class, and annotated or plain
        assignment in the stub, including `__version__`.

    """
    stub_path = Path(fhy_core.__file__).parent / "_rs.pyi"
    module = ast.parse(stub_path.read_text(), filename=str(stub_path))
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return names


def _read_public_extension_names() -> set[str]:
    """Return the public attribute names the built extension exports.

    Returns:
        The extension's attributes that do not start with an underscore,
        plus `__version__`.

    """
    extension = importlib.import_module("fhy_core._rs")
    return {
        name
        for name in dir(extension)
        if not name.startswith("_") or name == "__version__"
    }


@pytest.mark.skipif(
    not fhy_core.RUST_BACKEND_SELECTED, reason="the Rust backend is not selected"
)
def test_stub_names_match_the_built_extension() -> None:
    """Test the stub declares exactly the names the built extension exports."""
    stub_names = _read_public_stub_names()
    extension_names = _read_public_extension_names()

    missing_from_stub = sorted(extension_names - stub_names)
    missing_from_extension = sorted(stub_names - extension_names)

    assert not missing_from_stub and not missing_from_extension, (
        f"names exported by fhy_core._rs but missing from _rs.pyi: "
        f"{missing_from_stub}; names declared in _rs.pyi but not exported by "
        f"fhy_core._rs: {missing_from_extension}"
    )
