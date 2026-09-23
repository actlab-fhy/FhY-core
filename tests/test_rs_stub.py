"""Tests checking the `_rs.pyi` stub against the built extension.

The stub describes the public surface of the compiled extension
`fhy_core._rs` for type checkers. This suite parses the stub with `ast` and
compares its top-level public names against the names the built extension
actually exports, in both directions, and each stub function's parameters
against the signature PyO3 publishes for it, so the two cannot drift apart
silently. Types are not compared: PyO3's signatures carry none.
"""

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import fhy_core

# A parameter as both sides can describe it: its name, its kind, and whether
# it has a default.
_ParameterShape = tuple[str, inspect._ParameterKind, bool]


def _parse_stub() -> ast.Module:
    """Return the parsed `_rs.pyi` stub."""
    stub_path = Path(fhy_core.__file__).parent / "_rs.pyi"
    return ast.parse(stub_path.read_text(), filename=str(stub_path))


def _read_public_stub_names() -> set[str]:
    """Return the public top-level names declared in `_rs.pyi`.

    Returns:
        The name of every top-level function, class, and annotated or plain
        assignment in the stub, including `__version__`.

    """
    names: set[str] = set()
    for node in _parse_stub().body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return names


def _read_public_extension_names(extension: ModuleType) -> set[str]:
    """Return the public attribute names the built extension exports.

    Args:
        extension: The built extension module.

    Returns:
        The extension's attributes that do not start with an underscore,
        plus `__version__`.

    """
    return {
        name
        for name in dir(extension)
        if not name.startswith("_") or name == "__version__"
    }


def _describe_stub_parameters(function: ast.FunctionDef) -> list[_ParameterShape]:
    """Return the shape of each parameter a stub function declares, in order.

    Args:
        function: The stub's definition of the function.

    Returns:
        Each parameter's name, kind, and whether it has a default.

    """
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    first_defaulted = len(positional) - len(arguments.defaults)
    shapes: list[_ParameterShape] = [
        (
            argument.arg,
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD,
            index >= first_defaulted,
        )
        for index, argument in enumerate(positional)
    ]
    if arguments.vararg is not None:
        shapes.append((arguments.vararg.arg, inspect.Parameter.VAR_POSITIONAL, False))
    shapes.extend(
        (argument.arg, inspect.Parameter.KEYWORD_ONLY, default is not None)
        for argument, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        )
    )
    if arguments.kwarg is not None:
        shapes.append((arguments.kwarg.arg, inspect.Parameter.VAR_KEYWORD, False))
    return shapes


def _describe_extension_parameters(
    function: Callable[..., object],
) -> list[_ParameterShape]:
    """Return the shape of each parameter of an extension function, in order.

    Args:
        function: The extension's function, whose signature PyO3 publishes.

    Returns:
        Each parameter's name, kind, and whether it has a default.

    """
    return [
        (
            parameter.name,
            parameter.kind,
            parameter.default is not inspect.Parameter.empty,
        )
        for parameter in inspect.signature(function).parameters.values()
    ]


def test_stub_names_match_the_built_extension() -> None:
    """Test the stub declares exactly the names the built extension exports."""
    extension = pytest.importorskip("fhy_core._rs")
    stub_names = _read_public_stub_names()
    extension_names = _read_public_extension_names(extension)

    missing_from_stub = sorted(extension_names - stub_names)
    missing_from_extension = sorted(stub_names - extension_names)

    assert not missing_from_stub and not missing_from_extension, (
        f"names exported by fhy_core._rs but missing from _rs.pyi: "
        f"{missing_from_stub}; names declared in _rs.pyi but not exported by "
        f"fhy_core._rs: {missing_from_extension}"
    )


def test_stub_function_parameters_match_the_built_extension() -> None:
    """Test each stub function declares the parameters the extension's takes."""
    extension = pytest.importorskip("fhy_core._rs")
    stub_functions = [
        node for node in _parse_stub().body if isinstance(node, ast.FunctionDef)
    ]

    mismatches = {
        function.name: (stub_shape, extension_shape)
        for function in stub_functions
        if (stub_shape := _describe_stub_parameters(function))
        != (
            extension_shape := _describe_extension_parameters(
                getattr(extension, function.name)
            )
        )
    }

    assert stub_functions
    assert not mismatches, f"stub parameters (stub, extension): {mismatches}"
