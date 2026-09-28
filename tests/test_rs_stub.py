"""Tests checking the `_rs.pyi` stub against the built extension.

The stub describes the public surface of the compiled extension
`fhy_core._rs` for type checkers. This suite parses the stub with `ast` and
compares its top-level public names against the names the built extension
actually exports, in both directions, each stub function's parameters
against the signature PyO3 publishes for it, and each class's non-dunder
members and their kinds (method, class method, static method, property or
attribute), in both directions, so the two cannot drift apart silently.
Types are not compared: PyO3's signatures carry none.
"""

import ast
import importlib
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


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _stub_member_kind(node: ast.stmt) -> str:
    """Return the kind of member a stub class body statement declares."""
    if not isinstance(node, ast.FunctionDef):
        return "attribute"
    decorators = {_decorator_name(decorator) for decorator in node.decorator_list}
    if decorators & {"property", "setter", "deleter"}:
        return "property"
    if "classmethod" in decorators:
        return "classmethod"
    if "staticmethod" in decorators:
        return "staticmethod"
    return "method"


def _extension_member_kind(member: object) -> str:
    """Return the kind of an extension class's attribute, read statically."""
    kind = type(member).__name__
    if kind in {
        "method_descriptor",
        "wrapper_descriptor",
        "builtin_function_or_method",
    }:
        return "method"
    if kind in {"classmethod_descriptor", "classmethod"}:
        return "classmethod"
    if kind == "staticmethod":
        return "staticmethod"
    if kind in {"getset_descriptor", "member_descriptor", "property"}:
        return "property"
    return "attribute"


def _read_stub_imports(stub: ast.Module) -> dict[str, object]:
    """Return each name the stub imports from `fhy_core`, as its object."""
    imported: dict[str, object] = {}
    for node in stub.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "fhy_core"
        ):
            module = importlib.import_module(node.module or "")
            for alias in node.names:
                imported[alias.asname or alias.name] = getattr(module, alias.name)
    return imported


def _python_class_members(cls: type) -> dict[str, str]:
    """Return the non-dunder members `cls` and its package bases define.

    A field a base annotates, such as a dataclass field, counts too.
    """
    members: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if klass.__module__.startswith("fhy_core"):
            members.update(
                (name, "attribute")
                for name in inspect.get_annotations(klass)
                if not _is_dunder(name)
            )
            members.update(
                (name, _extension_member_kind(member))
                for name, member in vars(klass).items()
                if not _is_dunder(name)
            )
    return members


def _read_stub_class_members(
    node: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    imported: dict[str, object],
) -> tuple[dict[str, str], set[str]]:
    """Return the members a stub class declares, and those it inherits.

    The declared members are the non-dunder members of the class and of the
    bases the stub declares, with their kinds. The inherited ones are the
    non-dunder members of the package classes the stub names as bases
    (such as `SymbolTableFrame`, which the frame classes register with):
    they account for an extension member, but the extension need not
    define them, since they are that Python class's API.
    """
    declared: dict[str, str] = {}
    inherited: set[str] = set()
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else None
        if name in classes:
            base_declared, base_inherited = _read_stub_class_members(
                classes[name], classes, imported
            )
            declared.update(base_declared)
            inherited |= base_inherited
        elif name in imported and isinstance(imported[name], type):
            inherited |= set(_python_class_members(imported[name]))  # type: ignore[arg-type]
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef):
            names = [statement.name]
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            names = [statement.target.id]
        elif isinstance(statement, ast.Assign):
            names = [
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
        else:
            names = []
        for name in names:
            if not _is_dunder(name):
                declared[name] = _stub_member_kind(statement)
    return declared, inherited


def _read_extension_class_members(cls: type) -> dict[str, str]:
    """Return the non-dunder members an extension class and its bases define.

    Only the extension's own classes count; what a class inherits from a
    built-in base, such as `Exception`, is not the stub's to declare.
    """
    members: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if klass.__module__ == "fhy_core._rs":
            members.update(
                (name, _extension_member_kind(member))
                for name, member in vars(klass).items()
                if not _is_dunder(name)
            )
    return members


def _are_kinds_compatible(stub_kind: str, extension_kind: str) -> bool:
    """Return whether a stub kind describes an extension kind.

    A stub attribute annotation also describes a property, as it does a
    `#[pyo3(get)]` field.
    """
    return stub_kind == extension_kind or (
        stub_kind == "attribute" and extension_kind == "property"
    )


def test_stub_class_members_match_the_built_extension() -> None:
    """Test each stub class declares the members its extension class has.

    The non-dunder members are compared in both directions, through the
    bases on each side, with their kinds; dunders are left out, since PyO3
    adds its own.
    """
    extension = pytest.importorskip("fhy_core._rs")
    stub = _parse_stub()
    classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
    imported = _read_stub_imports(stub)

    mismatches: list[str] = []
    for name, node in classes.items():
        cls = getattr(extension, name)
        assert isinstance(cls, type), name
        stub_members, inherited = _read_stub_class_members(node, classes, imported)
        extension_members = _read_extension_class_members(cls)
        mismatches.extend(
            f"{name}.{member}: declared in the stub only"
            for member in sorted(stub_members.keys() - extension_members.keys())
        )
        mismatches.extend(
            f"{name}.{member}: defined by the extension only"
            for member in sorted(
                extension_members.keys() - stub_members.keys() - inherited
            )
        )
        mismatches.extend(
            f"{name}.{member}: the stub declares a {stub_members[member]}, "
            f"the extension defines a {extension_members[member]}"
            for member in sorted(stub_members.keys() & extension_members.keys())
            if not _are_kinds_compatible(
                stub_members[member], extension_members[member]
            )
        )

    assert classes
    assert not mismatches, "\n".join(mismatches)


def test_the_member_check_sees_a_member_declared_on_the_wrong_class() -> None:
    """Test the member check reports a member the extension class lacks.

    `rebuild_with_visit_children` is defined by the node classes only, so a
    stub that declared it on their base would be reported for the base.
    """
    extension = pytest.importorskip("fhy_core._rs")
    stub = ast.parse(
        "class Expression:\n    def rebuild_with_visit_children(self) -> None: ...\n"
    )
    node = stub.body[0]
    assert isinstance(node, ast.ClassDef)

    stub_members, _ = _read_stub_class_members(node, {"Expression": node}, {})
    extension_members = _read_extension_class_members(extension.Expression)

    assert "rebuild_with_visit_children" in stub_members
    assert "rebuild_with_visit_children" not in extension_members
    assert "rebuild_with_visit_children" in _read_extension_class_members(
        extension.BinaryExpression
    )


_PYO3_DUNDERS = frozenset(
    {
        "__module__",
        "__doc__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__eq__",
        "__ne__",
        "__annotations__",
    }
)
"""Dunders a class gets without the stub declaring them: its module and
docstring and the six comparison wrappers of one `__richcmp__`, which PyO3
adds, and `__annotations__`, which the interpreter creates in a class the
first time something reads it, as an `isinstance` check against a
runtime-checkable protocol does (R2-N5)."""


def _read_stub_class_dunders(
    node: ast.ClassDef, classes: dict[str, ast.ClassDef]
) -> set[str]:
    """Return the dunder methods a stub class and its stub bases declare."""
    dunders: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in classes:
            dunders |= _read_stub_class_dunders(classes[base.id], classes)
        if isinstance(base, ast.Subscript) and ast.unparse(base.value) == "Generic":
            dunders.add("__class_getitem__")
    dunders.update(
        statement.name
        for statement in node.body
        if isinstance(statement, ast.FunctionDef) and _is_dunder(statement.name)
    )
    return dunders


def _is_reflection_of_a_declared_operator(name: str, declared: set[str]) -> bool:
    """Return whether `name` is the reflected wrapper PyO3 adds for a declared
    binary operator: `__rand__` for `__and__`, and so on."""
    return name.startswith("__r") and f"__{name[3:]}" in declared


def test_stub_class_dunders_match_the_built_extension() -> None:
    """Test each stub class declares the dunders its extension class adds.

    An extension dunder that `object` lacks is declared, unless PyO3 adds it
    on its own (`_PYO3_DUNDERS`, the reflected operators, a generic class's
    `__class_getitem__`); and each dunder the stub declares exists on the
    class, except `__init__`, which stands for PyO3's `__new__`.
    """
    extension = pytest.importorskip("fhy_core._rs")
    stub = _parse_stub()
    classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}

    mismatches: list[str] = []
    for name, node in classes.items():
        cls = getattr(extension, name)
        declared = _read_stub_class_dunders(node, classes)
        defined = {
            member
            for klass in cls.__mro__
            if klass.__module__ == "fhy_core._rs"
            for member in vars(klass)
            if _is_dunder(member)
        }
        mismatches.extend(
            f"{name}.{member}: defined by the extension only"
            for member in sorted(defined - declared)
            if not hasattr(object, member)
            and member not in _PYO3_DUNDERS
            and not _is_reflection_of_a_declared_operator(member, declared)
        )
        mismatches.extend(
            f"{name}.{member}: declared in the stub only"
            for member in sorted(declared - {"__init__"})
            if not hasattr(cls, member)
        )

    assert not mismatches, "\n".join(mismatches)
