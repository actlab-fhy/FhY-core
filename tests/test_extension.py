"""Tests the import-time check of the required Rust extension.

The package runs on its compiled extension ``fhy_core._rs`` and cannot work
without it. Importing ``fhy_core`` raises ``ImportError`` naming the cause
and the fix when the extension is not installed, is built only for other
interpreters, fails to import, or reports a version other than the installed
package's. The check runs once, while the package is imported, so each
failure is simulated in a fresh interpreter.
"""

import importlib
import importlib.metadata
import os
import pathlib
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from types import ModuleType

import pytest

from fhy_core import _extension, _rs, native
from fhy_core._extension import (
    NATIVE_ENTRY_POINT_GROUP,
    NATIVE_MODULE_ENVIRONMENT_VARIABLE,
    Candidate,
    Selection,
    _find_foreign_extension_builds,
    _select_native_module,
    _verify_declared_layers,
    get_current_selection,
    get_layer_version_attribute,
    normalize_pep440_version,
)
from tests.native_modules import (
    read_import_error,
    run_python,
    write_distribution,
    write_native_module,
)

_IMPORT_PACKAGE_PROGRAM = (
    "import fhy_core\n"
    "from fhy_core.identifier import Identifier\n"
    "first = Identifier('first')\n"
    "second = Identifier('second')\n"
    "print(second.id - first.id)"
)
_REQUIREMENT = "fhy_core requires its Rust extension fhy_core._rs"
_REBUILD_ADVICE = "`uv sync`"


def _import_the_package(program_prefix: str = "") -> subprocess.CompletedProcess[str]:
    """Import the package in a fresh interpreter and return the result.

    Args:
        program_prefix: Source run before the package is imported.

    Returns:
        The completed process. On success its standard output holds the
        difference between two consecutively constructed ids.

    """
    # Python 3.13+ colors tracebacks when FORCE_COLOR is set, as CI and nox
    # set it; plain text keeps the final line starting with the exception
    # name. PYTHON_COLORS takes precedence over FORCE_COLOR.
    environment = {**os.environ, "PYTHON_COLORS": "0"}
    return subprocess.run(
        [sys.executable, "-c", program_prefix + _IMPORT_PACKAGE_PROGRAM],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def _read_import_error(completed: subprocess.CompletedProcess[str]) -> str:
    """Return the final ``ImportError`` line of a failed import.

    Args:
        completed: The process that failed to import the package.

    Returns:
        The last line of its standard error, the message of the uncaught
        ``ImportError``.

    """
    assert completed.returncode != 0
    assert completed.stdout == ""
    last_line = completed.stderr.strip().splitlines()[-1]
    assert last_line.startswith("ImportError: "), completed.stderr
    return last_line


def test_the_package_imported_its_extension() -> None:
    """Test importing the package imported the extension module."""
    assert importlib.import_module("fhy_core._rs") is _rs


@pytest.mark.slow
@pytest.mark.subprocess
def test_the_package_imports_in_a_fresh_interpreter() -> None:
    """Test a fresh interpreter imports the package and constructs identifiers."""
    completed = _import_the_package()

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["1"]


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_missing_extension_raises_import_error() -> None:
    """Test an extension that is not installed makes the import fail."""
    # A `None` entry in `sys.modules` makes importing that name raise
    # `ModuleNotFoundError` for it, as an extension that was never built
    # would.
    hide_extension = "import sys\nsys.modules['fhy_core._rs'] = None\n"

    message = _read_import_error(_import_the_package(hide_extension))

    assert _REQUIREMENT in message
    assert "which is not installed" in message
    assert _REBUILD_ADVICE in message


_THIS_INTERPRETER_SUFFIXES = [".cpython-313-darwin.so", ".abi3.so", ".so"]


@pytest.mark.parametrize(
    ("file_names", "expected_foreign_builds"),
    [
        (
            ["_rs.cpython-312-darwin.so", "_rs.cpython-310-darwin.so", "_rs.pyi"],
            ["_rs.cpython-310-darwin.so", "_rs.cpython-312-darwin.so"],
        ),
        (["_rs.cp312-win_amd64.pyd"], ["_rs.cp312-win_amd64.pyd"]),
        (["_rs.cpython-312-darwin.so", "_rs.cpython-313-darwin.so"], []),
        (["_rs.cpython-312-darwin.so", "_rs.abi3.so"], []),
        (["_rs.pyi", "_extension.py", "other.cpython-312-darwin.so"], []),
        ([], []),
    ],
)
def test_find_foreign_extension_builds_reports_only_unloadable_builds(
    tmp_path: pathlib.Path,
    file_names: list[str],
    expected_foreign_builds: list[str],
) -> None:
    """Test only builds with no loadable sibling count as foreign."""
    for file_name in file_names:
        (tmp_path / file_name).touch()

    foreign_builds = _find_foreign_extension_builds(
        tmp_path, _THIS_INTERPRETER_SUFFIXES
    )

    assert foreign_builds == expected_foreign_builds


@pytest.mark.slow
@pytest.mark.subprocess
def test_an_extension_built_for_other_interpreters_raises_import_error() -> None:
    """Test builds this interpreter skips make the import fail, naming them."""
    build_name = pathlib.Path(_rs.__file__).name
    # Hide the build this interpreter loads, as an import under another
    # interpreter would, and give the interpreter a suffix no build carries.
    hide_loadable_build = (
        "import importlib.machinery, sys\n"
        "sys.modules['fhy_core._rs'] = None\n"
        "importlib.machinery.EXTENSION_SUFFIXES[:] = ['.cpython-399-fake.so']\n"
    )

    message = _read_import_error(_import_the_package(hide_loadable_build))

    assert _REQUIREMENT in message
    assert build_name in message
    assert "_rs.cpython-399-fake.so" in message
    assert _REBUILD_ADVICE in message


def _create_failing_extension_prefix(raise_statement: str) -> str:
    """Return source that makes importing the extension run `raise_statement`."""
    return (
        "import sys\n"
        "class _FailingExtensionFinder:\n"
        "    @staticmethod\n"
        "    def find_spec(name, path=None, target=None):\n"
        "        if name == 'fhy_core._rs':\n"
        f"            {raise_statement}\n"
        "        return None\n"
        "sys.meta_path.insert(0, _FailingExtensionFinder)\n"
    )


_BROKEN_EXTENSION_PREFIXES = {
    "load-failure": _create_failing_extension_prefix(
        "raise ImportError('simulated dlopen failure')"
    ),
    "missing-dependency": _create_failing_extension_prefix(
        "raise ModuleNotFoundError("
        "\"No module named 'simulated_dependency'\", name='simulated_dependency')"
    ),
}


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("failure", "expected_detail"),
    [
        ("load-failure", "ImportError: simulated dlopen failure"),
        ("missing-dependency", "No module named 'simulated_dependency'"),
    ],
)
def test_a_broken_extension_raises_import_error_naming_the_cause(
    failure: str, expected_detail: str
) -> None:
    """Test an installed extension that fails to import makes the import fail."""
    message = _read_import_error(
        _import_the_package(_BROKEN_EXTENSION_PREFIXES[failure])
    )

    assert _REQUIREMENT in message
    assert "failed to import" in message
    assert expected_detail in message
    assert _REBUILD_ADVICE in message


def test_extension_version_matches_the_installed_package_version() -> None:
    """Test the extension's Cargo version normalizes to the package version."""
    assert normalize_pep440_version(_rs.__version__) == importlib.metadata.version(
        "fhy_core"
    )


@pytest.mark.parametrize(
    ("cargo_version", "package_version"),
    [
        ("0.2.0", "0.2.0"),
        ("0.3.0-rc.1", "0.3.0rc1"),
        ("0.3.0-rc1", "0.3.0rc1"),
        ("0.3.0-RC.1", "0.3.0rc1"),
        ("0.3.0-alpha.2", "0.3.0a2"),
        ("0.3.0-a.1", "0.3.0a1"),
        ("0.3.0-alpha", "0.3.0a0"),
        ("0.3.0-beta.3", "0.3.0b3"),
        ("0.3.0-c.1", "0.3.0rc1"),
        ("0.3.0-pre.1", "0.3.0rc1"),
        ("0.3.0-preview", "0.3.0rc0"),
        ("0.3.0-dev.1", "0.3.0.dev1"),
        ("0.3.0-rc.1.dev.2", "0.3.0rc1.dev2"),
        ("0.3.0-post.1", "0.3.0.post1"),
        ("0.3.0-1", "0.3.0.post1"),
        ("0.2.0+build.5", "0.2.0+build.5"),
        ("0.3.0-rc.1+Build-7", "0.3.0rc1+build.7"),
    ],
)
def test_cargo_version_normalizes_to_the_version_maturin_builds(
    cargo_version: str, package_version: str
) -> None:
    """Test a Cargo version normalizes to the package version maturin gives it."""
    assert normalize_pep440_version(cargo_version) == package_version


@pytest.mark.parametrize("version", ["0.3.0-foo.1", "0.0.0-stale", "", "stale"])
def test_a_non_pep440_version_does_not_normalize(version: str) -> None:
    """Test a version maturin cannot build a package from has no normal form."""
    assert normalize_pep440_version(version) is None


def _create_stale_extension_prefix(version_assignment: str) -> str:
    """Return source that fakes a built extension reporting a given version.

    Args:
        version_assignment: Source assigning the fake module's
            ``__version__`` attribute, or an empty string to omit it.

    Returns:
        Source that installs a fake ``fhy_core._rs`` module in
        ``sys.modules`` before ``fhy_core`` is imported.

    """
    return (
        "import sys, types\n"
        "_fake_extension = types.ModuleType('fhy_core._rs')\n"
        f"{version_assignment}\n"
        "sys.modules['fhy_core._rs'] = _fake_extension\n"
    )


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_stale_extension_raises_import_error() -> None:
    """Test an extension reporting another version makes the import fail."""
    message = _read_import_error(
        _import_the_package(
            _create_stale_extension_prefix(
                "_fake_extension.__version__ = '0.0.0-stale'"
            )
        )
    )

    assert _REQUIREMENT in message
    assert "'0.0.0-stale'" in message
    assert repr(importlib.metadata.version("fhy_core")) in message
    assert _REBUILD_ADVICE in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_versionless_extension_raises_import_error() -> None:
    """Test an extension with no `__version__` makes the import fail."""
    message = _read_import_error(
        _import_the_package(_create_stale_extension_prefix(""))
    )

    assert _REQUIREMENT in message
    assert "no __version__ attribute" in message
    assert _REBUILD_ADVICE in message


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="only a free-threaded interpreter can run without the GIL",
)
def test_the_extension_declares_that_it_uses_the_gil() -> None:
    """Test importing the extension on a free-threaded build enables the GIL.

    The module declares ``gil_used = true``, so CPython turns the
    GIL back on when it is imported.
    """
    is_gil_enabled = getattr(sys, "_is_gil_enabled")  # noqa: B009

    assert _rs is not None
    assert is_gil_enabled()


def _has_native_entry_points() -> bool:
    """Return whether a product advertises a combined extension here."""
    return bool(importlib.metadata.entry_points(group=NATIVE_ENTRY_POINT_GROUP))


@pytest.mark.skipif(
    _has_native_entry_points(),
    reason="a combined extension is installed, so the package does not run alone",
)
def test_the_package_alone_runs_on_its_own_extension() -> None:
    """Test with no combined extension installed the package uses `_rs`."""
    assert _rs.__name__ == "fhy_core._rs"
    assert importlib.import_module("fhy_core._rs") is _rs


_PRINT_EXTENSION_NAME = "import fhy_core\nprint(fhy_core._rs.__name__)"


@pytest.mark.slow
@pytest.mark.subprocess
def test_incomparable_native_modules_are_refused_naming_both(
    tmp_path: pathlib.Path,
) -> None:
    """Test with no module containing every installed layer the import fails."""
    write_distribution(tmp_path, "product-one", {"one": "native_one"})
    write_distribution(tmp_path, "product-two", {"two": "native_two"})

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "no installed module contains every installed layer" in message
    assert "'native_one' (entry point 'one' of 'product-one') lacks 'two'" in message
    assert "'native_two' (entry point 'two' of 'product-two') lacks 'one'" in message
    assert NATIVE_MODULE_ENVIRONMENT_VARIABLE in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_two_entry_points_naming_one_module_are_not_a_conflict(
    tmp_path: pathlib.Path,
) -> None:
    """Test the same module advertised twice loads once, or fails as one module."""
    write_distribution(tmp_path, "product-one", {"one": "native_shared"})
    write_distribution(tmp_path, "product-two", {"two": "native_shared"})

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "'native_shared'" in message
    assert "no installed module contains every installed layer" not in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_an_advertised_module_that_is_missing_is_not_replaced_by_the_own_one(
    tmp_path: pathlib.Path,
) -> None:
    """Test a missing advertised module raises rather than falling back."""
    write_distribution(tmp_path, "product", {"product": "native_missing"})

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert message == (
        "ImportError: fhy_core selected the native extension module "
        "'native_missing' (entry point 'product' of 'product') from the "
        "installed entry points, but it failed to import (ModuleNotFoundError: "
        "No module named 'native_missing'). Reinstall the distribution that "
        "provides it, or uninstall that distribution."
    )


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("version_assignment", "expected_detail"),
    [
        ("__fhy_core_version__ = '0.0.0-stale'", "version '0.0.0-stale'"),
        ("", "no __fhy_core_version__ attribute"),
    ],
)
def test_a_stale_advertised_module_is_refused(
    tmp_path: pathlib.Path, version_assignment: str, expected_detail: str
) -> None:
    """Test a combined extension of another fhy_core version is refused."""
    write_distribution(tmp_path, "product", {"product": "native_stale"})
    (tmp_path / "native_stale.py").write_text(version_assignment + "\n")

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "'native_stale'" in message
    assert expected_detail in message
    assert repr(importlib.metadata.version("fhy_core")) in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_the_environment_variable_overrides_the_advertised_modules(
    tmp_path: pathlib.Path,
) -> None:
    """Test naming `fhy_core._rs` in the environment selects the own module."""
    write_distribution(tmp_path, "product-one", {"one": "native_one"})
    write_distribution(tmp_path, "product-two", {"two": "native_two"})

    completed = run_python(
        _PRINT_EXTENSION_NAME,
        python_path=[tmp_path],
        environment={NATIVE_MODULE_ENVIRONMENT_VARIABLE: "fhy_core._rs"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["fhy_core._rs"]


def _entry_point(name: str, module: str) -> importlib.metadata.EntryPoint:
    """Return an entry point of the native group, as an installed product has."""
    return importlib.metadata.EntryPoint(
        name=name, value=module, group=NATIVE_ENTRY_POINT_GROUP
    )


def _select(
    entry_points: Iterable[importlib.metadata.EntryPoint],
    environ: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return the module and the reason the selection gives."""
    selection = _select_native_module(environ or {}, entry_points)
    return selection.module, selection.reason


def test_with_no_entry_points_the_own_module_is_selected() -> None:
    """Test a process with no advertised module runs on `fhy_core._rs`."""
    assert _select([]) == ("fhy_core._rs", "own")


def test_one_advertised_module_is_selected() -> None:
    """Test the one module the entry points name is selected."""
    assert _select([_entry_point("product", "native_product")]) == (
        "native_product",
        "entry_points",
    )


def test_entry_points_naming_one_module_are_merged_into_its_layers() -> None:
    """Test two entry points naming one module give it both layers, once."""
    selection = _select_native_module(
        {},
        [_entry_point("one", "native_shared"), _entry_point("two", "native_shared")],
    )

    assert selection.module == "native_shared"
    assert [candidate.module for candidate in selection.candidates] == ["native_shared"]
    assert selection.layers == {"fhy_core", "one", "two"}


_CHAINS = {
    "one": [("a", "native_a")],
    "two": [("a", "native_a"), ("b", "native_ab"), ("a", "native_ab")],
    "three": [
        ("a", "native_a"),
        ("a", "native_ab"),
        ("b", "native_ab"),
        ("b", "native_abc"),
        ("a", "native_abc"),
        ("c", "native_abc"),
        ("c", "native_abc"),
    ],
}
_CHAIN_TOPS = {
    "one": "native_a",
    "two": "native_ab",
    "three": "native_abc",
}


@pytest.mark.parametrize("chain", list(_CHAINS))
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_a_chain_of_layers_selects_its_top(chain: str, reverse: bool) -> None:
    """Test the module containing every layer is selected, whatever the order."""
    claims = _CHAINS[chain][::-1] if reverse else _CHAINS[chain]
    entry_points = [_entry_point(name, module) for name, module in claims]

    selection = _select_native_module({}, entry_points)

    assert (selection.module, selection.reason) == (
        _CHAIN_TOPS[chain],
        "entry_points",
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reversed"])
def test_a_diamond_selects_the_module_containing_every_layer(reverse: bool) -> None:
    """Test ``{a}``, ``{b}`` and ``{a, b}`` select the module with both layers."""
    claims = [
        ("a", "native_a"),
        ("b", "native_b"),
        ("a", "native_ab"),
        ("b", "native_ab"),
    ]
    entry_points = [
        _entry_point(name, module)
        for name, module in (claims[::-1] if reverse else claims)
    ]

    selection = _select_native_module({}, entry_points)

    assert (selection.module, selection.reason) == ("native_ab", "entry_points")


def test_separate_entry_points_naming_one_module_are_unioned() -> None:
    """Test separate entry points, one naming each layer, union on their module."""
    selection = _select_native_module(
        {},
        [
            _entry_point("product", "native_top"),
            _entry_point("base", "native_top"),
            _entry_point("base", "native_base"),
        ],
    )

    assert selection.module == "native_top"
    assert selection.layers == {"fhy_core", "product", "base"}


def test_two_modules_with_the_same_layers_are_a_tie() -> None:
    """Test equally complete modules are refused, naming both and the fix."""
    entry_points = [_entry_point("a", "native_one"), _entry_point("a", "native_two")]

    with pytest.raises(ImportError) as caught:
        _select_native_module({}, entry_points)

    message = str(caught.value)
    assert "several installed modules each contain every installed layer" in message
    assert "'native_one'" in message
    assert "'native_two'" in message
    assert "'a'" in message
    assert NATIVE_MODULE_ENVIRONMENT_VARIABLE in message


def test_incomparable_layer_sets_are_refused_naming_what_each_lacks() -> None:
    """Test with no module containing every installed layer, each lack is named."""
    entry_points = [
        _entry_point("one", "native_small"),
        _entry_point("two", "native_big"),
        _entry_point("three", "native_big"),
    ]

    with pytest.raises(ImportError) as caught:
        _select_native_module({}, entry_points)

    message = str(caught.value)
    assert "no installed module contains every installed layer" in message
    assert "'fhy_core', 'one', 'three', 'two'" in message
    # The module with more layers comes first.
    assert message.index("'native_big'") < message.index("'native_small'")
    assert "'native_big' (entry points 'three', 'two') lacks 'one'" in message
    assert "'native_small' (entry point 'one') lacks 'three', 'two'" in message
    assert NATIVE_MODULE_ENVIRONMENT_VARIABLE in message


@pytest.mark.parametrize(
    "entry_points",
    [
        [("a", "native_one"), ("a", "native_two")],
        [("one", "native_one"), ("two", "native_two")],
    ],
    ids=["tie", "incomparable"],
)
def test_the_environment_variable_overrides_a_tie_and_incomparable_sets(
    entry_points: list[tuple[str, str]],
) -> None:
    """Test the named module wins, where the entry points alone are refused."""
    advertised = [_entry_point(name, module) for name, module in entry_points]

    selection = _select_native_module(
        {NATIVE_MODULE_ENVIRONMENT_VARIABLE: " native_two "}, advertised
    )

    assert (selection.module, selection.reason) == ("native_two", "environment")
    assert len(selection.candidates) == 2


@pytest.mark.parametrize(
    "name", ["not-an-identifier", "1st", "a.b", "", "class", "None"]
)
def test_an_entry_point_name_that_is_not_a_layer_name_is_refused(name: str) -> None:
    """Test a name that cannot be a package's import name is an error."""
    with pytest.raises(ImportError) as caught:
        _select_native_module({}, [_entry_point(name, "native_bad")])

    message = str(caught.value)
    assert repr(name) in message
    assert "'native_bad'" in message
    assert "not a valid Python identifier" in message


_MALFORMED_MODULE_VALUES = [
    "",
    "my-product._native",
    "a b",
    "pkg.mod:attr",
    "pkg.mod [extra]",
    ".rel",
    "pkg..mod",
    "pkg.class",
]


@pytest.mark.parametrize("value", _MALFORMED_MODULE_VALUES)
def test_an_entry_point_value_that_is_not_a_module_path_is_refused(value: str) -> None:
    """Test a value that cannot be imported as a module is an error."""
    with pytest.raises(ImportError) as caught:
        _select_native_module({}, [_entry_point("product", value)])

    message = str(caught.value)
    assert "The entry point 'product' in the group 'fhy_core.native'" in message
    assert f"has the value {value!r}" in message
    assert "importable module path" in message
    assert "no attribute or extras" in message
    assert NATIVE_MODULE_ENVIRONMENT_VARIABLE in message


def test_a_malformed_entry_point_value_is_refused_naming_the_fix() -> None:
    """Test the refusal of a malformed value names the cause and the fix."""
    with pytest.raises(ImportError) as caught:
        _select_native_module({}, [_entry_point("product", "pkg.mod:attr")])

    assert str(caught.value) == (
        "The entry point 'product' in the group 'fhy_core.native' has the "
        "value 'pkg.mod:attr', but an entry point's value is the native "
        "module, an importable module path such as 'my_product._native' with "
        "no attribute or extras. Fix the entry point, or name the module to "
        "use in the environment variable FHY_CORE_NATIVE_MODULE."
    )


@pytest.mark.parametrize("value", _MALFORMED_MODULE_VALUES)
def test_a_malformed_entry_point_value_is_ignored_under_the_variable(
    value: str,
) -> None:
    """Test the variable overrides strict values, as it does strict names."""
    selection = _select_native_module(
        {NATIVE_MODULE_ENVIRONMENT_VARIABLE: "native_named"},
        [_entry_point("product", value), _entry_point("other", "native_other")],
    )

    assert selection.module == "native_named"
    assert [candidate.module for candidate in selection.candidates] == ["native_other"]


def test_an_invalid_name_with_a_malformed_value_is_refused_for_its_name() -> None:
    """Test an entry point bad in both ways is refused for its name."""
    with pytest.raises(ImportError) as caught:
        _select_native_module(
            {}, [_entry_point("not-an-identifier", "my-product._native")]
        )

    message = str(caught.value)
    assert "'not-an-identifier'" in message
    assert "names the module 'my-product._native'" in message
    assert "not a valid Python identifier" in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_malformed_entry_point_value_is_refused_end_to_end(
    tmp_path: pathlib.Path,
) -> None:
    """Test an installed malformed value fails the import naming its owner."""
    write_distribution(tmp_path, "product", {"product": "my-product._native"})

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "The entry point 'product' of 'product'" in message
    assert "has the value 'my-product._native'" in message


def test_a_module_value_is_read_without_surrounding_space() -> None:
    """Test a value padded with spaces names the module it surrounds."""
    selection = _select_native_module({}, [_entry_point("product", " native_p ")])

    assert selection.module == "native_p"


@pytest.mark.parametrize("layer", ["class", "None", "not-a-layer", "1st", ""])
def test_get_layer_version_attribute_refuses_a_name_that_is_not_a_layer(
    layer: str,
) -> None:
    """Test a keyword or non-identifier layer name raises `ValueError`."""
    with pytest.raises(ValueError, match="identifier"):
        get_layer_version_attribute(layer)


def test_an_invalid_entry_point_name_is_ignored_under_the_variable() -> None:
    """Test the variable overrides strict names too."""
    selection = _select_native_module(
        {NATIVE_MODULE_ENVIRONMENT_VARIABLE: "native_named"},
        [_entry_point("not-an-identifier", "native_bad")],
    )

    assert selection.module == "native_named"
    assert selection.candidates == ()


def test_the_environment_variable_selects_over_the_entry_points() -> None:
    """Test the named module wins, even over modules that conflict."""
    entry_points = [
        _entry_point("one", "native_one"),
        _entry_point("two", "native_two"),
    ]

    assert _select(entry_points, {NATIVE_MODULE_ENVIRONMENT_VARIABLE: " named "}) == (
        "named",
        "environment",
    )


@pytest.mark.parametrize("value", [".rel", "a b", "pkg.mod:attr", "my-product.x"])
def test_an_environment_variable_that_is_not_a_module_path_is_refused(
    value: str,
) -> None:
    """Test the variable must name an importable module."""
    with pytest.raises(ImportError) as caught:
        _select_native_module(
            {NATIVE_MODULE_ENVIRONMENT_VARIABLE: value},
            [_entry_point("product", "native_product")],
        )

    assert str(caught.value) == (
        f"The environment variable FHY_CORE_NATIVE_MODULE is set to {value!r}, "
        "which is not an importable module path (dot-separated identifiers "
        "that are not keywords). Set it to the module to use, such as "
        "'fhy_core._rs', or unset it to let fhy_core choose."
    )


def test_a_blank_environment_variable_is_ignored() -> None:
    """Test a variable that is empty or blank does not select a module."""
    assert _select([], {NATIVE_MODULE_ENVIRONMENT_VARIABLE: "  "}) == (
        "fhy_core._rs",
        "own",
    )


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_module_that_is_not_selected_is_never_imported(
    tmp_path: pathlib.Path,
) -> None:
    """Test a lower layer's module is not imported when a higher one is chosen."""
    version = importlib.metadata.version("fhy_core")
    write_distribution(tmp_path, "lower", {"base": "native_lower"})
    write_distribution(
        tmp_path, "upper", {"base": "native_upper", "top": "native_upper"}
    )
    write_native_module(
        tmp_path,
        "native_lower",
        {},
        fhy_core_version=version,
        on_import="raise RuntimeError('the lower module was imported')",
    )
    write_native_module(
        tmp_path, "native_upper", {"base": "0", "top": "0"}, fhy_core_version=None
    )

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "'native_upper'" in message
    assert "native_lower" not in message
    assert "was imported" not in message


def test_a_selected_module_raising_on_import_is_refused_with_its_cause(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test any exception importing the module becomes the `ImportError` cause."""
    write_native_module(
        tmp_path,
        "native_raising_on_import",
        {},
        fhy_core_version=importlib.metadata.version("fhy_core"),
        on_import="raise RuntimeError('broken at import')",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ImportError) as caught:
        _extension._load_aggregate_extension(
            Selection("native_raising_on_import", "environment", ())
        )

    assert "'native_raising_on_import'" in str(caught.value)
    assert "failed to import (RuntimeError: broken at import)" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "native_raising_on_import" not in sys.modules


def test_a_variable_module_that_fails_to_import_names_the_variable() -> None:
    """Test the advice for a module the variable chose is about the variable."""
    with pytest.raises(ImportError) as caught:
        _extension._load_aggregate_extension(
            Selection("native_absent_by_variable", "environment", ())
        )

    assert str(caught.value) == (
        "FHY_CORE_NATIVE_MODULE=native_absent_by_variable selected the native "
        "extension module 'native_absent_by_variable', but it failed to import "
        "(ModuleNotFoundError: No module named 'native_absent_by_variable'). "
        "Unset FHY_CORE_NATIVE_MODULE to let fhy_core choose, or set it to an "
        "importable module."
    )


def test_an_advertised_module_that_fails_to_import_names_its_entry_points() -> None:
    """Test the advice for an advertised module is about its distribution."""
    candidate = Candidate(
        "native_absent_advertised",
        (_entry_point("product", "native_absent_advertised"),),
    )

    with pytest.raises(ImportError) as caught:
        _extension._load_aggregate_extension(
            Selection("native_absent_advertised", "entry_points", (candidate,))
        )

    assert str(caught.value) == (
        "fhy_core selected the native extension module "
        "'native_absent_advertised' (entry point 'product') from the installed "
        "entry points, but it failed to import (ModuleNotFoundError: No module "
        "named 'native_absent_advertised'). Reinstall the distribution that "
        "provides it, or uninstall that distribution."
    )


def _candidate(*names: str) -> Candidate:
    """Return the candidate `native_fake` that the named layers claim."""
    return Candidate(
        "native_fake", tuple(_entry_point(n, "native_fake") for n in names)
    )


def _fake_module(**attributes: object) -> ModuleType:
    module = ModuleType("native_fake")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def test_the_version_attribute_of_a_layer_is_dunder_version() -> None:
    """Test a layer's attribute is its name between `__` and `_version__`."""
    assert get_layer_version_attribute("moga_sim") == "__moga_sim_version__"
    with pytest.raises(ValueError, match="identifier"):
        get_layer_version_attribute("not-a-layer")


def test_a_module_holding_every_claimed_layer_is_verified() -> None:
    """Test claims backed by version attributes pass; `fhy_core` is exempt."""
    module = _fake_module(__a_version__="1", __b_version__="2")

    _verify_declared_layers("native_fake", module, _candidate("a", "b", "fhy_core"))


def test_a_module_with_no_entry_points_is_not_verified() -> None:
    """Test a module no entry point advertises claims nothing to verify."""
    _verify_declared_layers("native_fake", _fake_module(), None)


@pytest.mark.parametrize(
    "attributes",
    [{}, {"__b_version__": 2}, {"__b_version__": None}],
    ids=["absent", "not-a-string", "none"],
)
def test_a_claimed_layer_without_a_version_string_is_refused(
    attributes: dict[str, object],
) -> None:
    """Test a lying claim names the entry point and the missing attribute."""
    module = _fake_module(__a_version__="1", **attributes)

    with pytest.raises(ImportError) as caught:
        _verify_declared_layers("native_fake", module, _candidate("a", "b"))

    message = str(caught.value)
    assert "'native_fake'" in message
    assert "the entry point 'b' claims the layer 'b'" in message
    assert "'__b_version__'" in message
    assert "__a_version__" not in message


def test_every_missing_layer_is_named() -> None:
    """Test a module that lacks several claimed layers is refused naming each."""
    with pytest.raises(ImportError) as caught:
        _verify_declared_layers("native_fake", _fake_module(), _candidate("a", "b"))

    message = str(caught.value)
    assert "'__a_version__'" in message
    assert "'__b_version__'" in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_claim_the_module_does_not_back_is_refused_end_to_end(
    tmp_path: pathlib.Path,
) -> None:
    """Test an entry point claiming a layer the module lacks fails the import."""
    version = importlib.metadata.version("fhy_core")
    write_distribution(
        tmp_path, "product", {"fhy_core": "native_lying", "extra": "native_lying"}
    )
    write_native_module(tmp_path, "native_lying", {}, fhy_core_version=version)

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "the entry point 'extra' of 'product' claims the layer 'extra'" in message
    assert "'__extra_version__'" in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_the_variable_naming_an_advertised_module_verifies_its_claims(
    tmp_path: pathlib.Path,
) -> None:
    """Test a module the variable names is verified against its entry points."""
    version = importlib.metadata.version("fhy_core")
    write_distribution(
        tmp_path, "product", {"fhy_core": "native_lying", "extra": "native_lying"}
    )
    write_native_module(tmp_path, "native_lying", {}, fhy_core_version=version)

    message = read_import_error(
        run_python(
            _PRINT_EXTENSION_NAME,
            python_path=[tmp_path],
            environment={NATIVE_MODULE_ENVIRONMENT_VARIABLE: "native_lying"},
        )
    )

    assert "the entry point 'extra' of 'product' claims the layer 'extra'" in message
    assert "'__extra_version__'" in message


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_claim_on_the_own_module_it_does_not_back_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """Test an entry point naming `fhy_core._rs` has its claim verified too."""
    write_distribution(tmp_path, "product", {"foo": "fhy_core._rs"})

    message = read_import_error(
        run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])
    )

    assert "the entry point 'foo' of 'product' claims the layer 'foo'" in message
    assert "'__foo_version__'" in message


def _write_proxy_aggregate(site_directory: pathlib.Path, module: str) -> None:
    """Write an aggregate that is a fresh load of this build of `fhy_core._rs`.

    Importing `module` loads the built extension file again, sets its
    ``__fhy_core_version__`` to its ``__version__`` and makes the loaded
    extension the module, so the package runs on it in full.
    """
    (site_directory / f"{module}.py").write_text(
        "import importlib.util, sys\n"
        "_spec = importlib.util.spec_from_file_location(\n"
        f"    'fhy_core._rs', {_rs.__file__!r}\n"
        ")\n"
        "_extension = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_extension)\n"
        "_extension.__fhy_core_version__ = _extension.__version__\n"
        "sys.modules[__name__] = _extension\n"
    )


_FAKE_PACKAGE_VERSION = (
    "import importlib.metadata\n"
    "_real_version = importlib.metadata.version\n"
    "importlib.metadata.version = (\n"
    "    lambda name: '9.9.9' if name == 'fhy_core' else _real_version(name)\n"
    ")\n"
)
_RESTORE_PACKAGE_VERSION = "importlib.metadata.version = _real_version\n"


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("entry_points", "before", "after", "expected_refusal"),
    [
        ({}, _FAKE_PACKAGE_VERSION, _RESTORE_PACKAGE_VERSION, "so it is stale"),
        (
            {"foo": "fhy_core._rs"},
            "",
            "",
            "the entry point 'foo' of 'product' claims the layer 'foo'",
        ),
    ],
    ids=["stale", "undeclared-layer"],
)
def test_a_rejected_own_extension_is_not_kept_for_a_retry(
    tmp_path: pathlib.Path,
    entry_points: dict[str, str],
    before: str,
    after: str,
    expected_refusal: str,
) -> None:
    """Test a retry under the variable loads its module, not the refused `_rs`."""
    write_distribution(tmp_path, "product", entry_points)
    _write_proxy_aggregate(tmp_path, "native_proxy")
    program = (
        "import os, sys\n"
        f"{before}"
        "try:\n"
        "    import fhy_core\n"
        "except ImportError as error:\n"
        "    print('refused:', error)\n"
        f"{after}"
        "print('fhy_core._rs' in sys.modules)\n"
        f"os.environ[{NATIVE_MODULE_ENVIRONMENT_VARIABLE!r}] = 'native_proxy'\n"
        "import fhy_core\n"
        "from fhy_core.identifier import Identifier\n"
        "print(fhy_core._rs is sys.modules['native_proxy'])\n"
        "print(fhy_core.native.get_selection().reason)\n"
        "first = Identifier('first')\n"
        "print(Identifier('second').id - first.id)\n"
    )

    completed = run_python(program, python_path=[tmp_path])

    assert completed.returncode == 0, completed.stderr
    refusal, *rest = completed.stdout.splitlines()
    assert refusal.startswith("refused: ")
    assert expected_refusal in refusal
    assert rest == ["False", "True", "environment", "1"]


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_rejected_extension_held_before_the_import_is_kept() -> None:
    """Test the loader removes only an extension the failed import loaded."""
    program = (
        _create_stale_extension_prefix("_fake_extension.__version__ = '0.0.0-stale'")
        + "try:\n"
        "    import fhy_core\n"
        "except ImportError:\n"
        "    pass\n"
        "print(sys.modules['fhy_core._rs'] is _fake_extension)\n"
    )

    completed = run_python(program)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["True"]


@pytest.mark.slow
@pytest.mark.subprocess
def test_an_entry_point_naming_the_own_module_as_fhy_core_is_accepted(
    tmp_path: pathlib.Path,
) -> None:
    """Test advertising `fhy_core._rs` for the layer `fhy_core` imports."""
    write_distribution(tmp_path, "product", {"fhy_core": "fhy_core._rs"})

    completed = run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["fhy_core._rs"]


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_claim_the_module_backs_is_accepted_end_to_end(
    tmp_path: pathlib.Path,
) -> None:
    """Test a module that reports a claimed layer passes verification."""
    version = importlib.metadata.version("fhy_core")
    write_distribution(tmp_path, "product", {"extra": "native_honest"})
    write_native_module(
        tmp_path, "native_honest", {"extra": "1"}, fhy_core_version=version
    )

    completed = run_python(_PRINT_EXTENSION_NAME, python_path=[tmp_path])

    # The fake holds none of the binding, so the package fails later, not in
    # the verification of the claims.
    assert completed.returncode != 0
    assert "claims the layer" not in completed.stderr
    assert "no attribute 'PartiallyOrderedSet'" in completed.stderr


def _install_selection(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    selection: Selection,
) -> None:
    """Make `module` the native module of the process, chosen as `selection`."""
    monkeypatch.setitem(sys.modules, "fhy_core._rs", module)
    monkeypatch.setattr(_extension, "_SELECTION", selection)


@pytest.fixture
def demo_package() -> Iterable[ModuleType]:
    """Provide a package ``demo_layer`` in `sys.modules`, restored after."""
    package = ModuleType("demo_layer")
    sys.modules["demo_layer"] = package
    yield package
    sys.modules.pop("demo_layer", None)
    sys.modules.pop("demo_layer._rs", None)


def test_the_selection_of_this_process_is_public() -> None:
    """Test `fhy_core.native.get_selection` reports what the loader chose."""
    chosen = native.get_selection()

    loaded = _extension._SELECTION
    assert loaded is not None
    assert (chosen.module, chosen.reason, chosen.layers) == (
        loaded.module,
        loaded.reason,
        _extension.read_module_layers(_rs),
    )
    assert chosen.module == _rs.__name__
    assert chosen.reason != "own" or chosen.module == "fhy_core._rs"
    assert "fhy_core" in chosen.layers
    assert native.ENTRY_POINT_GROUP == NATIVE_ENTRY_POINT_GROUP
    assert native.MODULE_ENVIRONMENT_VARIABLE == NATIVE_MODULE_ENVIRONMENT_VARIABLE


def test_the_current_selection_is_an_error_before_one_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test reading the selection before the loader made one raises."""
    monkeypatch.setattr(_extension, "_SELECTION", None)

    with pytest.raises(RuntimeError, match="has not selected its native module"):
        get_current_selection()
    with pytest.raises(RuntimeError, match="has not selected its native module"):
        native.get_selection()


def test_the_selection_reports_the_layers_its_module_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the layers are those the module reports, advertised or not."""
    module = _fake_module(__fhy_core_version__="0", __demo_version__="1")
    _install_selection(monkeypatch, module, Selection("native_fake", "environment", ()))

    assert native.get_selection().layers == {"fhy_core", "demo"}


def test_only_string_versions_of_layer_names_are_layers() -> None:
    """Test attributes that are not a layer's string version are ignored."""
    module = _fake_module(
        __version__="1",
        __a_version__="1",
        __b_version__=2,
        __c_version__=None,
        _d_version__="1",
        __e_version="1",
        **{
            "__class_version__": "1",
            "__not-a-layer_version__": "1",
            "__1st_version__": "1",
        },
    )

    assert _extension.read_module_layers(module) == {"fhy_core", "a"}


@pytest.mark.skipif(
    _has_native_entry_points(),
    reason="a combined extension is installed, so the package does not run alone",
)
def test_the_own_module_contains_the_fhy_core_layer_alone() -> None:
    """Test `fhy_core._rs`, which reports `__version__`, holds `fhy_core` only."""
    assert _extension.read_module_layers(_rs) == {"fhy_core"}
    assert native.get_selection().layers == {"fhy_core"}


def test_require_layer_installs_the_module_for_the_package(
    monkeypatch: pytest.MonkeyPatch, demo_package: ModuleType
) -> None:
    """Test a present layer returns the module and installs it as `_rs`."""
    module = _fake_module(__demo_layer_version__="1.2.3rc1")
    chosen = Selection("native_fake", "entry_points", (_candidate("demo_layer"),))
    _install_selection(monkeypatch, module, chosen)

    returned = native.require_layer("demo_layer", "1.2.3-rc.1")

    assert returned is module
    assert sys.modules["demo_layer._rs"] is module
    assert demo_package._rs is module


def test_require_layer_installs_under_the_named_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `package` names where the module is installed, and is set on it."""
    module = _fake_module(__demo_layer_version__="1")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))
    other = ModuleType("demo_other")
    monkeypatch.setitem(sys.modules, "demo_other", other)
    try:
        native.require_layer("demo_layer", "1", package="demo_other")

        assert sys.modules["demo_other._rs"] is module
        assert other._rs is module
        assert "demo_layer._rs" not in sys.modules
    finally:
        sys.modules.pop("demo_other._rs", None)


@pytest.mark.parametrize("package", ["", "a b", "pkg.", ".rel", "class"])
def test_require_layer_refuses_a_package_that_is_not_a_module_path(
    monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    """Test a package name that cannot be imported is the caller's error."""
    module = _fake_module(__demo_layer_version__="1")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))

    with pytest.raises(ValueError) as caught:
        native.require_layer("demo_layer", "1", package=package)

    assert str(caught.value) == (
        f"a package is named by an importable module path, not {package!r}"
    )
    assert f"{package}._rs" not in sys.modules
    assert "demo_layer._rs" not in sys.modules


@pytest.mark.parametrize(
    ("layer", "package"),
    [("demo_layer", "typo_pkg"), ("demo_unimported", None)],
    ids=["package", "layer"],
)
def test_require_layer_refuses_a_package_that_is_not_imported(
    monkeypatch: pytest.MonkeyPatch, layer: str, package: str | None
) -> None:
    """Test the owner must be imported, so its `_rs` is set on it."""
    module = _fake_module(**{f"__{layer}_version__": "1"})
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))
    owner = package or layer

    with pytest.raises(ValueError) as caught:
        native.require_layer(layer, "1", package=package)

    assert str(caught.value) == (
        f"the package {owner!r} is not in sys.modules; call require_layer "
        "while that package is being imported"
    )
    assert f"{owner}._rs" not in sys.modules


@pytest.mark.usefixtures("demo_package")
def test_require_layer_refuses_a_package_that_holds_another_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a package's own `_rs` is never replaced by another module."""
    module = _fake_module(__demo_layer_version__="1")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))
    other = ModuleType("demo_layer._rs")
    other.__file__ = "/elsewhere/demo_layer/_rs.so"
    monkeypatch.setitem(sys.modules, "demo_layer._rs", other)

    with pytest.raises(ImportError, match="already another module") as caught:
        native.require_layer("demo_layer", "1")

    message = str(caught.value)
    assert "(loaded from /elsewhere/demo_layer/_rs.so)" in message
    assert "'native_fake'" in message
    assert "Uninstall the package that provides the other module" in message


@pytest.mark.usefixtures("demo_package")
def test_require_layer_refuses_a_held_module_without_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the held-module error omits the location when there is no file."""
    module = _fake_module(__demo_layer_version__="1")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))
    monkeypatch.setitem(sys.modules, "demo_layer._rs", ModuleType("demo_layer._rs"))

    with pytest.raises(ImportError, match="already another module") as caught:
        native.require_layer("demo_layer", "1")

    message = str(caught.value)
    assert "loaded from" not in message
    assert "'demo_layer._rs' is already another module than" in message


def test_require_layer_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, demo_package: ModuleType
) -> None:
    """Test requiring a layer twice returns the same module and changes nothing."""
    module = _fake_module(__demo_layer_version__="1")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))

    first = native.require_layer("demo_layer", "1")
    second = native.require_layer("demo_layer", "1")

    assert first is second is module
    assert sys.modules["demo_layer._rs"] is module
    assert demo_package._rs is module


@pytest.mark.parametrize("layer", ["class", "not-a-layer"])
def test_require_layer_refuses_a_name_that_is_not_a_layer(layer: str) -> None:
    """Test a keyword or non-identifier layer raises `ValueError`."""
    with pytest.raises(ValueError, match="identifier"):
        native.require_layer(layer, "1")


@pytest.mark.usefixtures("demo_package")
def test_require_layer_names_the_variable_when_it_selected_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a layer the variable's module lacks names the modules that have it."""
    chosen = Selection(
        "native_low",
        "environment",
        (_candidate("demo_layer"), Candidate("native_low", ())),
    )
    _install_selection(monkeypatch, _fake_module(), chosen)

    with pytest.raises(ImportError) as caught:
        native.require_layer("demo_layer", "1")

    assert str(caught.value) == (
        "FHY_CORE_NATIVE_MODULE=native_low selected 'native_low', which does "
        "not contain the layer 'demo_layer' (no __demo_layer_version__). "
        "Installed modules that do: 'native_fake' (entry point 'demo_layer'). "
        "Unset the variable to let fhy_core choose, or set it to one of those."
    )


@pytest.mark.usefixtures("demo_package")
def test_require_layer_with_the_variable_and_no_holder_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test no installed module declaring the layer is said, not hidden."""
    chosen = Selection("native_low", "environment", ())
    _install_selection(monkeypatch, _fake_module(), chosen)

    with pytest.raises(ImportError, match="No installed distribution declares"):
        native.require_layer("demo_layer", "1")


@pytest.mark.usefixtures("demo_package")
@pytest.mark.parametrize("reason", ["entry_points", "own"])
def test_require_layer_names_the_missing_declaration(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """Test an undeclared layer says to reinstall the aggregate's package."""
    chosen = Selection(
        "native_fake",
        reason,  # type: ignore[arg-type]
        (_candidate("other_layer"),),
    )
    module = _fake_module(__other_layer_version__="1", __unadvertised_version__="1")
    _install_selection(monkeypatch, module, chosen)

    with pytest.raises(ImportError) as caught:
        native.require_layer("demo_layer", "1")

    assert str(caught.value) == (
        "The layer 'demo_layer' (version '1') is not in the native module "
        "'native_fake' that fhy_core runs on, which contains 'fhy_core', "
        "'other_layer', 'unadvertised', and no installed distribution "
        "declares it in the "
        "entry point group 'fhy_core.native'. Reinstall the package that "
        "provides the aggregate containing 'demo_layer'."
    )


@pytest.mark.usefixtures("demo_package")
def test_require_layer_refuses_a_stale_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a layer of another version names both versions."""
    module = _fake_module(__demo_layer_version__="0.9.0")
    chosen = Selection("native_fake", "entry_points", (_candidate("demo_layer"),))
    _install_selection(monkeypatch, module, chosen)

    with pytest.raises(ImportError) as caught:
        native.require_layer("demo_layer", "1.0.0")

    assert str(caught.value) == (
        "The layer 'demo_layer' needs version '1.0.0', but the native module "
        "'native_fake' that fhy_core runs on reports version '0.9.0' in "
        "__demo_layer_version__, so it is stale. Rebuild or reinstall the "
        "package that provides the module."
    )


@pytest.mark.parametrize("version", ["dev", "", "1.0.0-foo.1"])
def test_require_layer_refuses_a_version_that_is_not_pep440(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """Test a required version that is not PEP 440 is the caller's error."""
    module = _fake_module(__demo_layer_version__=version)
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))

    with pytest.raises(ValueError) as caught:
        native.require_layer("demo_layer", version)

    assert str(caught.value) == (
        f"a layer's version is a PEP 440 version, not {version!r}"
    )
    assert "demo_layer._rs" not in sys.modules


def test_require_layer_checks_the_version_before_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a bad version is refused even where the module lacks the layer."""
    _install_selection(monkeypatch, _fake_module(), Selection("native_fake", "own", ()))

    with pytest.raises(ValueError, match="PEP 440"):
        native.require_layer("demo_layer", "dev")


@pytest.mark.usefixtures("demo_package")
def test_require_layer_refuses_a_reported_version_that_is_not_pep440(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a module reporting a non-PEP 440 version of the layer is stale."""
    module = _fake_module(__demo_layer_version__="garbage")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))

    with pytest.raises(ImportError) as caught:
        native.require_layer("demo_layer", "1.0.0")

    assert str(caught.value) == (
        "The layer 'demo_layer' needs version '1.0.0', but the native module "
        "'native_fake' that fhy_core runs on reports 'garbage' in "
        "__demo_layer_version__, which is not a PEP 440 version, so it is "
        "stale. Rebuild or reinstall the package that provides the module."
    )
    assert "demo_layer._rs" not in sys.modules


def test_require_layer_accepts_versions_equal_in_normal_form(
    monkeypatch: pytest.MonkeyPatch, demo_package: ModuleType
) -> None:
    """Test a Cargo-form report meets a PEP 440 requirement, and the reverse."""
    module = _fake_module(__demo_layer_version__="1.2.3-rc.1")
    _install_selection(monkeypatch, module, Selection("native_fake", "own", ()))

    assert native.require_layer("demo_layer", "1.2.3rc1") is module
    assert native.require_layer("demo_layer", "v1.2.3-RC1") is module


@pytest.mark.usefixtures("demo_package")
def test_require_layer_treats_a_non_string_version_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a version attribute that is not a string does not count."""
    chosen = Selection("native_fake", "own", ())
    _install_selection(monkeypatch, _fake_module(__demo_layer_version__=1), chosen)

    with pytest.raises(ImportError, match="is not in the native module"):
        native.require_layer("demo_layer", "1")
