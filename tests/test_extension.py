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

import pytest

from fhy_core import _rs
from fhy_core._extension import (
    _find_foreign_extension_builds,
    _normalize_pep440_version,
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
    assert _normalize_pep440_version(_rs.__version__) == importlib.metadata.version(
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
    assert _normalize_pep440_version(cargo_version) == package_version


@pytest.mark.parametrize("version", ["0.3.0-foo.1", "0.0.0-stale", "", "stale"])
def test_a_non_pep440_version_does_not_normalize(version: str) -> None:
    """Test a version maturin cannot build a package from has no normal form."""
    assert _normalize_pep440_version(version) is None


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
