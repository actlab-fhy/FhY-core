"""Tests selecting the Rust-backed or pure-Python backend at import.

The package runs on the compiled extension ``fhy_core._rs`` iff it imports
and ``FHY_CORE_NO_EXTENSIONS`` does not disable it; the choice is reported
as ``fhy_core.RUST_BACKEND_AVAILABLE``. An extension that is not installed
selects the pure-Python backend silently; one that is installed but fails to
import selects it with a ``RuntimeWarning``. The selection happens once,
while the package is imported, so each case other than the running process's
own is checked in a fresh interpreter.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

import fhy_core

_NO_EXTENSIONS_VARIABLE = "FHY_CORE_NO_EXTENSIONS"

_REPORT_BACKEND_PROGRAM = (
    "import fhy_core\n"
    "from fhy_core.identifier import Identifier\n"
    "first = Identifier('first')\n"
    "second = Identifier('second')\n"
    "print(fhy_core.RUST_BACKEND_AVAILABLE, second.id - first.id)"
)


def _run_backend_report(
    variable_value: str | None,
    *,
    program_prefix: str = "",
    python_options: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the backend report in a fresh interpreter and return the result.

    Args:
        variable_value: Value for ``FHY_CORE_NO_EXTENSIONS``, or ``None`` to
            leave it unset.
        program_prefix: Source run before the package is imported.
        python_options: Interpreter options placed before ``-c``.

    Returns:
        The completed process, whose standard output holds the printed
        ``RUST_BACKEND_AVAILABLE`` value, then the printed difference
        between two consecutively constructed ids.

    """
    environment = dict(os.environ)
    environment.pop(_NO_EXTENSIONS_VARIABLE, None)
    environment.pop("PYTHONWARNINGS", None)
    if variable_value is not None:
        environment[_NO_EXTENSIONS_VARIABLE] = variable_value
    return subprocess.run(
        [
            sys.executable,
            *python_options,
            "-c",
            program_prefix + _REPORT_BACKEND_PROGRAM,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _report_backend(variable_value: str | None) -> tuple[str, str]:
    """Return a fresh interpreter's backend flag and its id step, as printed.

    Args:
        variable_value: Value for ``FHY_CORE_NO_EXTENSIONS``, or ``None`` to
            leave it unset.

    Returns:
        The printed ``RUST_BACKEND_AVAILABLE`` value, then the printed
        difference between two consecutively constructed ids.

    """
    completed = _run_backend_report(variable_value)
    assert completed.returncode == 0, completed.stderr
    backend_flag, id_step = completed.stdout.split()
    return backend_flag, id_step


def test_backend_flag_reflects_this_process_environment() -> None:
    """Test this process's flag follows the variable the test run was given.

    ``nox`` runs the suite once with the variable set to ``1`` and once with
    it set to ``0``; a plain ``pytest`` run leaves it unset.
    """
    variable_value = os.environ.get(_NO_EXTENSIONS_VARIABLE)
    is_installed = importlib.util.find_spec("fhy_core._rs") is not None

    if variable_value == "1":
        assert fhy_core.RUST_BACKEND_AVAILABLE is False
    elif variable_value in {None, "0"}:
        assert fhy_core.RUST_BACKEND_AVAILABLE is is_installed
    else:
        pytest.skip(f"{_NO_EXTENSIONS_VARIABLE}={variable_value!r} is not pinned here")


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    "variable_value",
    ["1", "true", "YES", "On", " 1 ", "anything-else"],
    ids=["one", "true", "yes-upper", "on-mixed", "padded-one", "other-text"],
)
def test_disabling_variable_value_selects_the_python_backend(
    variable_value: str,
) -> None:
    """Test a value other than empty, 0, false, no, or off disables Rust."""
    assert _report_backend(variable_value) == ("False", "1")


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    "variable_value",
    [None, "", "0", "false", "No", "OFF", " off "],
    ids=["unset", "empty", "zero", "false", "no-mixed", "off-upper", "padded-off"],
)
def test_unset_or_enabling_variable_value_selects_the_rust_backend(
    variable_value: str | None,
) -> None:
    """Test an unset, empty, 0, false, no, or off value keeps Rust selected."""
    pytest.importorskip("fhy_core._rs")

    assert _report_backend(variable_value) == ("True", "1")


@pytest.mark.slow
@pytest.mark.subprocess
def test_a_missing_extension_selects_the_python_backend_silently() -> None:
    """Test an extension that is not installed falls back without a warning."""
    # A `None` entry in `sys.modules` makes importing that name raise
    # `ModuleNotFoundError` for it, as an extension that was never built
    # would. `-W error::RuntimeWarning` turns any warning into a failure.
    hide_extension = "import sys\nsys.modules['fhy_core._rs'] = None\n"

    completed = _run_backend_report(
        None,
        program_prefix=hide_extension,
        python_options=("-W", "error::RuntimeWarning"),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["False", "1"]
    assert completed.stderr == ""


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
        ("load-failure", "simulated dlopen failure"),
        ("missing-dependency", "simulated_dependency"),
    ],
)
def test_a_broken_extension_selects_the_python_backend_with_a_warning(
    failure: str, expected_detail: str
) -> None:
    """Test an installed extension that fails to import warns, then falls back."""
    completed = _run_backend_report(
        None, program_prefix=_BROKEN_EXTENSION_PREFIXES[failure]
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["False", "1"]
    assert "RuntimeWarning" in completed.stderr
    assert "fhy_core._rs" in completed.stderr
    assert expected_detail in completed.stderr
    assert f"{_NO_EXTENSIONS_VARIABLE}=1" in completed.stderr


@pytest.mark.slow
@pytest.mark.subprocess
def test_disabling_the_extension_skips_the_broken_extension_warning() -> None:
    """Test a disabled extension is never imported, so a broken one never warns."""
    completed = _run_backend_report(
        "1",
        program_prefix=_BROKEN_EXTENSION_PREFIXES["load-failure"],
        python_options=("-W", "error::RuntimeWarning"),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["False", "1"]
    assert completed.stderr == ""
