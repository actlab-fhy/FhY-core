"""Tests selecting the Rust-backed or pure-Python backend at import.

The package runs on the compiled extension ``fhy_core._rs`` iff it imports
and ``FHY_CORE_NO_EXTENSIONS`` does not disable it; the choice is reported
as ``fhy_core.RUST_BACKEND_AVAILABLE``. The selection happens once, while
the package is imported, so each case other than the running process's own
is checked in a fresh interpreter.
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


def _report_backend(
    variable_value: str | None, *, program_prefix: str = ""
) -> tuple[str, str]:
    """Return a fresh interpreter's backend flag and its id step, as printed.

    Args:
        variable_value: Value for ``FHY_CORE_NO_EXTENSIONS``, or ``None`` to
            leave it unset.
        program_prefix: Source run before the package is imported.

    Returns:
        The printed ``RUST_BACKEND_AVAILABLE`` value, then the printed
        difference between two consecutively constructed ids.

    """
    environment = dict(os.environ)
    environment.pop(_NO_EXTENSIONS_VARIABLE, None)
    if variable_value is not None:
        environment[_NO_EXTENSIONS_VARIABLE] = variable_value
    completed = subprocess.run(
        [sys.executable, "-c", program_prefix + _REPORT_BACKEND_PROGRAM],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
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
def test_an_extension_that_fails_to_import_selects_the_python_backend() -> None:
    """Test a failing extension import falls back to the pure-Python backend."""
    # A `None` entry in `sys.modules` makes importing that name raise
    # `ImportError`, as a broken or missing extension would.
    block_extension = "import sys\nsys.modules['fhy_core._rs'] = None\n"

    assert _report_backend(None, program_prefix=block_extension) == ("False", "1")
