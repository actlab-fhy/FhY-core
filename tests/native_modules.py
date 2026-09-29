"""Helpers for the tests that run ``fhy_core`` on a chosen native module.

The loader in ``fhy_core._extension`` runs once, while the package is
imported, so each test runs a fresh interpreter. A test advertises a native
module the way an installed product does, with a ``fhy_core.native`` entry
point in a ``.dist-info`` directory on ``PYTHONPATH``.
"""

import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping, Sequence


def write_distribution(
    site_directory: pathlib.Path, name: str, entry_points: Mapping[str, str]
) -> None:
    """Write the metadata of an installed distribution with entry points.

    Args:
        site_directory: Directory that goes on the interpreter's path.
        name: Distribution name.
        entry_points: Entry points of the group ``fhy_core.native``, as
            entry point name to module name.

    """
    metadata_directory = site_directory / f"{name.replace('-', '_')}-0.dist-info"
    metadata_directory.mkdir(parents=True)
    (metadata_directory / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 0\n"
    )
    lines = ["[fhy_core.native]"]
    lines += [f"{key} = {module}" for key, module in entry_points.items()]
    (metadata_directory / "entry_points.txt").write_text("\n".join(lines) + "\n")


def run_python(
    program: str,
    *,
    python_path: Sequence[pathlib.Path] = (),
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a program in a fresh interpreter and return the result.

    Args:
        program: Source to run.
        python_path: Directories put ahead of the inherited ``PYTHONPATH``.
        environment: Variables added to the inherited environment.

    Returns:
        The completed process, whose output is decoded text.

    """
    # Python 3.13+ colors tracebacks when FORCE_COLOR is set, as CI and nox
    # set it; plain text keeps the final line starting with the exception
    # name. PYTHON_COLORS takes precedence over FORCE_COLOR.
    full_environment = {**os.environ, "PYTHON_COLORS": "0", **(environment or {})}
    search_path = [str(path) for path in python_path]
    if "PYTHONPATH" in os.environ:
        search_path.append(os.environ["PYTHONPATH"])
    if search_path:
        full_environment["PYTHONPATH"] = os.pathsep.join(search_path)
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        env=full_environment,
    )


def read_import_error(completed: subprocess.CompletedProcess[str]) -> str:
    """Return the ``ImportError`` message of a failed run.

    Args:
        completed: The process that failed with an uncaught ``ImportError``.

    Returns:
        The last line of its standard error, from the exception name on.

    """
    assert completed.returncode != 0
    last_line = completed.stderr.strip().splitlines()[-1]
    assert last_line.startswith("ImportError: "), completed.stderr
    return last_line
