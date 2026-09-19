"""Task automation for FhY Core, driven by uv-backed nox sessions."""

import pathlib

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "type_check", "tests", "coverage"]

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
ROOT = pathlib.Path(__file__).parent
SOURCES = ["src", "tests"]
# `FHY_CORE_NO_EXTENSIONS` value that selects each backend for a test run.
BACKEND_EXTENSION_SETTINGS = {"rust": "0", "python": "1"}


def _sync(session: nox.Session, *groups: str) -> None:
    """Install the project and the named dependency groups into the session env."""
    args = ["uv", "sync", "--no-default-groups"]
    for group in groups:
        args += ["--group", group]
    session.run_install(
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session(python=PYTHONS)
@nox.parametrize("backend", list(BACKEND_EXTENSION_SETTINGS))
def tests(session: nox.Session, backend: str) -> None:
    """Run the unit and integration test suite under coverage on one backend.

    ``FHY_CORE_NO_EXTENSIONS`` selects the backend, and the session fails
    before testing unless the package reports the backend it was asked for,
    so an extension that silently fails to import cannot pass as a Rust run.
    """
    _sync(session, "test")
    session.env["FHY_CORE_NO_EXTENSIONS"] = BACKEND_EXTENSION_SETTINGS[backend]
    is_rust_expected = backend == "rust"
    session.run(
        "python",
        "-c",
        "import sys, fhy_core; "
        f"sys.exit(None if fhy_core.RUST_BACKEND_AVAILABLE is {is_rust_expected} "
        f"else 'expected RUST_BACKEND_AVAILABLE to be {is_rust_expected}')",
    )
    # Start coverage inside pytest-xdist worker subprocesses.
    purelib = session.run(
        "python",
        "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))",
        silent=True,
    ).strip()
    pathlib.Path(purelib, "cov.pth").write_text(
        "import coverage; coverage.process_startup()"
    )
    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "-m",
        "slow or not slow",
        *session.posargs,
        env={"COVERAGE_PROCESS_START": str(ROOT / "pyproject.toml")},
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Check linting and formatting with ruff."""
    _sync(session, "lint")
    session.run("ruff", "check", *SOURCES)
    session.run("ruff", "format", "--check", *SOURCES)


@nox.session
def type_check(session: nox.Session) -> None:
    """Run static type analysis: ty (advisory) and mypy --strict (gate)."""
    _sync(session, "type")
    # ty is preview-stage; report its findings but do not fail the session
    # (exit 1 == diagnostics found). mypy --strict is the enforcing gate.
    session.run("ty", "check", *SOURCES, success_codes=[0, 1])
    session.run("mypy", *SOURCES)


@nox.session
def coverage(session: nox.Session) -> None:
    """Combine per-version coverage data and report.

    Run a ``tests`` session first to produce the ``.coverage.*`` data files;
    ``coverage combine`` consumes them, so this session has nothing to do on a
    clean tree (or if run twice in a row).
    """
    _sync(session, "test")
    if not list(ROOT.glob(".coverage.*")):
        session.skip(
            "No .coverage.* data to combine. Run `nox -s tests` (or `nox -s "
            "tests-3.12`) first; note that `coverage combine` consumes the data."
        )
    session.run("coverage", "combine")
    session.run("coverage", "report")
    session.run("coverage", "xml")


@nox.session
def property(session: nox.Session) -> None:
    """Run hypothesis-based property tests under the thorough profile.

    This is the CI release gate (opt-in locally); it forces
    ``HYPOTHESIS_PROFILE=thorough`` regardless of the caller's environment.
    """
    _sync(session, "property")
    # No success_codes override: exit 5 (nothing collected) must fail, so a
    # marker typo or a collection error cannot pass as a clean run.
    session.run(
        "pytest",
        "-m",
        "property",
        *session.posargs,
        env={"HYPOTHESIS_PROFILE": "thorough"},
    )


@nox.session
def mutation(session: nox.Session) -> None:
    """Run cosmic-ray mutation testing for one module (opt-in).

    `nox -s mutation -- lattice`.
    """
    module = session.posargs[0] if session.posargs else "lattice"
    config = ROOT / "cosmic-ray" / f"{module}.toml"
    if not config.is_file():
        available = ", ".join(
            sorted(path.stem for path in config.parent.glob("*.toml"))
        )
        session.error(f"no mutation config for {module!r}; available: {available}")
    _sync(session, "mutation")
    # Every mutant has to see the same Hypothesis draws, and none may replay a
    # counterexample saved while testing another, so the run uses the
    # derandomized, database-free `mutation` profile from tests/conftest.py.
    session.env["HYPOTHESIS_PROFILE"] = "mutation"
    # Cosmic-ray scores a mutant whose test run overruns the config's timeout as
    # killed, so stop before mutating anything if the unmutated suite fails or
    # does not finish within that timeout.
    session.run("cosmic-ray", "baseline", str(config))
    database = ROOT / "session.sqlite"
    database.unlink(missing_ok=True)
    session.run("cosmic-ray", "init", str(config), str(database))
    session.run("cr-filter-pragma", str(database))
    session.run("cosmic-ray", "exec", str(config), str(database))
    report = ROOT / "mutation-report.html"
    with report.open("w") as report_file:
        session.run("cr-html", str(database), stdout=report_file, stderr=None)
    session.run("cr-report", str(database))
    session.log(f"Report: {report}")
