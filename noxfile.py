"""Task automation for FhY Core, driven by uv-backed nox sessions."""

import pathlib

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "type_check", "tests", "coverage"]

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
ROOT = pathlib.Path(__file__).parent
SOURCES = ["src", "tests"]


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
def tests(session: nox.Session) -> None:
    """Run the unit and integration test suite under coverage."""
    _sync(session, "test")
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
    """Run hypothesis-based property tests (CI release gate; opt-in locally)."""
    _sync(session, "property")
    # An empty collection (pytest exit 5) is a failure: it means the property
    # suites lost their `property` marker and the gate would pass vacuously.
    session.run("pytest", "-m", "property", *session.posargs)


@nox.session
def mutation(session: nox.Session) -> None:
    """Run cosmic-ray mutation testing (opt-in)."""
    _sync(session, "mutation")
    session.run("cosmic-ray", "init", "cosmic-ray.toml", "cosmic-ray.sqlite")
    session.run("cosmic-ray", "exec", "cosmic-ray.toml", "cosmic-ray.sqlite")
    session.run("cr-report", "cosmic-ray.sqlite")
