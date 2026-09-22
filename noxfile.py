"""Task automation for FhY Core, driven by uv-backed nox sessions."""

import difflib
import json
import pathlib

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "type_check", "tests", "coverage"]

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
ROOT = pathlib.Path(__file__).parent
SOURCES = ["src", "tests"]
# `FHY_CORE_NO_EXTENSIONS` value that selects each backend for a test run.
BACKEND_EXTENSION_SETTINGS = {"rust": "0", "python": "1"}
GOLDEN_DIRECTORY = ROOT / "rust" / "tests" / "golden"
# Diff lines shown for each stale golden corpus; the rest are elided.
GOLDEN_DIFF_LINE_LIMIT = 60


def _sync(session: nox.Session, *groups: str) -> None:
    """Install the project and the named dependency groups into the session env."""
    args = ["uv", "sync", "--no-default-groups"]
    for group in groups:
        args += ["--group", group]
    session.run_install(
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


def _canonicalize_golden_corpus(path: pathlib.Path) -> str:
    """Return a golden corpus as sorted-key JSON text, without its provenance.

    The provenance block records the commit and interpreter that generated the
    corpus, so it differs between runs. Comparing text rather than parsed
    values keeps ``true`` distinct from ``1`` and ``1`` from ``1.0``.
    """
    with path.open(encoding="utf-8") as corpus_file:
        document = json.load(corpus_file)
    document.pop("provenance", None)
    return json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True)


def _select_backend(session: nox.Session, backend: str) -> None:
    """Select the session's backend and fail unless the package reports it."""
    session.env["FHY_CORE_NO_EXTENSIONS"] = BACKEND_EXTENSION_SETTINGS[backend]
    is_rust_expected = backend == "rust"
    session.run(
        "python",
        "-c",
        "import sys, fhy_core; "
        f"sys.exit(None if fhy_core.RUST_BACKEND_SELECTED is {is_rust_expected} "
        f"else 'expected RUST_BACKEND_SELECTED to be {is_rust_expected}')",
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
    _select_backend(session, backend)
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
def golden(session: nox.Session) -> None:
    """Check that the committed golden corpora match their generators.

    Each ``generate_*.py`` under ``rust/tests/golden/`` replays seeded scripts
    through the Python implementation, the oracle the Rust equivalence tests
    replay, and writes the corpus named after it (``generate_X.py`` writes
    ``X.json``). The session regenerates every corpus on the pure-Python
    backend into a temporary directory and fails if any differs from the
    committed one outside its provenance block.
    """
    _sync(session)
    _select_backend(session, "python")
    output_directory = pathlib.Path(session.create_tmp())
    stale_corpora = []
    for generator in sorted(GOLDEN_DIRECTORY.glob("generate_*.py")):
        committed = generator.with_name(
            generator.stem.removeprefix("generate_") + ".json"
        )
        if not committed.is_file():
            session.error(f"{generator.name} has no committed corpus {committed.name}")
        regenerated = output_directory / committed.name
        # Silent: the oracles log a warning per ignored re-registration. Nox
        # still prints the captured output if the generator fails.
        session.run("python", str(generator), "--output", str(regenerated), silent=True)
        committed_text = _canonicalize_golden_corpus(committed)
        regenerated_text = _canonicalize_golden_corpus(regenerated)
        if committed_text == regenerated_text:
            session.log(f"{committed.name} matches its generator")
            continue
        stale_corpora.append(committed.name)
        diff = list(
            difflib.unified_diff(
                committed_text.splitlines(),
                regenerated_text.splitlines(),
                fromfile=f"committed/{committed.name}",
                tofile=f"regenerated/{committed.name}",
                lineterm="",
            )
        )
        elided = len(diff) - GOLDEN_DIFF_LINE_LIMIT
        if elided > 0:
            diff = [*diff[:GOLDEN_DIFF_LINE_LIMIT], f"... {elided} more diff lines"]
        session.warn(
            f"{committed.name} is stale; regenerate it from the repository root "
            "with\n\n    FHY_CORE_NO_EXTENSIONS=1 uv run --no-sync python "
            f"{generator.relative_to(ROOT).as_posix()}\n\n"
            "Diff (provenance dropped, keys sorted):\n" + "\n".join(diff)
        )
    if stale_corpora:
        session.error(
            f"stale golden corpora: {', '.join(stale_corpora)} (see the diffs above)"
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
