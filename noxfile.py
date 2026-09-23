"""Task automation for FhY Core, driven by uv-backed nox sessions."""

import pathlib
import re
from typing import NamedTuple

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "type_check", "tests", "coverage"]

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
ROOT = pathlib.Path(__file__).parent
# The golden-corpus generators are the Rust port's equivalence oracle, so they
# pass the same lint and type gates as the package.
SOURCES = ["src", "tests", "rust/fhy-core/tests/golden"]
# `FHY_CORE_NO_EXTENSIONS` value that selects each backend for a test run.
BACKEND_EXTENSION_SETTINGS = {"rust": "0", "python": "1"}
GOLDEN_DIRECTORY = ROOT / "rust" / "fhy-core" / "tests" / "golden"
# `cargo test` summary of a run that replayed one expanded corpus.
_EXPANDED_REPLAY_PASSED = re.compile(r"^test result: ok\. 1 passed;", re.MULTILINE)


class ExpandedGoldenCorpus(NamedTuple):
    """How to generate and replay one generator's expanded random corpus."""

    options: str
    rust_test: str
    variable: str


# Expanded corpus settings for each generator under GOLDEN_DIRECTORY, keyed by
# file name: the generator options its docstring gives (space-separated), the
# Rust test target whose ignored test replays the corpus, and the variable that
# names the corpus file for that test.
EXPANDED_GOLDEN_CORPORA = {
    "generate_deterministic_identifier_cases.py": ExpandedGoldenCorpus(
        options="--seed 7 --random-count 2000 --max-ops 40 --hints a,b,c,d,e",
        rust_test="deterministic_identifiers_equivalence",
        variable="FHY_DETERMINISTIC_IDENTIFIER_CORPUS",
    ),
    "generate_interned_cases.py": ExpandedGoldenCorpus(
        options="--seed 7 --random-count 2000 --max-ops 60 --keys a,b,c,d,e",
        rust_test="interned_equivalence",
        variable="FHY_INTERNED_CORPUS",
    ),
    "generate_tag_type_cases.py": ExpandedGoldenCorpus(
        options="--seed 7 --random-count 2000 --max-ops 40 --slots s0,s1,s2,s3,s4",
        rust_test="tag_type_equivalence",
        variable="FHY_TAG_TYPE_CORPUS",
    ),
}


def _sync(session: nox.Session, *groups: str) -> None:
    """Install the project and the named dependency groups into the session env."""
    args = ["uv", "sync", "--no-default-groups"]
    for group in groups:
        args += ["--group", group]
    session.run_install(
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


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
@nox.parametrize("backend", list(BACKEND_EXTENSION_SETTINGS))
def property(session: nox.Session, backend: str) -> None:
    """Run hypothesis-based property tests under the thorough profile on one backend.

    This is the CI release gate (opt-in locally); it forces
    ``HYPOTHESIS_PROFILE=thorough`` regardless of the caller's environment.
    As in ``tests``, the session fails before testing unless the package
    reports the backend it was asked for, so the Rust run cannot silently skip
    the tests that need the extension.
    """
    _sync(session, "property")
    _select_backend(session, backend)
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
def golden_expanded(session: nox.Session) -> None:
    """Replay expanded random golden corpora through the Rust equivalence tests.

    The committed corpora under ``rust/fhy-core/tests/golden/`` are small enough to
    review; each generator can also write a much larger random corpus, which
    its equivalence test replays in an ignored test that reads the corpus
    path from an environment variable. The session writes every expanded
    corpus from the pure-Python oracle into a temporary directory and runs
    the matching ignored test on it. It needs ``cargo`` on ``PATH`` and fails
    if a generator has no expanded settings in ``EXPANDED_GOLDEN_CORPORA``.
    """
    generators = sorted(GOLDEN_DIRECTORY.glob("generate_*.py"))
    unconfigured = [
        generator.name
        for generator in generators
        if generator.name not in EXPANDED_GOLDEN_CORPORA
    ]
    if unconfigured:
        session.error(
            f"no expanded corpus settings for {', '.join(unconfigured)}; "
            "add them to EXPANDED_GOLDEN_CORPORA in noxfile.py"
        )
    _sync(session)
    _select_backend(session, "python")
    # Absolute: `cargo test -p fhy-core` runs its test binaries with the
    # fhy-core crate directory as the working directory, not the repository
    # root nox itself runs from, so a relative corpus path would miss.
    output_directory = pathlib.Path(session.create_tmp()).resolve()
    for generator in generators:
        corpus = EXPANDED_GOLDEN_CORPORA[generator.name]
        corpus_path = output_directory / (
            generator.stem.removeprefix("generate_") + ".json"
        )
        # Silent: the oracles log a warning per ignored re-registration. Nox
        # still prints the captured output if the generator fails.
        session.run(
            "python",
            str(generator),
            *corpus.options.split(),
            "--output",
            str(corpus_path),
            silent=True,
        )
        # `testing` is on for this crate's own tests already; naming it keeps
        # the deterministic-identifier test, which requires it, from being
        # skipped if that ever changes.
        output = session.run(
            "cargo",
            "test",
            "--locked",
            "-p",
            "fhy-core",
            "--features",
            "testing",
            "--test",
            corpus.rust_test,
            "--",
            "--ignored",
            env={corpus.variable: str(corpus_path)},
            external=True,
            silent=True,
        )
        # `cargo test` also succeeds when no test matches, so an expanded
        # test that lost its `#[ignore]` would pass without a replay.
        if not isinstance(output, str) or not _EXPANDED_REPLAY_PASSED.search(output):
            session.error(
                f"{corpus.rust_test} did not replay the expanded corpus in "
                f"exactly one ignored test:\n{output}"
            )
        session.log(f"{corpus.rust_test}: the expanded corpus replayed")


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
