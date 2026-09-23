"""Tests the committed golden corpora match the generators that record them.

Each ``generate_X.py`` under ``rust/fhy-core/tests/golden/`` replays seeded scripts
through the Python implementation, the oracle the Rust equivalence tests
replay, and writes the corpus ``X.json`` beside it. Every generator is rerun
here and its output compared with the committed corpus outside the
``provenance`` block, which records the commit and interpreter of a run and
so differs between runs.

The generators mutate process-global state (interned registries, the
identifier counter, the deterministic-identifier scope), so each one runs in
a fresh interpreter. That interpreter inherits this process's environment,
so a corpus is checked on whichever backend the test run selected.
"""

import difflib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_DIRECTORY = _REPOSITORY_ROOT / "rust" / "fhy-core" / "tests" / "golden"
_GENERATORS = sorted(_GOLDEN_DIRECTORY.glob("generate_*.py"))
# Diff lines shown for a stale corpus; the rest are elided.
_DIFF_LINE_LIMIT = 60


def _canonicalize_corpus(path: Path) -> str:
    """Return a golden corpus as sorted-key JSON text, without its provenance.

    Comparing text rather than parsed values keeps ``true`` distinct from
    ``1`` and ``1`` from ``1.0``.

    Args:
        path: The corpus file.

    Returns:
        The corpus without its ``provenance`` key, dumped with sorted keys
        and one-space indentation.

    """
    with path.open(encoding="utf-8") as corpus_file:
        document = json.load(corpus_file)
    document.pop("provenance", None)
    return json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True)


def _build_capped_diff(committed_text: str, regenerated_text: str, name: str) -> str:
    """Return a unified diff of two corpus texts, elided past the line limit.

    Args:
        committed_text: The canonical text of the committed corpus.
        regenerated_text: The canonical text of the regenerated corpus.
        name: The corpus file name, used in the diff headers.

    Returns:
        The diff lines joined by newlines.

    """
    diff = list(
        difflib.unified_diff(
            committed_text.splitlines(),
            regenerated_text.splitlines(),
            fromfile=f"committed/{name}",
            tofile=f"regenerated/{name}",
            lineterm="",
        )
    )
    elided = len(diff) - _DIFF_LINE_LIMIT
    if elided > 0:
        diff = [*diff[:_DIFF_LINE_LIMIT], f"... {elided} more diff lines"]
    return "\n".join(diff)


def test_golden_directory_has_generators() -> None:
    """Test the corpus check below has at least one generator to run."""
    assert _GENERATORS, f"no generate_*.py under {_GOLDEN_DIRECTORY}"


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "generator", _GENERATORS, ids=[generator.name for generator in _GENERATORS]
)
def test_committed_corpus_matches_its_generator(
    generator: Path, tmp_path: Path
) -> None:
    """Test a generator reproduces its committed corpus outside provenance."""
    committed = generator.with_name(generator.stem.removeprefix("generate_") + ".json")
    assert committed.is_file(), (
        f"{generator.name} has no committed corpus {committed.name}"
    )
    regenerated = tmp_path / committed.name

    completed = subprocess.run(
        [sys.executable, str(generator), "--output", str(regenerated)],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{generator.name} exited with {completed.returncode}:\n{completed.stderr}"
    )
    committed_text = _canonicalize_corpus(committed)
    regenerated_text = _canonicalize_corpus(regenerated)
    if committed_text != regenerated_text:
        pytest.fail(
            f"{committed.name} is stale; regenerate it from the repository root "
            "with\n\n    uv run --no-sync python "
            f"{generator.relative_to(_REPOSITORY_ROOT).as_posix()}\n\n"
            "Diff (provenance dropped, keys sorted):\n"
            + _build_capped_diff(committed_text, regenerated_text, committed.name),
            pytrace=False,
        )
