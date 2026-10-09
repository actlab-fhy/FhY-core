"""Command-line options, provenance and output shared by the golden generators.

Every ``generate_*.py`` beside this module runs as a script, so this module
is imported from the script's own directory rather than from a package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def add_corpus_arguments(
    parser: argparse.ArgumentParser,
    *,
    seed: int,
    random_count: int,
    max_ops: int,
    default_output: Path,
) -> None:
    """Add the options every generator takes to shape and place its corpus.

    Args:
        parser: The generator's argument parser.
        seed: Default seed for the random scripts (``--seed``).
        random_count: Default number of random scripts (``--random-count``).
        max_ops: Default largest operation count of one random script
            (``--max-ops``).
        default_output: Default path the corpus is written to (``--output``).

    """
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--random-count", type=int, default=random_count)
    parser.add_argument("--max-ops", type=int, default=max_ops)
    parser.add_argument("--output", type=Path, default=default_output)


def build_provenance(repository_root: Path, generator_command: str) -> dict[str, Any]:
    """Return the provenance block recording how a corpus was generated.

    Args:
        repository_root: The repository whose checked-out commit is recorded.
        generator_command: The command that regenerates the default corpus.

    Returns:
        The package and its version, the git commit, the interpreter version,
        and the generator command.

    """
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    return {
        "package": "fhy_core",
        "package_version": version("fhy_core"),
        "git_commit": git_commit,
        "python_version": sys.version,
        "generator_command": generator_command,
    }


def write_document(path: Path, document: dict[str, Any]) -> None:
    """Write a golden document as one-space-indented UTF-8 JSON.

    Args:
        path: The file to overwrite.
        document: The document to write.

    """
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, ensure_ascii=False, indent=1)
        output_file.write("\n")
