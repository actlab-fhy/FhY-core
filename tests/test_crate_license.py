"""Tests the Rust crate ships the repository's license text unchanged.

`cargo package` only packs files inside the crate's directory, and the
BSD-3-Clause license requires source redistributions to retain its text, so
`rust/fhy-core/LICENSE` is a copy of the repository's `LICENSE`.
"""

from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_crate_license_matches_the_repository_license() -> None:
    """Test the crate's license file is byte-identical to the repository's."""
    repository_license = (_REPOSITORY_ROOT / "LICENSE").read_bytes()
    crate_license = (_REPOSITORY_ROOT / "rust" / "fhy-core" / "LICENSE").read_bytes()

    assert crate_license == repository_license
