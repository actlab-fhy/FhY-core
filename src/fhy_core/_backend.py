"""Transitional flag of the retired backend selection.

``IS_RUST_BACKEND_SELECTED`` is always true, since the package requires its
Rust extension (:mod:`fhy_core._extension`). It remains only until the
modules that branched on it lose their pure-Python branches.
"""

__all__ = ["IS_RUST_BACKEND_SELECTED"]

from typing import Final

from . import _extension  # noqa: F401  # the extension is checked first

IS_RUST_BACKEND_SELECTED: Final[bool] = True
"""Always true: the package runs on the Rust extension."""
