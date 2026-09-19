"""Selection between the Rust extension and the pure-Python implementation.

The package runs on the compiled extension ``fhy_core._rs`` iff the
extension imports and the ``FHY_CORE_NO_EXTENSIONS`` environment variable
does not disable it. The variable disables the extension when it holds
anything other than an empty string or one of ``0``, ``false``, ``no``, and
``off`` (compared case-insensitively, ignoring surrounding whitespace). The
selection is made once, when this module is first imported.
"""

__all__ = ["IS_RUST_BACKEND_SELECTED"]

import importlib
import os

_NO_EXTENSIONS_VARIABLE = "FHY_CORE_NO_EXTENSIONS"
_EXTENSION_ENABLING_VALUES = frozenset({"", "0", "false", "no", "off"})


def _is_extension_disabled_by_environment() -> bool:
    value = os.environ.get(_NO_EXTENSIONS_VARIABLE, "")
    return value.strip().lower() not in _EXTENSION_ENABLING_VALUES


def _is_extension_importable() -> bool:
    try:
        importlib.import_module("fhy_core._rs")
    except ImportError:
        return False
    return True


IS_RUST_BACKEND_SELECTED: bool = (
    not _is_extension_disabled_by_environment() and _is_extension_importable()
)
"""Whether the package runs on the Rust extension in this process."""
