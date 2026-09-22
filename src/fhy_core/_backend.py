"""Selection between the Rust extension and the pure-Python implementation.

The package runs on the compiled extension ``fhy_core._rs`` iff the
extension imports and the ``FHY_CORE_NO_EXTENSIONS`` environment variable
does not disable it. The variable disables the extension when it holds
anything other than an empty string or one of ``0``, ``false``, ``no``, and
``off`` (compared case-insensitively, ignoring surrounding whitespace). The
selection is made once, when this module is first imported.

An extension that is not installed selects the pure-Python implementation
silently. An extension that is installed but fails to import, for example
because its shared library fails to load or was built for another
interpreter, selects it with a ``RuntimeWarning`` naming the error; a
disabled extension is never imported, so it never warns.
"""

__all__ = ["IS_RUST_BACKEND_SELECTED"]

import importlib
import os
import warnings

_EXTENSION_MODULE = "fhy_core._rs"
_NO_EXTENSIONS_VARIABLE = "FHY_CORE_NO_EXTENSIONS"
_EXTENSION_ENABLING_VALUES = frozenset({"", "0", "false", "no", "off"})


def _is_extension_disabled_by_environment() -> bool:
    value = os.environ.get(_NO_EXTENSIONS_VARIABLE, "")
    return value.strip().lower() not in _EXTENSION_ENABLING_VALUES


def _warn_extension_failed_to_import(error: ImportError) -> None:
    warnings.warn(
        f"The Rust extension {_EXTENSION_MODULE} is installed but failed to "
        f"import ({type(error).__name__}: {error}); falling back to the "
        f"pure-Python backend. Set {_NO_EXTENSIONS_VARIABLE}=1 to select the "
        "pure-Python backend without importing the extension.",
        RuntimeWarning,
        stacklevel=3,
    )


def _is_extension_importable() -> bool:
    try:
        importlib.import_module(_EXTENSION_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _EXTENSION_MODULE:
            _warn_extension_failed_to_import(error)
        return False
    except ImportError as error:
        _warn_extension_failed_to_import(error)
        return False
    return True


IS_RUST_BACKEND_SELECTED: bool = (
    not _is_extension_disabled_by_environment() and _is_extension_importable()
)
"""Whether the package runs on the Rust extension in this process."""
