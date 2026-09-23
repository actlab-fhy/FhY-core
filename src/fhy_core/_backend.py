"""Selection between the Rust extension and the pure-Python implementation.

The package runs on the compiled extension ``fhy_core._rs`` iff the
extension imports, its ``__version__`` matches the installed ``fhy_core``
package, and the ``FHY_CORE_NO_EXTENSIONS`` environment variable does not
disable it. The extension reports its Cargo version, from which maturin
derives the package version by PEP 440 normalization, so the two match when
the extension's version normalizes to the package's (``0.3.0-rc.1`` matches
``0.3.0rc1``). The variable disables the extension when it holds anything
other than an empty string or one of ``0``, ``false``, ``no``, and ``off``
(compared case-insensitively, ignoring surrounding whitespace). The
selection is made once, when this module is first imported.

An extension that is not installed selects the pure-Python implementation
silently. An extension that is installed but fails to import, for example
because its shared library fails to load or was built for another
interpreter, selects it with a ``RuntimeWarning`` naming the error. An
extension that imports but whose ``__version__`` does not match the
installed package, is not a PEP 440 version, or is missing, is stale: it
selects the pure-Python implementation with a ``RuntimeWarning`` naming
both versions. A disabled extension is never imported, so it never warns.
"""

__all__ = ["IS_RUST_BACKEND_SELECTED"]

import importlib
import importlib.metadata
import os
import re
import warnings

_EXTENSION_MODULE = "fhy_core._rs"
_PACKAGE_NAME = "fhy_core"
_NO_EXTENSIONS_VARIABLE = "FHY_CORE_NO_EXTENSIONS"
_EXTENSION_ENABLING_VALUES = frozenset({"", "0", "false", "no", "off"})

# PEP 440's version grammar without the epoch, which Cargo cannot express.
_PEP440_VERSION_PATTERN = re.compile(
    r"""
    v?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?:
        [-_.]?(?P<pre_label>alpha|a|beta|b|preview|pre|c|rc)
        [-_.]?(?P<pre_number>[0-9]+)?
    )?
    (?:
        -(?P<implicit_post_number>[0-9]+)
        |
        [-_.]?(?P<post_label>post|rev|r)[-_.]?(?P<post_number>[0-9]+)?
    )?
    (?:[-_.]?(?P<dev_label>dev)[-_.]?(?P<dev_number>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
    """,
    re.VERBOSE | re.IGNORECASE,
)
_PEP440_PRE_RELEASE_LABELS = {
    "alpha": "a",
    "a": "a",
    "beta": "b",
    "b": "b",
    "preview": "rc",
    "pre": "rc",
    "c": "rc",
    "rc": "rc",
}


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


def _warn_extension_version_mismatch(
    extension_version: str | None, package_version: str
) -> None:
    reported_version = (
        "no __version__ attribute"
        if extension_version is None
        else f"version {extension_version!r}"
    )
    warnings.warn(
        f"The Rust extension {_EXTENSION_MODULE} reports {reported_version}, "
        f"but the installed package {_PACKAGE_NAME} is version "
        f"{package_version!r}; the extension is stale, so the pure-Python "
        "backend is used instead. Rebuild the extension (`uv sync`) to "
        "select it again.",
        RuntimeWarning,
        stacklevel=3,
    )


def _normalize_pep440_version(version: str) -> str | None:
    """Return a version in PEP 440's normal form, the form maturin gives it.

    Args:
        version: Version to normalize, such as a Cargo version.

    Returns:
        The normalized version, or ``None`` if ``version`` is not a PEP 440
        version.

    """
    match = _PEP440_VERSION_PATTERN.fullmatch(version.strip())
    if match is None:
        return None
    normalized = ".".join(str(int(part)) for part in match["release"].split("."))
    if match["pre_label"] is not None:
        pre_label = _PEP440_PRE_RELEASE_LABELS[match["pre_label"].lower()]
        normalized += f"{pre_label}{int(match['pre_number'] or 0)}"
    if match["implicit_post_number"] is not None:
        normalized += f".post{int(match['implicit_post_number'])}"
    elif match["post_label"] is not None:
        normalized += f".post{int(match['post_number'] or 0)}"
    if match["dev_label"] is not None:
        normalized += f".dev{int(match['dev_number'] or 0)}"
    if match["local"] is not None:
        normalized += "+" + re.sub(r"[-_]", ".", match["local"].lower())
    return normalized


def _is_extension_importable() -> bool:
    try:
        extension = importlib.import_module(_EXTENSION_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _EXTENSION_MODULE:
            _warn_extension_failed_to_import(error)
        return False
    except ImportError as error:
        _warn_extension_failed_to_import(error)
        return False
    extension_version = getattr(extension, "__version__", None)
    package_version = importlib.metadata.version(_PACKAGE_NAME)
    if (
        not isinstance(extension_version, str)
        or _normalize_pep440_version(extension_version) != package_version
    ):
        _warn_extension_version_mismatch(extension_version, package_version)
        return False
    return True


IS_RUST_BACKEND_SELECTED: bool = (
    not _is_extension_disabled_by_environment() and _is_extension_importable()
)
"""Whether the package runs on the Rust extension in this process."""
