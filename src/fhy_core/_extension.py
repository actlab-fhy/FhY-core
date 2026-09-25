"""Import-time check of the required Rust extension ``fhy_core._rs``.

The package runs on its compiled extension, which it cannot work without.
Importing this module, which ``fhy_core`` does before anything else, imports
the extension and raises ``ImportError`` with the cause and the fix when the
extension is not installed, is built only for other interpreters, fails to
import, or reports a ``__version__`` that is missing, not a PEP 440 version,
or unequal to the installed package's version (a stale build).
"""

__all__: list[str] = []

import importlib
import importlib.machinery
import importlib.metadata
import re
from pathlib import Path

_EXTENSION_MODULE = "fhy_core._rs"
_EXTENSION_STEM = "_rs"
# File endings of a compiled extension module on any platform.
_EXTENSION_FILE_ENDINGS = (".so", ".pyd")
_PACKAGE_NAME = "fhy_core"
_REQUIREMENT = f"{_PACKAGE_NAME} requires its Rust extension {_EXTENSION_MODULE}"
_REBUILD_ADVICE = (
    "Rebuild the extension for this interpreter from the source checkout "
    "with `uv sync`, or reinstall the fhy_core wheel."
)

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


def _find_foreign_extension_builds(
    directory: Path, extension_suffixes: list[str]
) -> list[str]:
    """Return the extension builds in `directory` that this interpreter skips.

    Args:
        directory: Directory the extension module is imported from.
        extension_suffixes: File suffixes this interpreter loads extension
            modules from, such as ``importlib.machinery.EXTENSION_SUFFIXES``.

    Returns:
        The sorted file names of the builds found, or an empty list if none
        was found or one of them matches ``extension_suffixes``.

    """
    loadable_names = {_EXTENSION_STEM + suffix for suffix in extension_suffixes}
    build_names = sorted(
        path.name
        for path in directory.glob(f"{_EXTENSION_STEM}.*")
        if path.name.endswith(_EXTENSION_FILE_ENDINGS)
    )
    if any(name in loadable_names for name in build_names):
        return []
    return build_names


def _build_missing_extension_error() -> ImportError:
    return ImportError(
        f"{_REQUIREMENT}, which is not installed. Install a fhy_core wheel for "
        "this platform, or build the extension from the source checkout with "
        "`uv sync`.",
        name=_EXTENSION_MODULE,
    )


def _build_foreign_extension_error(
    build_names: list[str], extension_suffixes: list[str]
) -> ImportError:
    return ImportError(
        f"{_REQUIREMENT}, which is installed only as {', '.join(build_names)}; "
        f"this interpreter does not load those builds (it loads "
        f"{_EXTENSION_STEM}{extension_suffixes[0]}). {_REBUILD_ADVICE}",
        name=_EXTENSION_MODULE,
    )


def _build_failed_import_error(error: ImportError) -> ImportError:
    return ImportError(
        f"{_REQUIREMENT}, which is installed but failed to import "
        f"({type(error).__name__}: {error}). {_REBUILD_ADVICE}",
        name=_EXTENSION_MODULE,
    )


def _build_stale_extension_error(
    extension_version: object, package_version: str
) -> ImportError:
    reported_version = (
        "no __version__ attribute"
        if extension_version is None
        else f"version {extension_version!r}"
    )
    return ImportError(
        f"{_REQUIREMENT} at the installed package's version "
        f"{package_version!r}, but the extension reports {reported_version}, "
        f"so it is stale. {_REBUILD_ADVICE}",
        name=_EXTENSION_MODULE,
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


def _check_extension() -> None:
    """Import the extension and check it matches the installed package.

    Raises:
        ImportError: If the extension is not installed, is built only for
            other interpreters, fails to import, or reports another version
            than the installed package's. The message names the cause and
            the fix; the original import error, if any, is its cause.

    """
    try:
        extension = importlib.import_module(_EXTENSION_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _EXTENSION_MODULE:
            raise _build_failed_import_error(error) from error
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES
        build_names = _find_foreign_extension_builds(
            Path(__file__).parent, extension_suffixes
        )
        if build_names:
            raise _build_foreign_extension_error(
                build_names, extension_suffixes
            ) from error
        raise _build_missing_extension_error() from error
    except ImportError as error:
        raise _build_failed_import_error(error) from error
    extension_version = getattr(extension, "__version__", None)
    package_version = importlib.metadata.version(_PACKAGE_NAME)
    if (
        not isinstance(extension_version, str)
        or _normalize_pep440_version(extension_version) != package_version
    ):
        raise _build_stale_extension_error(extension_version, package_version)


_check_extension()
