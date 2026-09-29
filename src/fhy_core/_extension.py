"""Import-time check of the required Rust extension.

The package runs on its compiled extension, which it cannot work without.
Importing this module, which ``fhy_core`` does before anything else, imports
the extension and raises ``ImportError`` with the cause and the fix when the
extension is not installed, is built only for other interpreters, fails to
import, or reports a version that is missing, not a PEP 440 version, or
unequal to the installed package's version (a stale build).

Which native module is the extension depends on what is installed:

- ``fhy_core`` alone uses its own module, ``fhy_core._rs``.
- A downstream product that builds *one* combined extension for the whole
  process (``fhy-core-py``'s ``register`` called next to its own crates')
  advertises that module with an entry point in the group
  ``fhy_core.native`` (``name = "module.path"``). ``fhy_core`` then loads that
  module instead of its own and installs it under the name ``fhy_core._rs`` as
  well, so the classes, whose qualified names are ``fhy_core._rs.<Class>``,
  and the binding's lookups by that name reach it. The module reports the
  ``fhy_core`` version it holds as ``__fhy_core_version__``. It must be
  importable without importing ``fhy_core``.
- The environment variable ``FHY_CORE_NATIVE_MODULE`` names the module to
  load and overrides the entry points; ``fhy_core._rs`` names the module
  ``fhy_core`` ships.

A process holds one native module: two entry points that name different
modules, or an entry point whose module differs from the ``fhy_core._rs`` the
process already holds, raise ``ImportError`` naming both. A named module that
fails to import is never replaced by ``fhy_core._rs``, since a second copy of
the Rust code would then run beside the one the product expects.
"""

__all__: list[str] = []

import importlib
import importlib.machinery
import importlib.metadata
import os
import re
import sys
from pathlib import Path
from types import ModuleType

_EXTENSION_MODULE = "fhy_core._rs"
# The entry-point group in which a product's combined extension module is
# advertised: the entry point's value is the module's name.
NATIVE_ENTRY_POINT_GROUP = "fhy_core.native"
# The environment variable that names the native module to load, overriding
# the entry points.
NATIVE_MODULE_ENVIRONMENT_VARIABLE = "FHY_CORE_NATIVE_MODULE"
# The attribute of a combined extension module that holds the version of the
# `fhy_core` Rust code it links.
_AGGREGATE_VERSION_ATTRIBUTE = "__fhy_core_version__"
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


def _load_own_extension() -> None:
    """Import ``fhy_core._rs`` and check it matches the installed package.

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


def _describe_entry_point(entry_point: importlib.metadata.EntryPoint) -> str:
    """Return the module an entry point names, and the distribution it is in."""
    distribution = getattr(entry_point, "dist", None)
    owner = f" (entry point {entry_point.name!r}"
    owner += f" of {distribution.name!r})" if distribution is not None else ")"
    return f"{entry_point.module!r}{owner}"


def _build_two_native_modules_error(first: str, second: str) -> ImportError:
    return ImportError(
        f"{_PACKAGE_NAME} runs on one native extension module per process, "
        f"but two different ones are installed or loaded: {first} and "
        f"{second}. Uninstall one of them, or name the one to use in the "
        f"environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}. Two "
        "native modules would hold two copies of the Rust code, with "
        "colliding identifier ids, split registries and unrelated classes.",
        name=_EXTENSION_MODULE,
    )


def _select_native_module() -> str:
    """Return the name of the native module to load.

    Returns:
        The module named by ``FHY_CORE_NATIVE_MODULE``, else the module the
        entry points of the group ``fhy_core.native`` name, else
        ``fhy_core._rs``.

    Raises:
        ImportError: If the entry points name two different modules.

    """
    named = os.environ.get(NATIVE_MODULE_ENVIRONMENT_VARIABLE, "").strip()
    if named:
        return named
    by_module: dict[str, importlib.metadata.EntryPoint] = {}
    for entry_point in importlib.metadata.entry_points(group=NATIVE_ENTRY_POINT_GROUP):
        by_module.setdefault(entry_point.module, entry_point)
    if len(by_module) > 1:
        first, second, *_ = by_module.values()
        raise _build_two_native_modules_error(
            _describe_entry_point(first), _describe_entry_point(second)
        )
    return next(iter(by_module), _EXTENSION_MODULE)


def _build_aggregate_import_error(module_name: str, error: ImportError) -> ImportError:
    return ImportError(
        f"{_PACKAGE_NAME} is set to run on the native extension module "
        f"{module_name!r}, which failed to import ({type(error).__name__}: "
        f"{error}). Reinstall the package that provides it, or uninstall it "
        f"to use {_EXTENSION_MODULE}.",
        name=module_name,
    )


def _build_stale_aggregate_error(
    module_name: str, extension_version: object, package_version: str
) -> ImportError:
    reported_version = (
        f"no {_AGGREGATE_VERSION_ATTRIBUTE} attribute"
        if extension_version is None
        else f"version {extension_version!r}"
    )
    return ImportError(
        f"{_PACKAGE_NAME} needs the native extension module {module_name!r} "
        f"to hold {_PACKAGE_NAME} {package_version!r}, but it reports "
        f"{reported_version}, so it is stale or is not a {_PACKAGE_NAME} "
        "extension. Rebuild or reinstall the package that provides it.",
        name=module_name,
    )


def _load_aggregate_extension(module_name: str) -> None:
    """Load a combined extension module and install it as ``fhy_core._rs``.

    Args:
        module_name: Importable name of the combined extension module.

    Raises:
        ImportError: If the module fails to import, does not report the
            installed package's version as ``__fhy_core_version__``, or
            another native module is already installed as ``fhy_core._rs``.

    """
    try:
        extension = importlib.import_module(module_name)
    except ImportError as error:
        raise _build_aggregate_import_error(module_name, error) from error
    extension_version = getattr(extension, _AGGREGATE_VERSION_ATTRIBUTE, None)
    package_version = importlib.metadata.version(_PACKAGE_NAME)
    if (
        not isinstance(extension_version, str)
        or _normalize_pep440_version(extension_version) != package_version
    ):
        raise _build_stale_aggregate_error(
            module_name, extension_version, package_version
        )
    _install_as_own_extension(module_name, extension)


def _install_as_own_extension(module_name: str, extension: ModuleType) -> None:
    """Make ``fhy_core._rs`` name `extension`, refusing to replace another.

    Args:
        module_name: Name `extension` was imported as.
        extension: The combined extension module.

    Raises:
        ImportError: If ``fhy_core._rs`` names another module.

    """
    held = sys.modules.get(_EXTENSION_MODULE)
    if held is not None and held is not extension:
        held_file = getattr(held, "__file__", None)
        raise _build_two_native_modules_error(
            f"{module_name!r}",
            f"{_EXTENSION_MODULE!r}"
            + (f" (loaded from {held_file})" if held_file else ""),
        )
    sys.modules[_EXTENSION_MODULE] = extension
    setattr(sys.modules[_PACKAGE_NAME], _EXTENSION_STEM, extension)


def _check_extension() -> None:
    """Load the native module of this process and check it.

    Raises:
        ImportError: If no usable native module is installed, or two
            different ones are. The message names the cause and the fix.

    """
    module_name = _select_native_module()
    if module_name == _EXTENSION_MODULE:
        _load_own_extension()
    else:
        _load_aggregate_extension(module_name)


_check_extension()
