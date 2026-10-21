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
  advertises that module with entry points in the group ``fhy_core.native``.
  An entry point ``name = "module.path"`` means the module contains the
  *layer* ``name``, the import name of the Python package whose Rust binding
  the module holds. The loader groups the entry points by module across all
  distributions; a module's layers are its claims plus ``fhy_core``. The one
  module whose layers are all the installed layers is loaded instead of
  ``fhy_core._rs``, and installed under that name as well, so the classes,
  whose qualified names are ``fhy_core._rs.<Class>``, and the binding's
  lookups by that name reach it. The module reports the ``fhy_core`` version
  it holds as ``__fhy_core_version__``, and a layer ``L`` is present in it
  if and only if it has a ``str`` attribute ``__L_version__``. It must be
  importable without importing ``fhy_core``.
- The environment variable ``FHY_CORE_NATIVE_MODULE`` names the module to
  load and overrides the entry points; ``fhy_core._rs`` names the module
  ``fhy_core`` ships.

A process holds one native module. Several modules that each contain every
installed layer (a tie) and a set of modules of which none contains every
installed layer (incomparable) raise ``ImportError`` naming them and what
each lacks. Layer sets need not form a chain: a diamond such as ``{a}``,
``{b}``, ``{a, b}`` is accepted, since the last module contains every
installed layer. ``ImportError`` is raised too for an entry point whose
name is not a valid layer name (an identifier that is not a keyword) or whose
value is not an importable module path (dot-separated layer names, with no
attribute or extras), and for a selected module that differs from the
``fhy_core._rs`` the process already holds. A module that is not selected is
never imported. A selected module whose entry points claim a layer it does
not report is refused, naming the entry point and the missing attribute;
this holds for ``fhy_core._rs`` too when entry points advertise it. A named
module that fails to import is never replaced by ``fhy_core._rs``, since a
second copy of the Rust code would then run beside the one the product
expects. ``fhy_core.native`` is the public interface to the selection.
"""

__all__: list[str] = []

import importlib
import importlib.machinery
import importlib.metadata
import keyword
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

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


def _is_layer_name(name: str) -> bool:
    """Return whether `name` can be the import name of a package.

    Args:
        name: Candidate layer name.

    Returns:
        ``True`` if `name` is an identifier and not a Python keyword.

    """
    return name.isidentifier() and not keyword.iskeyword(name)


def _is_module_name(name: str) -> bool:
    """Return whether `name` is an absolute, importable module path.

    Args:
        name: Candidate module name, such as ``"my_product._native"``.

    Returns:
        ``True`` if every dot-separated part of `name` is a layer name (an
        identifier that is not a keyword); an empty part, as in a relative
        name, fails.

    """
    return all(_is_layer_name(part) for part in name.split("."))


def get_layer_version_attribute(layer: str) -> str:
    """Return the module attribute that holds the version of a layer.

    Args:
        layer: Import name of the package whose Rust binding the layer is.

    Returns:
        ``__<layer>_version__``, the attribute a native module sets when it
        contains the layer.

    Raises:
        ValueError: If `layer` is not a valid Python identifier or is a
            Python keyword.

    """
    if not _is_layer_name(layer):
        raise ValueError(
            f"a layer is named by an identifier that is not a keyword, not {layer!r}"
        )
    return f"__{layer}_version__"


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


def _load_own_extension() -> ModuleType:
    """Import ``fhy_core._rs`` and check it matches the installed package.

    Returns:
        The imported extension module.

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
    return extension


@dataclass(frozen=True)
class _Candidate:
    """A native module that entry points advertise, with its entry points.

    Each entry point's name is a layer the module claims to contain: the
    import name of a Python package whose Rust binding the module holds.
    """

    module: str
    entry_points: tuple[importlib.metadata.EntryPoint, ...]

    @property
    def layers(self) -> frozenset[str]:
        """The layers the module claims, which always include ``fhy_core``."""
        return frozenset({_PACKAGE_NAME, *(ep.name for ep in self.entry_points)})


@dataclass(frozen=True)
class _Selection:
    """The native module a process runs on, and why it was chosen.

    Attributes:
        module: Name of the selected module.
        reason: ``"environment"`` if the environment variable named it,
            ``"entry_points"`` if the entry points advertised it, ``"own"``
            if nothing was advertised and ``fhy_core._rs`` is used.
        candidates: Every module the entry points advertise, sorted by name,
            whether or not it was selected.

    """

    module: str
    reason: Literal["environment", "entry_points", "own"]
    candidates: tuple[_Candidate, ...]

    @property
    def candidate(self) -> _Candidate | None:
        """The advertised module that was selected, if it is advertised."""
        for candidate in self.candidates:
            if candidate.module == self.module:
                return candidate
        return None

    @property
    def layers(self) -> frozenset[str]:
        """The layers the selected module is declared to contain."""
        candidate = self.candidate
        return frozenset({_PACKAGE_NAME}) if candidate is None else candidate.layers


_SELECTION: _Selection | None = None


def _distribution_name(entry_point: importlib.metadata.EntryPoint) -> str:
    """Return the name of the distribution of an entry point, or ``""``."""
    distribution = getattr(entry_point, "dist", None)
    return "" if distribution is None else str(distribution.name)


def _describe_candidate(candidate: _Candidate) -> str:
    """Return the module of a candidate and the entry points that name it."""
    owners = sorted(
        {(ep.name, _distribution_name(ep)) for ep in candidate.entry_points},
        key=lambda owner: (owner[1], owner[0]),
    )
    parts = [
        f"{name!r} of {distribution!r}" if distribution else f"{name!r}"
        for name, distribution in owners
    ]
    noun = "entry point" if len(parts) == 1 else "entry points"
    return f"{candidate.module!r} ({noun} {', '.join(parts)})"


def _format_layers(layers: Iterable[str]) -> str:
    return ", ".join(repr(layer) for layer in sorted(layers))


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


def _build_tie_error(
    required: frozenset[str], tied: Sequence[_Candidate]
) -> ImportError:
    return ImportError(
        f"{_PACKAGE_NAME} runs on one native extension module per process, "
        f"but several installed modules each contain every installed layer "
        f"({_format_layers(required)}): "
        f"{'; '.join(_describe_candidate(candidate) for candidate in tied)}. "
        f"{_PACKAGE_NAME} cannot choose between them. Uninstall all but one "
        f"of the distributions that provide them, or name the one to use in "
        f"the environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}.",
        name=_EXTENSION_MODULE,
    )


def _build_incomparable_error(
    required: frozenset[str], candidates: Sequence[_Candidate]
) -> ImportError:
    ordered = sorted(candidates, key=lambda c: (-len(c.layers), c.module))
    lacks = "; ".join(
        f"{_describe_candidate(candidate)} lacks "
        f"{_format_layers(required - candidate.layers)}"
        for candidate in ordered
    )
    return ImportError(
        f"{_PACKAGE_NAME} runs on one native extension module per process, "
        f"but no installed module contains every installed layer "
        f"({_format_layers(required)}): {lacks}. Install a product that "
        f"contains all of those layers, or name the module to use in the "
        f"environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}, which "
        "leaves the layers it lacks without their native code.",
        name=_EXTENSION_MODULE,
    )


def _build_invalid_layer_name_error(
    entry_point: importlib.metadata.EntryPoint,
) -> ImportError:
    distribution = _distribution_name(entry_point)
    owner = f" of {distribution!r}" if distribution else ""
    return ImportError(
        f"The entry point {entry_point.name!r}{owner} in the group "
        f"{NATIVE_ENTRY_POINT_GROUP!r} names the module {entry_point.value!r}, "
        "but an entry point's name is the layer the module contains, the "
        "import name of a Python package, and "
        f"{entry_point.name!r} is not a valid Python identifier (a keyword is "
        "not one). Fix the "
        f"entry point, or name the module to use in the environment variable "
        f"{NATIVE_MODULE_ENVIRONMENT_VARIABLE}.",
        name=_EXTENSION_MODULE,
    )


def _build_invalid_module_value_error(
    entry_point: importlib.metadata.EntryPoint,
) -> ImportError:
    distribution = _distribution_name(entry_point)
    owner = f" of {distribution!r}" if distribution else ""
    return ImportError(
        f"The entry point {entry_point.name!r}{owner} in the group "
        f"{NATIVE_ENTRY_POINT_GROUP!r} has the value {entry_point.value!r}, "
        "but an entry point's value is the native module, an importable "
        "module path such as 'my_product._native' with no attribute or "
        "extras. Fix the entry point, or name the module to use in the "
        f"environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}.",
        name=_EXTENSION_MODULE,
    )


def _collect_candidates(
    entry_points: Iterable[importlib.metadata.EntryPoint], *, strict: bool
) -> tuple[_Candidate, ...]:
    """Group entry points by the module they name, across distributions.

    Args:
        entry_points: Entry points of the group ``fhy_core.native``.
        strict: Whether an entry point whose name is not a layer name, or
            whose value is not a module path, is an error. If not, such entry
            points are ignored.

    Returns:
        One candidate per distinct module, sorted by module name, with its
        entry points sorted by distribution and name.

    Raises:
        ImportError: If ``strict`` and an entry point's name is not a valid
            layer name (an identifier that is not a keyword), or its value,
            stripped of surrounding space, is not an absolute module path
            (dot-separated layer names, with no attribute or extras). The name
            is checked first.

    """
    by_module: dict[str, list[importlib.metadata.EntryPoint]] = {}
    for entry_point in entry_points:
        if not _is_layer_name(entry_point.name):
            if strict:
                raise _build_invalid_layer_name_error(entry_point)
            continue
        module = entry_point.value.strip()
        if not _is_module_name(module):
            if strict:
                raise _build_invalid_module_value_error(entry_point)
            continue
        by_module.setdefault(module, []).append(entry_point)
    return tuple(
        _Candidate(
            module,
            tuple(
                sorted(
                    claims,
                    key=lambda ep: (_distribution_name(ep), ep.name),
                )
            ),
        )
        for module, claims in sorted(by_module.items())
    )


def _choose_candidate(candidates: Sequence[_Candidate]) -> _Candidate | None:
    """Return the candidate that contains every layer any candidate does.

    Args:
        candidates: The advertised modules.

    Returns:
        The one module whose layers are all the installed layers, or
        ``None`` if there are no candidates.

    Raises:
        ImportError: If several modules contain every layer (a tie), or none
            does (the layer sets are incomparable).

    """
    if not candidates:
        return None
    required = frozenset().union(*(candidate.layers for candidate in candidates))
    complete = [candidate for candidate in candidates if candidate.layers == required]
    if not complete:
        raise _build_incomparable_error(required, candidates)
    if len(complete) > 1:
        raise _build_tie_error(required, complete)
    return complete[0]


def _select_native_module(
    environ: Mapping[str, str],
    entry_points: Iterable[importlib.metadata.EntryPoint],
) -> _Selection:
    """Select the native module to load.

    Args:
        environ: The process environment.
        entry_points: Entry points of the group ``fhy_core.native``.

    Returns:
        The module named by ``FHY_CORE_NATIVE_MODULE``, else the one
        advertised module that contains every installed layer, else
        ``fhy_core._rs``.

    Raises:
        ImportError: If the variable is not set and an entry point's name is
            not a valid layer name (an identifier that is not a keyword) or
            its value is not a module path, or the advertised modules tie or
            are incomparable.

    """
    named = environ.get(NATIVE_MODULE_ENVIRONMENT_VARIABLE, "").strip()
    candidates = _collect_candidates(entry_points, strict=not named)
    if named:
        return _Selection(named, "environment", candidates)
    chosen = _choose_candidate(candidates)
    if chosen is None:
        return _Selection(_EXTENSION_MODULE, "own", candidates)
    return _Selection(chosen.module, "entry_points", candidates)


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


def _build_undeclared_layer_error(
    module_name: str, missing: Sequence[importlib.metadata.EntryPoint]
) -> ImportError:
    claims = "; ".join(
        f"the entry point {ep.name!r}"
        + (f" of {_distribution_name(ep)!r}" if _distribution_name(ep) else "")
        + f" claims the layer {ep.name!r}, but the module has no string "
        f"attribute {get_layer_version_attribute(ep.name)!r}"
        for ep in missing
    )
    return ImportError(
        f"{_PACKAGE_NAME} is set to run on the native extension module "
        f"{module_name!r}, which does not contain every layer it is declared "
        f"to: {claims}. A layer is present in a module only if the module "
        "reports its version. Rebuild or reinstall the distribution that "
        "provides the module, or correct its entry points.",
        name=module_name,
    )


def _verify_declared_layers(
    module_name: str, extension: ModuleType, candidate: _Candidate | None
) -> None:
    """Check a module holds every layer its entry points claim for it.

    A layer ``L`` is present in a module if and only if the module has a
    ``str`` attribute ``__L_version__``. The layer ``fhy_core`` is checked
    separately, by its version.

    Args:
        module_name: Name `extension` was imported as.
        extension: The imported native module.
        candidate: The entry points that advertise the module, or ``None``
            if none does, as when the environment variable names it.

    Raises:
        ImportError: If a claimed layer has no version attribute. The message
            names each such entry point, its distribution and the attribute.

    """
    if candidate is None:
        return
    missing = [
        entry_point
        for entry_point in candidate.entry_points
        if entry_point.name != _PACKAGE_NAME
        and not isinstance(
            getattr(extension, get_layer_version_attribute(entry_point.name), None), str
        )
    ]
    if missing:
        raise _build_undeclared_layer_error(module_name, missing)


def _load_aggregate_extension(
    module_name: str, candidate: _Candidate | None = None
) -> None:
    """Load a combined extension module and install it as ``fhy_core._rs``.

    Args:
        module_name: Importable name of the combined extension module.
        candidate: The entry points that advertise the module, whose claimed
            layers are verified, or ``None`` if none does.

    Raises:
        ImportError: If the module fails to import, does not report the
            installed package's version as ``__fhy_core_version__``, lacks a
            layer its entry points claim, or another native module is already
            installed as ``fhy_core._rs``.

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
    _verify_declared_layers(module_name, extension, candidate)
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
    global _SELECTION  # noqa: PLW0603
    selection = _select_native_module(
        os.environ, importlib.metadata.entry_points(group=NATIVE_ENTRY_POINT_GROUP)
    )
    _SELECTION = selection
    if selection.module == _EXTENSION_MODULE:
        extension = _load_own_extension()
        _verify_declared_layers(_EXTENSION_MODULE, extension, selection.candidate)
    else:
        _load_aggregate_extension(selection.module, selection.candidate)


_check_extension()
