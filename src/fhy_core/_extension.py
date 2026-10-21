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
  ``fhy_core`` ships. A value that is not an importable module path raises
  ``ImportError``. Under the variable, invalid entry points are skipped and
  a tie or incomparable set is not an error; the import says nothing of
  either, and the selection records the skipped entry points so that
  ``fhy_core.native.require_layer`` can name them, and the conflict, if the
  module lacks a layer.

A process holds one native module. Several modules that each contain every
installed layer (a tie) raise ``ImportError`` naming the tied modules; a set
of modules of which none contains every installed layer (incomparable)
raises ``ImportError`` naming each module with the layers it lacks. Layer
sets need not form a chain: a diamond such as ``{a}``,
``{b}``, ``{a, b}`` is accepted, since the last module contains every
installed layer. ``ImportError`` is raised too for an entry point whose
name is not a valid layer name (an identifier that is not a keyword) or whose
value is not an importable module path (dot-separated layer names, with no
attribute or extras), and for a selected module that differs from the
``fhy_core._rs`` the process already holds. A module that is not selected is
never imported. A selected module whose entry points claim a layer it does
not report is refused, naming the entry point and the missing attribute;
this holds for ``fhy_core._rs`` too when entry points advertise it. A named
module that fails to import, whatever it raises, is never replaced by
``fhy_core._rs``, since a second copy of the Rust code would then run beside
the one the product expects. An ``fhy_core._rs`` that the failed import
loaded and then refused, as stale or lacking a claimed layer, is removed from
``sys.modules``, so a retry in the same process, such as one with the
variable set, can load another module. ``fhy_core.native`` is the public
interface to the selection; the layers it reports are those the loaded module
contains (`read_module_layers`), whether or not an entry point advertises
them.
"""

__all__: list[str] = []

import importlib
import importlib.machinery
import importlib.metadata
import itertools
import keyword
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

EXTENSION_MODULE = "fhy_core._rs"
# The entry-point group in which a product's combined extension module is
# advertised: the entry point's value is the module's name.
NATIVE_ENTRY_POINT_GROUP = "fhy_core.native"
# The environment variable that names the native module to load, overriding
# the entry points.
NATIVE_MODULE_ENVIRONMENT_VARIABLE = "FHY_CORE_NATIVE_MODULE"
# The attribute of a combined extension module that holds the version of the
# `fhy_core` Rust code it links.
_AGGREGATE_VERSION_ATTRIBUTE = "__fhy_core_version__"
EXTENSION_STEM = "_rs"
# File endings of a compiled extension module on any platform.
_EXTENSION_FILE_ENDINGS = (".so", ".pyd")
_PACKAGE_NAME = "fhy_core"
# A layer `L` reports its version as the attribute `__L_version__`.
_LAYER_VERSION_PREFIX = "__"
_LAYER_VERSION_SUFFIX = "_version__"
_REQUIREMENT = f"{_PACKAGE_NAME} requires its Rust extension {EXTENSION_MODULE}"
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


def is_module_name(name: str) -> bool:
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
    return f"{_LAYER_VERSION_PREFIX}{layer}{_LAYER_VERSION_SUFFIX}"


def read_module_layers(module: ModuleType) -> frozenset[str]:
    """Return the layers a native module contains, by what it reports.

    A layer ``L`` is present in a module if and only if the module has a
    ``str`` attribute ``__L_version__`` and ``L`` is a layer name (an
    identifier that is not a keyword). ``__version__``, which names no
    layer, is not one.

    Args:
        module: The native module.

    Returns:
        Every layer the module reports, and ``fhy_core``, which every native
        module contains.

    """
    layers = {_PACKAGE_NAME}
    for name, value in vars(module).items():
        if not (
            isinstance(value, str)
            and name.startswith(_LAYER_VERSION_PREFIX)
            and name.endswith(_LAYER_VERSION_SUFFIX)
        ):
            continue
        layer = name[len(_LAYER_VERSION_PREFIX) : -len(_LAYER_VERSION_SUFFIX)]
        if _is_layer_name(layer):
            layers.add(layer)
    return frozenset(layers)


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
    loadable_names = {EXTENSION_STEM + suffix for suffix in extension_suffixes}
    build_names = sorted(
        path.name
        for path in directory.glob(f"{EXTENSION_STEM}.*")
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
        name=EXTENSION_MODULE,
    )


def _build_foreign_extension_error(
    build_names: list[str], extension_suffixes: list[str]
) -> ImportError:
    return ImportError(
        f"{_REQUIREMENT}, which is installed only as {', '.join(build_names)}; "
        f"this interpreter does not load those builds (it loads "
        f"{EXTENSION_STEM}{extension_suffixes[0]}). {_REBUILD_ADVICE}",
        name=EXTENSION_MODULE,
    )


def _build_failed_import_error(error: ImportError) -> ImportError:
    return ImportError(
        f"{_REQUIREMENT}, which is installed but failed to import "
        f"({type(error).__name__}: {error}). {_REBUILD_ADVICE}",
        name=EXTENSION_MODULE,
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
        name=EXTENSION_MODULE,
    )


def normalize_pep440_version(version: str) -> str | None:
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
        extension = importlib.import_module(EXTENSION_MODULE)
    except ModuleNotFoundError as error:
        if error.name != EXTENSION_MODULE:
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
        or normalize_pep440_version(extension_version) != package_version
    ):
        raise _build_stale_extension_error(extension_version, package_version)
    return extension


SelectionReason = Literal["environment", "entry_points", "own"]
"""Why a native module was selected: the variable, the entry points, or neither."""


@dataclass(frozen=True)
class Candidate:
    """A native module that entry points advertise, with its entry points.

    Each entry point's name is a layer the module claims to contain: the
    import name of a Python package whose Rust binding the module holds. A
    candidate has at least one entry point, each named by a layer name and
    valued, stripped of surrounding space, with the module; ``ValueError``
    is raised otherwise.
    """

    module: str
    entry_points: tuple[importlib.metadata.EntryPoint, ...]

    def __post_init__(self) -> None:
        if not self.entry_points:
            raise ValueError(
                "a candidate is advertised by at least one entry point, but "
                f"{self.module!r} has none"
            )
        for entry_point in self.entry_points:
            if not _is_layer_name(entry_point.name):
                raise ValueError(
                    "an entry point of a candidate is named by a layer (an "
                    "identifier that is not a keyword), not "
                    f"{entry_point.name!r}"
                )
            if entry_point.value.strip() != self.module:
                raise ValueError(
                    "every entry point of a candidate names its module "
                    f"{self.module!r}, but the entry point {entry_point.name!r} "
                    f"names {entry_point.value!r}"
                )

    @property
    def layers(self) -> frozenset[str]:
        """The layers the module claims, which always include ``fhy_core``."""
        return frozenset({_PACKAGE_NAME, *(ep.name for ep in self.entry_points)})


@dataclass(frozen=True)
class Selection:
    """The native module a process runs on, and why it was chosen.

    Attributes:
        module: Name of the selected module.
        reason: ``"environment"`` if the environment variable named it,
            ``"entry_points"`` if the entry points advertised it, ``"own"``
            if nothing was advertised and ``fhy_core._rs`` is used.
        candidates: Every module the entry points advertise, sorted by name,
            whether or not it was selected.
        skipped: The entry points left out of the candidates because their
            name is not a layer name or their value is not a module path,
            which only the environment variable allows; without it they are
            refused.

    Only a state the loader can reach is built; ``ValueError`` is raised for
    a selection for the reason ``"own"`` of a module other than
    ``fhy_core._rs`` or with candidates, one for the reason
    ``"entry_points"`` of a module that is not a candidate, candidates that
    are not sorted by module or repeat one, and skipped entry points under a
    reason other than ``"environment"``.

    """

    module: str
    reason: SelectionReason
    candidates: tuple[Candidate, ...]
    skipped: tuple[importlib.metadata.EntryPoint, ...] = ()

    def __post_init__(self) -> None:
        if self.skipped and self.reason != "environment":
            names = [entry_point.name for entry_point in self.skipped]
            raise ValueError(
                "a selection skips entry points only under the environment "
                f"variable, but one for the reason {self.reason!r} skips {names}"
            )
        modules = [candidate.module for candidate in self.candidates]
        if any(first >= second for first, second in itertools.pairwise(modules)):
            raise ValueError(
                "a selection's candidates are sorted by module with none "
                f"repeated, not {modules}"
            )
        if self.reason == "own":
            if self.module != EXTENSION_MODULE:
                raise ValueError(
                    "a selection for the reason 'own' is of "
                    f"{EXTENSION_MODULE!r}, not {self.module!r}"
                )
            if modules:
                raise ValueError(
                    "a selection for the reason 'own' has no candidates, since "
                    f"an advertised module would be selected, but it has {modules}"
                )
        if self.reason == "entry_points" and self.module not in modules:
            raise ValueError(
                "a selection for the reason 'entry_points' is of an advertised "
                f"module, but {self.module!r} is not among the candidates "
                f"{modules}"
            )

    @property
    def candidate(self) -> Candidate | None:
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


_SELECTION: Selection | None = None


def get_current_selection() -> Selection:
    """Return the selection made while ``fhy_core`` was imported.

    Returns:
        The native module of the process, and why it was chosen.

    Raises:
        RuntimeError: If no native module has been selected.

    """
    if _SELECTION is None:
        raise RuntimeError("fhy_core has not selected its native module")
    return _SELECTION


def get_distribution_name(entry_point: importlib.metadata.EntryPoint) -> str:
    """Return the name of the distribution of an entry point, or ``""``."""
    distribution = getattr(entry_point, "dist", None)
    return "" if distribution is None else str(distribution.name)


def describe_candidate(candidate: Candidate) -> str:
    """Return the module of a candidate and the entry points that name it."""
    owners = sorted(
        {(ep.name, get_distribution_name(ep)) for ep in candidate.entry_points},
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
        name=EXTENSION_MODULE,
    )


def _describe_tie(required: frozenset[str], tied: Sequence[Candidate]) -> str:
    return (
        "several installed modules each contain every installed layer "
        f"({_format_layers(required)}): "
        f"{'; '.join(describe_candidate(candidate) for candidate in tied)}"
    )


def _describe_incomparable(
    required: frozenset[str], candidates: Sequence[Candidate]
) -> str:
    ordered = sorted(candidates, key=lambda c: (-len(c.layers), c.module))
    lacks = "; ".join(
        f"{describe_candidate(candidate)} lacks "
        f"{_format_layers(required - candidate.layers)}"
        for candidate in ordered
    )
    return (
        "no installed module contains every installed layer "
        f"({_format_layers(required)}): {lacks}"
    )


def _find_complete_candidates(
    candidates: Sequence[Candidate],
) -> tuple[frozenset[str], list[Candidate]]:
    """Return the installed layers and the candidates that contain them all."""
    required = frozenset().union(*(candidate.layers for candidate in candidates))
    complete = [candidate for candidate in candidates if candidate.layers == required]
    return required, complete


def describe_candidate_conflict(candidates: Sequence[Candidate]) -> str | None:
    """Return why no one candidate can be selected, if none can.

    Args:
        candidates: The advertised modules.

    Returns:
        A description of the tie, naming the tied modules, or of the
        incomparable set, naming what each module lacks; ``None`` if there
        are no candidates or exactly one contains every installed layer.

    """
    if not candidates:
        return None
    required, complete = _find_complete_candidates(candidates)
    if not complete:
        return _describe_incomparable(required, candidates)
    if len(complete) > 1:
        return _describe_tie(required, complete)
    return None


def _build_tie_error(
    required: frozenset[str], tied: Sequence[Candidate]
) -> ImportError:
    return ImportError(
        f"{_PACKAGE_NAME} runs on one native extension module per process, "
        f"but {_describe_tie(required, tied)}. "
        f"{_PACKAGE_NAME} cannot choose between them. Uninstall all but one "
        f"of the distributions that provide them, or name the one to use in "
        f"the environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}.",
        name=EXTENSION_MODULE,
    )


def _build_incomparable_error(
    required: frozenset[str], candidates: Sequence[Candidate]
) -> ImportError:
    return ImportError(
        f"{_PACKAGE_NAME} runs on one native extension module per process, "
        f"but {_describe_incomparable(required, candidates)}. Install a "
        "product that contains all of those layers, or name the module to use in the "
        f"environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}, which "
        "leaves the layers it lacks without their native code.",
        name=EXTENSION_MODULE,
    )


def _build_invalid_layer_name_error(
    entry_point: importlib.metadata.EntryPoint,
) -> ImportError:
    distribution = get_distribution_name(entry_point)
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
        name=EXTENSION_MODULE,
    )


def _build_invalid_module_value_error(
    entry_point: importlib.metadata.EntryPoint,
) -> ImportError:
    distribution = get_distribution_name(entry_point)
    owner = f" of {distribution!r}" if distribution else ""
    return ImportError(
        f"The entry point {entry_point.name!r}{owner} in the group "
        f"{NATIVE_ENTRY_POINT_GROUP!r} has the value {entry_point.value!r}, "
        "but an entry point's value is the native module, an importable "
        "module path such as 'my_product._native' with no attribute or "
        "extras. Fix the entry point, or name the module to use in the "
        f"environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE}.",
        name=EXTENSION_MODULE,
    )


def _collect_candidates(
    entry_points: Iterable[importlib.metadata.EntryPoint], *, strict: bool
) -> tuple[tuple[Candidate, ...], tuple[importlib.metadata.EntryPoint, ...]]:
    """Group entry points by the module they name, across distributions.

    Args:
        entry_points: Entry points of the group ``fhy_core.native``.
        strict: Whether an entry point whose name is not a layer name, or
            whose value is not a module path, is an error. If not, such entry
            points are skipped.

    Returns:
        One candidate per distinct module, sorted by module name, with its
        entry points sorted by distribution and name; and the skipped entry
        points, in the order given, which are none if ``strict``.

    Raises:
        ImportError: If ``strict`` and an entry point's name is not a valid
            layer name (an identifier that is not a keyword), or its value,
            stripped of surrounding space, is not an absolute module path
            (dot-separated layer names, with no attribute or extras). The name
            is checked first.

    """
    by_module: dict[str, list[importlib.metadata.EntryPoint]] = {}
    skipped: list[importlib.metadata.EntryPoint] = []
    for entry_point in entry_points:
        if not _is_layer_name(entry_point.name):
            if strict:
                raise _build_invalid_layer_name_error(entry_point)
            skipped.append(entry_point)
            continue
        module = entry_point.value.strip()
        if not is_module_name(module):
            if strict:
                raise _build_invalid_module_value_error(entry_point)
            skipped.append(entry_point)
            continue
        by_module.setdefault(module, []).append(entry_point)
    candidates = tuple(
        Candidate(
            module,
            tuple(
                sorted(
                    claims,
                    key=lambda ep: (get_distribution_name(ep), ep.name),
                )
            ),
        )
        for module, claims in sorted(by_module.items())
    )
    return candidates, tuple(skipped)


def _choose_candidate(candidates: Sequence[Candidate]) -> Candidate | None:
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
    required, complete = _find_complete_candidates(candidates)
    if not complete:
        raise _build_incomparable_error(required, candidates)
    if len(complete) > 1:
        raise _build_tie_error(required, complete)
    return complete[0]


def _build_invalid_variable_value_error(value: str) -> ImportError:
    return ImportError(
        f"The environment variable {NATIVE_MODULE_ENVIRONMENT_VARIABLE} is set "
        f"to {value!r}, which is not an importable module path (dot-separated "
        "identifiers that are not keywords). Set it to the module to use, such "
        f"as {EXTENSION_MODULE!r}, or unset it to let {_PACKAGE_NAME} choose.",
        name=EXTENSION_MODULE,
    )


def _select_native_module(
    environ: Mapping[str, str],
    entry_points: Iterable[importlib.metadata.EntryPoint],
) -> Selection:
    """Select the native module to load.

    Args:
        environ: The process environment.
        entry_points: Entry points of the group ``fhy_core.native``.

    Returns:
        A `Selection` of the module named by ``FHY_CORE_NATIVE_MODULE``,
        else of the one advertised module that contains every installed
        layer, else of ``fhy_core._rs``, with the reason it was chosen,
        every advertised candidate and, under the variable, the entry points
        skipped as invalid.

    Raises:
        ImportError: If the variable is set to a value that is not an
            importable module path; if the variable is not set and an entry
            point's name is not a valid layer name (an identifier that is not
            a keyword) or its value is not a module path; or if the
            advertised modules tie or are incomparable.

    """
    named = environ.get(NATIVE_MODULE_ENVIRONMENT_VARIABLE, "").strip()
    if named and not is_module_name(named):
        raise _build_invalid_variable_value_error(named)
    candidates, skipped = _collect_candidates(entry_points, strict=not named)
    if named:
        return Selection(named, "environment", candidates, skipped)
    chosen = _choose_candidate(candidates)
    if chosen is None:
        return Selection(EXTENSION_MODULE, "own", candidates)
    return Selection(chosen.module, "entry_points", candidates)


def _build_aggregate_import_error(
    selection: Selection, error: Exception
) -> ImportError:
    failure = f"failed to import ({type(error).__name__}: {error})"
    if selection.reason == "environment":
        return ImportError(
            f"{NATIVE_MODULE_ENVIRONMENT_VARIABLE}={selection.module} selected "
            f"the native extension module {selection.module!r}, but it {failure}. "
            f"Unset {NATIVE_MODULE_ENVIRONMENT_VARIABLE} to let {_PACKAGE_NAME} "
            "choose, or set it to an importable module.",
            name=selection.module,
        )
    candidate = selection.candidate
    described = (
        repr(selection.module) if candidate is None else describe_candidate(candidate)
    )
    return ImportError(
        f"{_PACKAGE_NAME} selected the native extension module {described} "
        f"from the installed entry points, but it {failure}. Reinstall the "
        "distribution that provides it, or uninstall that distribution.",
        name=selection.module,
    )


def _build_stale_aggregate_error(
    selection: Selection, extension_version: object, package_version: str
) -> ImportError:
    module_name = selection.module
    reported_version = (
        f"no {_AGGREGATE_VERSION_ATTRIBUTE} attribute"
        if extension_version is None
        else f"version {extension_version!r}"
    )
    stale = (
        f"which must hold {_PACKAGE_NAME} {package_version!r}, but it reports "
        f"{reported_version}, so it is stale or is not a {_PACKAGE_NAME} "
        "extension"
    )
    target = f"for {_PACKAGE_NAME} {package_version!r}"
    if selection.reason == "environment":
        return ImportError(
            f"{NATIVE_MODULE_ENVIRONMENT_VARIABLE}={module_name} selected the "
            f"native extension module {module_name!r}, {stale}. Rebuild or "
            f"reinstall the package that provides it {target}, or unset "
            f"{NATIVE_MODULE_ENVIRONMENT_VARIABLE} to let {_PACKAGE_NAME} "
            "choose.",
            name=module_name,
        )
    candidate = selection.candidate
    described = (
        repr(module_name) if candidate is None else describe_candidate(candidate)
    )
    return ImportError(
        f"{_PACKAGE_NAME} selected the native extension module {described} "
        f"from the installed entry points, {stale}. Rebuild or reinstall the "
        f"distribution that provides it {target}, or uninstall that "
        "distribution.",
        name=module_name,
    )


def _build_undeclared_layer_error(
    module_name: str, missing: Sequence[importlib.metadata.EntryPoint]
) -> ImportError:
    claims = "; ".join(
        f"the entry point {ep.name!r}"
        + (f" of {get_distribution_name(ep)!r}" if get_distribution_name(ep) else "")
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
    module_name: str, extension: ModuleType, candidate: Candidate | None
) -> None:
    """Check a module holds every layer its entry points claim for it.

    A layer ``L`` is present in a module if and only if the module has a
    ``str`` attribute ``__L_version__``. The layer ``fhy_core`` is checked
    separately, by its version.

    Args:
        module_name: Name `extension` was imported as.
        extension: The imported native module.
        candidate: The entry points that advertise the module, or ``None``
            only if no entry point names it, as when the environment variable
            names a module nothing advertises. A module the variable names
            that entry points advertise has a candidate, and is verified.

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


def _load_aggregate_extension(selection: Selection) -> None:
    """Load a combined extension module and install it as ``fhy_core._rs``.

    Args:
        selection: The selection of a module other than ``fhy_core._rs``.
            The claimed layers of its candidate, if any, are verified, and
            its reason decides the advice given if the module fails to
            import or is stale: the variable is named if it chose the
            module, else the entry points that advertise it.

    Raises:
        ImportError: If importing the module raises an ``Exception``, which
            becomes the cause of this error, or the module does not report the
            installed package's version as ``__fhy_core_version__``, lacks a
            layer its entry points claim, or another native module is already
            installed as ``fhy_core._rs``.

    """
    module_name = selection.module
    try:
        extension = importlib.import_module(module_name)
    except Exception as error:
        raise _build_aggregate_import_error(selection, error) from error
    extension_version = getattr(extension, _AGGREGATE_VERSION_ATTRIBUTE, None)
    package_version = importlib.metadata.version(_PACKAGE_NAME)
    if (
        not isinstance(extension_version, str)
        or normalize_pep440_version(extension_version) != package_version
    ):
        raise _build_stale_aggregate_error(
            selection, extension_version, package_version
        )
    _verify_declared_layers(module_name, extension, selection.candidate)
    _install_as_own_extension(module_name, extension)


def _install_as_own_extension(module_name: str, extension: ModuleType) -> None:
    """Make ``fhy_core._rs`` name `extension`, refusing to replace another.

    Args:
        module_name: Name `extension` was imported as.
        extension: The combined extension module.

    Raises:
        ImportError: If ``fhy_core._rs`` names another module.

    """
    held = sys.modules.get(EXTENSION_MODULE)
    if held is not None and held is not extension:
        held_file = getattr(held, "__file__", None)
        raise _build_two_native_modules_error(
            f"{module_name!r}",
            f"{EXTENSION_MODULE!r}"
            + (f" (loaded from {held_file})" if held_file else ""),
        )
    sys.modules[EXTENSION_MODULE] = extension
    setattr(sys.modules[_PACKAGE_NAME], EXTENSION_STEM, extension)


def _check_extension() -> None:
    """Load the native module of this process and check it.

    Raises:
        ImportError: If ``FHY_CORE_NATIVE_MODULE`` is set to a value that is
            not an importable module path; if, with the variable unset, an
            entry point's name is not a valid layer name or its value is not
            a module path, or the advertised modules tie or are
            incomparable; if the selected module is not installed, is built
            only for other interpreters, or fails to import; if it reports a
            missing, non-PEP 440 or stale version; if it does not report a
            layer its entry points claim; or if ``fhy_core._rs`` already
            names another module. The message names the cause and the fix.

    """
    global _SELECTION  # noqa: PLW0603
    selection = _select_native_module(
        os.environ, importlib.metadata.entry_points(group=NATIVE_ENTRY_POINT_GROUP)
    )
    _SELECTION = selection
    if selection.module == EXTENSION_MODULE:
        _load_and_verify_own_extension(selection.candidate)
    else:
        _load_aggregate_extension(selection)


def _load_and_verify_own_extension(candidate: Candidate | None) -> None:
    """Load ``fhy_core._rs`` and check its version and claimed layers.

    If this call imports the module and then refuses it, the module is
    removed from ``sys.modules`` before the error is raised, so a later
    import of ``fhy_core`` in the process can select another module. A module
    already in ``sys.modules`` before the call is left there.

    Args:
        candidate: The entry points that advertise ``fhy_core._rs``, or
            ``None`` if none does.

    Raises:
        ImportError: If the module cannot be loaded, reports another version
            than the installed package's, or lacks a layer its entry points
            claim.

    """
    was_held = EXTENSION_MODULE in sys.modules
    try:
        extension = _load_own_extension()
        _verify_declared_layers(EXTENSION_MODULE, extension, candidate)
    except ImportError:
        if not was_held:
            sys.modules.pop(EXTENSION_MODULE, None)
        raise


_check_extension()
