"""The native module ``fhy_core`` runs on, and the layers it holds.

A process runs on one native (Rust extension) module, installed as
``fhy_core._rs``. A downstream product that builds an aggregate module,
``fhy_core``'s binding plus its own, advertises it in the entry-point group
``fhy_core.native``: an entry point ``name = "module"`` means the module
contains the *layer* ``name``, the import name of the Python package whose
Rust binding the module holds. ``fhy_core`` is always a layer. A layer is
present in a module if and only if the module has a ``str`` attribute
``__<layer>_version__``.

This module is the interface a downstream package uses to claim its layer:
call `require_layer` while that package is being imported, to check the layer is in
the native module of the process and to install the module as the package's
own ``_rs``.
"""

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

from fhy_core import _extension
from fhy_core._extension import (
    _EXTENSION_MODULE,
    _EXTENSION_STEM,
    NATIVE_ENTRY_POINT_GROUP,
    NATIVE_MODULE_ENVIRONMENT_VARIABLE,
    _describe_candidate,
    _distribution_name,
    _is_module_name,
    _normalize_pep440_version,
    get_layer_version_attribute,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "MODULE_ENVIRONMENT_VARIABLE",
    "NativeSelection",
    "get_layer_version_attribute",
    "get_selection",
    "require_layer",
]

ENTRY_POINT_GROUP = NATIVE_ENTRY_POINT_GROUP
"""The entry-point group in which a product advertises its native module."""

MODULE_ENVIRONMENT_VARIABLE = NATIVE_MODULE_ENVIRONMENT_VARIABLE
"""The environment variable that names the native module, overriding the group."""


@dataclass(frozen=True)
class NativeSelection:
    """The native module this process runs on, and why it was chosen.

    Attributes:
        module: Name of the module installed as ``fhy_core._rs``.
        reason: ``"environment"`` if ``FHY_CORE_NATIVE_MODULE`` named it,
            ``"entry_points"`` if installed distributions advertised it, or
            ``"own"`` if nothing was advertised and it is ``fhy_core._rs``.
        layers: The layers its entry points declare it contains, always
            including ``fhy_core``.

    """

    module: str
    reason: Literal["environment", "entry_points", "own"]
    layers: frozenset[str]


def _current_selection() -> _extension._Selection:
    chosen = _extension._SELECTION
    if chosen is None:
        raise RuntimeError("fhy_core has not selected its native module")
    return chosen


def get_selection() -> NativeSelection:
    """Return the native module this process runs on.

    Returns:
        What was selected while ``fhy_core`` was imported, and why.

    """
    chosen = _current_selection()
    return NativeSelection(chosen.module, chosen.reason, chosen.layers)


def _build_missing_layer_error(layer: str, version: str) -> ImportError:
    chosen = _current_selection()
    attribute = get_layer_version_attribute(layer)
    if chosen.reason == "environment":
        holders = [c for c in chosen.candidates if layer in c.layers]
        advice = (
            "Installed modules that do: "
            + "; ".join(_describe_candidate(c) for c in holders)
            + ". Unset the variable to let fhy_core choose, or set it to one "
            "of those."
            if holders
            else "No installed distribution declares that layer; install the "
            "package that provides it, or unset the variable."
        )
        return ImportError(
            f"{NATIVE_MODULE_ENVIRONMENT_VARIABLE}={chosen.module} selected "
            f"{chosen.module!r}, which does not contain the layer {layer!r} "
            f"(no {attribute}). {advice}",
            name=chosen.module,
        )
    layers = ", ".join(repr(name) for name in sorted(chosen.layers))
    return ImportError(
        f"The layer {layer!r} (version {version!r}) is not in the native "
        f"module {chosen.module!r} that fhy_core runs on, which contains "
        f"{layers}, and no installed distribution declares it in the entry "
        f"point group {NATIVE_ENTRY_POINT_GROUP!r}. Reinstall the package that "
        f"provides the aggregate containing {layer!r}.",
        name=chosen.module,
    )


def _build_stale_layer_error(
    layer: str, version: str, reported: str, *, is_pep440: bool
) -> ImportError:
    chosen = _current_selection()
    candidate = chosen.candidate
    distributions = sorted(
        {name for ep in candidate.entry_points if (name := _distribution_name(ep))}
        if candidate is not None
        else set()
    )
    source = (
        "the distribution that advertised the module "
        f"({', '.join(map(repr, distributions))})"
        if distributions
        else "the package that provides the module"
    )
    report = (
        f"reports version {reported!r} in {get_layer_version_attribute(layer)}"
        if is_pep440
        else f"reports {reported!r} in {get_layer_version_attribute(layer)}, which "
        "is not a PEP 440 version"
    )
    return ImportError(
        f"The layer {layer!r} needs version {version!r}, but the native "
        f"module {chosen.module!r} that fhy_core runs on {report}, so it is "
        f"stale. Rebuild or reinstall {source}.",
        name=chosen.module,
    )


def require_layer(
    layer: str, version: str, *, package: str | None = None
) -> ModuleType:
    """Check the native module holds a layer, and install it for a package.

    Call this while the package that owns the layer is imported. The check
    is that the native module ``fhy_core`` runs on has the attribute
    ``__<layer>_version__`` and that its PEP 440 normal form equals that of
    `version`. The module is then installed as ``<package>._rs``, in
    ``sys.modules`` and as the attribute ``_rs`` of the package, so the
    package's binding finds its state there as ``fhy_core``'s finds its own
    in ``fhy_core._rs``. Nothing is installed unless every check passes.

    Args:
        layer: Import name of the package whose Rust binding is the layer.
        version: The version of the package, a PEP 440 version in any
            spelling (a Cargo version such as ``1.2.3-rc.1`` is one), which
            the native module's version of the layer must equal in normal
            form.
        package: Package to install the module under as ``_rs``, an
            importable module path that is in ``sys.modules``. Defaults to
            `layer`.

    Returns:
        The native module of the process.

    Raises:
        ImportError: If the module does not contain the layer, or reports
            another version of it or one that is not a PEP 440 version. The
            message names the reason the module was selected and what to do.
        ValueError: If `layer` is not a valid identifier or is a keyword,
            `version` is not a PEP 440 version, or the package (`package`, or
            `layer` if it is ``None``) is not an importable module path or is
            not in ``sys.modules``.

    """
    attribute = get_layer_version_attribute(layer)
    required = _normalize_pep440_version(version)
    if required is None:
        raise ValueError(f"a layer's version is a PEP 440 version, not {version!r}")
    owner = package if package is not None else layer
    if not _is_module_name(owner):
        raise ValueError(
            f"a package is named by an importable module path, not {owner!r}"
        )
    parent = sys.modules.get(owner)
    if parent is None:
        raise ValueError(
            f"the package {owner!r} is not in sys.modules; call require_layer "
            "while that package is being imported"
        )
    module = sys.modules[_EXTENSION_MODULE]
    reported = getattr(module, attribute, None)
    if not isinstance(reported, str):
        raise _build_missing_layer_error(layer, version)
    normalized = _normalize_pep440_version(reported)
    if normalized != required:
        raise _build_stale_layer_error(
            layer, version, reported, is_pep440=normalized is not None
        )
    name = f"{owner}.{_EXTENSION_STEM}"
    held = sys.modules.get(name)
    if held is not None and held is not module:
        held_file = getattr(held, "__file__", None)
        where = f" (loaded from {held_file})" if held_file else ""
        raise ImportError(
            f"{name!r} is already another module{where} than the native "
            f"module {module.__name__!r} that fhy_core runs on, and a process "
            "holds one native extension module. Uninstall the package that "
            "provides the other module.",
            name=name,
        )
    sys.modules[name] = module
    setattr(parent, _EXTENSION_STEM, module)
    return module
