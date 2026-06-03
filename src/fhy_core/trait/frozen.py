"""`Frozen` trait and mixin."""

__all__ = [
    "Frozen",
    "FrozenFieldTypeError",
    "FrozenMixin",
    "FrozenMutationError",
    "FrozenValidationError",
]

import collections.abc
import datetime
import decimal
import enum
import fractions
import functools
import inspect
import pathlib
import re
import sys
import types
import typing
from abc import ABC
from dataclasses import is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    ForwardRef,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from frozendict import frozendict

from fhy_core.error import register_error
from fhy_core.logger import get_logger
from fhy_core.utils import Self

_LOGGER = get_logger(__name__)

_FROZEN_FLAG = "_fhy_core_is_frozen"
_CONSTRUCTION_DEPTH_FLAG = "_fhy_core_construction_depth"
_CLASS_SETUP_DONE_FLAG = "_fhy_core_class_setup_done"
_INIT_WRAPPER_MARKER = "_fhy_core_init_wrapper"
_IS_NATIVE_FROZEN_DATACLASS_CACHE = "_fhy_core_is_native_frozen_dataclass"
_FIELD_TYPE_CHECK_DONE_FLAG = "_fhy_core_field_type_check_done"

_MAX_FIELD_TYPE_DEPTH = 8

_FrozenMixinT = TypeVar("_FrozenMixinT", bound="FrozenMixin")


@register_error
class FrozenMutationError(AttributeError):
    """Raised when mutating a frozen object.

    Inherits from :class:`AttributeError` so callers that follow Python's
    standard ``__setattr__``/``__delattr__`` convention keep working,
    including code that catches :class:`AttributeError` (the base class
    of :class:`dataclasses.FrozenInstanceError`) without explicit
    knowledge of FhY's error type. Note that catching
    :class:`dataclasses.FrozenInstanceError` itself will *not* catch this
    error, as the two are sibling subclasses of :class:`AttributeError`.

    """


@register_error
class FrozenValidationError(RuntimeError):
    """Raised when runtime frozen verification fails."""


@register_error
class FrozenFieldTypeError(TypeError):
    """Raised when a `FrozenMixin` subclass declares a mutable field type.

    Fires on first instantiation of a concrete subclass, mirroring the
    convention used by :class:`abc.ABC` when refusing to instantiate a
    concrete class with unimplemented abstract methods. The error message
    names every offending field and its declared type.

    """


@runtime_checkable
class Frozen(Protocol):
    """Protocol for objects that participate in the freezing lifecycle.

    A consumer that probes ``isinstance(obj, Frozen)`` to decide whether
    a value is safe to cache or hash by identity should additionally
    consult :attr:`is_frozen` before assuming the object is in its
    frozen state.

    """

    @property
    def is_frozen(self) -> bool: ...

    def freeze(self) -> None: ...

    def assert_frozen(self) -> None: ...


_IMMUTABLE_WHITELIST: tuple[type, ...] = (
    type(None),
    types.EllipsisType,
    types.NotImplementedType,
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    type,
    enum.Enum,
    enum.Flag,
    decimal.Decimal,
    fractions.Fraction,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    datetime.timezone,
    re.Pattern,
    pathlib.PurePath,
    tuple,
    frozenset,
    frozendict,
)
"""Concrete classes considered immutable for field-type validation.

Parameterized generics (``tuple[T, ...]``, ``frozenset[T]`` etc.) are
inspected recursively in :func:`_is_immutable_annotation`; the
container origin is checked against this whitelist and the type
parameters are checked against the same predicate.

``types.MappingProxyType`` is deliberately excluded: it is a read-only
*view* over a backing mapping that can still mutate underneath it, so it
is not a sound immutable field type. Use ``frozendict`` instead.
"""

_MUTABLE_OUTRIGHT: tuple[type, ...] = (
    list,
    set,
    dict,
    bytearray,
)
"""Concrete classes rejected as mutable regardless of parameterization."""


def _is_immutable_class(cls: type) -> bool:
    if issubclass(cls, _MUTABLE_OUTRIGHT):
        return False
    if issubclass(cls, FrozenMixin):
        return True
    for candidate in _IMMUTABLE_WHITELIST:
        if issubclass(cls, candidate):
            return True
    return False


_IMMUTABLE_LEAF_SINGLETONS: frozenset[Any] = frozenset(
    {None, type(None), Ellipsis, Any, Self}
)


def _is_immutable_leaf_annotation(annotation: Any) -> bool:
    """Return ``True`` for leaf-style annotations treated as immutable.

    Covers ``None``, ``NoneType``, ``Ellipsis``, ``typing.Self``,
    ``typing.Any`` (best-effort: the developer is trusted), and free
    type variables (``TypeVar``, bare ``ForwardRef``).
    """
    if annotation in _IMMUTABLE_LEAF_SINGLETONS:
        return True
    return isinstance(annotation, (TypeVar, ForwardRef))


def _is_immutable_parameterized_origin(
    origin: Any, args: tuple[Any, ...], depth: int
) -> bool | None:
    """Return ``True``/``False`` for known parameterized origins, ``None`` otherwise."""
    if origin is type or origin is collections.abc.Callable:
        return True
    elif origin is tuple or origin in {
        frozenset,
        frozendict,
    }:
        return all(_is_immutable_annotation(arg, depth - 1) for arg in args)
    else:
        return None


def _is_immutable_annotation(
    annotation: Any, depth: int = _MAX_FIELD_TYPE_DEPTH
) -> bool:
    """Return ``True`` iff ``annotation`` statically describes an immutable type.

    Recurses through ``Union``/``Optional``, parameterized containers,
    ``Annotated[T, ...]``, and ``Final[T]``. ``ClassVar`` is skipped at
    the caller level (it isn't an instance field). ``TypeVar``, bare
    ``ForwardRef``, and ``typing.Self`` are treated as immutable on a
    best-effort basis: the developer is trusted to fill them with
    immutable types. ``Any`` is allowed for the same reason a bare
    ``tuple`` is allowed: the author has chosen to leave the value type
    unspecified.

    Args:
        annotation: A resolved or unresolved type annotation.
        depth: Maximum recursion depth before bailing out as mutable.

    Returns:
        True when the annotation matches the immutability whitelist.

    """
    if depth <= 0:
        return False
    if _is_immutable_leaf_annotation(annotation):
        return True

    origin = get_origin(annotation)
    if origin is Annotated or origin is typing.Final:
        return _is_immutable_annotation(get_args(annotation)[0], depth - 1)
    if origin is Union or origin is types.UnionType:
        return all(
            _is_immutable_annotation(arg, depth - 1) for arg in get_args(annotation)
        )

    parameterized = _is_immutable_parameterized_origin(
        origin, get_args(annotation), depth
    )
    if parameterized is not None:
        return parameterized
    elif isinstance(origin, type):
        return _is_immutable_class(origin)
    elif isinstance(annotation, type):
        return _is_immutable_class(annotation)
    else:
        return False


def _iter_frozen_field_names(target_cls: type) -> list[str]:
    """Return field names declared on the subclass or its ``FrozenMixin`` ancestors.

    Annotations from non-``FrozenMixin`` mixins (such as ``EqualMixin``)
    are excluded since they're not the subclass's freezing contract.
    Annotations from ``FrozenMixin`` itself (the ``ClassVar`` machinery
    flags) are excluded.

    Uses :func:`inspect.get_annotations` so the lookup works under
    PEP 749 (Python 3.14+) deferred annotations, where the raw
    ``__annotations__`` slot in ``cls.__dict__`` may be absent and
    the dict is materialized lazily via the ``__annotate__`` callable.

    """
    seen: dict[str, None] = {}
    for klass in reversed(target_cls.__mro__):
        if klass is FrozenMixin or klass is object:
            continue
        if not issubclass(klass, FrozenMixin):
            continue
        for name in inspect.get_annotations(klass):
            seen.setdefault(name, None)
    return list(seen.keys())


def _resolve_frozen_field_hints(target_cls: type) -> tuple[dict[str, Any], bool]:
    """Resolve frozen-field annotations to types, degrading per field.

    Tries a single whole-class resolution first. On failure (typically
    an unresolved forward reference), falls back to resolving each frozen
    field individually so one unresolvable annotation does not suppress
    the immutability check for the rest of the class.

    Returns the resolved ``{field_name: type}`` mapping and a flag that
    is ``True`` only when every frozen field resolved. Unresolvable
    fields are logged and omitted from the mapping.

    """
    try:
        return get_type_hints(target_cls, include_extras=True), True
    except Exception as exc:  # noqa: BLE001 - resolution failure modes vary
        _LOGGER.warning(
            "field-type check for %s: whole-class annotation resolution failed "
            "(%s: %s); falling back to per-field resolution",
            target_cls.__name__,
            type(exc).__name__,
            exc,
        )

    resolved: dict[str, Any] = {}
    all_resolved = True
    for klass in reversed(target_cls.__mro__):
        if klass is FrozenMixin or klass is object:
            continue
        if not issubclass(klass, FrozenMixin):
            continue
        module = sys.modules.get(klass.__module__)
        globalns = getattr(module, "__dict__", {})
        localns = dict(vars(target_cls))
        for name, raw in inspect.get_annotations(klass).items():
            annotation: Any = ForwardRef(raw) if isinstance(raw, str) else raw
            try:
                resolved[name] = typing._eval_type(  # type: ignore[attr-defined]  # noqa: SLF001
                    annotation, globalns, localns
                )
            except Exception as exc:  # noqa: BLE001 - per-field failure modes vary
                all_resolved = False
                _LOGGER.warning(
                    "field-type check for %s: could not resolve field %r "
                    "(%s: %s); skipping that field",
                    target_cls.__name__,
                    name,
                    type(exc).__name__,
                    exc,
                )
    return resolved, all_resolved


def _check_field_types(target_cls: type) -> None:
    """Raise :class:`FrozenFieldTypeError` if a concrete subclass has mutable fields.

    No-op for abstract classes and for classes already checked. Latches the
    check as done only when every field resolved, so an unresolved forward
    reference is re-attempted on a later instantiation rather than skipped
    permanently.
    """
    if target_cls.__dict__.get(_FIELD_TYPE_CHECK_DONE_FLAG, False):
        return
    if getattr(target_cls, "__abstractmethods__", frozenset()):
        return

    resolved_hints, all_resolved = _resolve_frozen_field_hints(target_cls)

    offenders: list[tuple[str, Any]] = []
    for field_name in _iter_frozen_field_names(target_cls):
        field_type = resolved_hints.get(field_name)
        if field_type is None:
            continue
        if get_origin(field_type) is ClassVar:
            continue
        if not _is_immutable_annotation(field_type):
            offenders.append((field_name, field_type))

    if offenders:
        details = ", ".join(f"{name}: {annotation!r}" for name, annotation in offenders)
        raise FrozenFieldTypeError(
            f"{target_cls.__name__} declares mutable field type(s): {details}. "
            f"Replace each with an immutable equivalent (e.g. tuple, "
            f"frozenset, frozendict) or have the nested type inherit "
            f"FrozenMixin."
        )

    if all_resolved:
        type.__setattr__(target_cls, _FIELD_TYPE_CHECK_DONE_FLAG, True)


def _compute_is_native_frozen_dataclass(target_cls: type) -> bool:
    """Return ``True`` if the class is a ``@dataclass(frozen=True)``."""
    params = getattr(target_cls, "__dataclass_params__", None)
    return bool(
        is_dataclass(target_cls)
        and params is not None
        and getattr(params, "frozen", False)
    )


def _swap_dataclass_frozen_dispatch(target_cls: type) -> None:
    """Replace a dataclass-frozen ``__setattr__``/``__delattr__`` with the mixin's.

    Normalizes the raised error type to :class:`FrozenMutationError` and lets
    the mixin's bookkeeping-flag carve-outs work.
    """
    if "__setattr__" in target_cls.__dict__:
        type.__setattr__(target_cls, "__setattr__", FrozenMixin.__setattr__)
    if "__delattr__" in target_cls.__dict__:
        type.__setattr__(target_cls, "__delattr__", FrozenMixin.__delattr__)


def _install_init_wrap(target_cls: type) -> None:
    """Wrap the class's ``__init__`` to auto-freeze after the outermost call."""
    original_init = target_cls.__init__  # type: ignore[misc]

    @functools.wraps(original_init)
    def wrapped(self: "FrozenMixin", *args: Any, **kwargs: Any) -> None:
        try:
            depth = object.__getattribute__(self, _CONSTRUCTION_DEPTH_FLAG)
        except AttributeError:
            depth = 0
        object.__setattr__(self, _CONSTRUCTION_DEPTH_FLAG, depth + 1)
        try:
            original_init(self, *args, **kwargs)
        except BaseException:
            object.__setattr__(self, _CONSTRUCTION_DEPTH_FLAG, depth)
            raise
        object.__setattr__(self, _CONSTRUCTION_DEPTH_FLAG, depth)
        # Use the *runtime* class's policy so a most-derived class with
        # ``freeze_on_init=False`` overrides a parent wrap that would
        # otherwise freeze. The depth check ensures only the outermost wrap
        # evaluates the policy.
        if depth == 0 and type(self)._FREEZE_ON_INIT:
            self.freeze()

    setattr(wrapped, _INIT_WRAPPER_MARKER, True)
    type.__setattr__(target_cls, "__init__", wrapped)


def _setup_class(target_cls: type) -> None:
    """Run the one-time per-subclass setup: cache state, swap dispatch, check fields.

    Caches the native-frozen-dataclass flag, swaps in the mixin's
    ``__setattr__``/``__delattr__`` for dataclass-frozen subclasses, and
    runs the field-type immutability check. The auto-freeze ``__init__`` wrap
    is installed separately in :meth:`FrozenMixin.__init_subclass__`, not here.

    Called from :meth:`FrozenMixin.__new__` on the first instantiation of any
    concrete ``FrozenMixin`` subclass. Idempotent across repeated
    instantiations (guarded by ``_CLASS_SETUP_DONE_FLAG``) and across subclass
    chains (each subclass runs its own setup).

    """
    is_native = _compute_is_native_frozen_dataclass(target_cls)
    type.__setattr__(target_cls, _IS_NATIVE_FROZEN_DATACLASS_CACHE, is_native)

    if is_native:
        _swap_dataclass_frozen_dispatch(target_cls)

    _check_field_types(target_cls)

    type.__setattr__(target_cls, _CLASS_SETUP_DONE_FLAG, True)


class FrozenMixin(ABC):
    """Mixin that enforces runtime immutability after construction.

    Auto-freeze:
        Subclasses are auto-frozen at the end of the outermost
        ``__init__``. Auto-freeze is on by default (opt-out); pass
        ``freeze_on_init=False`` as a class-creation kwarg to keep the
        instance mutable after ``__init__`` (the escape hatch is
        provided for downstream extensibility and is not used by any
        consumer in ``fhy_core`` itself).

        The policy is sticky across subclasses: a descendant that does
        not pass ``freeze_on_init`` inherits its base's setting.

        Manual construction paths that bypass ``__init__``
        (e.g. ``cls.__new__(cls)`` followed by direct attribute
        assignment, used by serialization and the deterministic-id
        testing patch) do not trigger the auto-freeze wrap and must
        call :meth:`freeze` explicitly when construction is complete.

    Composing with ``@dataclass(frozen=True)``:
        Dataclass-frozen subclasses inherit the dataclass's own
        freezing semantics: instances are frozen the moment the
        auto-generated ``__init__`` returns, and ``__post_init__``
        runs against an already-frozen instance (derived-field
        assignment must go through ``object.__setattr__``). This mixin
        replaces the dataclass-installed ``__setattr__`` and
        ``__delattr__`` with its own so the raised error type is
        uniformly :class:`FrozenMutationError`.

    Field-type immutability:
        On first instantiation of a concrete subclass, every annotated
        field is verified against an immutability whitelist. A subclass
        that declares a mutable field type raises
        :class:`FrozenFieldTypeError`, following the
        :class:`abc.ABC`-style convention of raising at instantiation
        rather than at class definition. Abstract subclasses skip the
        check until a concrete descendant is constructed.

    ``__slots__`` compatibility:
        ``FrozenMixin`` declares its bookkeeping attributes
        (``_FROZEN_FLAG``, ``_CONSTRUCTION_DEPTH_FLAG``) in
        ``__slots__``, so slot-only subclasses can inherit them without
        having to know the private names. A subclass that opts into
        slots gets a working ``FrozenMixin`` for free; if the subclass
        also wants ``weakref`` support, add ``__weakref__`` to its own
        ``__slots__``.

    """

    __slots__ = (_FROZEN_FLAG, _CONSTRUCTION_DEPTH_FLAG)

    _FREEZE_ON_INIT: ClassVar[bool] = True

    def __init_subclass__(
        cls,
        *,
        freeze_on_init: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if freeze_on_init is not None:
            cls._FREEZE_ON_INIT = freeze_on_init
        # Install the auto-freeze wrap here, BEFORE any sibling mixin's
        # ``__init_subclass__`` (e.g. ``InternedMixin``) wraps on top.
        # Cooperative ``super().__init_subclass__(**kwargs)`` ordering
        # places FrozenMixin's wrap as the inner wrap and InternedMixin's
        # as the outer wrap, so the instance is frozen by the time the
        # interner's finalize hook runs.
        #
        # Only wrap when the subclass defines its own ``__init__``. The
        # "no own ``__init__``" case covers both ``@dataclass(frozen=True)``
        # subclasses (whose ``__init__`` is generated by the decorator
        # *after* ``__init_subclass__`` runs --- defining our own wrap
        # here would suppress dataclass's ``__init__`` generation) and
        # subclasses that simply inherit a wrapped ``__init__`` from a
        # ``FrozenMixin`` ancestor (no need to double-wrap; the inherited
        # wrap fires through MRO).
        if cls._FREEZE_ON_INIT and "__init__" in cls.__dict__:
            _install_init_wrap(cls)

    if not TYPE_CHECKING:
        # Defined only at runtime so it does not advertise a permissive
        # ``*args, **kwargs`` signature to type checkers, which would
        # silently disable constructor-call argument checking for every
        # ``FrozenMixin`` subclass. Subclass ``__init__`` signatures
        # (those written by hand or generated by ``@dataclass``) drive
        # type checking of constructor calls.
        def __new__(
            cls: type[_FrozenMixinT], *args: Any, **kwargs: Any
        ) -> _FrozenMixinT:
            del args, kwargs
            if not cls.__dict__.get(_CLASS_SETUP_DONE_FLAG, False):
                _setup_class(cls)
            return super().__new__(cls)

    @property
    def is_frozen(self) -> bool:
        if type(self).__dict__.get(_IS_NATIVE_FROZEN_DATACLASS_CACHE, False):
            return True
        try:
            return bool(object.__getattribute__(self, _FROZEN_FLAG))
        except AttributeError:
            return False

    def freeze(self) -> None:
        """Idempotently transition this instance to the frozen state."""
        if self.is_frozen:
            return
        object.__setattr__(self, _FROZEN_FLAG, True)

    def assert_frozen(self) -> None:
        """Assert that this instance is frozen and its dispatch is wired up.

        Two checks run:

        1. :attr:`is_frozen` is ``True``.
        2. The class's ``__setattr__`` resolves to
           :meth:`FrozenMixin.__setattr__` (or, equivalently, to a
           dataclass-frozen dispatch that this mixin's setup swapped in).

        Raises:
            FrozenValidationError: When either check fails.

        """
        if not self.is_frozen:
            raise FrozenValidationError(f"{type(self).__name__} is not frozen.")
        if type(self).__setattr__ is not FrozenMixin.__setattr__:
            raise FrozenValidationError(
                f'"{type(self).__name__}" __setattr__ dispatch is not wired '
                f"to FrozenMixin; the freeze invariant is unenforceable."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {_FROZEN_FLAG, _CONSTRUCTION_DEPTH_FLAG}:
            object.__setattr__(self, name, value)
            return
        if name == "__orig_class__":
            # ``typing.Generic.__class_getitem__`` writes ``__orig_class__``
            # on the instance after ``Foo[int]()`` returns. Allow the write
            # so frozen generic subclasses don't crash on subscript binding.
            object.__setattr__(self, name, value)
            return
        if self.is_frozen:
            raise FrozenMutationError(
                f'Cannot modify "{name}" on frozen {type(self).__name__}.'
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in {_FROZEN_FLAG, _CONSTRUCTION_DEPTH_FLAG}:
            object.__delattr__(self, name)
            return
        if self.is_frozen:
            raise FrozenMutationError(
                f'Cannot delete "{name}" on frozen {type(self).__name__}.'
            )
        object.__delattr__(self, name)
