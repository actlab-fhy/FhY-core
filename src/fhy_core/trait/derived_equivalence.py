"""Schema-derived structural and alpha equivalence.

This module exposes :class:`DerivedEquivalenceMixin`, which derives both
``is_structurally_equivalent`` and ``is_alpha_equivalent_under`` for a
``@dataclass`` from its fields. It is the equivalence analog of the
schema-derived ``Serializable`` trait: both methods are the same
field-walk differing only in the per-field operator.

For each comparable field the engine picks an operator. By default it
dispatches on the field value's capability -- an
``AlphaEquivalence``/``StructuralEquivalence`` sub-object recurses, a
sequence is compared element-wise, and a scalar or ``PartialEqual`` value
compares with ``==``. The three ``Identifier`` roles -- value, reference,
and binder -- are type-identical and so are declared with
``field(metadata=...)`` via :func:`compared_as_value`,
:func:`compared_as_reference`, and :func:`compared_as_binder`. An
undeclared ``Identifier`` field defaults to ``==``, which keeps the
derived alpha relation conservative (too strict, never unsound).

A field whose value is none of the recognized categories (and carries no
role metadata) raises :class:`EquivalenceDerivationError`, naming the
field; supply :func:`compared_with` or :func:`compared_as_value` to
resolve it.

See ``docs/design/derived-equivalence.md`` for the full design.
"""

__all__ = [
    "EQUIVALENCE_METADATA_KEY",
    "DerivedEquivalenceMixin",
    "EquivalenceDerivationError",
    "FieldComparator",
    "compared_as_binder",
    "compared_as_reference",
    "compared_as_value",
    "compared_with",
    "excluded_from_equivalence",
]

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

from fhy_core.error import register_error

from .alpha_equivalence import AlphaEquivalence, AlphaEquivalenceMixin, AlphaRenaming
from .equality import PartialEqual
from .structural_equivalence import StructuralEquivalence, StructuralEquivalenceMixin

EQUIVALENCE_METADATA_KEY: Final[str] = "fhy_equivalence"
"""``dataclasses.field(metadata=...)`` key holding an equivalence role override."""

_VALUE_SCALAR_TYPES: Final[tuple[type, ...]] = (bool, int, float, str)


@register_error
class EquivalenceDerivationError(Exception):
    """Raised when equivalence cannot be derived for a class.

    Raised on the first comparison of a class that is not a dataclass, or
    that has a field whose value is none of the recognized categories and
    carries no role metadata. The message names the class and field and
    lists the available fixes.
    """


@runtime_checkable
class FieldComparator(Protocol):
    """Compare one field's value structurally and under a renaming.

    A field is compared either by the default capability dispatch or by a
    ``FieldComparator`` supplied through ``field`` metadata via
    :func:`compared_with`. Binders are not comparators: they affect
    sibling fields and are handled by the mixin's field-walk.

    Implementations must agree with the trait contract:
    ``is_alpha_equivalent_under`` with an empty renaming and no binders in
    scope must equal ``is_structurally_equivalent`` for the same pair.
    """

    def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
        """Return whether ``left`` and ``right`` are structurally equivalent."""

    def is_alpha_equivalent_under(
        self, left: Any, right: Any, renaming: AlphaRenaming
    ) -> bool:
        """Return whether left and right are alpha-equivalent under the renaming."""


class _ComparatorInferenceError(Exception):
    """Internal: a field value fell into no recognized comparison category."""

    value_type: type

    def __init__(self, value_type: type) -> None:
        self.value_type = value_type
        super().__init__(value_type.__name__)


# ===========================================================================
# Role markers (stored under ``EQUIVALENCE_METADATA_KEY`` in field metadata)
# ===========================================================================


@dataclass(frozen=True)
class _ValueRole:
    key: Callable[[Any], Any] | None


@dataclass(frozen=True)
class _ReferenceRole:
    pass


@dataclass(frozen=True)
class _BinderRole:
    scopes_over: tuple[str, ...]


@dataclass(frozen=True)
class _ExcludeRole:
    pass


@dataclass(frozen=True)
class _ExplicitRole:
    comparator: FieldComparator


def compared_as_value(*, key: Callable[[Any], Any] | None = None) -> Mapping[str, Any]:
    """Build field metadata forcing equality comparison.

    Use for a field that should compare by ``==`` in both modes -- in
    particular a bare ``Identifier`` that is a plain value rather than a
    reference or binder, where inference would otherwise be ambiguous.

    Args:
        key: Optional normalizer applied to both values before comparison,
            so the field compares by ``key(left) == key(right)`` (e.g.
            ``_classify_literal_value`` for weak literal forms).

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _ValueRole(key)}


def compared_as_reference() -> Mapping[str, Any]:
    """Build field metadata marking an ``Identifier`` field as a reference.

    A reference identifier compares by ``==`` structurally and by
    ``renaming.are_identifiers_alpha_equivalent`` in alpha mode, so that
    a binder installed by an enclosing node renames it consistently.

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _ReferenceRole()}


def compared_as_binder(*, scopes_over: tuple[str, ...] = ()) -> Mapping[str, Any]:
    """Build field metadata marking an ``Identifier`` field as a binder.

    The field holds the bound names introduced by this node: a single
    ``Identifier`` or a sequence of them. Structurally the names compare
    nominally (``==``). In alpha mode the engine arity-checks the two
    sides, ``extend``s the renaming with their pairwise binding, and
    compares each field named in ``scopes_over`` under the extended
    renaming. The bound names themselves are not compared by identity in
    alpha mode.

    Args:
        scopes_over: Names of sibling fields whose comparison happens
            under the renaming extended by this binder. A field may be
            scoped by more than one binder (nested binders compose).

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    Raises:
        ValueError: At comparison time, if the two binders' pairwise
            binding is not injective (propagated from
            ``AlphaRenaming.extend``).

    """
    return {EQUIVALENCE_METADATA_KEY: _BinderRole(scopes_over)}


def excluded_from_equivalence() -> Mapping[str, Any]:
    """Build field metadata excluding a field from equivalence.

    Equivalent in effect to ``dataclasses.field(compare=False)``, which is
    also honored. Use the explicit form for a field that must remain
    ``compare=True`` for ``__eq__`` yet take no part in structural or
    alpha equivalence (e.g. human-readable metadata).

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _ExcludeRole()}


def compared_with(comparator: FieldComparator) -> Mapping[str, Any]:
    """Build field metadata supplying an explicit comparator.

    Use for a field whose type inference does not cover, in place of a
    hand-written equivalence method.

    Args:
        comparator: The strategy used to compare this field in both modes.

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _ExplicitRole(comparator)}


# ===========================================================================
# Default capability dispatch and built-in comparators
# ===========================================================================


def _is_value_comparable(value: object) -> bool:
    return isinstance(value, _VALUE_SCALAR_TYPES) or isinstance(
        value, (Enum, PartialEqual)
    )


def _auto_structural(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, StructuralEquivalence):
        return left.is_structurally_equivalent(right)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _auto_structural(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    if _is_value_comparable(left):
        return bool(left == right)
    raise _ComparatorInferenceError(type(left))


def _auto_alpha(left: Any, right: Any, renaming: AlphaRenaming) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, AlphaEquivalence):
        return left.is_alpha_equivalent_under(right, renaming)
    if isinstance(left, StructuralEquivalence):
        return left.is_structurally_equivalent(right)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _auto_alpha(item_left, item_right, renaming)
            for item_left, item_right in zip(left, right)
        )
    if _is_value_comparable(left):
        return bool(left == right)
    raise _ComparatorInferenceError(type(left))


class _ValueComparator:
    """Compare a field by ``==``, optionally through a normalizer."""

    def __init__(self, key: Callable[[Any], Any] | None) -> None:
        self._key = key

    def _normalize(self, value: Any) -> Any:
        return value if self._key is None else self._key(value)

    def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
        return bool(self._normalize(left) == self._normalize(right))

    def is_alpha_equivalent_under(
        self, left: Any, right: Any, renaming: AlphaRenaming
    ) -> bool:
        del renaming
        return bool(self._normalize(left) == self._normalize(right))


class _ReferenceComparator:
    """Compare a reference ``Identifier`` nominally or through the renaming."""

    def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
        return bool(left == right)

    def is_alpha_equivalent_under(
        self, left: Any, right: Any, renaming: AlphaRenaming
    ) -> bool:
        return renaming.are_identifiers_alpha_equivalent(left, right)


# ===========================================================================
# Plan (one per deriving class)
# ===========================================================================


@dataclass(frozen=True)
class _Plan:
    """Cached comparison plan for one deriving class.

    ``comparable_fields`` pairs each participating field name with its
    comparator, or ``None`` for the default capability dispatch.
    ``binders`` pairs each binder field name with the sibling field names
    it scopes over.
    """

    comparable_fields: tuple[tuple[str, FieldComparator | None], ...]
    binders: tuple[tuple[str, tuple[str, ...]], ...]


_PLAN_CACHE: dict[type, _Plan] = {}


def _build_plan(cls: type) -> _Plan:
    if not is_dataclass(cls):
        raise EquivalenceDerivationError(
            f'Cannot derive equivalence for "{cls.__name__}": it is not a '
            f"dataclass. Implement is_structurally_equivalent / "
            f"is_alpha_equivalent_under by hand."
        )
    comparable_fields: list[tuple[str, FieldComparator | None]] = []
    binders: list[tuple[str, tuple[str, ...]]] = []
    for field_definition in fields(cls):
        role = field_definition.metadata.get(EQUIVALENCE_METADATA_KEY)
        if isinstance(role, _ExcludeRole) or not field_definition.compare:
            continue
        if isinstance(role, _BinderRole):
            binders.append((field_definition.name, role.scopes_over))
        elif isinstance(role, _ValueRole):
            comparable_fields.append(
                (field_definition.name, _ValueComparator(role.key))
            )
        elif isinstance(role, _ReferenceRole):
            comparable_fields.append((field_definition.name, _ReferenceComparator()))
        elif isinstance(role, _ExplicitRole):
            comparable_fields.append((field_definition.name, role.comparator))
        else:
            comparable_fields.append((field_definition.name, None))
    return _Plan(tuple(comparable_fields), tuple(binders))


def _get_plan(cls: type) -> _Plan:
    plan = _PLAN_CACHE.get(cls)
    if plan is None:
        plan = _build_plan(cls)
        _PLAN_CACHE[cls] = plan
    return plan


def _as_identifier_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _inference_error(
    cls: type, field_name: str, exc: _ComparatorInferenceError
) -> EquivalenceDerivationError:
    return EquivalenceDerivationError(
        f'Cannot derive equivalence for field "{cls.__name__}.{field_name}" '
        f"of type {exc.value_type.__name__}. Tag it with compared_as_value(), "
        f"supply compared_with(...), exclude it with "
        f"excluded_from_equivalence(), or hand-write the comparison."
    )


# ===========================================================================
# Mixin
# ===========================================================================


class DerivedEquivalenceMixin(StructuralEquivalenceMixin, AlphaEquivalenceMixin):
    """Derive structural and alpha equivalence from a dataclass schema.

    Subclass this on a ``@dataclass``; both ``is_structurally_equivalent``
    and ``is_alpha_equivalent_under`` are derived from the fields.
    Inheritance is automatic: ``dataclasses.fields`` includes base fields
    in declaration order, so a subclass that adds fields derives the
    union.

    Override either method by hand to opt that one out of derivation; the
    other stays derived. ``is_alpha_equivalent`` is inherited from
    :class:`AlphaEquivalenceMixin`.

    Declare ``Identifier`` roles with :func:`compared_as_value`,
    :func:`compared_as_reference`, or :func:`compared_as_binder`; exclude a
    field with :func:`excluded_from_equivalence` or
    ``field(compare=False)``; supply a custom strategy with
    :func:`compared_with`.

    Raises:
        EquivalenceDerivationError: On first comparison, if the class is
            not a dataclass or a field's value is un-inferable and carries
            no role metadata.
    """

    def is_structurally_equivalent(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are structurally equivalent.

        Derived from the field schema: ``other`` must be of the same
        concrete type, and every participating field must compare equal
        under its comparator (equality for values, recursion for
        sub-objects, nominal equality for identifiers).

        Args:
            other: Candidate object.

        Returns:
            ``True`` if structurally equivalent, ``False`` otherwise
            (including when ``other`` is of an incompatible type).

        """
        if type(self) is not type(other):
            return False
        plan = _get_plan(type(self))
        for name, scopes in plan.binders:
            del scopes
            if _as_identifier_tuple(getattr(self, name)) != _as_identifier_tuple(
                getattr(other, name)
            ):
                return False
        for name, comparator in plan.comparable_fields:
            left = getattr(self, name)
            right = getattr(other, name)
            # The recurse case is inlined (rather than delegated to
            # ``_auto_structural``) so a deeply nested tree costs one stack
            # frame per level on the recursion spine, not two.
            if comparator is not None:
                try:
                    matched = comparator.is_structurally_equivalent(left, right)
                except _ComparatorInferenceError as exc:
                    raise _inference_error(type(self), name, exc) from exc
            elif left is None or right is None:
                matched = left is right
            elif isinstance(left, StructuralEquivalence):
                matched = left.is_structurally_equivalent(right)
            else:
                try:
                    matched = _auto_structural(left, right)
                except _ComparatorInferenceError as exc:
                    raise _inference_error(type(self), name, exc) from exc
            if not matched:
                return False
        return True

    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        Derived from the same field schema as
        :meth:`is_structurally_equivalent`, differing only in the
        per-field operator: ``AlphaEquivalence`` sub-objects recurse under
        ``renaming``, reference identifiers consult it, and binder fields
        ``extend`` it for the fields they scope over.

        Args:
            other: Candidate object.
            renaming: Binder-frame stack plus free-identifier bijection
                threaded from enclosing comparisons.

        Returns:
            ``True`` if alpha-equivalent under ``renaming``, ``False``
            otherwise (including when ``other`` is of an incompatible
            type).

        """
        if type(self) is not type(other):
            return False
        plan = _get_plan(type(self))
        scoped_renamings: dict[str, AlphaRenaming] = {}
        for name, scopes in plan.binders:
            self_ids = _as_identifier_tuple(getattr(self, name))
            other_ids = _as_identifier_tuple(getattr(other, name))
            if len(self_ids) != len(other_ids):
                return False
            binding = dict(zip(self_ids, other_ids))
            for scoped in scopes:
                base = scoped_renamings.get(scoped, renaming)
                scoped_renamings[scoped] = base.extend(binding)
        for name, comparator in plan.comparable_fields:
            left = getattr(self, name)
            right = getattr(other, name)
            field_renaming = scoped_renamings.get(name, renaming)
            try:
                if comparator is None:
                    matched = _auto_alpha(left, right, field_renaming)
                else:
                    matched = comparator.is_alpha_equivalent_under(
                        left, right, field_renaming
                    )
            except _ComparatorInferenceError as exc:
                raise _inference_error(type(self), name, exc) from exc
            if not matched:
                return False
        return True
