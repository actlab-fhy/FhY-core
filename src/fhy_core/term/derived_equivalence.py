"""Schema-derived structural and alpha equivalence.

This module exposes :class:`DerivedEquivalenceMixin`, which derives both
``is_structurally_equivalent`` and ``is_alpha_equivalent_under`` for a
``@dataclass`` from its fields. It is the equivalence analog of the
schema-derived ``Serializable`` trait: both methods walk the same field
schema, differing in their per-field operator and binder handling -- the
structural walk compares bound names nominally while the alpha walk
extends the renaming for the fields they scope over.

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

The engine runs in the Rust binding: it builds a class's plan from
``dataclasses.fields`` on the first comparison and keeps it in
``_PLAN_CACHE``, and walks nested derived values on its own stack, so a tree
of any depth compares. It compares identifiers, expressions, scalars and
nested derived values without calling their Python methods, and calls
everything else -- comparators, ``key`` normalizers, hand-written methods
and ``==`` -- as before. A binder field that repeats an identifier, on
either side, pairs with none, so its node is alpha-equivalent to no node,
itself included.
"""

from fhy_core.utils.override import override

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
from typing import Any, Final, Protocol, runtime_checkable

from fhy_core import _rs
from fhy_core.error import register_error
from fhy_core.traits import StructuralEquivalence

from .alpha_equivalence import AlphaEquivalenceMixin, AlphaRenaming

EQUIVALENCE_METADATA_KEY: Final[str] = "fhy_equivalence"
"""``dataclasses.field(metadata=...)`` key holding an equivalence role override."""

# Comparison plans are built by the binding once per class on first comparison
# and cached here forever, keyed by the class. This assumes a class's dataclass
# field schema is frozen after first use; redefining a class (or mutating its
# fields) under the same identity after it has been compared leaves the stale
# plan in place.
_PLAN_CACHE: dict[type, object] = {}


@register_error
class EquivalenceDerivationError(Exception):
    """Raised when equivalence cannot be derived for a class.

    Raised on the first comparison of a class that is not a dataclass,
    that has a field whose value is none of the recognized categories and
    carries no role metadata, or whose binder field declares a
    ``scopes_over`` name that does not match any field on the class. The
    message names the class and field and lists the available fixes.
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


# ===========================================================================
# Role markers (stored under ``EQUIVALENCE_METADATA_KEY`` in field metadata)
# ===========================================================================


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
    return {EQUIVALENCE_METADATA_KEY: _rs.EquivalenceRole.value(key)}


def compared_as_reference() -> Mapping[str, Any]:
    """Build field metadata marking an ``Identifier`` field as a reference.

    A reference identifier compares by ``==`` structurally and by
    ``renaming.are_identifiers_alpha_equivalent`` in alpha mode, so that
    a binder installed by an enclosing node renames it consistently.

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _rs.EquivalenceRole.reference()}


def compared_as_binder(*, scopes_over: tuple[str, ...] = ()) -> Mapping[str, Any]:
    """Build field metadata marking an ``Identifier`` field as a binder.

    The field holds the bound names introduced by this node: a single
    ``Identifier`` or a sequence of them. Structurally the names compare
    nominally (``==``). In alpha mode the engine arity-checks the two
    sides, ``extend``s the renaming with their pairwise binding, and
    compares each field named in ``scopes_over`` under the extended
    renaming. The bound names themselves are not compared by identity in
    alpha mode.

    If either side's names repeat an identifier, the binding pairs with
    none and ``is_alpha_equivalent_under`` returns ``False``, even
    against the same node.

    Args:
        scopes_over: Names of sibling fields whose comparison happens
            under the renaming extended by this binder. A field may be
            scoped by more than one binder (nested binders compose).
            Each name is validated against the class's fields at first
            comparison; an unknown name raises
            :class:`EquivalenceDerivationError`.

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _rs.EquivalenceRole.binder(scopes_over)}


def excluded_from_equivalence() -> Mapping[str, Any]:
    """Build field metadata excluding a field from equivalence.

    Equivalent in effect to ``dataclasses.field(compare=False)``, which is
    also honored. Use the explicit form for a field that must remain
    ``compare=True`` for ``__eq__`` yet take no part in structural or
    alpha equivalence (e.g. human-readable metadata).

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _rs.EquivalenceRole.excluded()}


def compared_with(comparator: FieldComparator) -> Mapping[str, Any]:
    """Build field metadata supplying an explicit comparator.

    Use for a field whose type inference does not cover, in place of a
    hand-written equivalence method.

    Args:
        comparator: The strategy used to compare this field in both modes.

    Returns:
        A one-entry mapping for ``dataclasses.field(metadata=...)``.

    """
    return {EQUIVALENCE_METADATA_KEY: _rs.EquivalenceRole.explicit(comparator)}


# ===========================================================================
# Mixin
# ===========================================================================


class DerivedEquivalenceMixin(StructuralEquivalence, AlphaEquivalenceMixin):
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

    @override
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
        return _rs.derived_is_structurally_equivalent(self, other)

    @override
    def is_alpha_equivalent_under(self, other: object, renaming: AlphaRenaming) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        Derived from the same field schema as
        :meth:`is_structurally_equivalent`, differing only in the
        per-field operator: ``AlphaEquivalence`` sub-objects recurse under
        ``renaming``, reference identifiers consult it, and binder fields
        extend it, pairing their identifiers by position, for the fields
        they scope over.

        Args:
            other: Candidate object.
            renaming: Binder-frame stack plus free-identifier bijection
                threaded from enclosing comparisons.

        Returns:
            ``True`` if alpha-equivalent under ``renaming``, ``False``
            otherwise (including when ``other`` is of an incompatible
            type).

        """
        return _rs.derived_is_alpha_equivalent_under(self, other, renaming)
