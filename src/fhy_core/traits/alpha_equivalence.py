"""`AlphaEquivalence` trait, mixin, ``AlphaRenaming``, and mapping helper.

This module exposes the alpha-equivalence contract for compiler-IR
objects: the binary relation that holds between two terms when they
agree up to a consistent renaming of their bound identifiers. The trait
is parallel to ``StructuralEquivalence``; structurally-equivalent pairs
are always alpha-equivalent, but alpha-equivalent pairs need not be
structurally equivalent.

The renaming state is carried by ``AlphaRenaming``: a stack of binder
frames plus a free-identifier bijection. Binder nodes ``extend`` the
renaming when they recurse into their bodies; identifier-reference
nodes consult the renaming to compare their identifiers.

For IR nodes whose attributes include ``Mapping[Identifier, V]``
collections (substitutions, parallel let-groups, identifier-keyed
catalogs), the module also exports
``is_identifier_mapping_alpha_equivalent_under``, a free function that
performs the bijection-finding and pairwise value comparison correctly
so implementors do not have to open-code it.
"""

__all__ = [
    "AlphaEquivalence",
    "AlphaEquivalenceMixin",
    "AlphaRenaming",
    "is_identifier_mapping_alpha_equivalent_under",
]

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, final, runtime_checkable

from frozendict import frozendict

from fhy_core.identifier import Identifier


def _check_injective(
    mapping: Mapping[Identifier, Identifier], parameter_name: str
) -> None:
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            f"`{parameter_name}` must be injective; got duplicate other-side "
            f"values in mapping."
        )


@runtime_checkable
class AlphaEquivalence(Protocol):
    """Protocol for objects that support alpha-equivalence comparison.

    Alpha-equivalence treats two binder-containing terms as equal when
    they agree up to a consistent renaming of bound identifiers. For
    objects without binders, the relation reduces to structural
    equivalence threaded through children; for objects that reference
    identifiers, the comparison consults an ``AlphaRenaming``.

    The two methods are the caller surface and the composition surface
    respectively. Callers ask ``is_alpha_equivalent``; recursive
    implementations call ``is_alpha_equivalent_under`` on their children
    so that binder frames installed by enclosing nodes remain in scope.

    Implementations must be reflexive, symmetric, and transitive on the
    domain of well-formed IR objects.
    """

    def is_alpha_equivalent(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are alpha-equivalent.

        Equivalent to ``self.is_alpha_equivalent_under(other,
        AlphaRenaming.empty())``: no binders are in scope and free
        identifiers compare by ``Identifier`` equality.

        Args:
            other: Candidate term.

        Returns:
            ``True`` if the two terms agree up to a consistent renaming
            of bound identifiers, ``False`` otherwise. Returns ``False``
            (not raises) when ``other`` is not of a compatible type.

        """

    def is_alpha_equivalent_under(
        self, other: object, renaming: "AlphaRenaming"
    ) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        Binder-bearing implementations call ``renaming.extend(...)``
        with the pairwise binding of their parameters and recurse on
        bodies. Identifier-reference implementations call
        ``renaming.are_identifiers_alpha_equivalent`` on their stored
        ``Identifier``s. All other implementations thread ``renaming``
        through unchanged into recursive calls on their children.

        Args:
            other: Candidate term.
            renaming: Binder-frame stack plus free-identifier bijection
                accumulated by enclosing comparison calls.

        Returns:
            ``True`` if ``self`` and ``other`` are alpha-equivalent
            under ``renaming``, ``False`` otherwise. Returns ``False``
            (not raises) when ``other`` is not of a compatible type.

        """


class AlphaEquivalenceMixin(ABC):
    """Mixin for objects that support alpha-equivalence comparison.

    Subclasses implement ``is_alpha_equivalent_under`` only.
    ``is_alpha_equivalent`` is provided as a concrete default that
    constructs the empty renaming and delegates.
    """

    def is_alpha_equivalent(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are alpha-equivalent.

        Delegates to ``is_alpha_equivalent_under`` with
        ``AlphaRenaming.empty()``. See :class:`AlphaEquivalence` for the
        contract.

        """
        return self.is_alpha_equivalent_under(other, AlphaRenaming.empty())

    @abstractmethod
    def is_alpha_equivalent_under(
        self, other: object, renaming: "AlphaRenaming"
    ) -> bool:
        """Return whether self and other are alpha-equivalent under the renaming.

        See :class:`AlphaEquivalence` for the contract.

        """


@final
@dataclass(frozen=True)
class AlphaRenaming:
    """Stack of binder-frame bijections plus a free-identifier bijection.

    An ``AlphaRenaming`` represents the renaming state accumulated by a
    recursive alpha-equivalence comparison: the stack of binder frames
    pushed by enclosing binder nodes, and an ambient bijection on free
    identifiers supplied by the original caller.

    Frame stack semantics:
        Each binder node pushes a single frame whose pairwise mapping
        is the bijection between its bound identifiers and the
        corresponding bound identifiers of the other term. Lookups walk
        the stack innermost-first, so an inner binder's mapping shadows
        an outer one. This is what makes a doubly-nested binder over
        ``x`` with body ``x`` alpha-equivalent to a doubly-nested
        binder over ``(a, b)`` with body ``b`` (and not to the same
        outer shape with body ``a``).

        Bijection is enforced per frame: two self-side keys in the same
        frame must not map to the same other-side value. Across frames,
        other-side values may repeat, which is the standard
        nominal-renaming semantics.

    Free-renaming semantics:
        Identifiers that are not in any frame are looked up in the
        free-identifier bijection (``with_free_renaming``). On a miss
        there, the comparison falls back to ``Identifier`` equality.
        The free bijection is not shadow-able by binder frames; the
        lookup order is frames-innermost-first, then free, then
        identity.

    Immutability:
        ``AlphaRenaming`` is a value object: ``extend`` returns a new
        instance with one additional frame and leaves the receiver
        unchanged. Equal-by-structure, hashable.

    """

    _frames: tuple[frozendict[Identifier, Identifier], ...]
    _free_renaming: frozendict[Identifier, Identifier]

    @classmethod
    def empty(cls) -> "AlphaRenaming":
        """Return the renaming with no binder frames and no free renaming.

        This is the renaming used by the default
        ``AlphaEquivalenceMixin.is_alpha_equivalent`` implementation.

        Returns:
            A fresh empty ``AlphaRenaming`` instance.

        """
        return cls(_frames=(), _free_renaming=frozendict())

    @classmethod
    def with_free_renaming(
        cls, free_renaming: Mapping[Identifier, Identifier]
    ) -> "AlphaRenaming":
        """Return a renaming seeded with a free-identifier bijection.

        Used by callers comparing two terms drawn from different
        identifier scopes who want to declare which free identifiers
        correspond. The renaming has no binder frames.

        Args:
            free_renaming: Bijection between self-side and other-side
                free identifiers. Must be injective on both sides;
                non-injective input raises ``ValueError``.

        Returns:
            A new ``AlphaRenaming`` with the given free bijection and
            no binder frames.

        Raises:
            ValueError: If ``free_renaming`` is not injective.

        """
        _check_injective(free_renaming, "free_renaming")
        return cls(_frames=(), _free_renaming=frozendict(free_renaming))

    def extend(self, bindings: Mapping[Identifier, Identifier]) -> "AlphaRenaming":
        """Push a new innermost binder frame and return the extended renaming.

        Binder implementations call this with the pairwise binding of
        their parameters before recursing into their bodies. The frame
        must be injective on both sides; across frames, other-side
        values may repeat (the shadowing case).

        Args:
            bindings: Bijection between self-side and other-side bound
                identifiers introduced by the binder. Must be injective
                on both sides; non-injective input raises ``ValueError``.

        Returns:
            A new ``AlphaRenaming`` with one additional innermost frame.
            The receiver is unchanged.

        Raises:
            ValueError: If ``bindings`` is not injective.

        """
        _check_injective(bindings, "bindings")
        return AlphaRenaming(
            _frames=self._frames + (frozendict(bindings),),
            _free_renaming=self._free_renaming,
        )

    def resolve(self, self_identifier: Identifier) -> Identifier:
        """Return the other-side identifier corresponding to ``self_identifier``.

        Resolution order:
            1. Innermost binder frame containing ``self_identifier`` as
               a key. If found, the resolved value is the frame's
               mapped value.
            2. Free-identifier bijection. If ``self_identifier`` is a
               key there, the resolved value is the bijection's mapped
               value.
            3. Identity fallback. ``self_identifier`` itself.

        ``resolve`` is total: it never raises and never returns
        ``None``. Identifiers that are not in any frame or in the free
        renaming are returned unchanged, which is the correct behavior
        for free identifiers under the trait's default semantics.

        Args:
            self_identifier: Identifier from the self side of the
                comparison.

        Returns:
            The corresponding identifier on the other side.

        """
        for frame in reversed(self._frames):
            if self_identifier in frame:
                return frame[self_identifier]
        if self_identifier in self._free_renaming:
            return self._free_renaming[self_identifier]
        else:
            return self_identifier

    def are_identifiers_alpha_equivalent(
        self, self_identifier: Identifier, other_identifier: Identifier
    ) -> bool:
        """Return whether ``self_identifier`` corresponds to ``other_identifier``.

        Performs a bidirectional check across the binder-frame stack
        and the free renaming. Agrees with ``self.resolve(self_id) ==
        other_id`` in the common case; diverges (returns ``False``) in
        the *capture* case, where ``self_id`` is free on the self side
        but ``other_id`` is bound by a frame on the other side, or
        symmetrically where ``other_id`` is the renaming target of some
        other free identifier.

        Args:
            self_identifier: Identifier from the self side of the
                comparison.
            other_identifier: Identifier from the other side.

        Returns:
            ``True`` if the two identifiers correspond under this
            renaming, ``False`` otherwise.

        """
        other_captured = False
        for frame in reversed(self._frames):
            if self_identifier in frame:
                return frame[self_identifier] == other_identifier
            if other_identifier in frame.values():
                other_captured = True
        if other_captured:
            return False
        elif self_identifier in self._free_renaming:
            return self._free_renaming[self_identifier] == other_identifier
        elif other_identifier in self._free_renaming.values():
            return False
        else:
            return self_identifier == other_identifier


_V_AlphaEquivalent = TypeVar("_V_AlphaEquivalent", bound=AlphaEquivalence)


def is_identifier_mapping_alpha_equivalent_under(
    self_mapping: Mapping[Identifier, _V_AlphaEquivalent],
    other_mapping: Mapping[Identifier, _V_AlphaEquivalent],
    renaming: AlphaRenaming,
) -> bool:
    """Return whether two Identifier-keyed mappings are alpha-equivalent.

    Helper for implementations of ``is_alpha_equivalent_under`` whose
    IR node stores a ``Mapping[Identifier, V]`` attribute, where ``V``
    is itself an ``AlphaEquivalence``. The helper treats the map's keys
    as references resolved through the current ``renaming``, not as
    binders that introduce new scope. Callers whose keys are binders
    should first call ``renaming.extend(...)`` with their chosen
    pairwise binding and then call this helper on the values, since the
    pairing is part of the enclosing IR node's contract and cannot be
    inferred from the map alone.

    Algorithm (O(n) in the number of entries):

    1. If the two mappings have different cardinality, return ``False``.
    2. For each ``self_key`` in ``self_mapping``, compute
       ``resolved = renaming.resolve(self_key)``. If ``resolved`` was
       already claimed by an earlier self-key (collision), return
       ``False``: two distinct self-side keys cannot honor a single
       other-side entry under a bijective renaming.
    3. If the set of resolved keys differs from ``other_mapping``'s key
       set, return ``False``.
    4. For each matched pair, verify
       ``renaming.are_identifiers_alpha_equivalent(self_key,
       resolved_key)`` (catches capture in keys) and then compare the
       values via
       ``self_value.is_alpha_equivalent_under(other_value, renaming)``.
       Short-circuit on the first mismatch.

    Args:
        self_mapping: Identifier-keyed mapping on the self side.
        other_mapping: Identifier-keyed mapping on the other side.
        renaming: Renaming threaded through the enclosing comparison.

    Returns:
        ``True`` if the two mappings are alpha-equivalent under
        ``renaming``, ``False`` otherwise. Returns ``False`` (not
        raises) on cardinality or shape mismatch.

    """
    if len(self_mapping) != len(other_mapping):
        return False
    resolved_self: dict[Identifier, tuple[Identifier, _V_AlphaEquivalent]] = {}
    for self_key, self_value in self_mapping.items():
        resolved_key = renaming.resolve(self_key)
        if resolved_key in resolved_self:
            return False
        resolved_self[resolved_key] = (self_key, self_value)
    if resolved_self.keys() != other_mapping.keys():
        return False
    for resolved_key, (self_key, self_value) in resolved_self.items():
        if not renaming.are_identifiers_alpha_equivalent(self_key, resolved_key):
            return False
        if not self_value.is_alpha_equivalent_under(
            other_mapping[resolved_key], renaming
        ):
            return False
    return True
