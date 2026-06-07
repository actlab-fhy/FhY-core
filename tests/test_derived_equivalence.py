"""Tests for schema-derived structural and alpha equivalence.

Covers the ``DerivedEquivalenceMixin`` field-walk: comparator inference from
resolved type hints (value, recurse, sequence, optional), the
``field(metadata=...)`` role channel (``compared_as_value`` with a normalizer,
``compared_as_reference``, ``compared_as_binder``, ``excluded_from_equivalence``,
``compared_with``), exclusion via ``field(compare=False)``, hand-written-override
precedence, inheritance, the ``EquivalenceDerivationError`` for un-inferable
fields, and the relational laws (reflexive / symmetric / transitive,
structural-implies-alpha, alpha-not-implies-structural for a renamed binder).

The synthetic dataclasses are deliberately minimal. The real migration target
(``Expression`` and friends) keeps its own structural- and alpha-equivalence
suites; these tests pin the engine itself.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.traits import AlphaEquivalence, AlphaRenaming, StructuralEquivalence
from fhy_core.traits.derived_equivalence import (
    EQUIVALENCE_METADATA_KEY,
    DerivedEquivalenceMixin,
    EquivalenceDerivationError,
    FieldComparator,
    compared_as_binder,
    compared_as_reference,
    compared_as_value,
    compared_with,
    excluded_from_equivalence,
)

from .conftest import mock_identifier

# ===========================================================================
# Module-level synthetic types (pure inference, no metadata helpers)
# ===========================================================================


@dataclass(frozen=True, eq=False)
class _Leaf(DerivedEquivalenceMixin):
    """A node whose fields are plain values compared by ``==``."""

    tag: str
    count: int


@dataclass(frozen=True, eq=False)
class _OtherLeaf(DerivedEquivalenceMixin):
    """A distinct type with the same field shape as ``_Leaf``."""

    tag: str
    count: int


@dataclass(frozen=True, eq=False)
class _Box(DerivedEquivalenceMixin):
    """A node with a recursive sub-object field."""

    label: str
    child: _Leaf


@dataclass(frozen=True, eq=False)
class _Seq(DerivedEquivalenceMixin):
    """A node with a homogeneous tuple of recursive sub-objects."""

    items: tuple[_Leaf, ...]


@dataclass(frozen=True, eq=False)
class _Maybe(DerivedEquivalenceMixin):
    """A node with an optional recursive sub-object field."""

    maybe: _Leaf | None = None


@dataclass(frozen=True, eq=False)
class _CompareFalse(DerivedEquivalenceMixin):
    """A node excluding a field via ``dataclasses.field(compare=False)``."""

    key: str
    note: str = field(compare=False)


# A small binder-free recursive IR used for the relational laws.
@dataclass(frozen=True, eq=False)
class _Term(DerivedEquivalenceMixin):
    """Abstract base for the synthetic term IR."""


@dataclass(frozen=True, eq=False)
class _Const(_Term):
    """A literal term."""

    value: int


@dataclass(frozen=True, eq=False)
class _Add(_Term):
    """A binary term recursing into two sub-terms."""

    left: _Term
    right: _Term


class _StructuralOnly:
    """Implements ``StructuralEquivalence`` but deliberately not ``AlphaEquivalence``.

    Exercises the ``_auto_alpha`` branch that falls back to structural recursion
    for a sub-object that has no alpha-equivalence capability.
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def is_structurally_equivalent(self, other: Any) -> bool:
        return isinstance(other, _StructuralOnly) and self.value == other.value


@dataclass(frozen=True, eq=False)
class _HoldsStructuralOnly(DerivedEquivalenceMixin):
    """A node whose sub-object only supports structural equivalence."""

    payload: _StructuralOnly


@dataclass(frozen=True, eq=False)
class _SeqOptional(DerivedEquivalenceMixin):
    """A node with a tuple that may contain ``None`` elements."""

    items: tuple[_Leaf | None, ...]


@dataclass(frozen=True, eq=False)
class _SeqOfComplex(DerivedEquivalenceMixin):
    """A node with a tuple of un-inferable ``complex`` elements."""

    items: tuple[complex, ...]


# ===========================================================================
# Protocol / metadata contract
# ===========================================================================


def test_mixin_satisfies_both_equivalence_protocols() -> None:
    """Test a derived class satisfies both runtime equivalence protocols."""
    leaf = _Leaf("a", 1)

    assert isinstance(leaf, StructuralEquivalence)
    assert isinstance(leaf, AlphaEquivalence)


def test_role_helpers_return_metadata_under_the_documented_key() -> None:
    """Test each role helper yields a mapping keyed by the metadata key."""
    for metadata in (
        compared_as_value(),
        compared_as_reference(),
        compared_as_binder(scopes_over=("body",)),
        excluded_from_equivalence(),
    ):
        assert EQUIVALENCE_METADATA_KEY in metadata


def test_field_comparator_is_a_runtime_checkable_protocol() -> None:
    """Test an object with both comparison methods satisfies ``FieldComparator``."""

    class _Always:
        def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
            return True

        def is_alpha_equivalent_under(
            self, left: Any, right: Any, renaming: AlphaRenaming
        ) -> bool:
            return True

    assert isinstance(_Always(), FieldComparator)


# ===========================================================================
# Value inference and the exact-type guard
# ===========================================================================


def test_value_fields_compare_equal_when_all_match() -> None:
    """Test a value-only node is structurally equivalent to an equal-valued node."""
    assert _Leaf("a", 1).is_structurally_equivalent(_Leaf("a", 1))


def test_value_fields_distinguish_a_differing_field() -> None:
    """Test a value-only node differs when any value field differs."""
    assert not _Leaf("a", 1).is_structurally_equivalent(_Leaf("a", 2))


def test_value_only_node_is_alpha_equivalent_when_values_match() -> None:
    """Test value-only nodes derive alpha equivalence identical to structural."""
    assert _Leaf("a", 1).is_alpha_equivalent(_Leaf("a", 1))
    assert not _Leaf("a", 1).is_alpha_equivalent(_Leaf("a", 2))


def test_different_concrete_types_are_not_structurally_equivalent() -> None:
    """Test the type guard rejects a same-shaped but different concrete type."""
    assert not _Leaf("a", 1).is_structurally_equivalent(_OtherLeaf("a", 1))


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param("a", id="str"),
        pytest.param(object(), id="object"),
    ],
)
def test_comparison_returns_false_for_foreign_objects(other: object) -> None:
    """Test comparison returns ``False`` (not raises) for an unrelated object."""
    assert not _Leaf("a", 1).is_structurally_equivalent(other)
    assert not _Leaf("a", 1).is_alpha_equivalent(other)


# ===========================================================================
# Recursion into sub-objects
# ===========================================================================


def test_sub_object_field_recurses_structurally() -> None:
    """Test a sub-object field is compared by structural recursion."""
    assert _Box("b", _Leaf("a", 1)).is_structurally_equivalent(_Box("b", _Leaf("a", 1)))
    assert not _Box("b", _Leaf("a", 1)).is_structurally_equivalent(
        _Box("b", _Leaf("a", 2))
    )


def test_sub_object_field_recurses_in_alpha_mode() -> None:
    """Test a sub-object field recurses through ``is_alpha_equivalent_under``."""
    assert _Box("b", _Leaf("a", 1)).is_alpha_equivalent(_Box("b", _Leaf("a", 1)))
    assert not _Box("b", _Leaf("a", 1)).is_alpha_equivalent(_Box("b", _Leaf("a", 9)))


def test_sub_object_node_distinguishes_its_own_value_field() -> None:
    """Test a recursive node still compares its own value fields."""
    assert not _Box("b", _Leaf("a", 1)).is_structurally_equivalent(
        _Box("c", _Leaf("a", 1))
    )


# ===========================================================================
# Sequence inference
# ===========================================================================


def test_sequence_field_matches_elementwise() -> None:
    """Test a tuple-of-sub-objects field matches element by element."""
    left = _Seq((_Leaf("a", 1), _Leaf("b", 2)))
    right = _Seq((_Leaf("a", 1), _Leaf("b", 2)))

    assert left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)


def test_sequence_field_distinguishes_length() -> None:
    """Test sequences of different length are not equivalent."""
    left = _Seq((_Leaf("a", 1),))
    right = _Seq((_Leaf("a", 1), _Leaf("b", 2)))

    assert not left.is_structurally_equivalent(right)


def test_sequence_field_distinguishes_an_element() -> None:
    """Test a single differing element breaks sequence equivalence."""
    left = _Seq((_Leaf("a", 1), _Leaf("b", 2)))
    right = _Seq((_Leaf("a", 1), _Leaf("b", 3)))

    assert not left.is_structurally_equivalent(right)


def test_empty_sequences_are_equivalent() -> None:
    """Test two empty-sequence nodes are equivalent."""
    assert _Seq(()).is_structurally_equivalent(_Seq(()))
    assert _Seq(()).is_alpha_equivalent(_Seq(()))


def test_structural_only_sub_object_recurses_in_alpha_mode() -> None:
    """Test an alpha comparison falls back to structural recursion for a
    sub-object that only implements ``StructuralEquivalence``."""
    assert _HoldsStructuralOnly(_StructuralOnly(1)).is_alpha_equivalent(
        _HoldsStructuralOnly(_StructuralOnly(1))
    )


def test_structural_only_sub_object_distinguishes_in_alpha_mode() -> None:
    """Test the structural fallback still distinguishes differing sub-objects."""
    assert not _HoldsStructuralOnly(_StructuralOnly(1)).is_alpha_equivalent(
        _HoldsStructuralOnly(_StructuralOnly(2))
    )


def test_sequence_field_distinguishes_length_in_alpha_mode() -> None:
    """Test sequences of different length are not alpha equivalent."""
    left = _Seq((_Leaf("a", 1), _Leaf("b", 2)))
    right = _Seq((_Leaf("a", 1),))

    assert not left.is_alpha_equivalent(right)


def test_sequence_with_none_elements_matches_in_alpha_mode() -> None:
    """Test ``None`` elements in a sequence match positionally in alpha mode."""
    left = _SeqOptional((None, _Leaf("a", 1)))
    right = _SeqOptional((None, _Leaf("a", 1)))

    assert left.is_alpha_equivalent(right)


def test_sequence_distinguishes_none_from_present_in_alpha_mode() -> None:
    """Test a ``None`` element does not match a present element in alpha mode."""
    left = _SeqOptional((None,))
    right = _SeqOptional((_Leaf("a", 1),))

    assert not left.is_alpha_equivalent(right)


# ===========================================================================
# Optional inference
# ===========================================================================


def test_optional_field_matches_when_both_absent() -> None:
    """Test optional sub-object fields match when both are ``None``."""
    assert _Maybe().is_structurally_equivalent(_Maybe())


def test_optional_field_matches_when_both_present() -> None:
    """Test optional sub-object fields match when both hold equal values."""
    assert _Maybe(_Leaf("a", 1)).is_structurally_equivalent(_Maybe(_Leaf("a", 1)))


def test_optional_field_distinguishes_present_from_absent() -> None:
    """Test an optional field present on one side only breaks equivalence."""
    assert not _Maybe(_Leaf("a", 1)).is_structurally_equivalent(_Maybe())
    assert not _Maybe().is_structurally_equivalent(_Maybe(_Leaf("a", 1)))


# ===========================================================================
# Field exclusion
# ===========================================================================


def test_compare_false_field_is_excluded() -> None:
    """Test a ``field(compare=False)`` field does not affect equivalence."""
    assert _CompareFalse("k", "note one").is_structurally_equivalent(
        _CompareFalse("k", "note two")
    )


def test_compare_false_field_still_compares_the_participating_field() -> None:
    """Test exclusion narrows comparison without disabling other fields."""
    assert not _CompareFalse("k", "n").is_structurally_equivalent(
        _CompareFalse("other", "n")
    )


def test_excluded_from_equivalence_helper_skips_a_compared_field() -> None:
    """Test the explicit exclusion helper drops a field from equivalence."""

    @dataclass(frozen=True, eq=False)
    class _Tagged(DerivedEquivalenceMixin):
        name: str
        description: str = field(metadata=excluded_from_equivalence())

    assert _Tagged("x", "alpha").is_structurally_equivalent(_Tagged("x", "beta"))
    assert not _Tagged("x", "alpha").is_structurally_equivalent(_Tagged("y", "alpha"))


# ===========================================================================
# Value normalizer
# ===========================================================================


def test_value_normalizer_compares_through_the_key() -> None:
    """Test ``compared_as_value(key=...)`` compares normalized field values."""

    @dataclass(frozen=True, eq=False)
    class _Number(DerivedEquivalenceMixin):
        text: str = field(metadata=compared_as_value(key=lambda v: v.strip()))

    assert _Number("1").is_structurally_equivalent(_Number(" 1 "))
    assert not _Number("1").is_structurally_equivalent(_Number("2"))


def test_value_role_forces_equality_on_a_sub_object_field() -> None:
    """Test ``compared_as_value`` overrides recursion with ``==`` on a sub-object.

    ``_Leaf`` is ``eq=False``, so ``==`` is identity: two equal-but-distinct
    leaves compare unequal once the field is forced to value semantics.
    """

    @dataclass(frozen=True, eq=False)
    class _ByValue(DerivedEquivalenceMixin):
        child: _Leaf = field(metadata=compared_as_value())

    shared = _Leaf("a", 1)

    assert _ByValue(shared).is_structurally_equivalent(_ByValue(shared))
    assert not _ByValue(_Leaf("a", 1)).is_structurally_equivalent(
        _ByValue(_Leaf("a", 1))
    )


# ===========================================================================
# Identifier reference role
# ===========================================================================


def test_reference_field_compares_by_equality_structurally() -> None:
    """Test a reference ``Identifier`` compares nominally in structural mode."""

    @dataclass(frozen=True, eq=False)
    class _Ref(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    assert _Ref(x).is_structurally_equivalent(_Ref(x))
    assert not _Ref(x).is_structurally_equivalent(_Ref(y))


def test_reference_field_consults_the_renaming_in_alpha_mode() -> None:
    """Test a reference ``Identifier`` matches its renamed counterpart in alpha mode."""

    @dataclass(frozen=True, eq=False)
    class _Ref(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    renaming = AlphaRenaming.empty().extend({x: y})

    assert _Ref(x).is_alpha_equivalent_under(_Ref(y), renaming)
    assert not _Ref(x).is_alpha_equivalent_under(_Ref(x), renaming)


def test_reference_field_requires_identifier_equality_without_renaming() -> None:
    """Test references fall back to identity when no renaming is in scope."""

    @dataclass(frozen=True, eq=False)
    class _Ref(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    assert _Ref(x).is_alpha_equivalent(_Ref(x))
    assert not _Ref(x).is_alpha_equivalent(_Ref(y))


def test_undeclared_identifier_defaults_to_conservative_equality() -> None:
    """Test an untagged ``Identifier`` field compares nominally, never renamed.

    The safe default: an undeclared identifier degrades alpha to structural for
    that field (too strict, never an unsound merge).
    """

    @dataclass(frozen=True, eq=False)
    class _Plain(DerivedEquivalenceMixin):
        identifier: Identifier

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    renaming = AlphaRenaming.empty().extend({x: y})

    assert _Plain(x).is_structurally_equivalent(_Plain(x))
    assert not _Plain(x).is_alpha_equivalent_under(_Plain(y), renaming)


# ===========================================================================
# Identifier binder role
# ===========================================================================


def test_binder_renames_bound_identifiers_in_the_scoped_body() -> None:
    """Test a binder makes its scoped body alpha-equivalent under a rename."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
        body: _Var

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    assert _Lam(x, _Var(x)).is_alpha_equivalent(_Lam(y, _Var(y)))


def test_binder_is_alpha_equivalent_to_itself() -> None:
    """Test a binder node is alpha-equivalent to itself (reflexivity with binders)."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
        body: _Var

    x = mock_identifier("x", 1)
    lam = _Lam(x, _Var(x))

    assert lam.is_alpha_equivalent(lam)


def test_binder_is_not_structurally_equivalent_under_a_rename() -> None:
    """Test structural equivalence treats bound identifiers nominally."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
        body: _Var

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    assert not _Lam(x, _Var(x)).is_structurally_equivalent(_Lam(y, _Var(y)))


def test_binder_distinguishes_free_identifiers_in_the_body() -> None:
    """Test a binder renames only its bound names, not free body identifiers."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
        body: _Var

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    free_a = mock_identifier("a", 4)
    free_b = mock_identifier("b", 5)

    assert _Lam(x, _Var(free_a)).is_alpha_equivalent(_Lam(y, _Var(free_a)))
    assert not _Lam(x, _Var(free_a)).is_alpha_equivalent(_Lam(y, _Var(free_b)))


def test_binder_distinguishes_bound_arity() -> None:
    """Test binders with different numbers of bound names are not equivalent."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        params: tuple[Identifier, ...] = field(
            metadata=compared_as_binder(scopes_over=("body",))
        )
        body: _Var

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)

    assert not _Lam((x,), _Var(x)).is_alpha_equivalent(_Lam((x, y), _Var(x)))


def test_nested_binders_shadow_outer_bindings() -> None:
    """Test an inner binder shadows an outer binder over the same body."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
        body: "_Var | _Lam"

    x = mock_identifier("x", 1)
    a = mock_identifier("a", 4)
    b = mock_identifier("b", 5)

    # \x. \x. x  is alpha-equivalent to  \a. \b. b  (inner binder shadows outer)
    left = _Lam(x, _Lam(x, _Var(x)))
    right = _Lam(a, _Lam(b, _Var(b)))
    # ...but not to \a. \b. a (which uses the outer binder).
    other = _Lam(a, _Lam(b, _Var(a)))

    assert left.is_alpha_equivalent(right)
    assert not left.is_alpha_equivalent(other)


def test_binder_with_non_injective_binding_returns_false() -> None:
    """Test a binder pairing that is not injective returns ``False``."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        params: tuple[Identifier, ...] = field(
            metadata=compared_as_binder(scopes_over=("body",))
        )
        body: _Var

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)

    # Two distinct self-side names binding to a single other-side name.
    assert not _Lam((x, y), _Var(x)).is_alpha_equivalent(_Lam((z, z), _Var(z)))


def test_binder_with_unknown_scopes_over_name_raises_on_first_comparison() -> None:
    """Test a typo in ``scopes_over`` raises ``EquivalenceDerivationError``."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(
            metadata=compared_as_binder(scopes_over=("bdy",))  # typo: "bdy" not "body"
        )
        body: _Var

    x = mock_identifier("x", 1)
    with pytest.raises(EquivalenceDerivationError, match='"bdy" is not a field'):
        _Lam(x, _Var(x)).is_alpha_equivalent(_Lam(x, _Var(x)))


# ===========================================================================
# Custom comparator
# ===========================================================================


def test_compared_with_supplies_an_explicit_comparator() -> None:
    """Test ``compared_with`` drives comparison for an otherwise un-inferable field."""

    class _MagnitudeComparator:
        def is_structurally_equivalent(self, left: Any, right: Any) -> bool:
            return bool(abs(left) == abs(right))

        def is_alpha_equivalent_under(
            self, left: Any, right: Any, renaming: AlphaRenaming
        ) -> bool:
            return bool(abs(left) == abs(right))

    @dataclass(frozen=True, eq=False)
    class _Signed(DerivedEquivalenceMixin):
        magnitude: complex = field(metadata=compared_with(_MagnitudeComparator()))

    assert _Signed(3 + 4j).is_structurally_equivalent(_Signed(4 + 3j))
    assert not _Signed(3 + 4j).is_structurally_equivalent(_Signed(1 + 0j))


# ===========================================================================
# Hand-written override precedence
# ===========================================================================


def test_hand_written_structural_method_takes_precedence() -> None:
    """Test a hand-written structural method overrides derivation."""

    @dataclass(frozen=True, eq=False)
    class _Manual(DerivedEquivalenceMixin):
        tag: str

        def is_structurally_equivalent(self, other: object) -> bool:
            return isinstance(other, _Manual)

    # The override ignores ``tag``, so differing tags still match.
    assert _Manual("a").is_structurally_equivalent(_Manual("b"))


def test_one_manual_one_derived_method_mix() -> None:
    """Test a class may hand-write one method and derive the other."""

    @dataclass(frozen=True, eq=False)
    class _HalfManual(DerivedEquivalenceMixin):
        tag: str

        def is_alpha_equivalent_under(
            self, other: object, renaming: AlphaRenaming
        ) -> bool:
            return isinstance(other, _HalfManual)

    # Derived structural still compares ``tag``...
    assert not _HalfManual("a").is_structurally_equivalent(_HalfManual("b"))
    # ...while the hand-written alpha method ignores it.
    assert _HalfManual("a").is_alpha_equivalent(_HalfManual("b"))


# ===========================================================================
# Inheritance
# ===========================================================================


def test_subclass_derives_the_union_of_inherited_and_new_fields() -> None:
    """Test a subclass compares both inherited and newly declared fields."""

    @dataclass(frozen=True, eq=False)
    class _Base(DerivedEquivalenceMixin):
        shared: str

    @dataclass(frozen=True, eq=False)
    class _Derived(_Base):
        extra: int

    assert _Derived("s", 1).is_structurally_equivalent(_Derived("s", 1))
    assert not _Derived("s", 1).is_structurally_equivalent(_Derived("s", 2))
    assert not _Derived("s", 1).is_structurally_equivalent(_Derived("t", 1))


# ===========================================================================
# Derivation errors
# ===========================================================================


def test_uninferable_field_without_metadata_raises_derivation_error() -> None:
    """Test an un-inferable field with no override raises a derivation error."""

    @dataclass(frozen=True, eq=False)
    class _Bad(DerivedEquivalenceMixin):
        payload: complex

    with pytest.raises(EquivalenceDerivationError):
        _Bad(1 + 2j).is_structurally_equivalent(_Bad(1 + 2j))


def test_derivation_error_names_the_offending_field() -> None:
    """Test the derivation error message identifies the field."""

    @dataclass(frozen=True, eq=False)
    class _BadNamed(DerivedEquivalenceMixin):
        payload: complex

    with pytest.raises(EquivalenceDerivationError, match="payload"):
        _BadNamed(1j).is_structurally_equivalent(_BadNamed(1j))


def test_uninferable_sequence_element_raises_with_sequence_message() -> None:
    """Test an un-inferable element inside a sequence reports the sequence cause."""
    with pytest.raises(EquivalenceDerivationError, match="sequence element"):
        _SeqOfComplex((1 + 2j,)).is_structurally_equivalent(_SeqOfComplex((1 + 2j,)))


def test_uninferable_sequence_element_raises_in_alpha_mode() -> None:
    """Test the in-sequence derivation error also fires through the alpha path."""
    with pytest.raises(EquivalenceDerivationError, match="sequence element"):
        _SeqOfComplex((1 + 2j,)).is_alpha_equivalent(_SeqOfComplex((1 + 2j,)))


def test_non_dataclass_cannot_be_derived() -> None:
    """Test a non-dataclass deriving class fails with a derivation error."""

    class _NotADataclass(DerivedEquivalenceMixin):
        pass

    with pytest.raises(EquivalenceDerivationError, match="dataclass"):
        _NotADataclass().is_structurally_equivalent(_NotADataclass())


# ===========================================================================
# Relational laws (user-story)
# ===========================================================================


def _mixed_tree() -> _Add:
    """Build a representative binder-free term tree."""
    return _Add(_Const(1), _Add(_Const(2), _Const(3)))


def test_structural_equivalence_is_reflexive() -> None:
    """Test structural equivalence holds between a tree and itself."""
    tree = _mixed_tree()

    assert tree.is_structurally_equivalent(tree)


def test_structural_equivalence_is_symmetric() -> None:
    """Test structural equivalence does not depend on argument order."""
    left = _mixed_tree()
    right = _mixed_tree()

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_structural_equivalence_is_transitive() -> None:
    """Test structural equivalence composes across three equal trees."""
    a = _mixed_tree()
    b = _mixed_tree()
    c = _mixed_tree()

    assert a.is_structurally_equivalent(b)
    assert b.is_structurally_equivalent(c)
    assert a.is_structurally_equivalent(c)


def test_structural_equivalence_implies_alpha_equivalence() -> None:
    """Test a structurally equivalent pair is also alpha-equivalent."""
    left = _mixed_tree()
    right = _mixed_tree()

    assert left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)


def test_alpha_equivalence_does_not_imply_structural_equivalence() -> None:
    """Test a renamed binder is alpha-equivalent but not structurally equivalent."""

    @dataclass(frozen=True, eq=False)
    class _Var(DerivedEquivalenceMixin):
        identifier: Identifier = field(metadata=compared_as_reference())

    @dataclass(frozen=True, eq=False)
    class _Lam(DerivedEquivalenceMixin):
        param: Identifier = field(metadata=compared_as_binder(scopes_over=("body",)))
        body: _Var

    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = _Lam(x, _Var(x))
    right = _Lam(y, _Var(y))

    assert left.is_alpha_equivalent(right)
    assert not left.is_structurally_equivalent(right)


def test_deeply_nested_tree_compares_without_recursion_error() -> None:
    """Test a deeply nested term tree compares cleanly at depth."""
    left: _Term = _Const(0)
    right: _Term = _Const(0)
    for _ in range(500):
        left = _Add(left, _Const(1))
        right = _Add(right, _Const(1))

    assert left.is_structurally_equivalent(right)
