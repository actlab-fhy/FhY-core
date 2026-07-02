# mypy: disable-error-code="misc"
"""Tests the basic compiler traits."""

import datetime
import decimal
import enum
import fractions
import pathlib
import re
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Final

import pytest
from immutabledict import immutabledict

from fhy_core.diagnostic import (
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationFailedError,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.provenance import Provenance
from fhy_core.traits import (
    Equal,
    EqualMixin,
    Frozen,
    FrozenFieldTypeError,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
    HasIdentifier,
    HasProvenance,
    Interned,
    InternedMixin,
    Orderable,
    OrderableMixin,
    PartialEqual,
    PartialEqualMixin,
    PartialOrderable,
    PartialOrderableMixin,
    VerifiableMixin,
)
from fhy_core.utils import Self
from fhy_core.utils.override import override
from fhy_core.value_domain import DATA_DOMAIN

from .conftest import mock_identifier


@dataclass
class _IdentifierCarrier(HasIdentifier):
    _identifier: Identifier

    @override
    def get_identifier(self) -> Identifier:
        return self._identifier


@dataclass
class _ProvenanceCarrier(HasProvenance):
    _provenance: Provenance

    @override
    def get_provenance(self) -> Provenance:
        return self._provenance


@dataclass(frozen=True)
class _FrozenNode(FrozenMixin):
    value: int
    items: tuple[int, ...]


@dataclass(frozen=True)
class _PartialOrderableValue(PartialOrderableMixin):
    value: int

    @override
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _PartialOrderableValue):
            return NotImplemented
        return self.value < other.value


@dataclass(eq=True)
class _AutoPartialEqualValue(PartialEqualMixin):
    value: int


@dataclass(eq=False)
class _NoPartialEqualValue(PartialEqualMixin):
    value: int


@dataclass(eq=True, frozen=True)
class _AutoEqualValue(EqualMixin):
    value: int


class _NoHashEqualValue(EqualMixin):
    pass


@dataclass(order=True)
class _AutoPartialOrderableValue(PartialOrderableMixin):  # type: ignore[override]
    value: int


@dataclass(order=True)
class _AutoOrderableValue(OrderableMixin):  # type: ignore[override]
    value: int


@dataclass(eq=True)
class _EqualButUnhashableValue(EqualMixin):
    """Declares total equality and defines ``__eq__`` but is not hashable.

    ``@dataclass(eq=True)`` without ``frozen=True`` sets ``__hash__`` to
    ``None``, modelling a class that supports partial equality yet cannot
    support total equality.
    """

    value: int


class _NoLtOrderableValue(OrderableMixin):
    """Declares total ordering via the mixin but never implements ``__lt__``."""


class _ManualEqualValue(EqualMixin):
    """Hand-rolled total-equality value with real ``__eq__`` and ``__hash__``."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ManualEqualValue) and self.value == other.value

    @override
    def __hash__(self) -> int:
        return hash(self.value)


class _ManualOrderableValue(OrderableMixin):
    """Hand-rolled total-ordering value with a real ``__lt__``."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ManualOrderableValue):
            return NotImplemented
        return self.value < other.value


class _OptOutAutoFreeze(FrozenMixin, freeze_on_init=False):
    """Escape-hatch fixture: opt-out of auto-freeze for tests that need to
    observe a non-frozen FrozenMixin instance.
    """

    def __init__(self, value: int) -> None:
        self.value = value


@dataclass(frozen=True)
class _AutoFrozenPoint(FrozenMixin):
    x: int
    y: int


class _InternedValue(InternedMixin[str]):
    def __init__(self, key: str, value: int) -> None:
        self.key = key
        self.value = value

    @override
    def get_intern_key(self) -> str:
        return self.key


class _BaseInternedValue(InternedMixin[str]):
    def __init__(self, key: str) -> None:
        self.key = key

    @override
    def get_intern_key(self) -> str:
        return self.key


class _DerivedInternedValue(_BaseInternedValue):
    def __init__(self, key: str, payload: int) -> None:
        super().__init__(key)
        self.payload = payload


class _VerifiedFrozenInternedValue(InternedMixin[str], FrozenMixin, VerifiableMixin):
    verify_calls = 0

    def __init__(self, key: str, payload: tuple[int, ...]) -> None:
        self.key = key
        self.payload = payload

    @override
    def get_intern_key(self) -> str:
        return self.key

    @override
    def verify(self) -> ValidationReport[object]:
        type(self).verify_calls += 1
        if not self.key:
            return ValidationReport(
                diagnostics=(
                    Diagnostic(
                        level=DiagnosticLevel.ERROR,
                        message=Note("missing intern key"),
                        source=type(self).__name__,
                    ),
                )
            )
        return ValidationReport()


class _RaisingInternedValue(InternedMixin[str]):
    def __init__(self, key: str) -> None:
        self.key = key
        raise RuntimeError("init failed")

    @override
    def get_intern_key(self) -> str:
        return self.key


@dataclass
class _DataclassInternedValue(InternedMixin[str]):
    key: str
    value: int

    def __post_init__(self) -> None:
        self.register_interned_instance()

    @override
    def get_intern_key(self) -> str:
        return self.key


@dataclass
class _NotedInternedValue(InternedMixin[str]):
    key: str
    note: str = field(compare=False)

    def __post_init__(self) -> None:
        self.register_interned_instance()

    @override
    def get_intern_key(self) -> str:
        return self.key


def test_has_identifier_runtime_protocol() -> None:
    """Test `HasIdentifier` runtime protocol."""
    carrier = _IdentifierCarrier(mock_identifier("x", 1))
    assert isinstance(carrier, HasIdentifier)


def test_has_provenance_runtime_protocol() -> None:
    """Test `HasProvenance` runtime protocol."""
    carrier = _ProvenanceCarrier(Provenance.unknown())
    assert isinstance(carrier, HasProvenance)


def test_partial_orderable_runtime_protocol() -> None:
    """Test `PartialOrderable` runtime protocol."""
    value = _PartialOrderableValue(3)
    assert isinstance(value, PartialOrderable)


def test_partial_orderable_supports_sorting() -> None:
    """Test `PartialOrderableMixin` implementations can be sorted."""
    values = [
        _PartialOrderableValue(3),
        _PartialOrderableValue(1),
        _PartialOrderableValue(2),
    ]
    sorted_values = sorted(values)
    assert [value.value for value in sorted_values] == [1, 2, 3]


def test_partial_equal_runtime_protocol() -> None:
    """Test `PartialEqual` runtime protocol."""
    value = _AutoPartialEqualValue(3)
    assert isinstance(value, PartialEqual)
    assert value.supports_partial_equality is True


def test_partial_equal_detects_eq_false_dataclass() -> None:
    """Test `PartialEqualMixin` detects `dataclass(eq=False)` as unsupported."""
    value = _NoPartialEqualValue(3)
    assert value.supports_partial_equality is False
    assert value.__eq__(_NoPartialEqualValue(3)) is NotImplemented


def test_equal_runtime_protocol() -> None:
    """Test `Equal` runtime protocol."""
    value = _AutoEqualValue(3)
    assert isinstance(value, Equal)
    assert value.supports_equality is True
    assert value.supports_partial_equality is True
    assert value == _AutoEqualValue(3)


def test_equal_mixin_requires_hash_implementation() -> None:
    """Test `EqualMixin` prompts subclasses to implement hash."""
    with pytest.raises(NotImplementedError):
        hash(_NoHashEqualValue())


def test_partial_orderable_detects_ordered_dataclass() -> None:
    """Test `PartialOrderableMixin` detects `dataclass(order=True)` support."""
    left = _AutoPartialOrderableValue(1)
    right = _AutoPartialOrderableValue(2)
    assert left.supports_partial_ordering is True
    assert left < right


def test_orderable_runtime_protocol() -> None:
    """Test `Orderable` runtime protocol."""
    value = _AutoOrderableValue(3)
    assert isinstance(value, Orderable)
    assert value.supports_ordering is True
    assert value.supports_partial_ordering is True


def test_orderable_mixin_defaults_to_total_order() -> None:
    """Test `OrderableMixin` defaults to total-order support."""
    value = _AutoOrderableValue(3)
    assert value.supports_partial_ordering is True
    assert value.supports_ordering is True


def test_equal_without_eq_or_hash_does_not_support_equality() -> None:
    """Test `EqualMixin` reports no equality support without `__eq__`/`__hash__`.

    A class that opts into total equality but supplies neither a real
    ``__eq__`` nor a real ``__hash__`` cannot honor the contract, so
    ``supports_equality`` must be ``False`` rather than advertising a
    capability that ``hash()`` then fails to provide.
    """
    value = _NoHashEqualValue()
    assert value.supports_partial_equality is False
    assert value.supports_equality is False


def test_equal_with_eq_but_no_hash_does_not_support_equality() -> None:
    """Test `EqualMixin` reports no equality support when not hashable.

    ``_EqualButUnhashableValue`` supports partial equality (it has a real
    ``__eq__``) but is unhashable (``__hash__`` is ``None``), so total
    equality is not supported.
    """
    value = _EqualButUnhashableValue(3)
    assert value.supports_partial_equality is True
    assert value.supports_equality is False


def test_orderable_without_lt_does_not_support_ordering() -> None:
    """Test `OrderableMixin` reports no ordering support without `__lt__`."""
    value = _NoLtOrderableValue()
    assert value.supports_partial_ordering is False
    assert value.supports_ordering is False


def test_manual_equal_with_eq_and_hash_supports_equality() -> None:
    """Test a hand-rolled `__eq__`/`__hash__` reports total equality support."""
    value = _ManualEqualValue(3)
    assert value.supports_partial_equality is True
    assert value.supports_equality is True
    assert hash(value) == hash(_ManualEqualValue(3))


def test_manual_orderable_with_lt_supports_ordering() -> None:
    """Test a hand-rolled `__lt__` reports total ordering support."""
    value = _ManualOrderableValue(3)
    assert value.supports_partial_ordering is True
    assert value.supports_ordering is True


@pytest.mark.parametrize(
    "value",
    [
        _AutoEqualValue(3),
        _NoHashEqualValue(),
        _EqualButUnhashableValue(3),
        _ManualEqualValue(3),
    ],
)
def test_total_equality_implies_partial_equality(value: Equal) -> None:
    """Test the subtype invariant: total equality entails partial equality.

    ``Equal`` is a subtype of ``PartialEqual``, so no object may report
    ``supports_equality`` while denying ``supports_partial_equality``.
    """
    assert not value.supports_equality or value.supports_partial_equality


@pytest.mark.parametrize(
    "value",
    [
        _AutoOrderableValue(3),
        _NoLtOrderableValue(),
        _ManualOrderableValue(3),
    ],
)
def test_total_ordering_implies_partial_ordering(value: Orderable) -> None:
    """Test the subtype invariant: total ordering entails partial ordering.

    ``Orderable`` is a subtype of ``PartialOrderable``, so no object may
    report ``supports_ordering`` while denying ``supports_partial_ordering``.
    """
    assert not value.supports_ordering or value.supports_partial_ordering


def test_identifier_returns_identifier() -> None:
    """Test `HasIdentifier.get_identifier` returns the carried identifier."""
    carrier = _IdentifierCarrier(mock_identifier("field", 2))
    assert carrier.get_identifier().name_hint == "field"
    assert carrier.get_identifier().id == 2


def test_provenance_returns_provenance() -> None:
    """Test `HasProvenance.get_provenance` returns the carried provenance."""
    carrier = _ProvenanceCarrier(Provenance.unknown())
    assert carrier.get_provenance() == Provenance.unknown()


def test_frozen_runtime_protocol() -> None:
    """Test `Frozen` runtime protocol."""
    node = _FrozenNode(1, (2, 3))
    assert isinstance(node, Frozen)


def test_frozen_blocks_attribute_updates_after_freeze() -> None:
    """Test `FrozenMixin` blocks direct attribute mutation after freezing."""
    node = _FrozenNode(1, (2, 3))
    with pytest.raises(FrozenMutationError):
        node.value = 42


def test_frozen_blocks_attribute_deletion_after_freeze() -> None:
    """Test `FrozenMixin` blocks attribute deletion after freezing."""
    node = _FrozenNode(1, (2, 3))
    with pytest.raises(FrozenMutationError):
        del node.value


def test_frozen_assert_frozen_passes_for_write_protected_instance() -> None:
    """Test `FrozenMixin.assert_frozen` passes for valid frozen instances."""
    node = _FrozenNode(1, (2, 3))
    node.assert_frozen()


def test_frozen_assert_frozen_fails_when_not_frozen() -> None:
    """Test `FrozenMixin.assert_frozen` fails if instance was not frozen."""
    instance = _OptOutAutoFreeze(value=1)
    with pytest.raises(FrozenValidationError):
        instance.assert_frozen()


def test_native_frozen_dataclass_runtime_protocol() -> None:
    """Test native frozen dataclass instances satisfy the `Frozen` protocol."""
    point = _AutoFrozenPoint(1, 2)
    assert isinstance(point, Frozen)
    assert point.is_frozen is True


def test_native_frozen_dataclass_blocks_mutation() -> None:
    """Test native frozen dataclass blocks direct attribute mutation."""
    point = _AutoFrozenPoint(1, 2)
    with pytest.raises(FrozenMutationError):
        point.x = 4


def test_native_frozen_dataclass_with_frozen_mixin() -> None:
    """Test native `dataclass(frozen=True)` integrates with `FrozenMixin`."""

    @dataclass(frozen=True)
    class _NativeFrozenPoint(FrozenMixin):
        x: int
        y: int

    point = _NativeFrozenPoint(1, 2)

    assert isinstance(point, Frozen)
    assert point.is_frozen is True
    point.assert_frozen()
    with pytest.raises(FrozenMutationError):
        point.x = 4


def test_native_frozen_dataclass_post_init_can_use_object_setattr() -> None:
    """Test `__post_init__` uses `object.__setattr__` to set derived fields.

    Documents the canonical ``@dataclass(frozen=True)`` pattern: the instance
    is already frozen when ``__post_init__`` runs, so derived-field
    assignment goes through ``object.__setattr__``. This is standard Python
    dataclass-frozen semantics; ``FrozenMixin`` does not override it.
    """

    @dataclass(frozen=True)
    class _DerivedFieldPoint(FrozenMixin):
        x: int
        y: int
        magnitude_squared: int = 0

        def __post_init__(self) -> None:
            object.__setattr__(
                self,
                "magnitude_squared",
                self.x * self.x + self.y * self.y,
            )

    point = _DerivedFieldPoint(3, 4)

    assert point.magnitude_squared == 25
    assert point.is_frozen is True


def test_frozen_dataclass_mixin_delete_raises_frozen_mutation_error() -> None:
    """Test `@dataclass(frozen=True) + FrozenMixin` raises on deletion."""

    @dataclass(frozen=True)
    class _NativeFrozenDeletable(FrozenMixin):
        value: int

    instance = _NativeFrozenDeletable(1)
    with pytest.raises(FrozenMutationError):
        delattr(instance, "value")


def test_frozen_dataclass_mixin_raises_consistent_error_for_existing_classes() -> None:
    """Test fhy_core's own frozen-dataclass classes raise `FrozenMutationError`."""
    with pytest.raises(FrozenMutationError):
        DATA_DOMAIN.description = "rewritten"


def test_interned_runtime_protocol() -> None:
    """Test `Interned` runtime protocol."""
    _InternedValue.clear_interned_registry()
    value = _InternedValue("x", 1)
    assert isinstance(value, Interned)


def test_interned_returns_first_registered_instance_for_key() -> None:
    """Test duplicate intern keys preserve the first registered instance."""
    _InternedValue.clear_interned_registry()
    first = _InternedValue("x", 1)
    second = _InternedValue("x", 2)

    assert _InternedValue.get_interned("x") is first
    assert _InternedValue.require_interned("x") is first
    assert second is not first


def test_interned_subclasses_share_family_registry() -> None:
    """Test descendants register into the same family registry as their base."""
    _BaseInternedValue.clear_interned_registry()
    value = _DerivedInternedValue("child", 7)

    assert _BaseInternedValue.get_interned("child") is value
    assert _DerivedInternedValue.get_interned("child") is value


def test_interned_supports_manual_registration_from_dataclass_post_init() -> None:
    """Test dataclass users can register canonical instances from `__post_init__`."""
    _DataclassInternedValue.clear_interned_registry()
    value = _DataclassInternedValue("dc", 9)

    assert _DataclassInternedValue.get_interned("dc") is value


def test_interned_finalize_verifies_and_freezes_once() -> None:
    """Test `InternedMixin` verifies/freezes exactly once per full init chain."""
    _VerifiedFrozenInternedValue.clear_interned_registry()
    _VerifiedFrozenInternedValue.verify_calls = 0
    value = _VerifiedFrozenInternedValue("frozen", (1, 2, 3))

    assert _VerifiedFrozenInternedValue.verify_calls == 1
    assert value.is_frozen is True
    assert value.payload == (1, 2, 3)


def test_interned_finalize_propagates_verification_errors() -> None:
    """Test failed verification prevents invalid interned registration."""
    _VerifiedFrozenInternedValue.clear_interned_registry()
    _VerifiedFrozenInternedValue.verify_calls = 0

    with pytest.raises(ValidationFailedError):
        _VerifiedFrozenInternedValue("", (1,))

    assert _VerifiedFrozenInternedValue.get_interned("") is None


def test_interned_does_not_register_when_init_raises() -> None:
    """Test failed initialization prevents partial interned registration."""
    _RaisingInternedValue.clear_interned_registry()

    with pytest.raises(RuntimeError, match="init failed"):
        _RaisingInternedValue("boom")

    assert _RaisingInternedValue.get_interned("boom") is None


def test_interned_construct_from_fields_returns_freshly_built_for_new_key() -> None:
    """Test the reconstruction hook registers and returns a brand-new instance."""
    _DataclassInternedValue.clear_interned_registry()

    restored = _DataclassInternedValue.construct_from_fields({"key": "new", "value": 1})

    assert restored == _DataclassInternedValue("new", 1)
    assert _DataclassInternedValue.get_interned("new") is restored


def test_interned_construct_from_fields_returns_canonical_for_existing_key() -> None:
    """Test the hook returns the canonical instance, not the rebuilt duplicate."""
    _DataclassInternedValue.clear_interned_registry()
    canonical = _DataclassInternedValue("dup", 1)

    restored = _DataclassInternedValue.construct_from_fields({"key": "dup", "value": 2})

    assert restored is canonical


def test_interned_construct_from_fields_warns_on_ignored_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an equality-excluded field diverging from the canonical is logged."""
    _NotedInternedValue.clear_interned_registry()
    canonical = _NotedInternedValue("noted", "original")

    with caplog.at_level("WARNING", logger="fhy_core.traits.interned"):
        restored = _NotedInternedValue.construct_from_fields(
            {"key": "noted", "note": "divergent"}
        )

    assert restored is canonical
    assert restored.note == "original"
    assert "already canonical" in caplog.text
    assert "divergent" in caplog.text


def test_interned_construct_from_fields_does_not_warn_when_metadata_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test no warning fires when the ignored field matches the canonical's."""
    _NotedInternedValue.clear_interned_registry()
    _NotedInternedValue("noted", "same")

    with caplog.at_level("WARNING", logger="fhy_core.traits.interned"):
        _NotedInternedValue.construct_from_fields({"key": "noted", "note": "same"})

    assert "already canonical" not in caplog.text


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({"key": "x"}, id="missing_field"),
        pytest.param({"key": "x", "value": 1, "extra": 2}, id="extra_field"),
    ],
)
def test_interned_construct_from_fields_rejects_malformed_field_map(
    malformed: dict[str, object],
) -> None:
    """Test the interned hook fails loudly on a field map that is not one-per-field."""
    _DataclassInternedValue.clear_interned_registry()

    with pytest.raises(TypeError):
        _DataclassInternedValue.construct_from_fields(malformed)


# =============================================================================
# FrozenMixin auto-freeze on init
# =============================================================================


class _AutoFreezeShallow(FrozenMixin):
    def __init__(self, value: int, items: tuple[int, ...]) -> None:
        self.value = value
        self.items = items


class _NoAutoFreeze(FrozenMixin, freeze_on_init=False):
    def __init__(self, value: int) -> None:
        self.value = value


class _BaseWithAutoFreeze(FrozenMixin):
    def __init__(self, value: int) -> None:
        self.value = value


class _InheritsAutoFreezePolicy(_BaseWithAutoFreeze):
    def __init__(self, value: int, extra: int) -> None:
        super().__init__(value)
        self.extra = extra


class _OverridesPolicyToFalse(_BaseWithAutoFreeze, freeze_on_init=False):
    def __init__(self, value: int) -> None:
        super().__init__(value)


class _InheritsWithoutOwnInit(_BaseWithAutoFreeze):
    pass


class _ExplicitSelfFreeze(FrozenMixin):
    def __init__(self, value: int) -> None:
        self.value = value
        self.freeze()


class _RaisingAutoFreeze(FrozenMixin):
    def __init__(self, value: int) -> None:
        self.value = value
        raise RuntimeError("init failed")


@dataclass(frozen=True)
class _AutoFreezeDataclass(FrozenMixin):
    value: int
    items: tuple[int, ...]


def test_auto_freeze_default_freezes_new_instance() -> None:
    """Test the default auto-freeze policy freezes new instances after `__init__`."""
    instance = _AutoFreezeShallow(1, (2, 3))

    assert instance.is_frozen is True


def test_auto_freeze_opt_out_leaves_instance_mutable() -> None:
    """Test ``freeze_on_init=False`` escape hatch leaves new instances mutable."""
    instance = _NoAutoFreeze(1)

    assert instance.is_frozen is False
    instance.value = 99
    assert instance.value == 99


def test_auto_freeze_policy_is_inherited_by_subclass() -> None:
    """Test a subclass inherits its parent's auto-freeze policy."""
    instance = _InheritsAutoFreezePolicy(1, 2)

    assert instance.is_frozen is True


def test_auto_freeze_policy_can_be_overridden_to_false() -> None:
    """Test a subclass can set ``freeze_on_init=False`` to opt out."""
    instance = _OverridesPolicyToFalse(1)

    assert instance.is_frozen is False


def test_auto_freeze_with_subclass_without_own_init_still_freezes() -> None:
    """Test a subclass that inherits ``__init__`` still ends up frozen."""
    instance = _InheritsWithoutOwnInit(1)

    assert instance.is_frozen is True


def test_auto_freeze_idempotent_when_init_explicitly_freezes() -> None:
    """Test the auto-freeze is a no-op when ``__init__`` already calls ``freeze``."""
    instance = _ExplicitSelfFreeze(1)

    assert instance.is_frozen is True


def test_auto_freeze_skipped_when_init_raises() -> None:
    """Test a failure inside ``__init__`` does not trigger the auto-freeze."""
    with pytest.raises(RuntimeError, match="init failed"):
        _RaisingAutoFreeze(1)


def test_auto_freeze_with_dataclass_frozen_subclass() -> None:
    """Test auto-freeze composes with a `@dataclass(frozen=True)` subclass."""
    instance = _AutoFreezeDataclass(1, (2, 3))

    assert instance.is_frozen is True
    assert instance.items == (2, 3)


def test_auto_freeze_subclass_can_set_own_state_before_outermost_freeze() -> None:
    """Test nested ``__init__`` chains let the most-derived class set state."""
    instance = _InheritsAutoFreezePolicy(value=1, extra=2)

    assert instance.is_frozen is True
    assert instance.value == 1
    assert instance.extra == 2


# =============================================================================
# Frozen protocol surface
# =============================================================================


def test_frozen_protocol_does_not_expose_assert_write_protected() -> None:
    """Test the ``Frozen`` protocol no longer exposes ``assert_write_protected``."""
    assert not hasattr(Frozen, "assert_write_protected")


def test_frozen_freeze_method_takes_no_arguments() -> None:
    """Test ``freeze()`` no longer accepts ``deep`` (or any) keyword argument."""

    class _SimpleFrozen(FrozenMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _SimpleFrozen(1)

    with pytest.raises(TypeError):
        instance.freeze(deep=True)  # type: ignore[call-arg]


def test_assert_frozen_takes_no_arguments() -> None:
    """Test ``assert_frozen()`` no longer accepts ``deep``/``strict`` kwargs."""

    class _SimpleFrozen(FrozenMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _SimpleFrozen(1)

    with pytest.raises(TypeError):
        instance.assert_frozen(deep=True)  # type: ignore[call-arg]


def test_assert_frozen_passes_for_dataclass_frozen_instance() -> None:
    """Test ``assert_frozen`` succeeds for a `@dataclass(frozen=True)` instance."""
    point = _AutoFrozenPoint(1, 2)
    point.assert_frozen()


def test_freeze_is_idempotent_on_already_frozen_instance() -> None:
    """Test calling ``freeze()`` on a frozen instance is a no-op."""

    class _SimpleFrozen(FrozenMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _SimpleFrozen(1)
    assert instance.is_frozen is True

    instance.freeze()
    instance.freeze()

    assert instance.is_frozen is True


def test_frozen_instance_cannot_be_unfrozen_by_setting_internal_flag() -> None:
    """Test writing the internal frozen flag on a frozen instance raises."""

    class _SimpleFrozen(FrozenMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _SimpleFrozen(1)

    with pytest.raises(FrozenMutationError):
        instance._fhy_core_is_frozen = False

    assert instance.is_frozen is True


def test_frozen_instance_cannot_be_unfrozen_by_deleting_internal_flag() -> None:
    """Test deleting the internal frozen flag on a frozen instance raises."""

    class _SimpleFrozen(FrozenMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _SimpleFrozen(1)

    with pytest.raises(FrozenMutationError):
        delattr(instance, "_fhy_core_is_frozen")

    assert instance.is_frozen is True


# =============================================================================
# Field-type immutability check
# =============================================================================


def test_field_type_check_rejects_list_field() -> None:
    """Test ``FrozenFieldTypeError`` fires for a `list[int]` field."""

    @dataclass(frozen=True)
    class _BadList(FrozenMixin):
        values: list[int]  # intentional violation

    with pytest.raises(FrozenFieldTypeError, match="values"):
        _BadList([1, 2, 3])


def test_field_type_check_rejects_set_field() -> None:
    """Test ``FrozenFieldTypeError`` fires for a `set[int]` field."""

    @dataclass(frozen=True)
    class _BadSet(FrozenMixin):
        values: set[int]

    with pytest.raises(FrozenFieldTypeError, match="values"):
        _BadSet(set())


def test_field_type_check_rejects_dict_field() -> None:
    """Test ``FrozenFieldTypeError`` fires for a `dict[str, int]` field."""

    @dataclass(frozen=True)
    class _BadDict(FrozenMixin):
        values: dict[str, int]

    with pytest.raises(FrozenFieldTypeError, match="values"):
        _BadDict({})


def test_field_type_check_rejects_bytearray_field() -> None:
    """Test ``FrozenFieldTypeError`` fires for a `bytearray` field."""

    @dataclass(frozen=True)
    class _BadBytearray(FrozenMixin):
        data: bytearray

    with pytest.raises(FrozenFieldTypeError, match="data"):
        _BadBytearray(bytearray())


def test_field_type_check_accepts_tuple_field() -> None:
    """Test ``tuple[int, ...]`` passes the field-type check."""

    @dataclass(frozen=True)
    class _GoodTuple(FrozenMixin):
        values: tuple[int, ...]

    instance = _GoodTuple((1, 2, 3))
    assert instance.values == (1, 2, 3)


def test_field_type_check_accepts_frozenset_field() -> None:
    """Test ``frozenset[int]`` passes the field-type check."""

    @dataclass(frozen=True)
    class _GoodFrozenset(FrozenMixin):
        values: frozenset[int]

    instance = _GoodFrozenset(frozenset([1, 2, 3]))
    assert instance.values == frozenset([1, 2, 3])


def test_field_type_check_accepts_union_of_immutables() -> None:
    """Test a Union of immutable arms passes the field-type check."""

    @dataclass(frozen=True)
    class _GoodUnion(FrozenMixin):
        value: int | str | None

    _GoodUnion(1)
    _GoodUnion("x")
    _GoodUnion(None)


def test_field_type_check_rejects_union_with_mutable_arm() -> None:
    """Test a Union with one mutable arm fails the field-type check."""

    @dataclass(frozen=True)
    class _BadUnion(FrozenMixin):
        value: int | list[int]

    with pytest.raises(FrozenFieldTypeError, match="value"):
        _BadUnion(1)


def test_field_type_check_rejects_tuple_with_mutable_element() -> None:
    """Test ``tuple[list[int], ...]`` fails the field-type check."""

    @dataclass(frozen=True)
    class _BadNestedTuple(FrozenMixin):
        rows: tuple[list[int], ...]

    with pytest.raises(FrozenFieldTypeError, match="rows"):
        _BadNestedTuple(())


def test_field_type_check_accepts_path_field() -> None:
    """Test a `pathlib.Path` field passes the check."""

    @dataclass(frozen=True)
    class _GoodPath(FrozenMixin):
        location: pathlib.Path

    _GoodPath(pathlib.Path("/tmp"))


def test_field_type_check_accepts_enum_field() -> None:
    """Test an `Enum` subclass field passes the check."""

    class _Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    @dataclass(frozen=True)
    class _GoodEnum(FrozenMixin):
        color: _Color

    _GoodEnum(_Color.RED)


def test_field_type_check_accepts_classvar_field() -> None:
    """Test ``ClassVar[list[int]]`` is skipped by the field-type check."""

    class _GoodClassVar(FrozenMixin):
        _scratch: ClassVar[list[int]] = []

        def __init__(self, value: int) -> None:
            self.value = value

    _GoodClassVar(1)


def test_field_type_check_warns_once_for_unresolvable_annotation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an unresolvable field annotation warns once instead of failing silently.

    The immutability of a field whose annotation cannot be resolved is left
    unverified; a one-time per-class warning makes that gap visible.
    """

    @dataclass(frozen=True)
    class _UnresolvableField(FrozenMixin):
        value: "_NameThatDoesNotExistAnywhere"  # type: ignore[name-defined]  # noqa: F821

    with caplog.at_level("WARNING", logger="fhy_core.traits.frozen"):
        _UnresolvableField(1)
        _UnresolvableField(2)

    matching = [r for r in caplog.records if "not enforced" in r.getMessage()]
    assert len(matching) == 1
    assert "value" in matching[0].getMessage()
    assert "_UnresolvableField" in matching[0].getMessage()


def test_field_type_check_skipped_for_abstract_class() -> None:
    """Test an abstract `FrozenMixin` subclass with mutable fields is allowed.

    The check fires only on concrete instantiation, matching the
    :class:`abc.ABC` convention. An abstract class with a mutable field
    is fine until a concrete descendant tries to instantiate it.
    """

    @dataclass(frozen=True)
    class _AbstractWithMutable(FrozenMixin):
        values: list[int]

        @abstractmethod
        def do_thing(self) -> None: ...

    # Instantiating the abstract class fails because it is abstract, not
    # because of the mutable field: the field-type check is skipped for
    # abstract classes, so ``TypeError`` is raised rather than
    # ``FrozenFieldTypeError``.
    with pytest.raises(TypeError) as exc_info:
        _AbstractWithMutable([1, 2])  # type: ignore[abstract]
    assert not isinstance(exc_info.value, FrozenFieldTypeError)

    @dataclass(frozen=True)
    class _ConcreteFix(_AbstractWithMutable):
        values: tuple[int, ...]  # type: ignore[assignment]

        @override
        def do_thing(self) -> None:
            pass

    instance = _ConcreteFix((1, 2, 3))
    assert instance.values == (1, 2, 3)


def test_field_type_check_accepts_callable_field() -> None:
    """Test ``Callable[..., T]`` passes the field-type check."""

    @dataclass(frozen=True)
    class _GoodCallable(FrozenMixin):
        action: Callable[[int], int]

    _GoodCallable(lambda x: x + 1)


def test_field_type_check_accepts_type_field() -> None:
    """Test ``type[T]`` passes the field-type check."""

    @dataclass(frozen=True)
    class _GoodType(FrozenMixin):
        cls: type[int]

    _GoodType(int)


def test_field_type_check_accepts_self_field() -> None:
    """Test a ``Self``-typed field passes the check across Python versions.

    ``Self`` is sourced from the ``fhy_core.utils`` compatibility shim so the
    module imports on Python 3.10, where ``typing.Self`` does not exist.
    """

    @dataclass(frozen=True)
    class _GoodSelf(FrozenMixin):
        peer: "Self | None" = None

    _GoodSelf()


def test_field_type_check_accepts_decimal_field() -> None:
    """Test a `decimal.Decimal` field passes the check."""

    @dataclass(frozen=True)
    class _GoodDecimal(FrozenMixin):
        value: decimal.Decimal

    _GoodDecimal(decimal.Decimal("1.5"))


def test_field_type_check_accepts_fraction_field() -> None:
    """Test a `fractions.Fraction` field passes the check."""

    @dataclass(frozen=True)
    class _GoodFraction(FrozenMixin):
        value: fractions.Fraction

    _GoodFraction(fractions.Fraction(1, 3))


def test_field_type_check_accepts_datetime_field() -> None:
    """Test a `datetime.datetime` field passes the check."""

    @dataclass(frozen=True)
    class _GoodDatetime(FrozenMixin):
        moment: datetime.datetime

    _GoodDatetime(datetime.datetime(2026, 1, 1))


def test_field_type_check_accepts_pattern_field() -> None:
    """Test a `re.Pattern` field passes the check."""

    @dataclass(frozen=True)
    class _GoodPattern(FrozenMixin):
        matcher: re.Pattern[str]

    _GoodPattern(re.compile("x"))


def test_field_type_check_accepts_immutabledict_field() -> None:
    """Test a parameterized ``immutabledict[str, int]`` field passes the check."""

    @dataclass(frozen=True)
    class _GoodFrozendict(FrozenMixin):
        mapping: immutabledict[str, int]

    _GoodFrozendict(immutabledict({"x": 1}))


def test_field_type_check_rejects_immutabledict_with_mutable_value() -> None:
    """Test ``immutabledict[str, list[int]]`` fails on its mutable value type."""

    @dataclass(frozen=True)
    class _BadFrozendict(FrozenMixin):
        mapping: immutabledict[str, list[int]]

    with pytest.raises(FrozenFieldTypeError, match="mapping"):
        _BadFrozendict(immutabledict())


def test_field_type_check_accepts_annotated_field() -> None:
    """Test ``Annotated[tuple[int, ...], ...]`` defers to the wrapped type."""

    @dataclass(frozen=True)
    class _GoodAnnotated(FrozenMixin):
        values: Annotated[tuple[int, ...], "metadata"]

    _GoodAnnotated((1, 2))


def test_field_type_check_rejects_annotated_mutable_field() -> None:
    """Test ``Annotated[list[int], ...]`` is rejected through the wrapped type."""

    @dataclass(frozen=True)
    class _BadAnnotated(FrozenMixin):
        values: Annotated[list[int], "metadata"]

    with pytest.raises(FrozenFieldTypeError, match="values"):
        _BadAnnotated([])


def test_field_type_check_accepts_final_field() -> None:
    """Test ``Final[int]`` defers to the wrapped immutable type."""

    @dataclass(frozen=True)
    class _GoodFinal(FrozenMixin):
        value: Final[int]

    _GoodFinal(1)


def test_field_type_check_accepts_optional_frozenset_field() -> None:
    """Test ``Optional[frozenset[int]]`` passes the check."""

    @dataclass(frozen=True)
    class _GoodOptional(FrozenMixin):
        values: frozenset[int] | None

    _GoodOptional(None)
    _GoodOptional(frozenset([1]))


def test_field_type_check_accepts_nested_frozen_mixin_field() -> None:
    """Test a field typed as another `FrozenMixin` subclass passes the check."""

    @dataclass(frozen=True)
    class _Inner(FrozenMixin):
        value: int

    @dataclass(frozen=True)
    class _Outer(FrozenMixin):
        inner: _Inner

    _Outer(_Inner(1))


def test_field_type_check_rejects_plain_user_class_field() -> None:
    """Test a field typed as a non-`FrozenMixin` user class is rejected.

    An arbitrary user class is mutable for all the check can tell, so it is
    refused unless it inherits `FrozenMixin`.
    """

    class _PlainUserClass:
        pass

    @dataclass(frozen=True)
    class _BadUserClass(FrozenMixin):
        payload: _PlainUserClass

    with pytest.raises(FrozenFieldTypeError, match="payload"):
        _BadUserClass(_PlainUserClass())


def test_field_type_check_reports_resolvable_violation_despite_unresolvable_field() -> (
    None
):
    """Test a resolvable mutable field is still caught when another won't resolve.

    A single unresolvable forward reference must not suppress the check for
    the rest of the class: the resolvable ``list[int]`` field is still
    reported.
    """

    @dataclass(frozen=True)
    class _MixedResolution(FrozenMixin):
        bad: list[int]
        ref: "_DefinitelyNotDefinedForwardRef"  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(FrozenFieldTypeError, match="bad"):
        _MixedResolution([1, 2], None)


def test_assert_frozen_raises_when_dispatch_not_wired() -> None:
    """Test ``assert_frozen`` fails when the class ``__setattr__`` is not wired.

    If something replaces the freeze-enforcing ``__setattr__`` on the class,
    the freeze invariant is no longer enforceable and ``assert_frozen``
    raises ``FrozenValidationError`` even though the frozen flag is set.
    """

    class _Tampered(FrozenMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _Tampered(1)
    type.__setattr__(_Tampered, "__setattr__", object.__setattr__)

    with pytest.raises(FrozenValidationError):
        instance.assert_frozen()


def test_dataclass_frozen_post_init_self_assignment_raises() -> None:
    """Test plain ``self.x = ...`` in a frozen dataclass ``__post_init__`` raises.

    The instance is already frozen when ``__post_init__`` runs, so a plain
    assignment raises ``FrozenMutationError``.
    """

    @dataclass(frozen=True)
    class _SelfAssignPostInit(FrozenMixin):
        x: int
        derived: int = 0

        def __post_init__(self) -> None:
            self.derived = self.x * 2

    with pytest.raises(FrozenMutationError):
        _SelfAssignPostInit(3)


# =============================================================================
# Wrap ordering with InternedMixin
# =============================================================================


class _InternedFrozenObservesFreeze(InternedMixin[str], FrozenMixin, VerifiableMixin):
    """Fixture that asserts ``is_frozen`` is true at finalize time."""

    observed_is_frozen_during_verify: bool | None = None

    def __init__(self, key: str, value: int) -> None:
        self.key = key
        self.value = value

    @override
    def get_intern_key(self) -> str:
        return self.key

    @override
    def verify(self) -> ValidationReport[object]:
        type(self).observed_is_frozen_during_verify = self.is_frozen
        return ValidationReport()


def test_interned_finalize_runs_against_frozen_instance() -> None:
    """Test ``InternedMixin._finalize_interned_instance`` sees a frozen instance.

    The wrap-ordering convention places ``InternedMixin``'s post-init work
    outside ``FrozenMixin``'s, so by the time finalize runs the instance
    is already frozen.
    """
    _InternedFrozenObservesFreeze.clear_interned_registry()
    _InternedFrozenObservesFreeze.observed_is_frozen_during_verify = None

    _InternedFrozenObservesFreeze("k", 1)

    assert _InternedFrozenObservesFreeze.observed_is_frozen_during_verify is True


# =============================================================================
# freeze_on_init=False escape hatch
# =============================================================================


def test_freeze_on_init_false_class_can_be_explicitly_frozen() -> None:
    """Test that an opt-out class can still be manually frozen via ``freeze()``."""

    class _OptOut(FrozenMixin, freeze_on_init=False):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = _OptOut(1)
    assert instance.is_frozen is False
    instance.value = 2  # still mutable
    instance.freeze()
    assert instance.is_frozen is True
    with pytest.raises(FrozenMutationError):
        instance.value = 3


def test_freeze_on_init_false_blocks_inherit_default_when_child_does_not_override() -> (
    None
):
    """Test inherited ``freeze_on_init=False`` propagates to descendants."""

    class _OptOutBase(FrozenMixin, freeze_on_init=False):
        def __init__(self, value: int) -> None:
            self.value = value

    class _ChildInheritsOptOut(_OptOutBase):
        pass

    instance = _ChildInheritsOptOut(1)
    assert instance.is_frozen is False
