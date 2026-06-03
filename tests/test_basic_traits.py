# mypy: disable-error-code="misc"
"""Tests the basic compiler traits."""

import enum
import pathlib
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import pytest

from fhy_core import DATA_DOMAIN
from fhy_core.diagnostic import (
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationFailedError,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.provenance import Provenance
from fhy_core.trait import (
    Equal,
    EqualMixin,
    Frozen,
    FrozenFieldTypeError,
    FrozenMixin,
    FrozenMutationError,
    FrozenValidationError,
    HasIdentifier,
    HasIdentifierMixin,
    HasProvenance,
    HasProvenanceMixin,
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

from .conftest import mock_identifier


@dataclass
class _IdentifierCarrier(HasIdentifierMixin):
    _identifier: Identifier

    def get_identifier(self) -> Identifier:
        return self._identifier


@dataclass
class _ProvenanceCarrier(HasProvenanceMixin):
    _provenance: Provenance

    def get_provenance(self) -> Provenance:
        return self._provenance


@dataclass(frozen=True)
class _FrozenNode(FrozenMixin):
    value: int
    items: tuple[int, ...]


@dataclass(frozen=True)
class _PartialOrderableValue(PartialOrderableMixin):
    value: int

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


class _NoHashEqualValue(EqualMixin):  # noqa: PLW1641
    pass


@dataclass(order=True)
class _AutoPartialOrderableValue(PartialOrderableMixin):  # type: ignore[override]
    value: int


@dataclass(order=True)
class _AutoOrderableValue(OrderableMixin):  # type: ignore[override]
    value: int


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

    def get_intern_key(self) -> str:
        return self.key


class _BaseInternedValue(InternedMixin[str]):
    def __init__(self, key: str) -> None:
        self.key = key

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

    def get_intern_key(self) -> str:
        return self.key

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

    def get_intern_key(self) -> str:
        return self.key


@dataclass
class _DataclassInternedValue(InternedMixin[str]):
    key: str
    value: int

    def __post_init__(self) -> None:
        self.register_interned_instance()

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


def test_identifier_mixin_contract() -> None:
    """Test `HasIdentifierMixin` contract."""
    carrier = _IdentifierCarrier(mock_identifier("field", 2))
    assert carrier.get_identifier().name_hint == "field"
    assert carrier.get_identifier().id == 2


def test_provenance_mixin_contract() -> None:
    """Test `HasProvenanceMixin` contract."""
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
        setattr(point, "x", 4)


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


# =============================================================================
# Field-type immutability check
# =============================================================================


def test_field_type_check_rejects_list_field() -> None:
    """Test ``FrozenFieldTypeError`` fires for a `list[int]` field."""

    @dataclass(frozen=True)
    class _BadList(FrozenMixin):
        values: list[int]  # noqa: ignore  # intentional violation

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

    # Class itself can exist without instantiation.
    assert _AbstractWithMutable is not None

    @dataclass(frozen=True)
    class _ConcreteFix(_AbstractWithMutable):
        values: tuple[int, ...]  # type: ignore[assignment]

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


# =============================================================================
# Wrap ordering with InternedMixin
# =============================================================================


class _InternedFrozenObservesFreeze(InternedMixin[str], FrozenMixin, VerifiableMixin):
    """Fixture that asserts ``is_frozen`` is true at finalize time."""

    observed_is_frozen_during_verify: bool | None = None

    def __init__(self, key: str, value: int) -> None:
        self.key = key
        self.value = value

    def get_intern_key(self) -> str:
        return self.key

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
