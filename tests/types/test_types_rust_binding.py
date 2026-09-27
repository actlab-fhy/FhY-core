"""Tests for the Python interface over the Rust-backed type system.

The four built-in classes are thin subclasses of their ``fhy_core._rs``
classes; the environment is ``fhy_core._rs.TypeUnificationEnvironment``
under a thin subclass; and the dispatchers' rules and defaults run in the
Rust core, which calls the handlers of a Python-defined type once per such
node it meets (S11a of ``docs/design/python-switch.md``). These tests cover
what the binding adds around the core: the class structure and freezing,
value semantics, the objects handed back, argument checks, pickling and
payloads, the environment's class and objects, and how the dispatchers
drive Python-defined types.
"""

import copy
import pickle
from typing import Any, ClassVar

import pytest
from immutabledict import immutabledict

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.serialization import DeserializationValueError, SerializedDict
from fhy_core.symbolic.expression import (
    IdentifierExpression,
    LiteralExpression,
    call,
    piecewise,
)
from fhy_core.traits import (
    FrozenMixin,
    FrozenMutationError,
    StructuralEquivalence,
    VerificationError,
)
from fhy_core.types import (
    CoreDataType,
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeQualifier,
    TypeUnificationEnvironment,
    bind_data_template,
    bind_template,
    get_core_data_type_bit_width,
    is_structurally_equivalent,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
    substitute_template,
    unify,
    unify_expression,
)

# ===========================================================================
# Python-defined types
# ===========================================================================


class _Unserializable:
    """Serialization hooks a test-only type never uses."""

    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def deserialize_data_from_dict(
        cls, data: SerializedDict
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


class _Tagged(_Unserializable, Type):
    """A type wrapping another type under a tag, counting its handler calls."""

    calls: ClassVar[list[str]] = []

    def __init__(self, tag: str, inner: Type) -> None:
        super().__init__()
        self.tag = tag
        self.inner = inner


@is_structurally_equivalent.register
def _(left: _Tagged, right: object) -> bool:
    _Tagged.calls.append("is_structurally_equivalent")
    return (
        isinstance(right, _Tagged)
        and left.tag == right.tag
        and is_structurally_equivalent(left.inner, right.inner)
    )


@bind_template.register
def _(
    pattern: _Tagged, actual: Type, environment: TypeUnificationEnvironment
) -> TypeUnificationEnvironment:
    _Tagged.calls.append("bind_template")
    if not isinstance(actual, _Tagged) or pattern.tag != actual.tag:
        raise VerificationError("tag mismatch")
    return bind_template(pattern.inner, actual.inner, environment)


@substitute_template.register
def _(type_: _Tagged, environment: TypeUnificationEnvironment) -> Type:
    _Tagged.calls.append("substitute_template")
    return _Tagged(type_.tag, substitute_template(type_.inner, environment))


class _Bare(_Unserializable, Type):
    """A type with no handler registered."""


class _Opaque(_Unserializable, DataType):
    """A data type whose structural equivalence is by a label."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label


@is_structurally_equivalent.register
def _(left: _Opaque, right: object) -> bool:
    return isinstance(right, _Opaque) and left.label == right.label


class _Raising(_Unserializable, Type):
    """A type whose handlers raise or answer wrongly, as its mode says."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode


_RAISED = ArithmeticError("raised by a handler")


@bind_template.register
def _(pattern: _Raising, actual: Type, environment: TypeUnificationEnvironment) -> Any:
    if pattern.mode == "raise":
        raise _RAISED
    if pattern.mode == "interrupt":
        raise KeyboardInterrupt
    return 42


@is_structurally_equivalent.register
def _(left: _Raising, right: object) -> bool:
    raise _RAISED


class _AnnotatedEnvironment(TypeUnificationEnvironment):
    """An environment subclass with an extra attribute set in its `__init__`."""

    def __init__(self, site: str) -> None:
        super().__init__()
        self.site = site


class _RaisingData(_Unserializable, DataType):
    """A data type whose `bind_data_template` handler raises or answers wrongly."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode


@bind_data_template.register
def _(
    pattern: _RaisingData, actual: DataType, environment: TypeUnificationEnvironment
) -> Any:
    if pattern.mode == "raise":
        raise _RAISED
    if pattern.mode == "interrupt":
        raise KeyboardInterrupt
    return 42


def _int32() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.INT32)


def _array(*extents: int) -> NumericalType:
    return NumericalType(_int32(), [LiteralExpression(extent) for extent in extents])


# ===========================================================================
# Class structure and freezing
# ===========================================================================


def test_built_in_classes_extend_their_rust_classes_and_the_public_bases() -> None:
    """Test each built-in class is its `_rs` class and a public base."""
    assert issubclass(PrimitiveDataType, _rs.PrimitiveDataType)
    assert issubclass(PrimitiveDataType, DataType)
    assert issubclass(NumericalType, _rs.NumericalType)
    assert issubclass(IndexType, Type)
    assert issubclass(TypeUnificationEnvironment, _rs.TypeUnificationEnvironment)


def test_every_class_is_frozen_and_a_virtual_frozen_mixin() -> None:
    """Test built-ins, environments and bases register as `FrozenMixin`."""
    for value in (_int32(), _array(2), TypeUnificationEnvironment.empty()):
        assert isinstance(value, FrozenMixin)
        assert isinstance(value, StructuralEquivalence)
        assert value.is_frozen
        with pytest.raises(FrozenMutationError):
            value.probe = 1  # type: ignore[union-attr]


def test_a_python_defined_type_constructs_with_its_init_and_freezes_after_it() -> None:
    """Test a subclass sets attributes in `__init__` and is frozen afterwards."""
    tagged = _Tagged("dense", _array(2))

    assert tagged.tag == "dense"
    assert tagged.is_frozen
    tagged.assert_frozen()
    with pytest.raises(FrozenMutationError):
        tagged.tag = "sparse"
    assert isinstance(tagged, FrozenMixin)


# ===========================================================================
# Value semantics and objects
# ===========================================================================


def test_built_in_types_compare_and_hash_structurally() -> None:
    """Test T-1: separately built equal types are `==` and hash alike."""
    assert _array(4, 8) == _array(4, 8)
    assert hash(_array(4, 8)) == hash(_array(4, 8))
    assert _array(4, 8) != _array(4, 9)
    assert TemplateDataType(Identifier("T")) != TemplateDataType(Identifier("T"))
    assert {_array(1): "a"}[_array(1)] == "a"


def test_a_python_defined_part_keeps_its_own_equality() -> None:
    """Test a type over a Python-defined data type is equal over the same object."""
    opaque = _Opaque("x")

    assert NumericalType(opaque) == NumericalType(opaque)
    assert NumericalType(opaque) != NumericalType(_Opaque("x"))
    assert NumericalType(opaque).is_structurally_equivalent(NumericalType(_Opaque("x")))
    assert hash(NumericalType(opaque)) == hash(NumericalType(opaque))


def test_properties_return_the_objects_the_type_was_built_from() -> None:
    """Test the fields are the given objects."""
    data_type, extent = _int32(), LiteralExpression(4)
    identifier = Identifier("T")
    array = NumericalType(data_type, [extent, ...])
    template = TemplateDataType(identifier, widths=[8, 16])

    assert array.data_type is data_type
    assert array.shape[0] is extent
    assert array.shape[1] is Ellipsis
    assert template.data_type is identifier
    assert template.widths == [8, 16]
    assert PrimitiveDataType(CoreDataType.INT8).core_data_type is CoreDataType.INT8


def test_the_default_stride_is_the_literal_one() -> None:
    """Test an index type without a stride steps by `1`."""
    stride = IndexType(LiteralExpression(0), LiteralExpression(4)).stride

    assert isinstance(stride, LiteralExpression)
    assert stride.value == 1


def test_arguments_of_the_wrong_type_are_refused() -> None:
    """Test T-3: the constructors check their arguments."""
    with pytest.raises(TypeError, match="data_type must be a DataType"):
        NumericalType(CoreDataType.INT32)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="shape dimension must be an Expression or Ellipsis"
    ):
        NumericalType(_int32(), [4])
    with pytest.raises(TypeError, match="lower_bound must be an Expression"):
        IndexType(0, LiteralExpression(4))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="data_type must be an Identifier"):
        TemplateDataType("T")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="core_data_type must be a CoreDataType"):
        PrimitiveDataType("int32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        TemplateDataType(Identifier("T"), widths=[0])


def test_reprs_list_every_field() -> None:
    """Test T-2: the reprs name the class and every field."""
    identifier = Identifier("T")

    assert repr(TemplateDataType(identifier, widths=[8])) == (
        f"TemplateDataType({identifier!r}, widths=[8])"
    )
    assert repr(TemplateDataType(identifier)) == f"TemplateDataType({identifier!r})"
    assert repr(_array(4)).startswith("NumericalType(PrimitiveDataType(")


@pytest.mark.parametrize(
    "value",
    [
        PrimitiveDataType(CoreDataType.FLOAT16),
        TemplateDataType(Identifier("T"), widths=[8]),
        NumericalType(
            PrimitiveDataType(CoreDataType.INT8), [LiteralExpression(3), ...]
        ),
        IndexType(LiteralExpression(0), IdentifierExpression(Identifier("N"))),
    ],
)
def test_types_pickle_copy_and_serialize_round_trip(value: Any) -> None:
    """Test pickling, copying and the payload keep the value."""
    assert pickle.loads(pickle.dumps(value)) == value
    assert copy.deepcopy(value) == value
    family = Type if isinstance(value, Type) else DataType
    assert family.deserialize_from_dict(value.serialize_to_dict()) == value


@pytest.mark.usefixtures("v1_wire")
def test_a_wildcard_dimension_serializes_as_the_sentinel() -> None:
    """Test `Ellipsis` becomes the sentinel payload."""
    payload: Any = NumericalType(_int32(), [...]).serialize_to_dict()

    assert payload["__data__"]["shape"] == [
        {"__type__": "__numerical_type_shape_ellipsis__", "__data__": {}}
    ]


@pytest.mark.usefixtures("v1_wire")
def test_a_payload_with_a_non_positive_width_is_refused() -> None:
    """Test deserialization refuses a width of zero."""
    payload = TemplateDataType(Identifier("T"), widths=[8]).serialize_to_dict()
    payload["__data__"]["widths"] = [0]  # type: ignore[index]

    with pytest.raises(DeserializationValueError):
        DataType.deserialize_from_dict(payload)


# ===========================================================================
# The environment
# ===========================================================================


def test_environment_lookups_return_the_bound_objects() -> None:
    """Test `get_*` and the tables return the objects that were bound."""
    name, value = Identifier("N"), LiteralExpression(4)
    environment = TypeUnificationEnvironment.empty().with_expression_binding(
        name, value
    )

    assert environment.get_expression_binding(name) is value
    assert isinstance(environment.expression_bindings, immutabledict)
    assert environment.expression_bindings[name] is value
    assert environment.get_expression_binding("N") is None  # type: ignore[arg-type]


def test_environment_constructor_takes_the_three_tables() -> None:
    """Test the keyword constructor, and its argument checks."""
    t = Identifier("T")
    environment = TypeUnificationEnvironment(data_type_bindings={t: _int32()})

    assert environment.get_data_type_binding(t) == _int32()
    with pytest.raises(TypeError, match="must be an Identifier"):
        TypeUnificationEnvironment(data_type_bindings={"T": _int32()})
    with pytest.raises(TypeError, match="must be a DataType"):
        TypeUnificationEnvironment(data_type_bindings={t: 3})
    with pytest.raises(TypeError, match="must be an Expression"):
        TypeUnificationEnvironment.empty().with_expression_binding(t, 4)


def test_environments_compare_hash_and_pickle_by_their_tables() -> None:
    """Test value semantics of environments."""
    name = Identifier("N")
    left = TypeUnificationEnvironment.empty().with_expression_binding(
        name, LiteralExpression(4)
    )
    right = TypeUnificationEnvironment.empty().with_expression_binding(
        name, LiteralExpression(4)
    )

    assert left == right
    assert hash(left) == hash(right)
    assert pickle.loads(pickle.dumps(left)) == left
    assert repr(left).startswith(
        "TypeUnificationEnvironment(data_type_bindings=immutabledict({})"
    )


def test_a_subclass_and_its_attributes_survive_every_derived_environment() -> None:
    """Test D-S11-13: `with_*` and the dispatchers keep the class and extras."""
    environment = _AnnotatedEnvironment("call-7")
    t, n = Identifier("T"), Identifier("N")
    pattern = NumericalType(TemplateDataType(t), [IdentifierExpression(n)])

    extended = environment.with_expression_binding(
        Identifier("M"), LiteralExpression(1)
    )
    bound = bind_template(pattern, _array(8), environment)
    _, unified = unify(pattern, _array(8), environment)

    for derived in (extended, bound, unified):
        assert type(derived) is _AnnotatedEnvironment
        assert derived.site == "call-7"
        assert derived.is_frozen
    assert pickle.loads(pickle.dumps(bound)).site == "call-7"
    with pytest.raises(FrozenMutationError):
        environment.site = "call-8"


def test_a_binding_that_learns_nothing_returns_the_environment_itself() -> None:
    """Test binding equal types returns the given environment object."""
    environment = TypeUnificationEnvironment.empty()

    assert bind_template(_array(2), _array(2), environment) is environment


def test_substitution_returns_the_type_itself_when_nothing_is_bound() -> None:
    """Test an unchanged type comes back as the same object."""
    pattern = NumericalType(
        TemplateDataType(Identifier("T")), [IdentifierExpression(Identifier("N"))]
    )

    assert substitute_template(pattern, TypeUnificationEnvironment.empty()) is pattern


def test_bound_objects_come_back_from_substitution() -> None:
    """Test a substituted dimension is the object that was bound."""
    n = Identifier("N")
    extent = LiteralExpression(8)
    environment = TypeUnificationEnvironment.empty().with_expression_binding(n, extent)

    substituted = substitute_template(
        NumericalType(_int32(), [IdentifierExpression(n)]), environment
    )

    assert isinstance(substituted, NumericalType)
    assert substituted.shape[0] is extent


def test_substitution_and_the_occurs_check_reach_calls_and_piecewise() -> None:
    """Test T-5 through the dispatchers."""
    n = Identifier("N")
    environment = TypeUnificationEnvironment.empty().with_expression_binding(
        n, LiteralExpression(4)
    )
    dimension = call("max", IdentifierExpression(n), 1)

    substituted = substitute_template(NumericalType(_int32(), [dimension]), environment)

    assert isinstance(substituted, NumericalType)
    assert substituted.shape[0] == call("max", 4, 1)
    with pytest.raises(VerificationError, match="occurs check failed"):
        unify_expression(
            IdentifierExpression(n),
            piecewise((IdentifierExpression(n) > 0, 1), otherwise=0),
            TypeUnificationEnvironment.empty(),
        )


def test_dispatchers_refuse_an_environment_of_the_wrong_type() -> None:
    """Test a dispatcher raises `TypeError` for a non-environment."""
    with pytest.raises(
        TypeError, match="environment must be a TypeUnificationEnvironment"
    ):
        bind_template(_array(1), _array(1), {})  # type: ignore[arg-type]


# ===========================================================================
# Driving Python-defined types
# ===========================================================================


def test_the_handlers_of_a_python_defined_type_are_called_once_per_node() -> None:
    """Test the core calls a nested Python-defined node's handlers."""
    t, n = Identifier("T"), Identifier("N")
    pattern = _Tagged(
        "dense", NumericalType(TemplateDataType(t), [IdentifierExpression(n)])
    )
    actual = _Tagged("dense", _array(8))
    _Tagged.calls.clear()

    environment = bind_template(pattern, actual, TypeUnificationEnvironment.empty())
    substituted = substitute_template(pattern, environment)

    assert _Tagged.calls == ["bind_template", "substitute_template"]
    assert isinstance(substituted, _Tagged)
    assert is_structurally_equivalent(substituted, actual)


def test_a_python_defined_type_nested_in_an_environment_is_driven_by_the_core() -> None:
    """Test a Python-defined type bound in an environment compares by its handler."""
    t = Identifier("T")
    tagged = _Tagged("dense", _array(2))
    environment = TypeUnificationEnvironment.empty().with_type_binding(t, tagged)
    _Tagged.calls.clear()

    assert environment.is_structurally_equivalent(
        TypeUnificationEnvironment.empty().with_type_binding(
            t, _Tagged("dense", _array(2))
        )
    )
    assert _Tagged.calls == ["is_structurally_equivalent"]


def test_a_python_defined_type_without_handlers_takes_the_default_rules() -> None:
    """Test the defaults of an unregistered class."""
    bare = _Bare()

    assert substitute_template(bare, TypeUnificationEnvironment.empty()) is bare
    assert not is_structurally_equivalent(bare, bare)
    with pytest.raises(VerificationError, match="structural mismatch"):
        bind_template(bare, bare, TypeUnificationEnvironment.empty())
    with pytest.raises(VerificationError, match="structural mismatch"):
        unify(bare, bare, TypeUnificationEnvironment.empty())


class _Int32Alias(_Unserializable, DataType):
    """A data type structurally equivalent to ``int32``."""


@is_structurally_equivalent.register
def _(left: _Int32Alias, right: object) -> bool:
    return isinstance(right, _Int32Alias) or (
        isinstance(right, PrimitiveDataType)
        and right.core_data_type is CoreDataType.INT32
    )


def test_a_built_in_type_against_a_python_defined_one_asks_its_handler() -> None:
    """Test equivalence is symmetric: the Python-defined side answers either way."""
    tagged = _Tagged("dense", _array(2))
    _Tagged.calls.clear()

    assert not is_structurally_equivalent(_array(2), tagged)
    assert _Tagged.calls == ["is_structurally_equivalent"]
    assert is_structurally_equivalent(
        NumericalType(_int32()), NumericalType(_Int32Alias())
    )
    assert is_structurally_equivalent(
        NumericalType(_Int32Alias()), NumericalType(_int32())
    )


def test_a_python_defined_data_type_binds_to_a_template_and_back() -> None:
    """Test a Python-defined data type is bound and handed back as itself."""
    t = Identifier("T")
    opaque = _Opaque("x")

    environment = bind_data_template(
        TemplateDataType(t), opaque, TypeUnificationEnvironment.empty()
    )

    assert environment.get_data_type_binding(t) is opaque


def test_a_handler_s_exception_propagates_as_the_same_object() -> None:
    """Test an exception from a handler the core calls reaches the caller unchanged."""
    with pytest.raises(ArithmeticError) as raised:
        bind_template(
            NumericalType(_RaisingData("raise")),
            NumericalType(_int32()),
            TypeUnificationEnvironment.empty(),
        )
    assert raised.value is _RAISED


def test_a_deferred_comparison_error_is_raised_after_the_core_returns() -> None:
    """Test an exception inside an infallible comparison is raised by the call."""
    t = Identifier("T")
    left = TypeUnificationEnvironment.empty().with_type_binding(t, _Raising("raise"))
    right = TypeUnificationEnvironment.empty().with_type_binding(t, _Raising("raise"))

    with pytest.raises(ArithmeticError) as raised:
        left.is_structurally_equivalent(right)
    assert raised.value is _RAISED


def test_a_keyboard_interrupt_from_a_handler_passes_through() -> None:
    """Test a `KeyboardInterrupt` reaches the caller."""
    with pytest.raises(KeyboardInterrupt):
        bind_template(
            NumericalType(_RaisingData("interrupt")),
            NumericalType(_int32()),
            TypeUnificationEnvironment.empty(),
        )


def test_a_handler_result_of_the_wrong_type_raises_type_error() -> None:
    """Test a handler's wrong result is refused with the handler named."""
    with pytest.raises(
        TypeError, match="bind_data_template handler for _RaisingData must return"
    ):
        bind_template(
            NumericalType(_RaisingData("wrong")),
            NumericalType(_int32()),
            TypeUnificationEnvironment.empty(),
        )


def test_a_handler_for_a_subclass_of_a_built_in_serves_direct_calls_only() -> None:
    """Test T-6: the core compares a built-in node it meets itself."""

    class _Counted(NumericalType):
        """A built-in subclass with its own structural-equivalence handler."""

    seen: list[str] = []

    @is_structurally_equivalent.register
    def _(left: _Counted, right: object) -> bool:
        seen.append("counted")
        return False

    t = Identifier("T")
    counted = _Counted(_int32(), [LiteralExpression(1)])
    left = TypeUnificationEnvironment.empty().with_type_binding(t, counted)
    right = TypeUnificationEnvironment.empty().with_type_binding(t, _array(1))

    assert not is_structurally_equivalent(counted, _array(1))
    assert left.is_structurally_equivalent(right)
    assert seen == ["counted"]


# ===========================================================================
# The helpers
# ===========================================================================


def test_the_helpers_refuse_a_value_of_the_wrong_type() -> None:
    """Test the promotion helpers raise `TypeError` for a non-member."""
    with pytest.raises(TypeError, match="must be a CoreDataType"):
        get_core_data_type_bit_width("int32")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a CoreDataType"):
        is_weak_core_data_type(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a CoreDataType"):
        promote_core_data_types(CoreDataType.INT8, 8)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a TypeQualifier"):
        promote_type_qualifiers(TypeQualifier.PARAM, "param")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a bool, int or float"):
        resolve_literal_core_data_type("1", CoreDataType.INT)  # type: ignore[arg-type]


def test_the_helpers_return_the_enum_members() -> None:
    """Test the helpers hand back the Python enum members themselves."""
    assert (
        promote_core_data_types(CoreDataType.UINT, CoreDataType.INT) is CoreDataType.INT
    )
    assert resolve_literal_core_data_type(2**40, CoreDataType.INT) is CoreDataType.INT64
    assert (
        promote_type_qualifiers(TypeQualifier.PARAM, TypeQualifier.PARAM)
        is TypeQualifier.PARAM
    )
