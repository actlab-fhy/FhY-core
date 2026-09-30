"""Tests the frozen contract of every Rust-backed class of ``fhy_core._rs``.

Each class that defines the frozen methods itself stands in for a
``FrozenMixin`` subclass: its instances are always frozen, and setting or
deleting any attribute raises ``FrozenMutationError`` with the mixin's
message. The classes are enumerated from the extension, so a new one
without an entry in the table of instance builders fails the suite. Each
builder goes through the public API, so an instance is of the public
Python subclass wherever one exists.
"""

from collections.abc import Callable

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import (
    OTHER_NOTE_KIND,
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationReport,
)
from fhy_core.op_attribute import COMMUTATIVE
from fhy_core.pass_infrastructure import (
    FixpointGroupRecord,
    FixpointIterationRecord,
    PassManagerResult,
    PassResult,
    PassRunRecord,
    PreservedAnalyses,
    ValidatorRecord,
)
from fhy_core.provenance import Position, Span, UnknownProvenance
from fhy_core.symbol_table import (
    FunctionKeyword,
    FunctionSymbolTableFrame,
    ImportSymbolTableFrame,
    VariableSymbolTableFrame,
)
from fhy_core.symbolic.constraint import (
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    Expression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
)
from fhy_core.symbolic.expression.pattern import (
    Capture,
    FiredRule,
    MatchBindings,
    RewriteRule,
    WildcardPattern,
)
from fhy_core.symbolic.param import (
    CategoricalDomain,
    IntegerDomain,
    IntervalIntegerDomain,
    OrdinalDomain,
    ParamAssignment,
    PermutationDomain,
    RealDomain,
    create_integer_param,
)
from fhy_core.term import AlphaRenaming
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TypeQualifier,
)
from fhy_core.value_domain import DATA_DOMAIN, ValueDomain

from .conftest import mock_identifier


def _rewrite_to_zero(_bindings: MatchBindings) -> Expression:
    return LiteralExpression(0)


def _negate(value: float) -> float:
    return -value


def _build_int32_scalar_type() -> NumericalType:
    return NumericalType(PrimitiveDataType(CoreDataType.INT32))


_BUILD_INSTANCE_BY_RUST_CLASS_NAME: dict[str, Callable[[], object]] = {
    "AlphaRenaming": AlphaRenaming.empty,
    "Capture": lambda: Capture("x"),
    "CategoricalDomain": lambda: CategoricalDomain(("b", "a")),
    "ConstraintSystem": lambda: create_constraint_system(
        InSetConstraint(mock_identifier("x", 1), {1, 2})
    ),
    "Diagnostic": lambda: Diagnostic(DiagnosticLevel.ERROR, Note("bad"), "v"),
    "EquationConstraint": lambda: EquationConstraint(
        IdentifierExpression(mock_identifier("x", 1)) > 0
    ),
    "Expression": lambda: LiteralExpression(1),
    "FiredRule": lambda: FiredRule(0, "r"),
    "FixpointGroupRecord": lambda: FixpointGroupRecord(
        mock_identifier("g", 1), (), True
    ),
    "FixpointIterationRecord": lambda: FixpointIterationRecord(1, False, ()),
    "FunctionSymbolTableFrame": lambda: FunctionSymbolTableFrame(
        mock_identifier("f", 1),
        FunctionKeyword.PROCEDURE,
        [(TypeQualifier.INPUT, _build_int32_scalar_type())],
    ),
    "ImportSymbolTableFrame": lambda: ImportSymbolTableFrame(mock_identifier("f", 1)),
    "InSetConstraint": lambda: InSetConstraint(mock_identifier("x", 1), {1, 2}),
    "IndexType": lambda: IndexType(
        LiteralExpression(0), LiteralExpression(4), LiteralExpression(1)
    ),
    "IntegerDomain": IntegerDomain,
    "IntervalIntegerDomain": IntervalIntegerDomain,
    "MatchBindings": MatchBindings,
    "NativeConstant": lambda: NativeConstant(
        name="frozen_test_constant", sort=FunctionSort.REAL, value=2.5
    ),
    "NativeFunction": lambda: NativeFunction(
        name="frozen_test_native",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=_negate,
    ),
    "NotInSetConstraint": lambda: NotInSetConstraint(mock_identifier("x", 1), {3}),
    "Note": lambda: Note("hello", OTHER_NOTE_KIND),
    "NoteKind": lambda: OTHER_NOTE_KIND,
    "NumericalType": _build_int32_scalar_type,
    "OpAttribute": lambda: COMMUTATIVE,
    "OrdinalDomain": lambda: OrdinalDomain((1, 2.5, 3)),
    "Param": lambda: create_integer_param(name=mock_identifier("p", 1)),
    "ParamAssignment": lambda: ParamAssignment(
        create_integer_param(name=mock_identifier("p", 1)), 5
    ),
    "PassManagerResult": lambda: PassManagerResult(0, ()),
    "PassResult": lambda: PassResult(0, changed=False),
    "PassRunRecord": lambda: PassRunRecord("p", False, (), PreservedAnalyses.all()),
    "Pattern": WildcardPattern,
    "PermutationDomain": lambda: PermutationDomain(("n", "c", "h", "w")),
    "Position": lambda: Position(2, 8),
    "PreservedAnalyses": PreservedAnalyses.all,
    "PrimitiveDataType": lambda: PrimitiveDataType(CoreDataType.INT32),
    "Provenance": UnknownProvenance,
    "RealDomain": RealDomain,
    "RegisteredFunction": lambda: RegisteredFunction(
        name="frozen_test_function",
        parameters=(mock_identifier("x", 1),),
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(mock_identifier("x", 1)),
    ),
    "RewriteRule": lambda: RewriteRule(WildcardPattern(), _rewrite_to_zero),
    "Span": Span,
    "TemplateDataType": lambda: TemplateDataType(mock_identifier("T", 1)),
    "ValidationReport": ValidationReport,
    "ValidatorRecord": lambda: ValidatorRecord("v", False, ()),
    "ValueDomain": lambda: ValueDomain(
        mock_identifier("domain", 1), "a child", DATA_DOMAIN
    ),
    "VariableSymbolTableFrame": lambda: VariableSymbolTableFrame(
        mock_identifier("x", 1), _build_int32_scalar_type(), TypeQualifier.INPUT
    ),
}


def _get_frozen_rust_class_names() -> frozenset[str]:
    return frozenset(
        name
        for name, value in vars(_rs).items()
        if isinstance(value, type) and "is_frozen" in vars(value)
    )


def _get_attribute_names_to_mutate(value: object) -> tuple[str, ...]:
    slot_names = tuple(
        slot_name
        for cls in type(value).__mro__
        for slot_name in vars(cls).get("__slots__", ())
    )
    return ("anything", "is_frozen", *slot_names)


def test_every_frozen_rust_class_has_an_instance_builder() -> None:
    """Test the builder table covers exactly the classes defining the methods."""
    assert _get_frozen_rust_class_names() == set(_BUILD_INSTANCE_BY_RUST_CLASS_NAME)
    assert len(_BUILD_INSTANCE_BY_RUST_CLASS_NAME) == 45


@pytest.fixture(params=sorted(_BUILD_INSTANCE_BY_RUST_CLASS_NAME))
def frozen_value(request: pytest.FixtureRequest) -> object:
    """Return an instance of one frozen Rust-backed class, built publicly."""
    rust_class_name: str = request.param
    value = _BUILD_INSTANCE_BY_RUST_CLASS_NAME[rust_class_name]()
    assert isinstance(value, getattr(_rs, rust_class_name))
    return value


def test_value_is_a_frozen_mixin_reporting_frozen(frozen_value: object) -> None:
    """Test the value is a virtual ``FrozenMixin`` and reports being frozen."""
    assert isinstance(frozen_value, FrozenMixin)
    assert frozen_value.is_frozen is True


def test_freeze_and_assert_frozen_do_nothing(frozen_value: object) -> None:
    """Test ``freeze`` and ``assert_frozen`` return ``None`` and keep it frozen."""
    assert isinstance(frozen_value, FrozenMixin)

    freeze_result = frozen_value.freeze()  # type: ignore[func-returns-value]
    assert_result = frozen_value.assert_frozen()  # type: ignore[func-returns-value]

    assert freeze_result is None
    assert assert_result is None
    assert frozen_value.is_frozen is True


def test_setattr_raises_the_frozen_mutation_error(frozen_value: object) -> None:
    """Test setting any attribute raises ``FrozenMutationError``."""
    type_name = type(frozen_value).__name__
    for name in _get_attribute_names_to_mutate(frozen_value):
        with pytest.raises(FrozenMutationError) as caught:
            setattr(frozen_value, name, 1)

        assert isinstance(caught.value, AttributeError)
        assert str(caught.value) == f'Cannot modify "{name}" on frozen {type_name}.'


def test_delattr_raises_the_frozen_mutation_error(frozen_value: object) -> None:
    """Test deleting any attribute raises ``FrozenMutationError``."""
    type_name = type(frozen_value).__name__
    for name in _get_attribute_names_to_mutate(frozen_value):
        with pytest.raises(FrozenMutationError) as caught:
            delattr(frozen_value, name)

        assert isinstance(caught.value, AttributeError)
        assert str(caught.value) == f'Cannot delete "{name}" on frozen {type_name}.'
