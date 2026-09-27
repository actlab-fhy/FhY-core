"""Tests for the Python interface over the Rust-backed type checker.

The checker, the sort tables, the body check and the registry sweep run in
the Rust core (S11b of ``docs/design/python-switch.md``).
``ExpressionTypeChecker`` is a ``CompilerPass`` over it, and the core
calls back the two lookups: the identifier types once per identifier
occurrence, and the call-target resolver once per call node, unless the
resolver is the registry's ``get_registered_entry``, which resolves
through the registry without a Python call. These tests cover what the
binding adds around the core: the pass shape, the lookups' protocol, the
objects handed back, the error classes and their framing, the body pass
and the sweep's report, and a deep expression.
"""

import time
from typing import Any

import pytest

from fhy_core.diagnostic import DiagnosticLevel, ValidationReport
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, VisitablePass
from fhy_core.serialization import SerializedDict
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    EntryLookupError,
    EntryRegistrationError,
    Expression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    get_registered_entry,
    pformat_expression,
    register_function,
)
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)
from fhy_core.types.checking import (
    ExpressionTypeChecker,
    RegisteredFunctionBodyTypeChecker,
    check_all_registered_function_bodies,
    check_expression_type,
    get_core_data_type_from_literal_type,
    get_result_core_data_type_for_sort,
    is_core_data_type_compatible_with_sort,
    synthesize_expression_type,
)
from fhy_core.utils.override import override

from .conftest import mock_identifier


def _scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


class _Recorder:
    """An identifier lookup over fixed bindings, recording each call."""

    def __init__(self, bindings: dict[Identifier, tuple[Type, TypeQualifier]]) -> None:
        self.bindings = bindings
        self.calls: list[Identifier] = []

    def __call__(self, identifier: Identifier) -> tuple[Type, TypeQualifier]:
        self.calls.append(identifier)
        return self.bindings[identifier]


class _CountingResolver:
    """A call-target resolver delegating to the registry, counting calls."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def __call__(self, name: str) -> Any:
        self.names.append(name)
        return get_registered_entry(name)


class _Opaque(Type):
    """A Python-defined type no rule of the checker accepts as a value."""

    @override
    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    @override
    def deserialize_data_from_dict(
        cls, data: SerializedDict
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


# ===========================================================================
# The pass
# ===========================================================================


def test_checker_is_a_compiler_pass_but_no_visitable_pass() -> None:
    """Test `ExpressionTypeChecker` is a plain `CompilerPass` (T-14)."""
    assert issubclass(ExpressionTypeChecker, CompilerPass)
    assert not issubclass(ExpressionTypeChecker, VisitablePass)


def test_subclass_defining_a_visit_method_is_refused() -> None:
    """Test a subclass with a per-node hook is refused when it is created."""
    with pytest.raises(TypeError, match=r"visit_binary_expression.*D-S11-20"):

        class _Hooked(ExpressionTypeChecker):  # pyright: ignore[reportUnusedClass]
            def visit_binary_expression(self, node: Any) -> Any:
                return node


def test_subclass_without_visit_methods_is_accepted() -> None:
    """Test a subclass that adds no per-node hook checks as the base does."""

    class _Quiet(ExpressionTypeChecker):
        pass

    x = mock_identifier("x", 0)
    checker = _Quiet(
        _Recorder({x: (_scalar(CoreDataType.INT8), TypeQualifier.PARAM)}),
        resolve_call_target=get_registered_entry,
    )

    result_type, _ = checker.synthesize(IdentifierExpression(x))

    assert result_type == _scalar(CoreDataType.INT8)


def test_pass_call_synthesizes() -> None:
    """Test calling the pass synthesizes, as `synthesize` and `visit` do."""
    x = mock_identifier("x", 0)
    checker = ExpressionTypeChecker(
        _Recorder({x: (_scalar(CoreDataType.INT16), TypeQualifier.INPUT)}),
        resolve_call_target=get_registered_entry,
    )
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
    )

    expected = (_scalar(CoreDataType.INT16), TypeQualifier.TEMP)
    assert checker(expression) == expected
    assert checker.synthesize(expression) == expected
    assert checker.visit(expression) == expected


# ===========================================================================
# The identifier lookup
# ===========================================================================


def test_identifier_lookup_is_called_once_per_occurrence_with_the_callers_objects() -> (
    None
):
    """Test the lookup sees each occurrence, with the identifier object used."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    lookup = _Recorder(
        {
            x: (_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            y: (_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    reference = IdentifierExpression(x)
    expression = BinaryExpression(
        BinaryOperation.MULTIPLY,
        BinaryExpression(BinaryOperation.ADD, reference, IdentifierExpression(y)),
        reference,
    )

    synthesize_expression_type(expression, lookup)

    assert [identifier.name_hint for identifier in lookup.calls] == ["x", "y", "x"]
    assert lookup.calls[0] is x
    assert lookup.calls[1] is y


def test_identifier_type_is_handed_back_as_the_looked_up_object() -> None:
    """Test a looked-up type comes back as the same object."""
    x = mock_identifier("x", 0)
    index = IndexType(LiteralExpression(0), LiteralExpression(8), LiteralExpression(1))

    result_type, qualifier = synthesize_expression_type(
        IdentifierExpression(x), _Recorder({x: (index, TypeQualifier.STATE)})
    )

    assert result_type is index
    assert qualifier is TypeQualifier.STATE


def test_key_error_from_the_lookup_means_unbound() -> None:
    """Test a `KeyError` is an unbound identifier, framed as a type error."""
    x = mock_identifier("x", 0)

    with pytest.raises(FhYCoreTypeError, match=r"is not bound"):
        synthesize_expression_type(IdentifierExpression(x), _Recorder({}))


@pytest.mark.parametrize("error_class", [RuntimeError, KeyboardInterrupt])
def test_other_lookup_exception_propagates_as_the_same_object(
    error_class: type[BaseException],
) -> None:
    """Test any other exception of the lookup propagates unchanged."""
    error = error_class("lookup failed")

    def lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
        raise error

    with pytest.raises(error_class) as exc_info:
        synthesize_expression_type(
            IdentifierExpression(mock_identifier("x", 0)), lookup
        )

    assert exc_info.value is error


@pytest.mark.parametrize(
    "result",
    [
        _scalar(CoreDataType.INT32),
        (_scalar(CoreDataType.INT32),),
        (object(), TypeQualifier.PARAM),
        (_scalar(CoreDataType.INT32), "param"),
    ],
    ids=["bare_type", "one_tuple", "no_type", "no_qualifier"],
)
def test_lookup_result_of_the_wrong_shape_raises_type_error(result: Any) -> None:
    """Test a lookup result that is no `(Type, TypeQualifier)` pair (T-15)."""
    with pytest.raises(TypeError, match=r"get_identifier_type"):
        synthesize_expression_type(
            IdentifierExpression(mock_identifier("x", 0)), lambda _: result
        )


def test_python_defined_type_from_the_lookup_is_no_value_type() -> None:
    """Test a Python-defined type reaches the core and is refused as a value."""
    x = mock_identifier("x", 0)

    with pytest.raises(FhYCoreTypeError, match=r"must resolve to a scalar numerical"):
        synthesize_expression_type(
            IdentifierExpression(x), _Recorder({x: (_Opaque(), TypeQualifier.PARAM)})
        )


# ===========================================================================
# The call-target resolver
# ===========================================================================


def _call_twice() -> Expression:
    return BinaryExpression(
        BinaryOperation.ADD,
        CallExpression("sqrt", (LiteralExpression(4.0),)),
        CallExpression("sqrt", (LiteralExpression(9.0),)),
    )


def test_custom_resolver_is_called_once_per_call_node() -> None:
    """Test a resolver that is not the registry's is called per call node."""
    resolver = _CountingResolver()
    checker = ExpressionTypeChecker(_Recorder({}), resolve_call_target=resolver)

    result_type, _ = checker.synthesize(_call_twice())

    assert resolver.names == ["sqrt", "sqrt"]
    assert result_type == _scalar(CoreDataType.FLOAT64)


def test_registry_resolver_agrees_with_a_custom_one() -> None:
    """Test the registry's own resolver, which takes the fast path, agrees."""
    fast = ExpressionTypeChecker(
        _Recorder({}), resolve_call_target=get_registered_entry
    )
    slow = ExpressionTypeChecker(_Recorder({}), resolve_call_target=_CountingResolver())

    assert fast.synthesize(_call_twice()) == slow.synthesize(_call_twice())


@pytest.mark.parametrize("resolver_kind", ["registry", "custom"])
def test_unknown_call_is_framed_or_deferred(resolver_kind: str) -> None:
    """Test an unknown call is a framed type error, or the raw lookup error."""
    resolver = (
        get_registered_entry if resolver_kind == "registry" else _CountingResolver()
    )
    expression = CallExpression(
        "test_binding_never_registered", (LiteralExpression(1),)
    )

    with pytest.raises(FhYCoreTypeError) as framed:
        ExpressionTypeChecker(_Recorder({}), resolve_call_target=resolver).synthesize(
            expression
        )
    with pytest.raises(EntryLookupError) as deferred:
        ExpressionTypeChecker(
            _Recorder({}), resolve_call_target=resolver, defer_on_unknown_call=True
        ).synthesize(expression)

    assert str(framed.value) == (
        "type error while inferring the type of "
        "`test_binding_never_registered(1)`: call to unknown function "
        "'test_binding_never_registered': No entry is registered under the name "
        "'test_binding_never_registered'."
    )
    assert deferred.value.args == (
        "No entry is registered under the name 'test_binding_never_registered'.",
    )


def test_deferred_unknown_call_raises_the_resolvers_own_error() -> None:
    """Test a custom resolver's `EntryLookupError` propagates as the same object."""
    error = EntryLookupError("nothing here")

    def resolver(name: str) -> Any:
        raise error

    checker = ExpressionTypeChecker(
        _Recorder({}), resolve_call_target=resolver, defer_on_unknown_call=True
    )

    with pytest.raises(EntryLookupError) as exc_info:
        checker.synthesize(CallExpression("f", (LiteralExpression(1),)))

    assert exc_info.value is error


def test_other_resolver_exception_propagates_as_the_same_object() -> None:
    """Test a resolver's other exception propagates unchanged."""
    error = ValueError("resolver failed")

    def resolver(name: str) -> Any:
        raise error

    checker = ExpressionTypeChecker(_Recorder({}), resolve_call_target=resolver)

    with pytest.raises(ValueError) as exc_info:
        checker.synthesize(CallExpression("f", (LiteralExpression(1),)))

    assert exc_info.value is error


def test_resolver_result_that_is_no_entry_raises_type_error() -> None:
    """Test a resolver returning something other than an entry (T-15)."""

    def resolver(name: str) -> Any:
        return 42

    checker = ExpressionTypeChecker(_Recorder({}), resolve_call_target=resolver)

    with pytest.raises(TypeError, match=r"resolve_call_target result must be"):
        checker.synthesize(CallExpression("f", (LiteralExpression(1),)))


def test_resolved_constant_cannot_be_called() -> None:
    """Test a call whose target is a constant is refused."""
    checker = ExpressionTypeChecker(
        _Recorder({}), resolve_call_target=_CountingResolver()
    )

    with pytest.raises(FhYCoreTypeError, match=r"'pi' is a registered constant"):
        checker.synthesize(CallExpression("pi", ()))


# ===========================================================================
# Arguments and errors
# ===========================================================================


def test_non_type_expected_type_raises_type_error() -> None:
    """Test `check` refuses an expected type that is no `Type`."""
    with pytest.raises(TypeError, match=r"expected_type must be a Type"):
        check_expression_type(
            LiteralExpression(1),
            CoreDataType.INT32,  # type: ignore[arg-type]
            _Recorder({}),
        )


def test_rule_error_is_framed_with_the_sub_expression() -> None:
    """Test a rule broken below the root names both, with ids (D-S11-21)."""
    flag = mock_identifier("flag", 0)
    inner = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(flag), LiteralExpression(1)
    )
    root = BinaryExpression(BinaryOperation.MULTIPLY, inner, LiteralExpression(2))

    with pytest.raises(FhYCoreTypeError) as exc_info:
        synthesize_expression_type(
            root, _Recorder({flag: (_scalar(CoreDataType.BOOL), TypeQualifier.PARAM)})
        )

    message = str(exc_info.value)
    root_text = pformat_expression(root, show_id=True)
    inner_text = pformat_expression(inner, show_id=True)
    prefix = (
        f"type error while inferring the type of `{root_text}`"
        f" at sub-expression `{inner_text}`: "
    )
    assert message.startswith(prefix)


def test_unsupported_construct_raises_not_implemented_error() -> None:
    """Test an unsupported construct is `NotImplementedError`, still framed."""
    with pytest.raises(
        NotImplementedError,
        match=r"^type error while inferring the type of `.*`: decimal literals",
    ):
        synthesize_expression_type(LiteralExpression("1.5"), _Recorder({}))


# ===========================================================================
# The sort tables and literals
# ===========================================================================


def test_sort_tables() -> None:
    """Test the sort tables' answers and their argument checks."""
    assert is_core_data_type_compatible_with_sort(CoreDataType.UINT8, FunctionSort.NAT)
    assert not is_core_data_type_compatible_with_sort(
        CoreDataType.INT8, FunctionSort.NAT
    )
    assert get_result_core_data_type_for_sort(FunctionSort.NAT) is CoreDataType.UINT32
    with pytest.raises(TypeError, match=r"sort must be a FunctionSort"):
        get_result_core_data_type_for_sort("nat")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"core_data_type must be a CoreDataType"):
        is_core_data_type_compatible_with_sort("int8", FunctionSort.INT)  # type: ignore[arg-type]


def test_literal_type_texts() -> None:
    """Test the literal helper's texts for what it refuses."""
    with pytest.raises(NotImplementedError, match=r"^string literals are not yet"):
        get_core_data_type_from_literal_type("1")
    with pytest.raises(ValueError, match=r"^unsupported literal type: <class 'list'>$"):
        get_core_data_type_from_literal_type([1])  # type: ignore[arg-type]
    assert get_core_data_type_from_literal_type(2**100) is CoreDataType.UINT


# ===========================================================================
# The body pass and the sweep
# ===========================================================================


def test_body_pass_calls_a_custom_resolver() -> None:
    """Test the body pass resolves calls through the resolver it is given."""
    x = mock_identifier("x", 0)
    resolver = _CountingResolver()
    body_pass = RegisteredFunctionBodyTypeChecker(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        resolve_call_target=resolver,
    )

    body_pass.check(CallExpression("sqrt", (IdentifierExpression(x),)))

    assert resolver.names == ["sqrt"]


def test_body_pass_error_names_the_function_with_the_checker_error_as_cause() -> None:
    """Test a body failure's text and its cause."""
    x = mock_identifier("x", 0)
    body_pass = RegisteredFunctionBodyTypeChecker(
        name="test_binding_flag",
        parameters=(x,),
        parameter_sorts=(FunctionSort.BOOL,),
        result_sort=FunctionSort.INT,
        resolve_call_target=get_registered_entry,
    )

    with pytest.raises(EntryRegistrationError) as exc_info:
        body_pass.check(
            BinaryExpression(
                BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
            )
        )

    assert str(exc_info.value).startswith(
        "function 'test_binding_flag' body failed to type-check: type error while "
    )
    assert isinstance(exc_info.value.__cause__, FhYCoreTypeError)


def test_body_pass_refuses_mismatched_parameter_sorts() -> None:
    """Test the body pass refuses parameters and sorts of different lengths."""
    body_pass = RegisteredFunctionBodyTypeChecker(
        name="f",
        parameters=(mock_identifier("x", 0),),
        parameter_sorts=(),
        result_sort=FunctionSort.INT,
        resolve_call_target=get_registered_entry,
    )

    with pytest.raises(ValueError, match=r"1 parameters but 0 parameter sorts"):
        body_pass.check(LiteralExpression(1))


def test_sweep_report_is_one_error_per_failing_body_in_registration_order(
    function_registry_snapshot: None,
) -> None:
    """Test the sweep's report, source and messages."""
    x = mock_identifier("x", 0)
    for name in ("test_binding_sweep_b", "test_binding_sweep_a"):
        register_function(
            name,
            parameters=[x],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.NAT,
            body=IdentifierExpression(x),
        )

    report = check_all_registered_function_bodies()

    assert isinstance(report, ValidationReport)
    assert [diagnostic.level for diagnostic in report.diagnostics] == [
        DiagnosticLevel.ERROR,
        DiagnosticLevel.ERROR,
    ]
    assert {diagnostic.source for diagnostic in report.diagnostics} == {
        "fhy_core.types.checking.check_all_registered_function_bodies"
    }
    assert [diagnostic.message_text for diagnostic in report.diagnostics] == [
        f"function '{name}' body synthesized type float64 is not compatible with "
        "the declared result sort nat"
        for name in ("test_binding_sweep_b", "test_binding_sweep_a")
    ]


# ===========================================================================
# Depth
# ===========================================================================


def test_deep_expression_checks_without_recursion_error() -> None:
    """Test a 10,000-level expression checks on a small stack (T-9)."""
    x = mock_identifier("x", 0)
    expression: Expression = IdentifierExpression(x)
    for _ in range(10_000):
        expression = BinaryExpression(
            BinaryOperation.ADD, expression, LiteralExpression(1)
        )
    lookup = _Recorder({x: (_scalar(CoreDataType.INT64), TypeQualifier.PARAM)})

    result_type, qualifier = synthesize_expression_type(expression, lookup)

    assert result_type == _scalar(CoreDataType.INT64)
    assert qualifier is TypeQualifier.PARAM
    assert len(lookup.calls) == 1


def test_a_doubling_dag_of_depth_40_checks_in_under_a_second() -> None:
    """Test a DAG with 2**40 paths checks once per distinct node (F2-001)."""
    x = mock_identifier("x", 0)
    expression: Expression = IdentifierExpression(x)
    for _ in range(40):
        expression = BinaryExpression(BinaryOperation.ADD, expression, expression)
    lookup = _Recorder({x: (_scalar(CoreDataType.INT32), TypeQualifier.PARAM)})

    start = time.perf_counter()
    result_type, qualifier = synthesize_expression_type(expression, lookup)
    elapsed = time.perf_counter() - start

    assert result_type == _scalar(CoreDataType.INT32)
    assert qualifier is TypeQualifier.PARAM
    assert elapsed < 1.0
    assert len(lookup.calls) == 2
