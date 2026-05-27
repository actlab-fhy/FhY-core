"""Direct tests for ``RegisteredFunctionBodyTypeChecker``."""

from unittest.mock import patch

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    EntryRegistrationError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.expression.passes.body_type_checker import (
    RegisteredFunctionBodyTypeChecker,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.types import (
    IndexType,
    NumericalType,
    TypeQualifier,
)


def _make_int_pass(name: str = "f") -> RegisteredFunctionBodyTypeChecker:
    """Build a pass with one INT parameter ``x`` and INT result."""
    x = Identifier("x")
    return RegisteredFunctionBodyTypeChecker(
        name=name,
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
    )


def _make_less_than_body() -> BinaryExpression:
    """Build a body expression ``x < y`` over fresh INT parameters."""
    x = Identifier("x")
    y = Identifier("y")
    return BinaryExpression(
        BinaryOperation.LESS,
        IdentifierExpression(x),
        IdentifierExpression(y),
    )


def _make_two_int_pass(name: str) -> RegisteredFunctionBodyTypeChecker:
    """Build a pass with two INT parameters ``x``, ``y`` and INT result."""
    x = Identifier("x")
    y = Identifier("y")
    return RegisteredFunctionBodyTypeChecker(
        name=name,
        parameters=(x, y),
        parameter_sorts=(FunctionSort.INT, FunctionSort.INT),
        result_sort=FunctionSort.INT,
    )


def test_check_accepts_body_with_matching_result_sort(
    function_registry_snapshot: None,
) -> None:
    """Test a scalar-numerical body matching the declared result sort passes."""
    x = Identifier("x")
    checker = RegisteredFunctionBodyTypeChecker(
        name="identity",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
    )

    checker.check(IdentifierExpression(x))


def test_check_rejects_body_whose_synthesized_type_clashes_with_result_sort(
    function_registry_snapshot: None,
) -> None:
    """Test a boolean body cannot satisfy an INT result sort."""
    checker = _make_two_int_pass("lt_as_int")

    with pytest.raises(EntryRegistrationError, match="lt_as_int"):
        checker.check(_make_less_than_body())


def test_check_rejects_body_referencing_undeclared_identifier(
    function_registry_snapshot: None,
) -> None:
    """Test identifiers outside parameter list and not a registered constant fail."""
    declared = Identifier("x")
    stray = Identifier("y")
    checker = RegisteredFunctionBodyTypeChecker(
        name="captures",
        parameters=(declared,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
    )

    with pytest.raises(EntryRegistrationError, match=r"captures.*y"):
        checker.check(IdentifierExpression(stray))


def test_check_tolerates_forward_reference_to_unregistered_function(
    function_registry_snapshot: None,
) -> None:
    """Test calls to as-yet-unregistered functions inside the body are accepted."""
    x = Identifier("x")
    checker = RegisteredFunctionBodyTypeChecker(
        name="uses_forward",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
    )

    checker.check(
        CallExpression(
            function_name="not_yet_registered",
            arguments=(IdentifierExpression(x),),
        )
    )


def test_check_accepts_literal_body_when_sort_compatible(
    function_registry_snapshot: None,
) -> None:
    """Test a literal body whose value's type matches the result sort passes."""
    checker = RegisteredFunctionBodyTypeChecker(
        name="const_zero",
        parameters=(),
        parameter_sorts=(),
        result_sort=FunctionSort.INT,
    )

    checker.check(LiteralExpression(0))


def test_run_pass_surfaces_function_registration_error_unwrapped(
    function_registry_snapshot: None,
) -> None:
    """Test ``run_pass`` raises ``EntryRegistrationError`` directly."""
    checker = _make_two_int_pass("lt_as_int")

    with pytest.raises(EntryRegistrationError, match="lt_as_int"):
        checker.run_pass(_make_less_than_body())


def test_pass_framework_call_wraps_domain_error_in_pass_execution_error(
    function_registry_snapshot: None,
) -> None:
    """Test ``__call__`` wraps the domain error per the pass-framework contract."""
    checker = _make_two_int_pass("lt_as_int")

    with pytest.raises(PassExecutionError) as exc_info:
        checker(_make_less_than_body())

    assert isinstance(exc_info.value.__cause__, EntryRegistrationError)


def test_get_noop_output_returns_none(function_registry_snapshot: None) -> None:
    """Test ``get_noop_output`` is ``None``; the pass is validation-only."""
    checker = _make_int_pass()

    # ``get_noop_output`` is typed to return ``None``; calling it once is
    # sufficient to assert it does not raise. mypy rejects equality
    # comparisons against ``None`` for ``-> None`` callables, so no
    # ``assert ... is None`` is needed here.
    checker.get_noop_output(LiteralExpression(0))


def test_did_change_is_always_false(function_registry_snapshot: None) -> None:
    """Test ``did_change`` is always ``False``; the pass never rewrites the input."""
    checker = _make_int_pass()
    body = LiteralExpression(0)

    assert checker.did_change(body, None) is False


def test_pass_is_registered_under_canonical_name(
    function_registry_snapshot: None,
) -> None:
    """Test ``@register_pass`` registers the pass under its stable, qualified name."""
    pass_name = "fhy_core.expression.check_registered_function_body"

    registry = CompilerPass.get_registered_passes()

    assert pass_name in registry
    assert registry[pass_name].pass_type is RegisteredFunctionBodyTypeChecker


def test_check_rejects_body_synthesizing_non_numerical_type(
    function_registry_snapshot: None,
) -> None:
    """Test a body synthesizing a non-NumericalType raises ``EntryRegistrationError``.

    The check guards the result-sort comparison from receiving an
    ``IndexType`` or any other non-numerical type. The path is
    defensive (today's expression IR never synthesizes such types from
    plain scalar arithmetic) so the test injects the synthesized type
    via patching the type-checker's ``synthesize`` method.
    """
    checker = _make_int_pass()
    body = LiteralExpression(0)
    index_type = IndexType(LiteralExpression(0), LiteralExpression(1))

    with patch(
        "fhy_core.expression.passes.type_checker.ExpressionTypeChecker.synthesize",
        return_value=(index_type, TypeQualifier.PARAM),
    ):
        with pytest.raises(
            EntryRegistrationError, match="must synthesize a scalar numerical type"
        ):
            checker.check(body)


def test_check_rejects_body_synthesizing_non_primitive_data_type(
    function_registry_snapshot: None,
) -> None:
    """Test a body whose ``NumericalType.data_type`` is non-primitive is rejected.

    The path guards against template / tensor types entering the
    result-sort comparison. As with the previous test, today's
    expression IR cannot reach this branch from plain scalar arithmetic
    so the synthesized type is patched in.
    """
    from unittest.mock import MagicMock  # noqa: PLC0415

    checker = _make_int_pass()
    body = LiteralExpression(0)

    fake_numerical = MagicMock(spec=NumericalType)
    fake_numerical.data_type = MagicMock()  # not a PrimitiveDataType

    with patch(
        "fhy_core.expression.passes.type_checker.ExpressionTypeChecker.synthesize",
        return_value=(fake_numerical, TypeQualifier.PARAM),
    ):
        with pytest.raises(
            EntryRegistrationError, match="must synthesize a scalar numerical type"
        ):
            checker.check(body)
