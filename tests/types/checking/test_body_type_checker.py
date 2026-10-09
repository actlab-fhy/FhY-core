"""Tests for ``check_registered_function_body`` and its backing pass class."""

import pytest

from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    EntryLookupError,
    EntryRegistrationError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    get_native_constant_identifier,
    get_registered_entry,
)
from fhy_core.types.checking.body_type_checker import (
    RegisteredFunctionBodyTypeChecker,
    check_registered_function_body,
)

from ..conftest import mock_identifier


def _make_int_pass(name: str = "f") -> RegisteredFunctionBodyTypeChecker:
    """Build a pass with one INT parameter ``x`` and INT result."""
    x = mock_identifier("x", 0)
    return RegisteredFunctionBodyTypeChecker(
        name=name,
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        resolve_call_target=get_registered_entry,
    )


def _make_less_than_body() -> BinaryExpression:
    """Build a body expression ``x < y`` over fresh INT parameters."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    return BinaryExpression(
        BinaryOperation.LESS,
        IdentifierExpression(x),
        IdentifierExpression(y),
    )


def test_check_accepts_body_with_matching_result_sort(
    function_registry_snapshot: None,
) -> None:
    """Test a scalar-numerical body matching the declared result sort passes."""
    x = mock_identifier("x", 0)
    check_registered_function_body(
        name="identity",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
        resolve_call_target=get_registered_entry,
    )


def test_check_rejects_body_whose_synthesized_type_clashes_with_result_sort(
    function_registry_snapshot: None,
) -> None:
    """Test a boolean body cannot satisfy an INT result sort."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    with pytest.raises(PassExecutionError) as exc_info:
        check_registered_function_body(
            name="lt_as_int",
            parameters=(x, y),
            parameter_sorts=(FunctionSort.INT, FunctionSort.INT),
            result_sort=FunctionSort.INT,
            body=_make_less_than_body(),
            resolve_call_target=get_registered_entry,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, EntryRegistrationError)
    assert "lt_as_int" in str(cause)


def test_check_rejects_body_referencing_undeclared_identifier(
    function_registry_snapshot: None,
) -> None:
    """Test identifiers outside parameter list and not a registered constant fail."""
    import re  # noqa: PLC0415

    declared = mock_identifier("x", 0)
    stray = mock_identifier("y", 1)
    with pytest.raises(PassExecutionError) as exc_info:
        check_registered_function_body(
            name="captures",
            parameters=(declared,),
            parameter_sorts=(FunctionSort.INT,),
            result_sort=FunctionSort.INT,
            body=IdentifierExpression(stray),
            resolve_call_target=get_registered_entry,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, EntryRegistrationError)
    assert re.search(r"captures.*y", str(cause))


def test_check_tolerates_forward_reference_to_unregistered_function(
    function_registry_snapshot: None,
) -> None:
    """Test a call to an as-yet-unregistered function is accepted, not judged.

    The body checker cannot type a call it cannot resolve, so it
    declines to rule on the body at all. Holding such a body to its
    declared result sort is
    :func:`check_all_registered_function_bodies`'s job, once the target
    exists.
    """
    x = mock_identifier("x", 0)
    check_registered_function_body(
        name="uses_forward",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=CallExpression(
            function_name="not_yet_registered",
            arguments=(IdentifierExpression(x),),
        ),
        resolve_call_target=get_registered_entry,
    )


def test_check_method_tolerates_a_forward_reference_directly(
    function_registry_snapshot: None,
) -> None:
    """Test calling ``.check`` directly also tolerates an unresolved call target.

    ``check_registered_function_body`` invokes the pass through the
    ``__call__`` / ``execute`` framework path; this pins that the raw
    ``.check`` method behaves the same way.
    """
    x = mock_identifier("x", 0)
    checker = RegisteredFunctionBodyTypeChecker(
        name="uses_forward_direct",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        resolve_call_target=get_registered_entry,
    )

    checker.check(
        CallExpression(
            function_name="still_not_registered",
            arguments=(IdentifierExpression(x),),
        )
    )


def test_check_accepts_literal_body_when_sort_compatible(
    function_registry_snapshot: None,
) -> None:
    """Test a literal body whose value's type matches the result sort passes."""
    check_registered_function_body(
        name="const_zero",
        parameters=(),
        parameter_sorts=(),
        result_sort=FunctionSort.INT,
        body=LiteralExpression(0),
        resolve_call_target=get_registered_entry,
    )


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
    pass_name = "fhy_core.types.checking.check_registered_function_body"

    registry = CompilerPass.get_registered_passes()

    assert pass_name in registry
    assert registry[pass_name].pass_type is RegisteredFunctionBodyTypeChecker


def test_check_rejects_body_using_an_unsupported_construct(
    function_registry_snapshot: None,
) -> None:
    """Test a body using an unsupported construct names it as such.

    Rewrites the test that patched ``ExpressionTypeChecker.synthesize`` to
    reach the non-numerical guard: the body is checked in Rust, where that
    guard is specified by a Rust story, since no body over scalar
    parameters synthesizes a non-scalar type. The unsupported-construct
    branch it sat beside is reachable, through a decimal literal.
    """
    x = mock_identifier("x", 0)

    with pytest.raises(PassExecutionError) as exc_info:
        check_registered_function_body(
            name="f",
            parameters=(x,),
            parameter_sorts=(FunctionSort.REAL,),
            result_sort=FunctionSort.REAL,
            body=LiteralExpression("1.5"),
            resolve_call_target=get_registered_entry,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, EntryRegistrationError)
    assert "does not support" in str(cause)
    assert isinstance(cause.__cause__, NotImplementedError)


def test_check_refuses_an_unresolved_call_without_deferral(
    function_registry_snapshot: None,
) -> None:
    """Test a call no entry resolves fails the check when not deferred.

    Rewrites the test that patched ``ExpressionTypeChecker.synthesize`` to
    reach the non-primitive guard (see the previous test). The lookup's
    ``EntryLookupError`` is the registration error's cause.
    """
    x = mock_identifier("x", 0)

    with pytest.raises(PassExecutionError) as exc_info:
        check_registered_function_body(
            name="f",
            parameters=(x,),
            parameter_sorts=(FunctionSort.INT,),
            result_sort=FunctionSort.INT,
            body=CallExpression(
                "test_body_never_registered", (IdentifierExpression(x),)
            ),
            resolve_call_target=get_registered_entry,
            defer_unresolved_calls=False,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, EntryRegistrationError)
    assert "calls a function that is not registered" in str(cause)
    assert "test_body_never_registered" in str(cause)
    assert isinstance(cause.__cause__, EntryLookupError)


def test_check_accepts_body_referencing_a_native_constant(
    function_registry_snapshot: None,
) -> None:
    """Test a body may reference a constant through its canonical identifier.

    The constant is not a declared parameter, so the body checker's
    identifier lookup misses and the registry fallback has to resolve it
    by identity for the body to type at all. Passing is not raising.
    """
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")

    check_registered_function_body(
        name="scaled_by_pi",
        parameters=(x,),
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(x) * pi,
        resolve_call_target=get_registered_entry,
    )


def test_check_rejects_body_identifier_merely_named_like_a_constant(
    function_registry_snapshot: None,
) -> None:
    """Test an identifier that only shares ``pi``'s name is an undeclared identifier.

    The registry fallback resolves the canonical identifier alone, so
    this body captures a free variable and fails the check rather than
    silently typing as the constant.
    """
    x = mock_identifier("x", 0)
    pi_lookalike = mock_identifier("pi", 1152)

    with pytest.raises(PassExecutionError) as exc_info:
        check_registered_function_body(
            name="scaled_by_a_pi_lookalike",
            parameters=(x,),
            parameter_sorts=(FunctionSort.REAL,),
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(x) * pi_lookalike,
            resolve_call_target=get_registered_entry,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, EntryRegistrationError)
    assert "pi" in str(cause)
