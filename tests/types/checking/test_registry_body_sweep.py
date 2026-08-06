"""Tests for the whole-registry body-check sweep.

`check_all_registered_function_bodies` is where a function body is
finally held to its declared result sort. Registration never inspects a
body, so a body may name a call target that does not exist yet; the
per-function checker skips such a body rather than guessing. The sweep
closes that gap by running after registration is complete, when every
name a body uses is resolvable.

The sweep aggregates: it checks every entry and reports one ERROR
diagnostic per offender rather than stopping at the first. These tests
pin both halves of that contract -- that an incompatible body is
reported at all, and that reporting one does not suppress another.
"""

import math

import pytest

from fhy_core.diagnostic import DiagnosticLevel, ValidationFailedError
from fhy_core.symbolic.expression import (
    CallExpression,
    FunctionSort,
    IdentifierExpression,
    register_function,
    register_native_function,
)
from fhy_core.types.checking import check_all_registered_function_bodies

from .conftest import mock_identifier


def _sweep_error_messages() -> tuple[str, ...]:
    """Return the ERROR diagnostic messages produced by a fresh sweep."""
    report = check_all_registered_function_bodies()
    return tuple(diagnostic.message_text for diagnostic in report.errors())


def test_sweep_reports_nothing_for_a_registry_of_well_typed_bodies(
    function_registry_snapshot: None,
) -> None:
    """Test a registry whose bodies all match their result sorts is clean."""
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_identity",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )

    report = check_all_registered_function_bodies()

    assert report.has_errors() is False, report.format()


def test_sweep_accepts_a_body_whose_forward_reference_became_resolvable(
    function_registry_snapshot: None,
) -> None:
    """Test a compatible body registered before its callee passes the sweep.

    ``test_sweep_dependent`` is registered while its callee is still
    absent, so the per-function check has nothing to resolve at that
    moment. By the time the sweep runs the callee exists and the body
    is checked for real.
    """
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_dependent",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=CallExpression("test_sweep_dependency", (IdentifierExpression(x),)),
    )
    y = mock_identifier("y", 1)
    register_function(
        "test_sweep_dependency",
        parameters=[y],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(y),
    )

    report = check_all_registered_function_bodies()

    assert report.has_errors() is False, report.format()


def test_sweep_rejects_a_body_its_forward_referenced_callee_contradicts(
    function_registry_snapshot: None,
) -> None:
    """Test an incompatible forward-referencing body is eventually rejected.

    ``test_sweep_incompatible`` declares an INT result sort, but its
    body compares the (INT) call result against a literal, which
    synthesizes BOOL. Nothing can catch that at registration time --
    the callee does not exist yet -- so the sweep is the guarantee that
    the mismatch is not tolerated forever.
    """
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_incompatible",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=CallExpression("test_sweep_callee", (IdentifierExpression(x),)) < 5,
    )
    y = mock_identifier("y", 1)
    register_function(
        "test_sweep_callee",
        parameters=[y],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(y),
    )

    messages = _sweep_error_messages()

    assert len(messages) == 1
    assert "test_sweep_incompatible" in messages[0]
    assert "not compatible with the declared result sort" in messages[0]


def test_sweep_rejects_a_body_holding_two_forward_references(
    function_registry_snapshot: None,
) -> None:
    """Test a body naming two absent callees is still checked once both exist.

    ``test_sweep_two_refs`` compares the results of two calls, neither
    of which is registered when it is. Resolving only the first
    reference must not be mistaken for the body having been checked:
    the comparison result cannot satisfy the declared INT result sort,
    and the sweep must say so.
    """
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_two_refs",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=CallExpression("test_sweep_first", (IdentifierExpression(x),))
        < CallExpression("test_sweep_second", (IdentifierExpression(x),)),
    )
    for index, name in enumerate(("test_sweep_first", "test_sweep_second"), start=1):
        parameter = mock_identifier("p", index)
        register_function(
            name,
            parameters=[parameter],
            parameter_sorts=[FunctionSort.INT],
            result_sort=FunctionSort.INT,
            body=IdentifierExpression(parameter),
        )

    messages = _sweep_error_messages()

    assert len(messages) == 1
    assert "test_sweep_two_refs" in messages[0]


def test_sweep_reports_every_incompatible_sibling_sharing_a_dependency(
    function_registry_snapshot: None,
) -> None:
    """Test one incompatible body does not hide another sharing its callee.

    Both siblings forward-reference ``test_sweep_shared`` and both turn
    out incompatible once it exists. The sweep aggregates, so the first
    offender must not end the walk before the second is checked.
    """
    for index, name in enumerate(
        ("test_sweep_sibling_first", "test_sweep_sibling_second")
    ):
        parameter = mock_identifier("p", index)
        register_function(
            name,
            parameters=[parameter],
            parameter_sorts=[FunctionSort.INT],
            result_sort=FunctionSort.INT,
            body=CallExpression("test_sweep_shared", (IdentifierExpression(parameter),))
            < 5,
        )
    z = mock_identifier("z", 2)
    register_function(
        "test_sweep_shared",
        parameters=[z],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(z),
    )

    messages = _sweep_error_messages()

    assert len(messages) == 2
    assert any("test_sweep_sibling_first" in message for message in messages)
    assert any("test_sweep_sibling_second" in message for message in messages)


def test_sweep_accepts_a_body_forward_referencing_a_native_function(
    function_registry_snapshot: None,
) -> None:
    """Test a call target supplied later as a native function resolves in the sweep."""
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_dependent_on_native",
        parameters=[x],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=CallExpression("test_sweep_native", (IdentifierExpression(x),)),
    )
    register_native_function(
        "test_sweep_native",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    report = check_all_registered_function_bodies()

    assert report.has_errors() is False, report.format()


def test_sweep_skips_a_body_whose_call_target_is_never_registered(
    function_registry_snapshot: None,
) -> None:
    """Test an unresolvable call target leaves the body unjudged, not condemned.

    The sweep cannot decide a body it cannot type, and a missing call
    target is a caller error the call site reports on its own terms.
    Reporting it here would turn every partially-populated registry
    into a failure.
    """
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_dangling",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=CallExpression("test_sweep_never_registered", (IdentifierExpression(x),))
        < 5,
    )

    report = check_all_registered_function_bodies()

    assert report.has_errors() is False, report.format()


def test_sweep_diagnostics_are_error_level_and_name_the_sweep_as_their_source(
    function_registry_snapshot: None,
) -> None:
    """Test each reported body failure is an ERROR attributed to the sweep."""
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_source",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=CallExpression("test_sweep_source_callee", (IdentifierExpression(x),)) < 5,
    )
    y = mock_identifier("y", 1)
    register_function(
        "test_sweep_source_callee",
        parameters=[y],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(y),
    )

    report = check_all_registered_function_bodies()

    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert diagnostic.level is DiagnosticLevel.ERROR
    assert (
        diagnostic.source
        == "fhy_core.types.checking.check_all_registered_function_bodies"
    )


def test_sweep_report_escalates_to_a_validation_failure_on_demand(
    function_registry_snapshot: None,
) -> None:
    """Test a caller that wants the sweep fatal gets a raise from the report."""
    x = mock_identifier("x", 0)
    register_function(
        "test_sweep_fatal",
        parameters=[x],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=CallExpression("test_sweep_fatal_callee", (IdentifierExpression(x),)) < 5,
    )
    y = mock_identifier("y", 1)
    register_function(
        "test_sweep_fatal_callee",
        parameters=[y],
        parameter_sorts=[FunctionSort.INT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(y),
    )

    report = check_all_registered_function_bodies()

    with pytest.raises(ValidationFailedError, match="test_sweep_fatal"):
        report.raise_if_failed()
