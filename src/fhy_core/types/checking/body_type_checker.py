"""Validate that a registered function's body matches its declared result sort.

Callers invoke the check explicitly, through
:func:`check_registered_function_body`, on a function that is already in
the registry (so self-recursive bodies resolve their own call site). The
pass synthesizes the body type under a parameter-lookup that maps each
parameter identifier to the concrete core data type derived from its
sort, then checks that the synthesized core data type is compatible
with the declared ``result_sort``.

The check runs in the Rust core (D-S11-22 of
``docs/design/python-switch.md``), which reports each failure with one
lowercase line naming the function. Forward-declared calls inside the
body (calls to functions not yet registered) are tolerated: with
``defer_unresolved_calls`` set, an unresolved call name abandons the
check and the pass returns ``None``: "trust the declared target sort; the
call-site check enforces the actual signature at use time."

:func:`check_all_registered_function_bodies` applies that per-function
check to the whole registry at once, with ``defer_unresolved_calls``
off. Run after registration is complete, it is where a forward-
referencing body is finally held to its declared result sort. Deferral
is what makes registration order irrelevant; turning it off at the sweep
is what stops a misspelled callee from abandoning the check silently,
which would take every other error in that body down with it.
"""

from fhy_core.utils.override import override

__all__ = [
    "RegisteredFunctionBodyTypeChecker",
    "check_all_registered_function_bodies",
    "check_registered_function_body",
]

from collections.abc import Sequence
from typing import Any

from fhy_core import _rs
from fhy_core.diagnostic import ValidationReport
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, register_pass
from fhy_core.symbolic.expression.core import Expression
from fhy_core.symbolic.expression.registry import CallTargetResolver
from fhy_core.symbolic.expression.sort import FunctionSort


@register_pass(
    "fhy_core.types.checking.check_registered_function_body",
    "Validate that a registered function's body synthesizes a type "
    "compatible with its declared result sort.",
)
class RegisteredFunctionBodyTypeChecker(CompilerPass[Expression, None]):
    """Pass that checks a registered function's body against its result sort.

    Construct the pass with the function's registration context (name,
    parameters, parameter sorts, declared result sort), then call it on
    the body expression. The pass either returns ``None`` (validation
    succeeded, or the body forward-references an unregistered function)
    or raises :class:`EntryRegistrationError`.

    Invoke the pass either via :meth:`check` (raises
    ``EntryRegistrationError`` directly) or via the standard
    pass-framework path ``__call__`` / ``execute`` (which wraps the
    domain error in ``PassExecutionError``). The registry uses
    :meth:`check`.

    Raises:
        EntryRegistrationError: When the body synthesizes a non-scalar
            or non-numerical type; when its synthesized core data type
            is not compatible with ``result_sort``; when it references
            an identifier that is neither a declared parameter nor a
            registered native constant; when it calls an unregistered
            function and ``defer_unresolved_calls`` is off; or when type
            synthesis fails, with the checker's ``FhYCoreTypeError`` or
            ``NotImplementedError``, or the resolver's
            ``EntryLookupError``, as ``__cause__``. A resolver's other
            exceptions propagate unchanged.

    Notes:
        With ``defer_unresolved_calls`` on (the default), a
        forward-referenced call inside the body is tolerated: the pass
        returns ``None`` without raising, and the call-site type checker
        enforces the actual signature when the body is later evaluated.
        Because that abandons the whole body, it also hides any other
        error in it, so a caller running after registration is complete
        should turn deferral off.

    """

    _name: str
    _parameters: tuple[Identifier, ...]
    _parameter_sorts: tuple[FunctionSort, ...]
    _result_sort: FunctionSort
    _resolve_call_target: CallTargetResolver
    _defer_unresolved_calls: bool

    def __init__(
        self,
        name: str,
        parameters: Sequence[Identifier],
        parameter_sorts: Sequence[FunctionSort],
        result_sort: FunctionSort,
        resolve_call_target: CallTargetResolver,
        defer_unresolved_calls: bool = True,
    ) -> None:
        super().__init__()
        self._name = name
        self._parameters = tuple(parameters)
        self._parameter_sorts = tuple(parameter_sorts)
        self._result_sort = result_sort
        self._resolve_call_target = resolve_call_target
        self._defer_unresolved_calls = defer_unresolved_calls

    def check(self, body: Expression) -> None:
        """Validate ``body`` against the declared parameter and result sorts.

        Args:
            body: Body expression to check.

        Raises:
            EntryRegistrationError: When the body does not satisfy
                the result-sort contract; see the class docstring.

        """
        _rs.types_check_function_body(
            self._name,
            self._parameters,
            self._parameter_sorts,
            self._result_sort,
            body,
            self._resolve_call_target,
            self._defer_unresolved_calls,
        )

    @override
    def run_pass(self, ir: Expression) -> None:
        self.check(ir)

    @override
    def get_noop_output(self, ir: Expression) -> None:
        _ = ir

    @override
    def did_change(self, input_ir: Expression, output: None) -> bool:
        _ = (input_ir, output)
        return False


def check_registered_function_body(
    name: str,
    parameters: Sequence[Identifier],
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
    body: Expression,
    resolve_call_target: CallTargetResolver,
    defer_unresolved_calls: bool = True,
) -> None:
    """Validate a registered function's body against its declared result sort.

    Args:
        name: Function name; used in error messages.
        parameters: Formal parameter identifiers in declaration order.
        parameter_sorts: Per-parameter declared sorts.
        result_sort: Declared result sort.
        body: Body expression to check.
        resolve_call_target: Lookup for call-site name resolution.
        defer_unresolved_calls: Whether a call to an unregistered
            function abandons the check instead of failing it. ``True``
            (the default) suits a check run during registration, when
            the target may still be registered later. Pass ``False``
            once registration is complete, so a name that will never
            resolve is reported rather than silently skipped.

    Raises:
        PassExecutionError: When the body does not satisfy the declared
            result-sort contract. The underlying
            :class:`EntryRegistrationError` is available as ``__cause__``.
            See :class:`RegisteredFunctionBodyTypeChecker` for the full
            list of failure conditions.

    """
    RegisteredFunctionBodyTypeChecker(
        name=name,
        parameters=parameters,
        parameter_sorts=parameter_sorts,
        result_sort=result_sort,
        resolve_call_target=resolve_call_target,
        defer_unresolved_calls=defer_unresolved_calls,
    )(body)


def check_all_registered_function_bodies() -> ValidationReport[Any]:
    """Validate every registered function body against its declared result sort.

    Walks a snapshot of the process-wide registry, in one call into the
    Rust core, and holds the body of each :class:`RegisteredFunction`
    (the composed built-ins, then the user functions) to its declared
    result sort, resolving call sites against that same registry. Run this
    after registration is complete: a body that calls a function
    registered after it resolves normally and is held to its declared
    result sort like any other, and a call to a name that was never
    registered is reported as an error rather than skipped. Run
    early instead, and a target that has simply not been registered yet
    is reported as missing.

    Every entry is checked. A body that fails does not stop the walk,
    so one broken body cannot hide another. Callers that want the sweep
    to be fatal escalate the returned report with
    :meth:`ValidationReport.raise_if_failed`.

    Returns:
        A :class:`ValidationReport` carrying one ERROR diagnostic per
        function whose body check failed, in registration order. In
        practice that is a result-sort mismatch or a call to an
        unregistered function; any other body-check failure is reported
        the same way. Each diagnostic's message is the failure's text,
        which names the offending function. The report is empty when
        every body checks out and when no expression-bodied function is
        registered.

    """
    return _rs.types_check_all_function_bodies()
