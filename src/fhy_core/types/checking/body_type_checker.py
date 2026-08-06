"""Validate that a registered function's body matches its declared result sort.

Callers invoke the check explicitly, through
:func:`check_registered_function_body`, on a function that is already in
the registry (so self-recursive bodies resolve their own call site). The
pass synthesizes the body type under a parameter-lookup that maps each
parameter identifier to the concrete core data type derived from its
sort, then checks that the synthesized core data type is compatible
with the declared ``result_sort``.

Forward-declared calls inside the body (calls to functions not yet
registered) are tolerated: the body checker constructs its
:class:`ExpressionTypeChecker` with ``defer_on_unknown_call=True`` so
that an unresolved call name raises :class:`EntryLookupError` directly
instead of being framed as a type error. The body checker catches the
raw lookup error and returns ``None``: "trust the declared target sort;
the call-site check enforces the actual signature at use time."

:func:`check_all_registered_function_bodies` applies that per-function
check to the whole registry at once. Run after registration is
complete, it is where a forward-referencing body is finally held to its
declared result sort: every call target it names is resolvable by then,
so nothing is skipped.
"""

from fhy_core.utils.override import override

__all__ = [
    "RegisteredFunctionBodyTypeChecker",
    "check_all_registered_function_bodies",
    "check_registered_function_body",
]

from collections.abc import Sequence
from typing import Any

from immutabledict import immutabledict

from fhy_core.diagnostic import (
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassExecutionError,
    register_pass,
)
from fhy_core.symbolic.expression.core import Expression
from fhy_core.symbolic.expression.errors import (
    EntryLookupError,
    EntryRegistrationError,
)
from fhy_core.symbolic.expression.registry import (
    CallTargetResolver,
    RegisteredFunction,
    get_registered_entries,
    get_registered_entry,
)
from fhy_core.symbolic.expression.sort import FunctionSort

from ..core import (
    CoreDataType,
    FhYCoreTypeError,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)
from .sort_compatibility import is_core_data_type_compatible_with_sort
from .type_checker import ExpressionTypeChecker

_REGISTRY_SWEEP_SOURCE = "fhy_core.types.checking.check_all_registered_function_bodies"

# Concrete core data types used as the parameter lookup when body-
# checking a registered function. Concrete (non-weak) types so
# downstream arithmetic on the parameter value triggers the
# type-checker's weak-literal rescue against this operand.
_BODY_CHECK_CONCRETE_TYPES: immutabledict[FunctionSort, CoreDataType] = immutabledict(
    {
        FunctionSort.BOOL: CoreDataType.BOOL,
        FunctionSort.NAT: CoreDataType.UINT32,
        FunctionSort.INT: CoreDataType.INT64,
        FunctionSort.REAL: CoreDataType.FLOAT64,
    }
)


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
            registered native constant; or when type synthesis fails
            with a recognized error (``FhYCoreTypeError``,
            ``NotImplementedError``, ``ValueError`` from an unsupported
            literal kind), wrapped with the original as ``__cause__``.

    Notes:
        Forward-referenced calls inside the body (calls to functions
        not yet registered) are tolerated: the pass returns ``None``
        without raising. The call-site type checker enforces the
        actual signature when the body is later evaluated.

    """

    _name: str
    _parameters: tuple[Identifier, ...]
    _parameter_sorts: tuple[FunctionSort, ...]
    _result_sort: FunctionSort
    _resolve_call_target: CallTargetResolver

    def __init__(
        self,
        name: str,
        parameters: Sequence[Identifier],
        parameter_sorts: Sequence[FunctionSort],
        result_sort: FunctionSort,
        resolve_call_target: CallTargetResolver,
    ) -> None:
        super().__init__()
        self._name = name
        self._parameters = tuple(parameters)
        self._parameter_sorts = tuple(parameter_sorts)
        self._result_sort = result_sort
        self._resolve_call_target = resolve_call_target

    def check(self, body: Expression) -> None:
        """Validate ``body`` against the declared parameter and result sorts.

        Args:
            body: Body expression to check.

        Raises:
            EntryRegistrationError: When the body does not satisfy
                the result-sort contract; see the class docstring.

        """
        parameter_to_type = self._make_parameter_lookup_table()
        checker = self._make_body_type_checker(parameter_to_type)
        body_type = self._synthesize_body_type(checker, body)
        if body_type is None:
            return
        self._check_body_core_data_type_against_sort(body_type)

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

    def _make_parameter_lookup_table(
        self,
    ) -> immutabledict[Identifier, tuple[NumericalType, TypeQualifier]]:
        return immutabledict(
            {
                identifier: (
                    NumericalType(PrimitiveDataType(_BODY_CHECK_CONCRETE_TYPES[sort])),
                    TypeQualifier.PARAM,
                )
                for identifier, sort in zip(
                    self._parameters, self._parameter_sorts, strict=True
                )
            }
        )

    def _make_body_type_checker(
        self,
        parameter_to_type: immutabledict[
            Identifier, tuple[NumericalType, TypeQualifier]
        ],
    ) -> ExpressionTypeChecker:
        def lookup(identifier: Identifier) -> tuple[NumericalType, TypeQualifier]:
            if identifier in parameter_to_type:
                return parameter_to_type[identifier]
            raise KeyError(identifier.name_hint)

        return ExpressionTypeChecker(
            lookup,
            resolve_call_target=self._resolve_call_target,
            defer_on_unknown_call=True,
        )

    def _synthesize_body_type(
        self,
        checker: ExpressionTypeChecker,
        body: Expression,
    ) -> NumericalType | None:
        try:
            body_type, _ = checker.synthesize(body)
        except EntryLookupError:
            return None
        except (FhYCoreTypeError, NotImplementedError, ValueError) as exc:
            raise EntryRegistrationError(
                f"Function {self._name!r} body failed to type-check: {exc}"
            ) from exc
        return body_type  # type: ignore[return-value]

    def _check_body_core_data_type_against_sort(
        self,
        body_type: NumericalType,
    ) -> None:
        if not isinstance(body_type, NumericalType) or not isinstance(
            body_type.data_type, PrimitiveDataType
        ):
            raise EntryRegistrationError(
                f"Function {self._name!r} body must synthesize a scalar "
                f"numerical type, but got {body_type}."
            )
        body_core_data_type = body_type.data_type.core_data_type
        if not is_core_data_type_compatible_with_sort(
            body_core_data_type, self._result_sort
        ):
            raise EntryRegistrationError(
                f"Function {self._name!r} body synthesized type "
                f"{body_core_data_type} is not compatible with the declared "
                f"result sort {self._result_sort}."
            )


def check_registered_function_body(
    name: str,
    parameters: Sequence[Identifier],
    parameter_sorts: Sequence[FunctionSort],
    result_sort: FunctionSort,
    body: Expression,
    resolve_call_target: CallTargetResolver,
) -> None:
    """Validate a registered function's body against its declared result sort.

    Args:
        name: Function name; used in error messages.
        parameters: Formal parameter identifiers in declaration order.
        parameter_sorts: Per-parameter declared sorts.
        result_sort: Declared result sort.
        body: Body expression to check.
        resolve_call_target: Lookup for call-site name resolution.

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
    )(body)


def check_all_registered_function_bodies() -> ValidationReport[Any]:
    """Validate every registered function body against its declared result sort.

    Walks a snapshot of the process-wide registry and runs
    :func:`check_registered_function_body` on each
    :class:`RegisteredFunction`, resolving call sites against that same
    registry. Registration is complete by the time the sweep runs, so a
    body that calls a function registered after it resolves normally
    and is held to its declared result sort like any other; the
    per-function check skips a body only while its call target is still
    absent.

    Every entry is checked. A body that fails does not stop the walk,
    so one broken body cannot hide another. Callers that want the sweep
    to be fatal escalate the returned report with
    :meth:`ValidationReport.raise_if_failed`.

    Returns:
        A :class:`ValidationReport` carrying one ERROR diagnostic per
        function whose body does not satisfy its declared result sort,
        in registration order. Each diagnostic's message names the
        offending function. The report is empty when every body checks
        out and when no expression-bodied function is registered.

    """
    diagnostics: list[Diagnostic] = []
    for entry in get_registered_entries().values():
        if not isinstance(entry, RegisteredFunction):
            continue
        try:
            check_registered_function_body(
                name=entry.name,
                parameters=entry.parameters,
                parameter_sorts=entry.parameter_sorts,
                result_sort=entry.result_sort,
                body=entry.body,
                resolve_call_target=get_registered_entry,
            )
        except PassExecutionError as exc:
            cause = exc.__cause__
            diagnostics.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    message=Note(str(cause) if cause is not None else str(exc)),
                    source=_REGISTRY_SWEEP_SOURCE,
                )
            )
    return ValidationReport(diagnostics=tuple(diagnostics))
