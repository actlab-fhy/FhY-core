"""Validate that a registered function's body matches its declared result sort.

The pass is invoked by :func:`fhy_core.expression.registry.register_function`
after the function has been inserted as a placeholder under its declared
sorts (so self-recursive bodies resolve their own call site). The pass
synthesizes the body type under a parameter-lookup that maps each
parameter identifier to the concrete core data type derived from its
sort, then checks that the synthesized core data type is compatible
with the declared ``result_sort``.

Forward-declared calls inside the body (calls to functions not yet
registered) are tolerated: synthesis raises a chain culminating in
:class:`FunctionLookupError`, which the pass treats as "trust the
declared target sort; the call-site check enforces the actual signature
at use time."
"""

__all__ = ["RegisteredFunctionBodyTypeChecker"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, register_pass
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)

from ..core import Expression
from ..errors import FunctionLookupError, FunctionRegistrationError
from ..sort import FunctionSort, is_core_data_type_compatible_with_sort

if TYPE_CHECKING:
    from .type_checker import ExpressionTypeChecker

# Concrete core data types used as the parameter lookup when body-
# checking a registered function. Concrete (rather than weak) types let
# the arithmetic weak-literal rescue triggers for numeric literals
# inside the body.
_BODY_CHECK_CONCRETE_TYPES: frozendict[FunctionSort, CoreDataType] = frozendict(
    {
        FunctionSort.BOOL: CoreDataType.BOOL,
        FunctionSort.NAT: CoreDataType.UINT32,
        FunctionSort.INT: CoreDataType.INT64,
        FunctionSort.REAL: CoreDataType.FLOAT64,
        FunctionSort.COMPLEX: CoreDataType.COMPLEX128,
    }
)


def _has_unknown_function_cause(exc: BaseException) -> bool:
    """Return whether ``exc`` (or its cause chain) is a ``FunctionLookupError``."""
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, FunctionLookupError):
            return True
        current = current.__cause__
    return False


@register_pass(
    "fhy_core.expression.check_registered_function_body",
    "Validate that a registered function's body synthesizes a type "
    "compatible with its declared result sort.",
)
class RegisteredFunctionBodyTypeChecker(CompilerPass[Expression, None]):
    """Pass that checks a registered function's body against its result sort.

    Construct the pass with the function's registration context (name,
    parameters, parameter sorts, declared result sort), then call it on
    the body expression. The pass either returns ``None`` (validation
    succeeded, or the body forward-references an unregistered function)
    or raises :class:`FunctionRegistrationError`.

    Invoke the pass either via :meth:`check` (raises
    ``FunctionRegistrationError`` directly) or via the standard
    pass-framework path ``__call__`` / ``execute`` (which wraps the
    domain error in ``PassExecutionError``). The registry uses
    :meth:`check`.

    Raises:
        FunctionRegistrationError: When the body's synthesized core data
            type is not compatible with ``result_sort``, when the body
            synthesizes a non-scalar / non-numerical type, or when the
            body references an undeclared identifier whose name does not
            match any registered constant.

    """

    _name: str
    _parameters: tuple[Identifier, ...]
    _parameter_sorts: tuple[FunctionSort, ...]
    _result_sort: FunctionSort

    def __init__(
        self,
        name: str,
        parameters: Sequence[Identifier],
        parameter_sorts: Sequence[FunctionSort],
        result_sort: FunctionSort,
    ) -> None:
        super().__init__()
        self._name = name
        self._parameters = tuple(parameters)
        self._parameter_sorts = tuple(parameter_sorts)
        self._result_sort = result_sort

    def check(self, body: Expression) -> None:
        """Validate ``body`` against the declared parameter and result sorts.

        Args:
            body: Body expression to check.

        Raises:
            FunctionRegistrationError: When the body does not satisfy
                the result-sort contract; see the class docstring.

        """
        parameter_to_type = self._make_parameter_lookup_table()
        checker = self._make_body_type_checker(parameter_to_type)
        body_type = self._synthesize_body_type(checker, body)
        if body_type is None:
            return
        self._check_body_core_data_type_against_sort(body_type)

    def run_pass(self, ir: Expression) -> None:
        self.check(ir)

    def get_noop_output(self, ir: Expression) -> None:
        _ = ir

    def did_change(self, input_ir: Expression, output: None) -> bool:
        _ = (input_ir, output)
        return False

    def _make_parameter_lookup_table(
        self,
    ) -> frozendict[Identifier, tuple[NumericalType, TypeQualifier]]:
        return frozendict(
            {
                identifier: (
                    NumericalType(PrimitiveDataType(_BODY_CHECK_CONCRETE_TYPES[sort])),
                    TypeQualifier.PARAM,
                )
                for identifier, sort in zip(self._parameters, self._parameter_sorts)
            }
        )

    def _make_body_type_checker(
        self,
        parameter_to_type: frozendict[Identifier, tuple[NumericalType, TypeQualifier]],
    ) -> "ExpressionTypeChecker":
        # The `ExpressionTypeChecker` import lives inside this method to
        # break a one-step import cycle: `type_checker` imports
        # `RegisteredFunction`, `NativeFunction`, `NativeConstant`, and
        # `get_registered_function` from `..registry`, and `..registry`
        # imports this pass at module-top to call it from
        # `register_function`. Deferring this import lets the registry
        # finish loading before `type_checker` is pulled in.
        from .type_checker import ExpressionTypeChecker  # noqa: PLC0415

        def lookup(identifier: Identifier) -> tuple[NumericalType, TypeQualifier]:
            if identifier in parameter_to_type:
                return parameter_to_type[identifier]
            raise FunctionRegistrationError(
                f"Function {self._name!r} body uses identifier "
                f"{identifier.name_hint!r} that is not declared as a parameter."
            )

        return ExpressionTypeChecker(lookup)

    def _synthesize_body_type(
        self,
        checker: "ExpressionTypeChecker",
        body: Expression,
    ) -> NumericalType | None:
        try:
            body_type, _ = checker.synthesize(body)
        except FunctionRegistrationError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _has_unknown_function_cause(exc):
                return None
            raise FunctionRegistrationError(
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
            raise FunctionRegistrationError(
                f"Function {self._name!r} body must synthesize a scalar "
                f"numerical type, but got {body_type}."
            )
        body_core_data_type = body_type.data_type.core_data_type
        if not is_core_data_type_compatible_with_sort(
            body_core_data_type, self._result_sort
        ):
            raise FunctionRegistrationError(
                f"Function {self._name!r} body synthesized type "
                f"{body_core_data_type} is not compatible with the declared "
                f"result sort {self._result_sort}."
            )
