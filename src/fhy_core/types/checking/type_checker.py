"""Bidirectional type checking and inference for expressions.

:func:`synthesize_expression_type` infers a type purely from the leaves of
an expression; :func:`check_expression_type` propagates an expected type
into the expression so weak literals can adopt the surrounding context,
then verifies the synthesized result is assignment-compatible with the
expected type. Both entry points return ``(Type, TypeQualifier)``.

A weak core data type (``UINT``, ``INT``, ``FLOAT``) names a family, not
a width, so it carries nothing for a literal to adopt: a literal checked
against a weak expected type keeps the weak type it synthesizes on its
own, and only a concrete expected type gives it a width. Checking against
a weak expected type therefore reduces to synthesis followed by the
assignment-compatibility check, so a literal checked against the weak
type :func:`synthesize_expression_type` returned for it succeeds and
yields that same type. Checking an expression against a concrete type it
synthesized can still fail, because checking hands that type to literals
synthesis left weak.

A weak literal stays unbounded while it stays weak: neither synthesizing
a literal nor checking it against a weak expected type range-checks its
value, though its weak type still applies (a negative integer literal
synthesizes ``INT``, which an expected ``UINT`` rejects). An integer
literal is range-checked only where it meets a concrete integer type:
as a direct operand of a binary arithmetic or comparison expression whose
other operand has a concrete numerical type, or where checking hands it a
concrete expected type, which checking passes through unary operands,
piecewise branch values, and the literal operands of an arithmetic binary
expression. Float and complex types bound no literal. Any other literal
stays unchecked even when the enclosing result becomes concrete: with
``x`` of type ``int32``, ``x`` plus the negation of ``2**200``
synthesizes ``int32`` and checks against ``int32``.

The rules run in the Rust core (S11b of ``docs/design/python-switch.md``).
A broken rule raises :class:`FhYCoreTypeError`, and a construct the
checker does not support yet (a decimal literal, a tensor operand) raises
:class:`NotImplementedError`. Every such error is framed by the expression
checked and, when different, the sub-expression where the rule failed,
with identifier ids:

- "type error while inferring the type of `<root>`: <reason>"
- "type error while inferring the type of `<root>` at sub-expression
  `<sub>`: <reason>"

The two lookups are Python callables, called once per identifier
occurrence and once per call node the walk meets; a sub-expression shared
by several parents, other than a lone identifier or literal, is checked
once, so the lookups inside it run once. Their exceptions propagate unchanged,
and a result of the wrong shape raises :class:`TypeError`. When the
call-target resolver is the registry's ``get_registered_entry``, calls
resolve through the registry without calling it.
"""

from fhy_core.utils.override import override

__all__ = [
    "ExpressionTypeChecker",
    "check_expression_type",
    "get_core_data_type_from_literal_type",
    "synthesize_expression_type",
]

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassExecutionError,
    register_pass,
)
from fhy_core.symbolic.expression.core import Expression, LiteralType
from fhy_core.symbolic.expression.registry.entries import CallTargetResolver
from fhy_core.symbolic.expression.registry.storage import get_registered_entry

from ..core import CoreDataType, Type, TypeQualifier

IdentifierTypeLookup = Callable[[Identifier], tuple[Type, TypeQualifier]]


def get_core_data_type_from_literal_type(
    literal: LiteralType | Decimal | str,
) -> CoreDataType:
    """Return the core data type assigned to a literal.

    Numeric literals (``int`` and ``float``) participate in the type
    system via the weak ``UINT``/``INT``/``FLOAT`` types; ``bool``
    literals are concrete ``BOOL``. A decimal literal, whose value is a
    ``decimal.Decimal`` (the value a float-grammar ``str`` normalizes to),
    has no core data type yet and is rejected here with
    :class:`NotImplementedError`, as is a raw ``str`` value. Callers that
    may receive decimal literals should either filter them earlier or
    catch ``NotImplementedError`` explicitly.

    Raises:
        NotImplementedError: If ``literal`` is a ``Decimal`` or a ``str``.
        ValueError: If ``literal`` is none of the supported literal types.

    """
    return _rs.get_core_data_type_from_literal_type(literal)


@register_pass(
    "fhy_core.types.checking.type_checker",
    "Bidirectionally synthesize and check expression types.",
)
class ExpressionTypeChecker(CompilerPass[Expression, tuple[Type, TypeQualifier]]):
    """Bidirectional type checker for expressions.

    Calling the pass synthesizes the type of its expression. The rules run
    in the Rust core (D-S11-20 of ``docs/design/python-switch.md``), so the
    checker has no per-node hook: a subclass defining a ``visit_*`` method
    is refused when it is created, rather than having its override ignored.

    Args:
        get_identifier_type: Callable mapping an :class:`Identifier` to
            its IR type and qualifier. Raises :class:`KeyError` to
            signal an unbound identifier; the type checker catches this
            and falls back to resolving the identifier as a registered
            ``NativeConstant`` before raising a typed-error. Any other
            exception propagates unchanged. A type supplied for a
            registered ``NativeConstant``'s canonical identifier is
            rejected as a type error instead of being honored: the
            constant's type is fixed by its sort, not by the caller.
        resolve_call_target: Callable that maps a call-site function
            name to its registered entry. Injected rather than hard-
            wired to the global registry so the type checker stays
            decoupled from registry-load ordering and is independently
            testable. Constant references are not names and do not go
            through it; they resolve by identifier identity against the
            registry's canonical constant identifiers.
        defer_on_unknown_call: When ``True``, an unresolved call name
            propagates its raw :class:`EntryLookupError` instead of being
            framed as a type error. Used by
            :class:`RegisteredFunctionBodyTypeChecker` to tolerate
            forward references inside a function body. Defaults to
            ``False`` (raise framed type errors).

    """

    _get_identifier_type: IdentifierTypeLookup
    _resolve_call_target: CallTargetResolver
    _defer_on_unknown_call: bool

    def __init__(
        self,
        get_identifier_type: IdentifierTypeLookup,
        *,
        resolve_call_target: CallTargetResolver,
        defer_on_unknown_call: bool = False,
    ) -> None:
        super().__init__()
        self._get_identifier_type = get_identifier_type
        self._resolve_call_target = resolve_call_target
        self._defer_on_unknown_call = defer_on_unknown_call

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        overrides = sorted(name for name in vars(cls) if name.startswith("visit_"))
        if overrides:
            raise TypeError(
                f"{cls.__name__} defines {', '.join(overrides)}, but "
                "ExpressionTypeChecker checks through the Rust core and has "
                "no per-node hooks (D-S11-20 of docs/design/python-switch.md)."
            )
        super().__init_subclass__(**kwargs)

    def synthesize(self, expression: Expression) -> tuple[Type, TypeQualifier]:
        """Synthesize a type for an expression."""
        return _rs.types_check_expression(
            expression,
            None,
            self._get_identifier_type,
            self._resolve_call_target,
            self._defer_on_unknown_call,
        )

    def check(
        self, expression: Expression, expected_type: Type
    ) -> tuple[Type, TypeQualifier]:
        """Check an expression against an expected type.

        A literal keeps the weak type it synthesizes on its own when the
        expected core data type is weak (``UINT``, ``INT``, ``FLOAT``);
        only a concrete expected type gives it a width.

        Raises:
            TypeError: If ``expected_type`` is not a :class:`Type`.

        """
        return _rs.types_check_expression(
            expression,
            expected_type,
            self._get_identifier_type,
            self._resolve_call_target,
            self._defer_on_unknown_call,
        )

    def visit(self, expression: Expression) -> tuple[Type, TypeQualifier]:
        """Synthesize a type for an expression, as :meth:`synthesize` does."""
        return self.synthesize(expression)

    @override
    def run_pass(self, ir: Expression) -> tuple[Type, TypeQualifier]:
        return self.synthesize(ir)

    @override
    def get_noop_output(self, ir: Expression) -> tuple[Type, TypeQualifier]:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output for {ir!r}.'
        )


def synthesize_expression_type(
    expression: Expression,
    get_identifier_type: IdentifierTypeLookup,
) -> tuple[Type, TypeQualifier]:
    """Synthesize a type for an expression.

    Args:
        expression: The expression to synthesize a type for.
        get_identifier_type: A function that returns the type of an identifier.

    Returns:
        A tuple containing the synthesized type and the type qualifier.

    """
    return _rs.types_check_expression(
        expression, None, get_identifier_type, get_registered_entry, False
    )


def check_expression_type(
    expression: Expression,
    expected_type: Type,
    get_identifier_type: IdentifierTypeLookup,
) -> tuple[Type, TypeQualifier]:
    """Check an expression against an expected type.

    Args:
        expression: The expression to check.
        expected_type: The expected type to check against.
        get_identifier_type: A function that returns the type of an identifier.

    Returns:
        A tuple containing the checked type and the type qualifier.

    """
    return _rs.types_check_expression(
        expression, expected_type, get_identifier_type, get_registered_entry, False
    )
