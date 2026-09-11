"""Expression passes that interface with SymPy."""

from fhy_core.utils.override import override

__all__ = [
    "convert_expression_to_sympy_expression",
    "convert_sympy_expression_to_expression",
    "simplify_expression",
    "substitute_sympy_expression_variables",
]

import operator
from collections.abc import Callable
from decimal import Decimal
from typing import Any, ClassVar

import sympy  # type: ignore
import sympy.logic  # type: ignore
import sympy.logic.boolalg  # type: ignore
from immutabledict import immutabledict

from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassExecutionError,
    VisitablePass,
    register_pass,
)

from ..core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    is_integer_valued_literal,
    validate_logical_operands,
)
from ..errors import (
    ComplexInfinityLiftError,
    NativeConstantBindingError,
    PartialPiecewiseError,
)
from ..registry import (
    EntryLookupError,
    NativeConstant,
    RegisteredFunction,
    get_native_constant_identifier,
    get_registered_entry,
    try_get_native_constant_for_identifier,
)


def _sympy_exp2(value: Any) -> Any:
    return sympy.Pow(2, value)


def _sympy_log2(value: Any) -> Any:
    return sympy.log(value, 2)


def _sympy_log10(value: Any) -> Any:
    return sympy.log(value, 10)


def _sympy_round(value: Any) -> Any:
    return sympy.Function("round")(value)


# Native-function name <-> sympy operator. Entries appearing in
# ``_NATIVE_FUNCTION_LIFT_DISPATCH`` below round-trip back to their
# named form. The following do not round-trip: ``log2``/``log10`` lower
# through ``sympy.log(arg, base)`` (sympy rewrites these to a Mul-of-
# logs and they lift back as that mul rather than as ``log2``/``log10``);
# ``exp2`` lowers to ``sympy.Pow(2, value)`` and lifts as ``Pow`` (or
# as ``sqrt`` when the exponent is exactly 1/2); ``round`` lowers to an
# opaque ``sympy.Function("round")`` and has no inverse lifting entry.
_NATIVE_FUNCTION_LOWER: dict[str, Callable[..., Any]] = {
    "exp": sympy.exp,
    "exp2": _sympy_exp2,
    "log": sympy.log,
    "log2": _sympy_log2,
    "log10": _sympy_log10,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "arcsin": sympy.asin,
    "arccos": sympy.acos,
    "arctan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "erf": sympy.erf,
    "round": _sympy_round,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
}

# Native-function lift dispatch: each sympy function class maps to the
# native name it lifts to.
_NATIVE_FUNCTION_LIFT_DISPATCH: tuple[tuple[type, str], ...] = (
    (sympy.exp, "exp"),
    (sympy.sin, "sin"),
    (sympy.cos, "cos"),
    (sympy.tan, "tan"),
    (sympy.asin, "arcsin"),
    (sympy.acos, "arccos"),
    (sympy.atan, "arctan"),
    (sympy.sinh, "sinh"),
    (sympy.cosh, "cosh"),
    (sympy.tanh, "tanh"),
    (sympy.erf, "erf"),
    (sympy.log, "log"),
    (sympy.floor, "floor"),
    (sympy.ceiling, "ceil"),
)

# Native-constant lowering / lifting. SymPy folds a negated ``oo`` into a
# separate ``-oo`` atom, which ``_try_lift_native_constant`` handles.
_NATIVE_CONSTANT_LOWER: dict[str, Any] = {
    "pi": sympy.pi,
    "e": sympy.E,
    "inf": sympy.oo,
    "nan": sympy.nan,
}
_NATIVE_CONSTANT_LIFT: dict[Any, str] = {
    sympy.pi: "pi",
    sympy.E: "e",
    sympy.oo: "inf",
    sympy.nan: "nan",
}


def _try_get_native_constant_sympy_value(identifier: Identifier) -> Any | None:
    """Return the sympy value for the constant ``identifier`` denotes, or ``None``.

    Resolution is by identifier identity, so an identifier that merely
    shares a constant's ``name_hint`` lowers to a sympy ``Symbol`` like
    any other free variable.
    """
    entry = try_get_native_constant_for_identifier(identifier)
    if entry is None:
        return None
    symbolic_value = _NATIVE_CONSTANT_LOWER.get(entry.name)
    if symbolic_value is not None:
        return symbolic_value
    if isinstance(entry.value, bool):
        return sympy.true if entry.value else sympy.false
    if isinstance(entry.value, int):
        return sympy.Integer(entry.value)
    return sympy.Float(entry.value)


def _split_off_prime_factor(value: int, prime: int) -> tuple[int, int]:
    """Return the multiplicity of ``prime`` in ``value`` and the remaining cofactor.

    Args:
        value: Strictly positive integer to factor.
        prime: Prime to divide out.

    Returns:
        The exponent of ``prime`` in ``value``, and ``value`` with every
        factor of ``prime`` removed.

    """
    exponent = 0
    while value % prime == 0:
        value //= prime
        exponent += 1
    return exponent, value


def _try_format_rational_as_exact_decimal(
    numerator: int, denominator: int
) -> str | None:
    """Return the exact decimal text for a non-negative rational, or ``None``.

    A rational in lowest terms has a terminating decimal expansion
    exactly when the only prime factors of its denominator are 2 and 5,
    the prime factors of ten. In that case ``n / (2**a * 5**b)`` equals
    ``n * 2**(k-a) * 5**(k-b) / 10**k`` for ``k = max(a, b)``, which is
    written exactly with ``k`` fractional digits; every other rational
    repeats forever and has no finite decimal text.

    Args:
        numerator: Non-negative numerator, in lowest terms with
            ``denominator``.
        denominator: Strictly positive denominator.

    Returns:
        Exact fixed-point decimal text, or ``None`` when the expansion
        does not terminate.

    """
    two_exponent, cofactor = _split_off_prime_factor(denominator, 2)
    five_exponent, cofactor = _split_off_prime_factor(cofactor, 5)
    if cofactor != 1:
        return None
    fractional_digits = max(two_exponent, five_exponent)
    scaled = (
        numerator
        * 2 ** (fractional_digits - two_exponent)
        * 5 ** (fractional_digits - five_exponent)
    )
    # Decimal's string constructor is exact regardless of the ambient
    # context precision, and the ``f`` format never falls back to
    # scientific notation, which the float grammar does not accept.
    return format(Decimal(f"{scaled}e-{fractional_digits}"), "f")


def _try_lift_native_constant(expr: sympy.Expr) -> Expression | None:
    """Return the IR expression for a sympy constant atom, or ``None``.

    Lifts to the registry's canonical identifier for the constant, so a
    lowered constant lifts back as the same identifier it came from and
    the round trip is idempotent. ``-oo`` lifts as the ``NEGATE`` of the
    canonical ``inf``, the IR's own spelling of a negative infinity. A
    sympy constant whose IR counterpart is not registered has no
    canonical identifier to lift to and is left to the ordinary
    dispatch.
    """
    if expr == sympy.S.NegativeInfinity:
        infinity = _try_lift_native_constant(sympy.oo)
        if infinity is None:
            return None
        return UnaryExpression(UnaryOperation.NEGATE, infinity)
    for sympy_value, name in _NATIVE_CONSTANT_LIFT.items():
        if expr == sympy_value:
            try:
                canonical = get_native_constant_identifier(name)
            except EntryLookupError:
                return None
            return IdentifierExpression(canonical)
    return None


_LOGGER = get_logger(__name__)


@register_pass(
    "fhy_core.symbolic.expression.to_sympy",
    "Lower expression IR into an equivalent SymPy expression.",
)
class ExpressionToSympyConverter(VisitablePass[Expression, Any]):
    """Transforms an expression to SymPy expression."""

    _UNARY_OPERATION_SYMPY_OPERATORS: immutabledict[
        UnaryOperation, Callable[[Any], Any]
    ] = immutabledict(
        {
            UnaryOperation.NEGATE: operator.neg,
            UnaryOperation.POSITIVE: operator.pos,
            # ``sympy.Not``, not ``operator.not_``: the latter calls ``bool()``,
            # and every SymPy object other than a ``Relational`` is truthy, so
            # it would decide the negation at lowering time and emit the
            # constant ``False`` -- discarding the operand entirely.
            UnaryOperation.LOGICAL_NOT: sympy.Not,
        }
    )
    _BINARY_OPERATION_SYMPY_OPERATORS: immutabledict[
        BinaryOperation, Callable[[Any, Any], Any]
    ] = immutabledict(
        {
            BinaryOperation.ADD: operator.add,
            BinaryOperation.SUBTRACT: operator.sub,
            BinaryOperation.MULTIPLY: operator.mul,
            BinaryOperation.DIVIDE: operator.truediv,
            BinaryOperation.FLOOR_DIVIDE: lambda x, y: sympy.floor(x / y),
            BinaryOperation.MODULO: operator.mod,
            BinaryOperation.POWER: operator.pow,
            # ``sympy.And``/``sympy.Or``, not ``operator.and_``/``operator.or_``:
            # the latter two are SymPy's ``&``/``|``, which are *bitwise* on
            # ``sympy.Integer`` operands, so a numeric operand would fold to a
            # numerically wrong literal instead of being refused. The sympy
            # constructors reject a non-Boolean operand; the bridge screens for
            # that shape before lowering so the refusal is this package's
            # ``NonBooleanLogicalOperandError`` rather than SymPy's own error.
            BinaryOperation.LOGICAL_AND: sympy.And,
            BinaryOperation.LOGICAL_OR: sympy.Or,
            BinaryOperation.EQUAL: sympy.Eq,
            BinaryOperation.NOT_EQUAL: sympy.Ne,
            BinaryOperation.LESS: operator.lt,
            BinaryOperation.LESS_EQUAL: operator.le,
            BinaryOperation.GREATER: operator.gt,
            BinaryOperation.GREATER_EQUAL: operator.ge,
        }
    )

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        return self._BINARY_OPERATION_SYMPY_OPERATORS[binary_expression.operation](
            left, right
        )

    def visit_unary_expression(
        self, unary_expression: UnaryExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        operand = self.visit(unary_expression.operand)
        return self._UNARY_OPERATION_SYMPY_OPERATORS[unary_expression.operation](
            operand
        )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        identifier = identifier_expression.identifier
        constant_value = _try_get_native_constant_sympy_value(identifier)
        if constant_value is not None:
            return constant_value
        return sympy.Symbol(self.format_identifier(identifier))

    def visit_piecewise_expression(
        self, piecewise_expression: PiecewiseExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        """Lower to a ``sympy.Piecewise`` with an unconditional trailing branch.

        Each case becomes a ``(value, condition)`` branch, in case
        order; ``otherwise`` becomes a final branch guarded by an
        unconditional ``True``, so the result is total and
        first-match-wins by construction (SymPy's ``Piecewise``
        evaluates to the first branch whose condition holds). Every
        condition must be boolean-valued; SymPy raises ``TypeError``
        for a branch condition it can determine is not a Boolean.
        """
        branches = [
            (self.visit(value), self.visit(condition))
            for condition, value in piecewise_expression.get_cases()
        ]
        branches.append((self.visit(piecewise_expression.otherwise), True))
        return sympy.Piecewise(*branches, evaluate=False)

    def visit_call_expression(
        self, call_expression: CallExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        name = call_expression.function_name
        if name in _NATIVE_FUNCTION_LOWER:
            sympy_operator = _NATIVE_FUNCTION_LOWER[name]
            sympy_arguments = [
                self.visit(argument) for argument in call_expression.arguments
            ]
            return sympy_operator(*sympy_arguments)
        raise self._make_unsupported_call_error(name)

    @staticmethod
    def _make_unsupported_call_error(name: str) -> TypeError:
        """Construct a precise error for an unsupported ``CallExpression``."""
        try:
            entry = get_registered_entry(name)
        except EntryLookupError:
            return TypeError(f"cannot lower call to unknown function {name!r} to SymPy")
        if isinstance(entry, RegisteredFunction):
            return TypeError(
                f"Cannot lower an expression-bodied function call to SymPy; "
                f"call `inline_functions` first to expand {name!r}."
            )
        if isinstance(entry, NativeConstant):
            return TypeError(
                f"call to {name!r} is to a registered native constant, which "
                f"is not callable"
            )
        return TypeError(f"native function {name!r} has no SymPy lowering registered")

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        """Lower a literal to the SymPy number its IR form denotes exactly.

        ``LiteralExpression`` gives each literal form its own precision
        contract, and each form reaches SymPy as the exact value that
        contract names:

        - ``bool`` becomes ``sympy.true``/``sympy.false``.
        - An integer -- a Python ``int`` or an integer-grammar ``str`` --
          becomes a ``sympy.Integer``. Both are in one equivalence class,
          so both have to reach SymPy as one number kind.
        - A Python ``float`` is an IEEE-754 binary value, and
          ``sympy.Float`` carries exactly that value. A non-finite one
          becomes ``oo``, ``-oo``, or ``nan``, which lift back as the
          registered ``inf``/``nan`` constants rather than as literals.
        - A float-grammar ``str`` is exact decimal, so it becomes a
          ``sympy.Rational`` built from the text, which is that decimal
          exactly. ``sympy.Float`` would instead round the text to binary,
          making ``"0.1" + "0.1" + "0.1" == "0.3"`` simplify to False for
          the same reason the binary form does.

        The Z3 bridge lowers each of those forms to the same value, so a
        single literal denotes the same number on both bridges. Past a
        single literal the two diverge: this bridge evaluates binary-float
        arithmetic in SymPy's binary floating point, while the solver seam
        reasons over it in exact rational arithmetic, so a ground
        comparison that does float arithmetic can come out differently --
        for example, ``(1e16 + 1.0) == 1e16`` simplifies to ``True`` but
        the solver seam finds it ``False``.

        A float-grammar string whose decimal value is a whole number
        (``"2."``, ``"2.0"``) yields a ``sympy.Integer``, since
        ``sympy.Rational`` normalizes a unit denominator away; lifting it
        back therefore lands in the integer bucket rather than the
        float-decimal one. An unsupported literal type raises
        ``TypeError``.
        """
        value = literal_expression.value
        if isinstance(value, bool):
            return sympy.true if value else sympy.false
        if isinstance(value, int):
            return sympy.Integer(value)
        if isinstance(value, float):
            return sympy.Float(value)
        if isinstance(value, str):
            if is_integer_valued_literal(value):
                return sympy.Integer(int(value))
            return sympy.Rational(value)
        raise TypeError(f"Unsupported literal type: {type(value)}")

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        return f"{identifier.name_hint}_{identifier.id}"

    @override
    def get_noop_output(self, ir: Expression) -> Any:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def convert_expression_to_sympy_expression(
    expression: Expression,
) -> sympy.Expr | sympy.logic.boolalg.Boolean:
    """Convert an expression to a SymPy expression.

    Screens the expression before lowering: a provably numeric operand of
    a logical connective, or a provably numeric piecewise case condition,
    is refused here rather than handed to SymPy, whose ``And``/``Or``
    raise a raw ``TypeError`` on such an operand.

    Args:
        expression: Expression to convert.

    Returns:
        SymPy expression.

    Raises:
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in ``expression`` provably denotes a number.

    """
    validate_logical_operands(expression)
    converter = ExpressionToSympyConverter()
    return converter(expression)


@register_pass(
    "fhy_core.symbolic.expression.substitute_sympy_variables",
    "Replace bound symbols in a SymPy expression via xreplace.",
)
class SympyVariableSubstitutionPass(
    CompilerPass[
        sympy.Expr | sympy.logic.boolalg.Boolean,
        sympy.Expr | sympy.logic.boolalg.Boolean,
    ]
):
    """Applies a symbol-to-value replacement, so a bridge failure is wrapped.

    An ``xreplace`` rebuilds every substituted node bottom-up, so a
    replacement can make SymPy auto-evaluate a relational it cannot
    represent -- for example a comparison against ``zoo`` (SymPy's complex
    infinity) or against NaN -- and raise a raw ``TypeError`` from deep
    inside SymPy. Running the replacement as a pass, rather than calling
    ``xreplace`` directly, lets the pass infrastructure wrap that failure
    as ``PassExecutionError`` like every other bridge failure.
    """

    def __init__(self, replacements: dict[sympy.Symbol, Any]) -> None:
        super().__init__()
        self._replacements = replacements

    @override
    def run_pass(
        self, ir: sympy.Expr | sympy.logic.boolalg.Boolean
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        return ir.xreplace(self._replacements)

    @override
    def get_noop_output(
        self, ir: sympy.Expr | sympy.logic.boolalg.Boolean
    ) -> sympy.Expr | sympy.logic.boolalg.Boolean:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def _raise_for_bound_native_constants(bound_constants: list[Identifier]) -> None:
    """Raise ``NativeConstantBindingError`` naming each already-sorted identifier."""
    if not bound_constants:
        return
    raise NativeConstantBindingError(
        f"cannot bind the native constant(s) {bound_constants}: a constant's "
        "value is fixed by the registry, and a binding for its canonical "
        "identifier is refused here because the SymPy bridge resolves the "
        "constant by identity before a substitution ever runs, silently "
        "dropping the binding rather than applying it."
    )


def _raise_if_environment_binds_a_referenced_native_constant(
    expression: Expression, environment: dict[Identifier, Expression]
) -> None:
    """Raise if ``environment`` binds a native constant ``expression`` references."""
    referenced = expression.get_free_identifiers()
    bound_constants = sorted(
        (
            identifier
            for identifier in environment
            if identifier in referenced
            and try_get_native_constant_for_identifier(identifier) is not None
        ),
        key=lambda identifier: identifier.id,
    )
    _raise_for_bound_native_constants(bound_constants)


def _raise_if_sympy_expression_binds_a_referenced_native_constant(
    sympy_expression: sympy.Expr | sympy.logic.boolalg.Boolean,
    environment: dict[Identifier, Expression],
) -> None:
    """Raise if ``environment`` binds a native constant free in ``sympy_expression``.

    A native constant's canonical identifier never lowers to a ``Symbol``:
    ``visit_identifier_expression`` resolves it to the constant's own
    SymPy value instead, so this checks the symbol name a caller's
    binding would target against ``sympy_expression``'s free symbols,
    which only matches a hand-built SymPy expression that still carries
    such a symbol.
    """
    referenced_symbol_names = frozenset(
        symbol.name for symbol in sympy_expression.free_symbols
    )
    bound_constants = sorted(
        (
            identifier
            for identifier in environment
            if ExpressionToSympyConverter.format_identifier(identifier)
            in referenced_symbol_names
            and try_get_native_constant_for_identifier(identifier) is not None
        ),
        key=lambda identifier: identifier.id,
    )
    _raise_for_bound_native_constants(bound_constants)


def substitute_sympy_expression_variables(
    sympy_expression: sympy.Expr | sympy.logic.boolalg.Boolean,
    environment: dict[Identifier, Expression],
) -> sympy.Expr | sympy.logic.boolalg.Boolean:
    """Substitute variables in a SymPy expression.

    Every binding in ``environment`` is applied simultaneously: a
    replacement value is never itself re-substituted by another binding in
    the same call, matching the IR-level ``Expression.substitute``
    semantics that this bridge must agree with. For example, substituting
    ``{x: y, y: 5}`` into ``x < 5`` yields the residual ``y < 5``, not
    ``5 < 5``.

    Args:
        sympy_expression: SymPy expression to substitute variables in.
        environment: Environment to substitute variables from.

    Returns:
        SymPy expression with substituted variables.

    Raises:
        NativeConstantBindingError: If ``environment`` binds a native
            constant's canonical identifier that is free in
            ``sympy_expression`` as a symbol.
        NonBooleanLogicalOperandError: If a replacement value in
            ``environment`` contains a ``LOGICAL_AND``, ``LOGICAL_OR``, or
            ``LOGICAL_NOT`` node whose operand, or a piecewise whose case
            condition, provably denotes a number.
        PassExecutionError: Wrapping the originating ``TypeError`` as
            ``__cause__`` if applying a replacement makes SymPy
            auto-evaluate a relational it cannot represent, for example a
            comparison against ``zoo`` or against NaN.

    """
    # SymPy can fold boolean-valued subexpressions to plain Python `bool`
    # instances (notably ``True``/``False`` after simplification of a
    # ``sympy.logic.boolalg.Boolean``). These instances lack ``.subs`` and
    # also have nothing to substitute, so we short-circuit the no-op case.
    if isinstance(sympy_expression, bool):
        return sympy_expression
    _raise_if_sympy_expression_binds_a_referenced_native_constant(
        sympy_expression, environment
    )
    # ``.subs(..., simultaneous=True)`` is deliberately avoided here.
    # Internally it masks every replacement behind a synthetic
    # ``Dummy() * Dummy()`` product before unmasking it with a final
    # ``xreplace`` -- a trick that exists solely to disambiguate bound vs.
    # free occurrences in calculus constructs (``Derivative``, ``Integral``,
    # ...) that this IR never produces (see
    # ``sympy.core.basic.Basic.subs``). When a replacement value is a
    # SymPy ``Boolean`` (e.g. ``sympy.true``, or a ``Relational`` such as
    # ``y > 0``), unmasking reconstructs a ``Mul`` with a non-``Expr``
    # argument: at best a ``SymPyDeprecationWarning`` (deprecated since
    # SymPy 1.7), at worst a hard ``TypeError`` (e.g. substituting a
    # ``Piecewise`` condition symbol, or substituting with a
    # ``Relational``, both raise outright).
    #
    # Every substitution key here is an atomic ``Symbol`` (never a
    # compound pattern), and the IR never constructs bound-variable
    # expressions, so ``xreplace`` is a safe, exact replacement: it
    # matches each tree node against the full replacement mapping in a
    # single bottom-up pass, so a replacement value is never itself
    # re-substituted by another binding -- the same simultaneous,
    # non-chaining semantics, without the deprecated masking path.
    replacements = {
        sympy.Symbol(
            ExpressionToSympyConverter.format_identifier(k)
        ): convert_expression_to_sympy_expression(v)
        for k, v in environment.items()
    }
    return SympyVariableSubstitutionPass(replacements)(sympy_expression)


@register_pass(
    "fhy_core.symbolic.expression.from_sympy",
    "Lift a SymPy expression into the FhY expression IR.",
)
class SymPyToExpressionConverter(
    CompilerPass[sympy.Expr | sympy.logic.boolalg.Boolean, Expression]
):
    """Converts a SymPy expression to an expression tree."""

    # First match wins, so a subclass entry must precede its base:
    # ``sympy.Integer`` subclasses ``sympy.Rational``, and an ``Integer``
    # placed after ``Rational`` would be lifted as a quotient instead of
    # as an integer literal.
    _EXPR_DISPATCH: ClassVar[tuple[tuple[type, str], ...]] = (
        (sympy.Piecewise, "_convert_piecewise"),
        (sympy.Add, "_convert_add"),
        (sympy.Mul, "_convert_mul"),
        (sympy.Mod, "_convert_mod"),
        (sympy.Pow, "_convert_pow"),
        (sympy.Symbol, "_convert_symbol"),
        (sympy.Integer, "_convert_integer"),
        (sympy.Float, "_convert_float"),
        (sympy.Rational, "_convert_rational"),
        (sympy.core.numbers.ComplexInfinity, "_refuse_complex_infinity"),
    )
    _BOOL_DISPATCH: ClassVar[tuple[tuple[type, str], ...]] = (
        (sympy.logic.boolalg.Not, "_convert_not"),
        (sympy.logic.boolalg.And, "_convert_and"),
        (sympy.logic.boolalg.Or, "_convert_or"),
        (sympy.logic.boolalg.Xor, "_convert_xor"),
        (sympy.logic.boolalg.Nor, "_convert_nor"),
        (sympy.logic.boolalg.Nand, "_convert_nand"),
        (sympy.core.relational.Relational, "convert_relational"),
        (sympy.logic.boolalg.Implies, "_convert_implies"),
        (sympy.logic.boolalg.BooleanTrue, "_convert_boolean_true"),
        (sympy.logic.boolalg.BooleanFalse, "_convert_boolean_false"),
    )
    _RELATIONAL_DISPATCH: ClassVar[tuple[tuple[type, str], ...]] = (
        (sympy.Equality, "_convert_equality"),
        (sympy.Unequality, "_convert_unequality"),
        (sympy.StrictLessThan, "_convert_strict_less_than"),
        (sympy.LessThan, "_convert_less_than"),
        (sympy.StrictGreaterThan, "_convert_strict_greater_than"),
        (sympy.GreaterThan, "_convert_greater_than"),
    )

    @override
    def run_pass(self, ir: sympy.Expr | sympy.logic.boolalg.Boolean) -> Expression:
        return self.convert(ir)

    @override
    def get_noop_output(
        self, ir: sympy.Expr | sympy.logic.boolalg.Boolean
    ) -> Expression:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )

    def convert(self, node: sympy.Expr | sympy.logic.boolalg.Boolean) -> Expression:
        """Convert a SymPy node.

        Args:
            node: SymPy node to convert.

        Returns:
            Expression tree.

        """
        if isinstance(node, sympy.Expr):
            return self.convert_expr(node)
        elif isinstance(node, sympy.logic.boolalg.Boolean):
            return self.convert_bool(node)
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")

    def convert_expr(self, expr: sympy.Expr) -> Expression:
        constant_lift = _try_lift_native_constant(expr)
        if constant_lift is not None:
            return constant_lift
        native_lift = self._try_lift_native_function_call(expr)
        if native_lift is not None:
            return native_lift
        sqrt_lift = self._try_lift_sqrt_pow(expr)
        if sqrt_lift is not None:
            return sqrt_lift
        return self._dispatch(expr, self._EXPR_DISPATCH, "Unsupported expression type")

    def _try_lift_native_function_call(self, expr: sympy.Expr) -> Expression | None:
        for sympy_class, native_name in _NATIVE_FUNCTION_LIFT_DISPATCH:
            if isinstance(expr, sympy_class):
                lifted_arguments = tuple(self.convert(arg) for arg in expr.args)  # type: ignore[attr-defined]
                return CallExpression(native_name, lifted_arguments)
        return None

    def _try_lift_sqrt_pow(self, expr: sympy.Expr) -> Expression | None:
        if isinstance(expr, sympy.Pow) and expr.args[1] == sympy.Rational(1, 2):
            return CallExpression("sqrt", (self.convert(expr.args[0]),))
        return None

    def convert_bool(
        self, boolean_expression: sympy.logic.boolalg.Boolean
    ) -> Expression:
        return self._dispatch(
            boolean_expression,
            self._BOOL_DISPATCH,
            "Unsupported boolean expression type",
        )

    def convert_relational(
        self, relational: sympy.core.relational.Relational
    ) -> Expression:
        """Convert a SymPy relational node to an expression.

        Args:
            relational: SymPy relational node to convert.

        Returns:
            Expression.

        """
        return self._dispatch(
            relational, self._RELATIONAL_DISPATCH, "Unsupported relational type"
        )

    def _dispatch(
        self,
        node: Any,
        table: tuple[tuple[type, str], ...],
        error_label: str,
    ) -> Expression:
        for sympy_type, method_name in table:
            if isinstance(node, sympy_type):
                method = getattr(self, method_name)
                return method(node)  # type: ignore[no-any-return]
        raise TypeError(f"{error_label}: {type(node)}")

    def _convert_add(self, add: sympy.Add) -> Expression:
        if len(add.args) == 0:
            return LiteralExpression(0)
        elif len(add.args) == 1:
            return self.convert(add.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.ADD, add
            )

    def _convert_mul(self, mul: sympy.Mul) -> Expression:
        if len(mul.args) == 0:
            return LiteralExpression(1)
        elif len(mul.args) == 1:
            return self.convert(mul.args[0])
        else:
            return self._convert_commutative_and_associative_binary_operation(
                BinaryOperation.MULTIPLY, mul
            )

    def _convert_mod(self, mod: sympy.Mod) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(BinaryOperation.MODULO, mod)

    def _convert_pow(self, pow_: sympy.Pow) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(BinaryOperation.POWER, pow_)

    def _convert_not(self, not_: sympy.logic.boolalg.Not) -> UnaryExpression:
        operand = self.convert(not_.args[0])
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, operand)

    def _convert_and(self, and_: sympy.logic.boolalg.And) -> BinaryExpression:
        return self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_AND, and_
        )

    def _convert_or(self, or_: sympy.logic.boolalg.Or) -> BinaryExpression:
        return self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_OR, or_
        )

    def _convert_xor(self, xor: sympy.logic.boolalg.Xor) -> BinaryExpression:
        left = self.convert(xor.args[0])
        right = self.convert(sympy.Xor(*xor.args[1:], evaluate=False))
        return BinaryExpression(
            BinaryOperation.LOGICAL_AND,
            BinaryExpression(BinaryOperation.LOGICAL_OR, left, right),
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                BinaryExpression(BinaryOperation.LOGICAL_AND, left, right),
            ),
        )

    def _convert_nor(self, nor: sympy.logic.boolalg.Nor) -> Expression:
        or_statement = self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_OR, nor
        )
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, or_statement)

    def _convert_nand(self, nand: sympy.logic.boolalg.Nand) -> Expression:
        and_statement = self._convert_commutative_and_associative_binary_operation(
            BinaryOperation.LOGICAL_AND, nand
        )
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, and_statement)

    def _convert_equality(self, equivalent: sympy.Equality) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.EQUAL, equivalent
        )

    def _convert_unequality(self, unequality: sympy.Unequality) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.NOT_EQUAL, unequality
        )

    def _convert_strict_less_than(
        self, strict_less_than: sympy.StrictLessThan
    ) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.LESS, strict_less_than
        )

    def _convert_less_than(self, less_than: sympy.LessThan) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.LESS_EQUAL, less_than
        )

    def _convert_strict_greater_than(
        self, strict_greater_than: sympy.StrictGreaterThan
    ) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.GREATER, strict_greater_than
        )

    def _convert_greater_than(
        self, greater_than: sympy.GreaterThan
    ) -> BinaryExpression:
        return self._convert_two_argument_binary_operation(
            BinaryOperation.GREATER_EQUAL, greater_than
        )

    def _convert_implies(
        self, implies: sympy.logic.boolalg.Implies
    ) -> BinaryExpression:
        _LOGGER.warning("encountered unsupported Implies node %r", implies)
        _ = implies
        raise NotImplementedError("Implies is not supported.")

    def _convert_boolean_true(
        self, node: sympy.logic.boolalg.BooleanTrue
    ) -> LiteralExpression:
        _ = node
        return LiteralExpression(True)

    def _convert_boolean_false(
        self, node: sympy.logic.boolalg.BooleanFalse
    ) -> LiteralExpression:
        _ = node
        return LiteralExpression(False)

    def _convert_commutative_and_associative_binary_operation(
        self,
        operation: BinaryOperation,
        sympy_operation: sympy.Expr | sympy.logic.boolalg.Boolean,
    ) -> BinaryExpression:
        left = self.convert(sympy_operation.args[0])
        right = self.convert(
            sympy_operation.func(*sympy_operation.args[1:], evaluate=False)
        )
        return BinaryExpression(operation, left, right)

    def _convert_two_argument_binary_operation(
        self,
        operation: BinaryOperation,
        sympy_operation: sympy.Expr | sympy.logic.boolalg.Boolean,
    ) -> BinaryExpression:
        NUM_REQUIRED_ARGS = 2
        if len(sympy_operation.args) != NUM_REQUIRED_ARGS:
            raise ValueError(
                "Expected a binary operation to have exactly two arguments."
            )
        left = self.convert(sympy_operation.args[0])
        right = self.convert(sympy_operation.args[1])
        return BinaryExpression(operation, left, right)

    def _convert_symbol(self, symbol: sympy.Symbol) -> IdentifierExpression:
        symbol_name = symbol.name
        last_underscore_index = symbol_name.rfind("_")
        if last_underscore_index == -1:
            raise RuntimeError(
                "When converting a symbol from SymPy to an identifier, the "
                "symbol did not contain an underscore. This typically means "
                "that the symbol was not produced by the "
                "ExpressionToSympyConverter, whose `format_identifier` "
                "encodes identifiers as '<name_hint>_<id>'."
            )
        identifier_id = int(symbol_name[last_underscore_index + 1 :])
        identifier_name_hint = symbol_name[:last_underscore_index]
        identifier = Identifier.deserialize_from_dict(
            {"id": identifier_id, "name_hint": identifier_name_hint}
        )
        return IdentifierExpression(identifier)

    def _convert_integer(self, int_: sympy.Integer) -> LiteralExpression:
        return LiteralExpression(int(int_))

    def _convert_float(self, float_: sympy.Float) -> LiteralExpression:
        return LiteralExpression(float(float_))

    def _convert_rational(self, rational: sympy.Rational) -> Expression:
        """Lift a non-integer rational to whichever exact IR form represents it.

        A rational whose decimal expansion terminates becomes a
        float-grammar string literal, the form ``LiteralExpression``
        stores as an exact ``decimal.Decimal``; that is what lets a
        decimal-string literal survive the round trip through SymPy. Every
        other rational -- ``1/3``, say -- has no finite decimal text, so
        it becomes a ``DIVIDE`` of its numerator and denominator, which is
        exact for every rational SymPy can hand over and so keeps lifting
        total.

        The float grammar is unsigned, so a negative terminating rational
        becomes a ``NEGATE`` of its magnitude's decimal text -- the IR's
        own spelling of a negative decimal. A non-terminating one carries
        the sign on its integer numerator instead.
        """
        numerator = int(rational.p)
        denominator = int(rational.q)
        decimal_text = _try_format_rational_as_exact_decimal(
            abs(numerator), denominator
        )
        if decimal_text is None:
            return BinaryExpression(
                BinaryOperation.DIVIDE,
                LiteralExpression(numerator),
                LiteralExpression(denominator),
            )
        magnitude = LiteralExpression(decimal_text)
        if numerator < 0:
            return UnaryExpression(UnaryOperation.NEGATE, magnitude)
        return magnitude

    def _refuse_complex_infinity(
        self, complex_infinity: sympy.core.numbers.ComplexInfinity
    ) -> Expression:
        """Refuse SymPy's complex infinity, which no IR expression denotes."""
        raise ComplexInfinityLiftError(
            f"cannot lift {complex_infinity!r} to an expression: SymPy folds a "
            "quotient by zero to its directionless complex infinity, and no "
            "expression denotes that value."
        )

    def _convert_piecewise(self, piecewise: sympy.Piecewise) -> Expression:
        """Lift a ``sympy.Piecewise`` to a flat ``PiecewiseExpression``.

        All branches but the last become cases, in branch order; the
        last branch's value becomes ``otherwise``. The last branch's
        condition must be ``sympy.true``: ``PiecewiseExpression`` is a
        total function, so a ``Piecewise`` whose final condition is not
        ``sympy.true`` -- meaning it does not cover its domain -- has no
        faithful IR representation and raises :class:`PartialPiecewiseError`
        rather than being silently totalized. A single-branch
        ``Piecewise`` (the only shape that survives construction with a
        non-``True`` condition) is rejected the same way; the degenerate
        case of a single branch whose condition is already ``sympy.true``
        converts directly to the branch's bare value.
        """
        branches = tuple(piecewise.args)
        if not branches:
            raise ValueError("Cannot convert an empty Piecewise expression.")
        *case_branches, (otherwise_value, final_condition) = branches
        if final_condition is not sympy.true:
            raise PartialPiecewiseError(
                "cannot lift a partial sympy.Piecewise to PiecewiseExpression: "
                f"the final branch's condition is {final_condition!r}, not "
                "sympy.true. PiecewiseExpression is a total function and has "
                "no faithful representation for a Piecewise that does not "
                "cover its domain."
            )
        if not case_branches:
            return self.convert(otherwise_value)
        conditions = [self.convert(condition) for _, condition in case_branches]
        values = [self.convert(value) for value, _ in case_branches]
        otherwise = self.convert(otherwise_value)
        return PiecewiseExpression(tuple(conditions), tuple(values), otherwise)


def convert_sympy_expression_to_expression(
    sympy_expression: sympy.Expr | sympy.logic.boolalg.Boolean,
) -> Expression:
    """Convert a SymPy expression to an expression.

    Lifting a ``sympy.Piecewise`` requires its last branch's condition
    to be ``sympy.true``: ``PiecewiseExpression`` is a total function,
    so a ``Piecewise`` whose final condition is not ``sympy.true`` --
    meaning it does not cover its domain -- has no faithful
    representation and raises :class:`PartialPiecewiseError`.

    A ``sympy.Rational`` lifts exactly: to a float-grammar string literal
    when its decimal expansion terminates, and otherwise to a ``DIVIDE``
    of its numerator and denominator. ``sympy.oo`` and ``sympy.nan``
    lift to the canonical identifiers of the registered ``inf`` and
    ``nan`` constants, and ``-oo`` to the negation of ``inf``.
    ``sympy.zoo`` is the one numeric value with no IR counterpart and
    raises :class:`ComplexInfinityLiftError`.

    Args:
        sympy_expression: SymPy expression to convert.

    Returns:
        Expression.

    Raises:
        PassExecutionError: Wrapping :class:`PartialPiecewiseError` as
            ``__cause__`` if ``sympy_expression`` contains a
            ``sympy.Piecewise`` whose final branch condition is not
            ``sympy.true``, or :class:`ComplexInfinityLiftError` if it
            contains ``sympy.zoo``.

    """
    converter = SymPyToExpressionConverter()
    return converter(sympy_expression)


def simplify_expression(
    expression: Expression, environment: dict[Identifier, Expression] | None = None
) -> Expression:
    """Simplify an expression.

    Args:
        expression: Expression to simplify.
        environment: Environment to simplify the expression in. Defaults to None.

    Returns:
        Simplified expression.

    Raises:
        NativeConstantBindingError: If ``environment`` binds a registered
            native constant's canonical identifier that ``expression``
            references. The SymPy bridge resolves such an identifier by
            identity to the constant's own value before any substitution
            runs, so the binding would otherwise be silently dropped
            rather than applied.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, provably denotes a number, counting an operand
            ``environment`` binds to one. Simplification refuses the shape
            before lowering rather than letting SymPy's ``And``/``Or``
            raise a raw ``TypeError`` on it.
        PassExecutionError: Wrapping the originating exception as
            ``__cause__``: a ``TypeError`` if substituting ``environment``
            makes SymPy auto-evaluate a relational it cannot represent
            (for example a comparison against ``zoo`` or against NaN);
            :class:`PartialPiecewiseError` if simplification yields a
            ``sympy.Piecewise`` whose final branch condition is not
            ``sympy.true``; or :class:`ComplexInfinityLiftError` if it
            yields ``sympy.zoo``, which a quotient by zero folds to.

    """
    validate_logical_operands(expression, environment)
    if environment is not None:
        _raise_if_environment_binds_a_referenced_native_constant(
            expression, environment
        )
    sympy_expression = convert_expression_to_sympy_expression(expression)
    if environment is not None:
        sympy_expression = substitute_sympy_expression_variables(
            sympy_expression, environment
        )
    _LOGGER.debug("pre-simplify=%r", sympy_expression)
    result = sympy.simplify(sympy_expression)
    _LOGGER.debug("post-simplify=%r", result)
    return convert_sympy_expression_to_expression(result)
