"""Tests for the ground simplifier and its chain, through the public Python API.

`GroundSimplifier` folds an expression with no free identifier in exact
arithmetic, with no SymPy: its result is exactly SymPy's wherever it folds,
and the expression itself wherever it cannot match SymPy. Given a fallback
it is the chain, asking the fallback for what it declines.
"""

import pickle
from collections.abc import Iterator

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
    UnaryExpression,
    UnaryOperation,
    get_native_constant_identifier,
    piecewise,
)
from fhy_core.symbolic.solver import (
    GroundSimplifier,
    Simplifier,
    Solver,
    SolverBackend,
    SolverCapabilityError,
    SolverQueryKind,
    _resolve_adapter,
    _solver_of,
    check_expression_satisfiability,
    get_backend_capabilities,
    get_default_solver,
    is_backend_available,
    set_default_solver,
    simplify_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.utils.override import override

from .conftest import mock_identifier


def _literal(value: object) -> LiteralExpression:
    return LiteralExpression(value)  # type: ignore[arg-type]


def _binary(
    operation: BinaryOperation, left: Expression, right: Expression
) -> BinaryExpression:
    return BinaryExpression(operation, left, right)


def _int(value: int) -> LiteralExpression:
    return LiteralExpression(value)


_ADD = BinaryOperation.ADD
_DIVIDE = BinaryOperation.DIVIDE
_FLOOR_DIVIDE = BinaryOperation.FLOOR_DIVIDE
_MODULO = BinaryOperation.MODULO
_POWER = BinaryOperation.POWER


class _RecordingSimplifier(Simplifier):
    """A Python simplifier recording its inputs and returning a fixed result."""

    def __init__(self, result: Expression | None = None) -> None:
        super().__init__()
        self.result = result
        self.inputs: list[Expression] = []

    @override
    def simplify(self, expression: Expression) -> Expression:
        self.inputs.append(expression)
        return expression if self.result is None else self.result


class _FailingSimplifier(Simplifier):
    @override
    def simplify(self, expression: Expression) -> Expression:
        raise ValueError("the fallback failed")


class _TimeoutRecordingSimplifier(Simplifier):
    """A Python simplifier recording the timeout it is asked under."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []

    @override
    def simplify(self, expression: Expression) -> Expression:
        self.timeouts.append(self.context.timeout)
        return expression


def _heavy_sum() -> Expression:
    """Return a sum of powers, each taking milliseconds, that nothing shares."""
    total: Expression = _int(0)
    for index in range(20):
        total = _binary(
            _ADD,
            total,
            _binary(_POWER, _int(1_000_003 + 2 * index), _int(52_000)),
        )
    return total


@pytest.fixture
def x() -> Identifier:
    return mock_identifier("x", 0)


@pytest.fixture
def restore_default_solver() -> Iterator[None]:
    original = get_default_solver()
    yield
    set_default_solver(original)


# A table of ground expressions the simplifier folds, and the literal each
# gives in SymPy's lifted form: a rational a binary float equals is a decimal
# literal, negated when negative, and any other is a quotient of integers.
_FOLDS: list[tuple[str, Expression, Expression]] = [
    ("sum", _binary(_ADD, _int(2), _int(3)), _int(5)),
    ("integer quotient", _binary(_DIVIDE, _int(6), _int(3)), _int(2)),
    ("float quotient", _binary(_DIVIDE, _int(1), _int(2)), _literal("0.5")),
    (
        "negative float quotient",
        _binary(_DIVIDE, _int(-3), _int(8)),
        UnaryExpression(UnaryOperation.NEGATE, _literal("0.375")),
    ),
    (
        "other quotient",
        _binary(_DIVIDE, _int(1), _int(3)),
        _binary(_DIVIDE, _int(1), _int(3)),
    ),
    (
        "negative other quotient",
        _binary(_DIVIDE, _int(-1), _int(3)),
        _binary(_DIVIDE, _int(-1), _int(3)),
    ),
    ("floor division", _binary(_FLOOR_DIVIDE, _int(-7), _int(2)), _int(-4)),
    ("modulo", _binary(_MODULO, _int(7), _int(-3)), _int(-2)),
    ("power", _binary(_POWER, _int(2), _int(10)), _int(1024)),
    ("negative power", _binary(_POWER, _int(2), _int(-2)), _literal("0.25")),
    (
        "exact root",
        _binary(_POWER, _int(8), _binary(_DIVIDE, _int(2), _int(3))),
        _int(4),
    ),
    ("comparison", _binary(BinaryOperation.LESS, _int(1), _int(2)), _literal(True)),
    (
        "bound check",
        LogicalExpression(
            LogicalOperation.AND,
            [
                _binary(BinaryOperation.GREATER_EQUAL, _int(3), _int(0)),
                _binary(BinaryOperation.LESS, _int(3), _int(10)),
            ],
        ),
        _literal(True),
    ),
    (
        "piecewise",
        piecewise(
            (_binary(BinaryOperation.LESS, _int(3), _int(2)), _int(10)),
            otherwise=_int(20),
        ),
        _int(20),
    ),
    (
        "decimal arithmetic",
        _binary(_ADD, _literal("0.1"), _literal("0.2")),
        _binary(_DIVIDE, _int(3), _int(10)),
    ),
    (
        "large integers",
        _binary(BinaryOperation.MULTIPLY, _int(2**70), _int(2**70)),
        _int(2**140),
    ),
]

# A table of expressions the simplifier declines: it returns each unchanged.
_DECLINES: list[tuple[str, Expression]] = [
    ("float", _literal(1.5)),
    ("float sum", _binary(_ADD, _literal(1.5), _int(1))),
    ("division by zero", _binary(_DIVIDE, _int(1), _int(0))),
    ("modulo by zero", _binary(_MODULO, _int(1), _int(0))),
    ("irrational root", _binary(_POWER, _int(2), _binary(_DIVIDE, _int(1), _int(2)))),
    (
        "native constant",
        _binary(
            BinaryOperation.GREATER,
            IdentifierExpression(get_native_constant_identifier("pi")),
            _int(3),
        ),
    ),
    ("huge power", _binary(_POWER, _int(3), _int(1_000_000))),
]


# =============================================================================
# The class
# =============================================================================


def test_ground_simplifier_is_a_simplifier() -> None:
    """Test the Rust class is a registered `Simplifier`, named `ground`."""
    simplifier = GroundSimplifier()

    assert isinstance(simplifier, Simplifier)
    assert simplifier.name == "ground"
    assert simplifier.fallback is None
    assert repr(simplifier) == "GroundSimplifier()"


def test_a_fallback_is_kept_and_named() -> None:
    """Test the chain's name, repr and fallback."""
    fallback = _RecordingSimplifier()

    chain = GroundSimplifier(fallback)

    assert chain.fallback is fallback
    assert chain.name == "ground+_RecordingSimplifier"
    assert repr(chain) == f"GroundSimplifier({fallback!r})"
    assert GroundSimplifier(None).fallback is None


@pytest.mark.parametrize("fallback", [1, "sympy", object()])
def test_a_fallback_that_is_not_a_simplifier_is_refused(fallback: object) -> None:
    """Test a non-`Simplifier` fallback raises `TypeError`."""
    with pytest.raises(TypeError, match="Simplifier"):
        GroundSimplifier(fallback)  # type: ignore[arg-type]


def test_simplify_refuses_what_is_not_an_expression() -> None:
    """Test `simplify` raises `TypeError` for a non-expression."""
    with pytest.raises(TypeError, match="Expression"):
        GroundSimplifier().simplify(1)  # type: ignore[arg-type]


def test_pickling_round_trips_the_class_and_its_fallback() -> None:
    """Test a ground simplifier pickles as a call of the class."""
    fallback = GroundSimplifier()

    plain = pickle.loads(pickle.dumps(GroundSimplifier()))
    chain = pickle.loads(pickle.dumps(GroundSimplifier(fallback)))

    assert (plain.fallback, chain.fallback.name) == (None, "ground")
    assert chain.name == "ground+ground"


@pytest.mark.parametrize(("label", "expression", "expected"), _FOLDS)
def test_simplify_folds_to_the_literal_sympy_lifts(
    label: str, expression: Expression, expected: Expression
) -> None:
    """Test each foldable expression becomes its literal, in SymPy's form."""
    assert GroundSimplifier().simplify(expression) == expected, label


@pytest.mark.parametrize(("label", "expression"), _DECLINES)
def test_simplify_returns_a_declined_expression_unchanged(
    label: str, expression: Expression
) -> None:
    """Test what the simplifier cannot match SymPy on comes back as it was."""
    assert GroundSimplifier().simplify(expression) == expression, label


# =============================================================================
# Agreement with SymPy
# =============================================================================


@pytest.mark.sympy
@pytest.mark.parametrize(("label", "expression", "expected"), _FOLDS)
def test_every_fold_is_what_the_sympy_backend_returns(
    label: str, expression: Expression, expected: Expression
) -> None:
    """Test the folds equal `SolverBackend.SYMPY`'s answers."""
    assert simplify_expression(expression, backend=SolverBackend.SYMPY) == expected, (
        label
    )
    assert simplify_expression(expression, backend=SolverBackend.GROUND) == expected, (
        label
    )


_COMPOSED_CALLS: list[tuple[str, Expression]] = [
    ("max", CallExpression("max", (_int(2), _int(5)))),
    ("min", CallExpression("min", (_int(2), _int(5)))),
    ("abs", CallExpression("abs", (_int(-5),))),
    ("sign", CallExpression("sign", (_int(-5),))),
    ("clamp", CallExpression("clamp", (_int(15), _int(0), _int(10)))),
    ("relu", CallExpression("relu", (_int(-3),))),
    ("xor", CallExpression("xor", (_literal(True), _literal(False)))),
    ("iff", CallExpression("iff", (_literal(False), _literal(False)))),
]


def _outcome(expression: Expression, backend: SolverBackend) -> object:
    """Return SymPy's answer, or the type of the error the backend raises."""
    try:
        return simplify_expression(expression, backend=backend)
    except Exception as error:
        return type(error)


@pytest.mark.sympy
@pytest.mark.parametrize(("label", "expression"), _COMPOSED_CALLS)
def test_the_chain_and_sympy_agree_on_a_composed_built_in(
    label: str, expression: Expression
) -> None:
    """Test the chain answers or fails as SymPy does: the ground fold declines it."""
    assert _outcome(expression, SolverBackend.GROUND) == expression, label
    assert _outcome(expression, SolverBackend.GROUND_THEN_SYMPY) == _outcome(
        expression, SolverBackend.SYMPY
    ), label


# =============================================================================
# Selecting the backends
# =============================================================================


def test_ground_backends_answer_only_simplification() -> None:
    """Test the capability table of the two ground members."""
    for backend in (SolverBackend.GROUND, SolverBackend.GROUND_THEN_SYMPY):
        assert get_backend_capabilities(backend) == {SolverQueryKind.SIMPLIFICATION}


@pytest.mark.parametrize(
    "backend", [SolverBackend.GROUND, SolverBackend.GROUND_THEN_SYMPY]
)
def test_a_ground_backend_cannot_answer_a_logical_question(
    backend: SolverBackend, x: Identifier
) -> None:
    """Test `SolverCapabilityError` for satisfiability."""
    with pytest.raises(SolverCapabilityError):
        check_expression_satisfiability(
            IdentifierExpression(x) > 0, {x: SymbolType.INT}, backend=backend
        )


def test_the_ground_backend_needs_no_package() -> None:
    """Test `SolverBackend.GROUND` is always available, and resolves once."""
    adapter = _resolve_adapter(SolverBackend.GROUND)

    assert is_backend_available(SolverBackend.GROUND) is True
    assert isinstance(adapter, GroundSimplifier)
    assert adapter.fallback is None
    assert _resolve_adapter(SolverBackend.GROUND) is adapter


@pytest.mark.sympy
def test_the_chain_backend_holds_the_sympy_backend() -> None:
    """Test `SolverBackend.GROUND_THEN_SYMPY` is the ground fold over SymPy."""
    from fhy_core.symbolic.expression.passes.sympy import (  # noqa: PLC0415
        SympySimplifier,
    )

    adapter = _resolve_adapter(SolverBackend.GROUND_THEN_SYMPY)

    assert is_backend_available(SolverBackend.GROUND_THEN_SYMPY) is True
    assert isinstance(adapter, GroundSimplifier)
    assert isinstance(adapter.fallback, SympySimplifier)
    assert adapter.name == "ground+sympy"
    assert _resolve_adapter(SolverBackend.GROUND_THEN_SYMPY) is adapter


def test_the_environment_is_substituted_before_the_fold(x: Identifier) -> None:
    """Test a bound check is decided once its identifier is bound."""
    bound = IdentifierExpression(x) >= 0

    holds = simplify_expression(bound, {x: _int(3)}, backend=SolverBackend.GROUND)
    fails = simplify_expression(bound, {x: _int(-3)}, backend=SolverBackend.GROUND)
    free = simplify_expression(bound, backend=SolverBackend.GROUND)

    assert (holds, fails, free) == (_literal(True), _literal(False), bound)


def test_a_free_expression_is_not_folded(x: Identifier) -> None:
    """Test SymPy's symbolic simplifications are left to SymPy."""
    expression = _binary(
        BinaryOperation.SUBTRACT, IdentifierExpression(x), IdentifierExpression(x)
    )

    assert simplify_expression(expression, backend=SolverBackend.GROUND) == expression


@pytest.mark.sympy
def test_the_chain_simplifies_what_the_ground_fold_declines(x: Identifier) -> None:
    """Test SymPy answers the free expression, and the ground fold the ground one."""
    free = _binary(
        BinaryOperation.SUBTRACT,
        _binary(_ADD, IdentifierExpression(x), IdentifierExpression(x)),
        IdentifierExpression(x),
    )
    ground = _binary(_ADD, _int(2), _int(3))

    chained_free = simplify_expression(free, backend=SolverBackend.GROUND_THEN_SYMPY)
    chained_ground = simplify_expression(
        ground, backend=SolverBackend.GROUND_THEN_SYMPY
    )

    assert chained_free == IdentifierExpression(x)
    assert chained_ground == _int(5)


# =============================================================================
# The chain
# =============================================================================


def test_the_chain_does_not_ask_its_fallback_for_a_ground_expression() -> None:
    """Test a folded expression never reaches the fallback."""
    fallback = _RecordingSimplifier(_int(99))

    result = GroundSimplifier(fallback).simplify(_binary(_ADD, _int(2), _int(3)))

    assert result == _int(5)
    assert fallback.inputs == []


def test_the_chain_asks_its_fallback_for_what_it_declines(x: Identifier) -> None:
    """Test a declined expression reaches the fallback, and its answer returns."""
    fallback = _RecordingSimplifier(_int(99))
    expression = _binary(_ADD, IdentifierExpression(x), _int(1))

    result = GroundSimplifier(fallback).simplify(expression)

    assert result == _int(99)
    assert fallback.inputs == [expression]


def test_a_solver_holding_the_chain_asks_the_fallback(x: Identifier) -> None:
    """Test a `Solver` substitutes, folds, and falls back, in Rust."""
    fallback = _RecordingSimplifier(_int(99))
    solver = Solver(simplifier=GroundSimplifier(fallback))
    expression = _binary(_ADD, IdentifierExpression(x), _int(1))

    folded = solver.simplify_expression(expression, {x: _int(1)})
    declined = solver.simplify_expression(expression)

    assert folded == _int(2)
    assert declined == _int(99)
    assert fallback.inputs == [expression]


def test_the_fallback_s_exception_propagates(x: Identifier) -> None:
    """Test the Python fallback's own exception reaches the caller."""
    chain = GroundSimplifier(_FailingSimplifier())

    with pytest.raises(ValueError, match="the fallback failed"):
        chain.simplify(IdentifierExpression(x))
    with pytest.raises(ValueError, match="the fallback failed"):
        Solver(simplifier=chain).simplify_expression(IdentifierExpression(x))


def test_chains_nest(x: Identifier) -> None:
    """Test a chain is itself a valid fallback."""
    fallback = _RecordingSimplifier(_int(7))
    chain = GroundSimplifier(GroundSimplifier(fallback))

    assert chain.simplify(_binary(_ADD, _int(1), _int(1))) == _int(2)
    assert chain.simplify(IdentifierExpression(x)) == _int(7)
    assert chain.name == "ground+ground+_RecordingSimplifier"


# =============================================================================
# Timeouts
# =============================================================================


def test_a_timeout_applies_to_the_ground_backend() -> None:
    """Test a timeout the run exceeds declines it, and none folds it."""
    solver = _solver_of(SolverBackend.GROUND)
    expression = _heavy_sum()

    bounded = solver.simplify_expression(expression, timeout_milliseconds=1)
    unbounded = solver.simplify_expression(expression)

    assert bounded == expression
    assert isinstance(unbounded, LiteralExpression)


@pytest.mark.sympy
def test_a_timeout_is_accepted_by_the_sympy_chain() -> None:
    """Test ``GROUND_THEN_SYMPY`` answers under a timeout, ground or not."""
    solver = _solver_of(SolverBackend.GROUND_THEN_SYMPY)

    result = solver.simplify_expression(
        _binary(_ADD, _int(2), _int(3)), timeout_milliseconds=60_000
    )

    assert result == _int(5)


def test_a_normal_expression_is_unaffected_by_a_timeout() -> None:
    """Test a generous timeout leaves the answer as it is."""
    solver = Solver(simplifier=GroundSimplifier())
    expression = _binary(_ADD, _binary(_POWER, _int(2), _int(10)), _int(1))

    assert solver.simplify_expression(expression, timeout_milliseconds=60_000) == _int(
        1025
    )


def test_the_chain_asks_its_fallback_under_the_remaining_timeout(
    x: Identifier,
) -> None:
    """Test the fallback gets what the ground part left, not a fresh timeout."""
    fallback = _TimeoutRecordingSimplifier()
    solver = Solver(simplifier=GroundSimplifier(fallback))

    solver.simplify_expression(IdentifierExpression(x), timeout_milliseconds=60_000)
    solver.simplify_expression(_heavy_sum(), timeout_milliseconds=1)
    solver.simplify_expression(IdentifierExpression(x))

    first, second, third = fallback.timeouts
    assert first is not None
    assert 59.0 < first <= 60.0
    assert second == 0.0
    assert third is None


# =============================================================================
# The default solver
# =============================================================================


@pytest.mark.usefixtures("restore_default_solver")
def test_the_default_solver_can_be_the_ground_simplifier(x: Identifier) -> None:
    """Test `set_default_solver` routes unnamed questions to the ground fold."""
    set_default_solver(Solver(simplifier=GroundSimplifier()))
    expression = _binary(BinaryOperation.SUBTRACT, _int(7), IdentifierExpression(x))

    folded = simplify_expression(expression, {x: _int(2)})
    kept = simplify_expression(expression)

    assert folded == _int(5)
    assert kept == expression


def test_the_default_solver_is_still_sympy() -> None:
    """Test the ground backends are opt-in: the default solver is unchanged."""
    simplifier = get_default_solver().simplifier

    assert simplifier is not None
    assert simplifier.name == "sympy"
