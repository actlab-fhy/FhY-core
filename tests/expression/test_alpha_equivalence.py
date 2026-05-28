"""Tests `AlphaEquivalence` adoption on the `Expression` hierarchy and
`RegisteredFunction`."""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    RegisteredFunction,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
    register_native_function,
)
from fhy_core.expression.sort import FunctionSort
from fhy_core.identifier import Identifier
from fhy_core.trait import AlphaEquivalence, AlphaRenaming

from .conftest import function_registry_snapshot  # noqa: F401  # fixture re-export


def _make_identifiers(*names: str) -> tuple[Identifier, ...]:
    """Construct one `Identifier` per ``names`` entry."""
    return tuple(Identifier(name) for name in names)


def _make_registered_function(
    name: str,
    parameters: tuple[Identifier, ...],
    parameter_sorts: tuple[FunctionSort, ...],
    result_sort: FunctionSort,
    body: Expression,
) -> RegisteredFunction:
    """Build a `RegisteredFunction` directly (no registry insertion)."""
    return RegisteredFunction(
        name=name,
        parameters=parameters,
        parameter_sorts=parameter_sorts,
        result_sort=result_sort,
        body=body,
    )


# ===========================================================================
# Runtime-protocol checks
# ===========================================================================


def test_expression_implements_alpha_equivalence_protocol() -> None:
    """Test every concrete `Expression` node satisfies the `AlphaEquivalence`
    runtime protocol."""
    (x,) = _make_identifiers("x")
    expressions: list[Expression] = [
        LiteralExpression(1),
        IdentifierExpression(x),
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
        BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
        ),
        TernaryExpression(
            LiteralExpression(1), LiteralExpression(2), LiteralExpression(3)
        ),
        CallExpression("f", (LiteralExpression(1),)),
    ]

    for expression in expressions:
        assert isinstance(expression, AlphaEquivalence)


def test_registered_function_implements_alpha_equivalence_protocol() -> None:
    """Test `RegisteredFunction` satisfies the `AlphaEquivalence` runtime
    protocol."""
    (x,) = _make_identifiers("x")
    function = _make_registered_function(
        name="identity_int",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )

    assert isinstance(function, AlphaEquivalence)


# ===========================================================================
# LiteralExpression
# ===========================================================================


def test_literal_expression_is_alpha_equivalent_to_same_value() -> None:
    """Test two `LiteralExpression`s with the same value are alpha-equivalent."""
    left = LiteralExpression(7)
    right = LiteralExpression(7)

    assert left.is_alpha_equivalent(right)


def test_literal_expression_is_not_alpha_equivalent_for_value_mismatch() -> None:
    """Test `LiteralExpression`s with different values are not alpha-equivalent."""
    left = LiteralExpression(7)
    right = LiteralExpression(8)

    assert not left.is_alpha_equivalent(right)


def test_literal_expression_distinguishes_int_from_float_value_type() -> None:
    """Test int and float literals with equal numeric value are not alpha-equivalent."""
    left = LiteralExpression(5)
    right = LiteralExpression(5.0)

    assert not left.is_alpha_equivalent(right)


def test_literal_expression_distinguishes_bool_from_int_value_type() -> None:
    """Test bool and int literals with equal numeric value are not alpha-equivalent."""
    left = LiteralExpression(True)
    right = LiteralExpression(1)

    assert not left.is_alpha_equivalent(right)


def test_literal_expression_is_not_alpha_equivalent_to_non_expression() -> None:
    """Test `LiteralExpression` alpha-equivalence returns False (not raises) for
    non-Expression `other`."""
    literal = LiteralExpression(1)

    assert not literal.is_alpha_equivalent("not an expression")
    assert not literal.is_alpha_equivalent(42)


# ===========================================================================
# IdentifierExpression
# ===========================================================================


def test_identifier_expression_is_alpha_equivalent_under_identity() -> None:
    """Test two `IdentifierExpression`s on the same `Identifier` are alpha-equivalent
    under the empty renaming."""
    (x,) = _make_identifiers("x")
    left = IdentifierExpression(x)
    right = IdentifierExpression(x)

    assert left.is_alpha_equivalent(right)


def test_identifier_expression_distinguishes_distinct_free_identifiers() -> None:
    """Test two distinct free `Identifier`s with the same name_hint are not
    alpha-equivalent under the empty renaming."""
    x, x_again = _make_identifiers("x", "x")
    left = IdentifierExpression(x)
    right = IdentifierExpression(x_again)

    assert not left.is_alpha_equivalent(right)


def test_identifier_expression_uses_supplied_free_renaming() -> None:
    """Test the free renaming maps distinct identifiers to alpha-equivalence."""
    x, xprime = _make_identifiers("x", "x")
    renaming = AlphaRenaming.with_free_renaming({x: xprime})

    assert IdentifierExpression(x).is_alpha_equivalent_under(
        IdentifierExpression(xprime), renaming
    )


def test_identifier_expression_resolves_through_binder_frame() -> None:
    """Test the renaming's binder frame is consulted when resolving."""
    x, y = _make_identifiers("x", "y")
    renaming = AlphaRenaming.empty().extend({x: y})

    assert IdentifierExpression(x).is_alpha_equivalent_under(
        IdentifierExpression(y), renaming
    )


def test_identifier_expression_returns_false_when_renaming_resolves_differently() -> (
    None
):
    """Test alpha-equivalence fails when the renaming maps self to a third
    identifier."""
    x, y, z = _make_identifiers("x", "y", "z")
    renaming = AlphaRenaming.empty().extend({x: y})

    assert not IdentifierExpression(x).is_alpha_equivalent_under(
        IdentifierExpression(z), renaming
    )


# ===========================================================================
# UnaryExpression
# ===========================================================================


def test_unary_expression_is_alpha_equivalent_for_matching_operation_and_operand() -> (
    None
):
    """Test `UnaryExpression`s match under same operation and alpha-equivalent
    operand."""
    left = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(3))
    right = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(3))

    assert left.is_alpha_equivalent(right)


def test_unary_expression_distinguishes_operation_kind() -> None:
    """Test `UnaryExpression`s differ when operations differ."""
    left = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(3))
    right = UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(3))

    assert not left.is_alpha_equivalent(right)


def test_unary_expression_threads_renaming_into_operand() -> None:
    """Test `UnaryExpression` propagates the renaming to its operand."""
    x, y = _make_identifiers("x", "y")
    renaming = AlphaRenaming.empty().extend({x: y})
    left = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))
    right = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(y))

    assert left.is_alpha_equivalent_under(right, renaming)


@pytest.mark.parametrize("operation", list(UnaryOperation))
def test_unary_expression_alpha_equivalence_holds_per_operation(
    operation: UnaryOperation,
) -> None:
    """Test alpha-equivalence holds for every `UnaryOperation` on matching operands."""
    left = UnaryExpression(operation, LiteralExpression(1))
    right = UnaryExpression(operation, LiteralExpression(1))

    assert left.is_alpha_equivalent(right)


# ===========================================================================
# BinaryExpression
# ===========================================================================


def test_binary_expression_is_alpha_equivalent_for_matching_shape() -> None:
    """Test `BinaryExpression`s match when operation and both operands match."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    assert left.is_alpha_equivalent(right)


def test_binary_expression_does_not_normalize_operand_order() -> None:
    """Test `BinaryExpression` does not treat reordered operands as
    alpha-equivalent."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(2), LiteralExpression(1)
    )

    assert not left.is_alpha_equivalent(right)


def test_binary_expression_distinguishes_operation_kind() -> None:
    """Test `BinaryExpression`s with different operations are not alpha-equivalent."""
    left = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    right = BinaryExpression(
        BinaryOperation.MULTIPLY, LiteralExpression(1), LiteralExpression(2)
    )

    assert not left.is_alpha_equivalent(right)


def test_binary_expression_threads_renaming_into_both_operands() -> None:
    """Test `BinaryExpression` propagates the renaming to both operands."""
    x, y = _make_identifiers("x", "y")
    renaming = AlphaRenaming.empty().extend({x: y})
    left = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(x)
    )
    right = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), IdentifierExpression(y)
    )

    assert left.is_alpha_equivalent_under(right, renaming)


@pytest.mark.parametrize("operation", list(BinaryOperation))
def test_binary_expression_alpha_equivalence_holds_per_operation(
    operation: BinaryOperation,
) -> None:
    """Test alpha-equivalence holds for every `BinaryOperation` on matching operands."""
    left = BinaryExpression(operation, LiteralExpression(1), LiteralExpression(2))
    right = BinaryExpression(operation, LiteralExpression(1), LiteralExpression(2))

    assert left.is_alpha_equivalent(right)


# ===========================================================================
# TernaryExpression
# ===========================================================================


def test_ternary_expression_is_alpha_equivalent_for_matching_shape() -> None:
    """Test `TernaryExpression`s match when all three branches alpha-match."""
    left = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    right = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )

    assert left.is_alpha_equivalent(right)


def test_ternary_expression_distinguishes_condition_branch() -> None:
    """Test `TernaryExpression`s with different conditions differ."""
    left = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    right = TernaryExpression(
        LiteralExpression(False), LiteralExpression(1), LiteralExpression(2)
    )

    assert not left.is_alpha_equivalent(right)


def test_ternary_expression_distinguishes_true_branch() -> None:
    """Test `TernaryExpression`s differ when their true-branches differ."""
    left = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    right = TernaryExpression(
        LiteralExpression(True), LiteralExpression(99), LiteralExpression(2)
    )

    assert not left.is_alpha_equivalent(right)


def test_ternary_expression_distinguishes_false_branch() -> None:
    """Test `TernaryExpression`s differ when their false-branches differ."""
    left = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    right = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(99)
    )

    assert not left.is_alpha_equivalent(right)


def test_ternary_expression_threads_renaming_into_all_branches() -> None:
    """Test `TernaryExpression` propagates the renaming to each branch."""
    x, y = _make_identifiers("x", "y")
    renaming = AlphaRenaming.empty().extend({x: y})
    left = TernaryExpression(
        IdentifierExpression(x), IdentifierExpression(x), IdentifierExpression(x)
    )
    right = TernaryExpression(
        IdentifierExpression(y), IdentifierExpression(y), IdentifierExpression(y)
    )

    assert left.is_alpha_equivalent_under(right, renaming)


# ===========================================================================
# CallExpression
# ===========================================================================


def test_call_expression_is_alpha_equivalent_for_matching_shape() -> None:
    """Test `CallExpression`s match when function name and all arguments
    match."""
    left = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    right = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    assert left.is_alpha_equivalent(right)


def test_call_expression_distinguishes_function_name() -> None:
    """Test `CallExpression`s with different function names differ."""
    left = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    right = CallExpression("min", (LiteralExpression(1), LiteralExpression(2)))

    assert not left.is_alpha_equivalent(right)


def test_call_expression_distinguishes_arity() -> None:
    """Test `CallExpression`s with different argument counts differ."""
    left = CallExpression("max", (LiteralExpression(1),))
    right = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    assert not left.is_alpha_equivalent(right)


def test_call_expression_threads_renaming_into_arguments() -> None:
    """Test `CallExpression` propagates the renaming to each argument."""
    x, y = _make_identifiers("x", "y")
    renaming = AlphaRenaming.empty().extend({x: y})
    left = CallExpression("max", (IdentifierExpression(x), LiteralExpression(0)))
    right = CallExpression("max", (IdentifierExpression(y), LiteralExpression(0)))

    assert left.is_alpha_equivalent_under(right, renaming)


def test_call_expression_zero_argument_form_is_alpha_equivalent_to_itself() -> None:
    """Test two zero-argument calls with the same name are alpha-equivalent."""
    left = CallExpression("f", ())
    right = CallExpression("f", ())

    assert left.is_alpha_equivalent(right)


def test_call_expression_zero_argument_form_distinguishes_function_name() -> None:
    """Test zero-arg `CallExpression`s with different names are not alpha-equivalent."""
    left = CallExpression("f", ())
    right = CallExpression("g", ())

    assert not left.is_alpha_equivalent(right)


# ===========================================================================
# Cross-type
# ===========================================================================


@pytest.mark.parametrize(
    "left,right",
    [
        (LiteralExpression(1), IdentifierExpression(Identifier("x"))),
        (
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
            LiteralExpression(1),
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
            ),
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
        ),
    ],
)
def test_expression_returns_false_for_cross_type_comparison(
    left: Expression, right: Expression
) -> None:
    """Test alpha-equivalence returns False when comparing different concrete
    Expression types."""
    assert not left.is_alpha_equivalent(right)


# ===========================================================================
# RegisteredFunction binder behavior
# ===========================================================================


def test_registered_function_is_alpha_equivalent_to_itself() -> None:
    """Test alpha-equivalence is reflexive on `RegisteredFunction`."""
    (x,) = _make_identifiers("x")
    function = _make_registered_function(
        name="square_int",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(x),
            IdentifierExpression(x),
        ),
    )

    assert function.is_alpha_equivalent(function)


def test_registered_function_is_alpha_equivalent_under_parameter_rename() -> None:
    """Test two `RegisteredFunction`s identical except for parameter
    `Identifier`s are alpha-equivalent."""
    x, y = _make_identifiers("x", "y")
    left = _make_registered_function(
        name="square_int",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(x),
            IdentifierExpression(x),
        ),
    )
    right = _make_registered_function(
        name="square_int",
        parameters=(y,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(y),
            IdentifierExpression(y),
        ),
    )

    assert left.is_alpha_equivalent(right)


def test_registered_function_distinguishes_parameter_arity() -> None:
    """Test alpha-equivalence fails when the parameter count differs."""
    x, y = _make_identifiers("x", "y")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )
    right = _make_registered_function(
        name="f",
        parameters=(x, y),
        parameter_sorts=(FunctionSort.INT, FunctionSort.INT),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )

    assert not left.is_alpha_equivalent(right)


def test_registered_function_distinguishes_parameter_sort() -> None:
    """Test alpha-equivalence fails when a parameter sort differs."""
    (x,) = _make_identifiers("x")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )
    right = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )

    assert not left.is_alpha_equivalent(right)


def test_registered_function_distinguishes_result_sort() -> None:
    """Test alpha-equivalence fails when the result sort differs."""
    (x,) = _make_identifiers("x")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )
    right = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(x),
    )

    assert not left.is_alpha_equivalent(right)


def test_registered_function_ignores_name_for_alpha_equivalence() -> None:
    """Test alpha-equivalence treats `name` as registry identity, not body
    semantics."""
    (x,) = _make_identifiers("x")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )
    right = _make_registered_function(
        name="g",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )

    assert left.is_alpha_equivalent(right)


def test_registered_function_requires_matching_free_identifiers_by_default() -> None:
    """Test free identifiers in the body must be equal by default."""
    x, free_a, free_b = _make_identifiers("x", "a", "b")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(free_a)
        ),
    )
    right = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(free_b)
        ),
    )

    assert not left.is_alpha_equivalent(right)


def test_registered_function_honors_free_identifier_renaming() -> None:
    """Test free identifiers in the body match via the supplied free renaming."""
    x, y, free_a, free_b = _make_identifiers("x", "y", "a", "b")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(free_a)
        ),
    )
    right = _make_registered_function(
        name="f",
        parameters=(y,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(y), IdentifierExpression(free_b)
        ),
    )
    renaming = AlphaRenaming.with_free_renaming({free_a: free_b})

    assert left.is_alpha_equivalent_under(right, renaming)


def test_registered_function_distinguishes_body_shape() -> None:
    """Test alpha-equivalence fails when bodies have different shapes."""
    (x,) = _make_identifiers("x")
    left = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
        ),
    )
    right = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=BinaryExpression(
            BinaryOperation.MULTIPLY, IdentifierExpression(x), LiteralExpression(1)
        ),
    )

    assert not left.is_alpha_equivalent(right)


def test_registered_function_is_not_alpha_equivalent_to_unrelated_type() -> None:
    """Test alpha-equivalence returns False (not raises) when comparing a
    `RegisteredFunction` to a non-`RegisteredFunction` object."""
    (x,) = _make_identifiers("x")
    function = _make_registered_function(
        name="f",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(x),
    )

    assert not function.is_alpha_equivalent(LiteralExpression(1))
    assert not function.is_alpha_equivalent("not a function")


# ===========================================================================
# Structural-implies-alpha property over the Expression hierarchy
# ===========================================================================


def _structurally_paired_expressions() -> list[tuple[Expression, Expression]]:
    (x,) = _make_identifiers("x")
    return [
        (LiteralExpression(1), LiteralExpression(1)),
        (LiteralExpression(2.5), LiteralExpression(2.5)),
        (IdentifierExpression(x), IdentifierExpression(x)),
        (
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
            ),
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
            ),
        ),
        (
            TernaryExpression(
                LiteralExpression(True),
                LiteralExpression(1),
                LiteralExpression(2),
            ),
            TernaryExpression(
                LiteralExpression(True),
                LiteralExpression(1),
                LiteralExpression(2),
            ),
        ),
        (
            CallExpression("max", (LiteralExpression(1), LiteralExpression(2))),
            CallExpression("max", (LiteralExpression(1), LiteralExpression(2))),
        ),
    ]


@pytest.mark.parametrize(
    "left,right",
    _structurally_paired_expressions(),
    ids=[
        "literal-int",
        "literal-float",
        "identifier",
        "unary",
        "binary",
        "ternary",
        "call",
    ],
)
def test_structurally_equivalent_implies_alpha_equivalent(
    left: Expression, right: Expression
) -> None:
    """Test structural equivalence implies alpha-equivalence across the
    Expression hierarchy."""
    assert left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)


# ===========================================================================
# Reflexivity on every concrete Expression node
# ===========================================================================


@pytest.fixture()
def every_expression_node() -> list[Expression]:
    """Return one instance of each concrete Expression node."""
    (x,) = _make_identifiers("x")
    return [
        LiteralExpression(1),
        LiteralExpression(2.5),
        LiteralExpression(True),
        IdentifierExpression(x),
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1)),
        BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
        ),
        TernaryExpression(
            LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
        ),
        CallExpression("f", (LiteralExpression(1),)),
    ]


def test_alpha_equivalence_is_reflexive_on_each_expression_node(
    every_expression_node: list[Expression],
) -> None:
    """Test alpha-equivalence is reflexive on every concrete Expression node."""
    for expression in every_expression_node:
        assert expression.is_alpha_equivalent(expression)


# ===========================================================================
# Deep nesting
# ===========================================================================


def test_deeply_nested_expression_is_alpha_equivalent_under_uniform_rename() -> None:
    """Test alpha-equivalence threads a renaming through a deeply nested
    Expression tree."""
    x, y = _make_identifiers("x", "y")
    left: Expression = IdentifierExpression(x)
    right: Expression = IdentifierExpression(y)
    for _ in range(10):
        left = UnaryExpression(UnaryOperation.NEGATE, left)
        right = UnaryExpression(UnaryOperation.NEGATE, right)
    renaming = AlphaRenaming.empty().extend({x: y})

    assert left.is_alpha_equivalent_under(right, renaming)


# ===========================================================================
# Integration with the registry: a registered native function used by name
# ===========================================================================


def test_registered_function_alpha_equivalence_works_when_body_uses_native_call(
    function_registry_snapshot: None,  # noqa: F811
) -> None:
    """Test alpha-equivalence over `RegisteredFunction` whose body references a
    native function via `CallExpression`."""
    del function_registry_snapshot
    register_native_function(
        name="square_native",
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        implementation=lambda value: value * value,
    )
    x, y = _make_identifiers("x", "y")
    left = _make_registered_function(
        name="apply_square",
        parameters=(x,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=CallExpression("square_native", (IdentifierExpression(x),)),
    )
    right = _make_registered_function(
        name="apply_square",
        parameters=(y,),
        parameter_sorts=(FunctionSort.INT,),
        result_sort=FunctionSort.INT,
        body=CallExpression("square_native", (IdentifierExpression(y),)),
    )

    assert left.is_alpha_equivalent(right)
