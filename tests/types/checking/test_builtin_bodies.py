"""Tests that every expression-bodied built-in type-checks against its sorts.

Registering a function stores its body; it does not type-check it. That
keeps the expression IR independent of the IR type system, but it means
nothing at runtime notices when a built-in's body stops agreeing with
its declared result sort.

Those bodies are this repository's own source, so the agreement is a
self-consistency property of the package rather than a property of
caller input, and it is checked here. This module sits in
`tests/types/checking` because it is a test of
`check_registered_function_body` -- the layer that owns the question --
applied to the built-ins, and this is the only test package that already
sees both the seeded registry and the checker.
"""

import pytest

from fhy_core.symbolic.expression import (
    BUILTIN_FUNCTIONS,
    RegisteredFunction,
    get_registered_entry,
)
from fhy_core.types.checking import check_registered_function_body

# The composed built-ins: every entry in ``BUILTIN_FUNCTIONS`` whose body
# is an expression in the IR rather than a Python callable.
_EXPECTED_EXPRESSION_BODIED_BUILTINS = frozenset(
    {
        "abs",
        "clamp",
        "clamp_symmetric",
        "gelu",
        "iff",
        "implies",
        "leaky_relu",
        "max",
        "min",
        "nand",
        "nor",
        "relu",
        "sigmoid",
        "sign",
        "silu",
        "xor",
    }
)

_EXPRESSION_BODIED_BUILTINS: dict[str, RegisteredFunction] = {
    name: entry
    for name, entry in BUILTIN_FUNCTIONS.items()
    if isinstance(entry, RegisteredFunction)
}


def test_expression_bodied_builtins_are_exactly_the_documented_set() -> None:
    """Test the discovered composed built-ins match the documented names.

    Guards the parametrization below: if the discovery predicate ever
    stopped matching, the per-body tests would silently shrink to
    nothing instead of failing.
    """
    assert (
        frozenset(_EXPRESSION_BODIED_BUILTINS) == _EXPECTED_EXPRESSION_BODIED_BUILTINS
    )
    assert len(_EXPRESSION_BODIED_BUILTINS) == 16


@pytest.mark.parametrize("name", sorted(_EXPRESSION_BODIED_BUILTINS))
def test_builtin_body_type_checks_against_its_declared_result_sort(name: str) -> None:
    """Test the built-in's body synthesizes a type compatible with its sort.

    `check_registered_function_body` raises `PassExecutionError` when the
    body does not satisfy the declared parameter and result sorts, so
    completing the call is what this test asserts.
    """
    entry = _EXPRESSION_BODIED_BUILTINS[name]

    check_registered_function_body(
        name=entry.name,
        parameters=entry.parameters,
        parameter_sorts=entry.parameter_sorts,
        result_sort=entry.result_sort,
        body=entry.body,
        resolve_call_target=get_registered_entry,
    )
