"""Tests that every expression-bodied built-in type-checks against its sorts.

Registering a function stores its body; it does not type-check it. That
keeps the expression IR independent of the IR type system, but it means
nothing at runtime notices when a built-in's body stops agreeing with
its declared result sort.

Those bodies are this repository's own source, so the agreement is a
self-consistency property of the package rather than a property of
caller input, and it is checked here.
`check_all_registered_function_bodies` answers exactly that question for
every entry at once, so the check is a single sweep over the seeded
registry rather than a per-built-in loop; the report names every
offender when it fails.
"""

from fhy_core.symbolic.expression import BUILTIN_FUNCTIONS, RegisteredFunction
from fhy_core.types.checking import check_all_registered_function_bodies

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

    Guards the sweep below: the sweep asserts that nothing in the
    registry is broken, which a registry holding no expression-bodied
    entries would satisfy vacuously.
    """
    assert (
        frozenset(_EXPRESSION_BODIED_BUILTINS) == _EXPECTED_EXPRESSION_BODIED_BUILTINS
    )
    assert len(_EXPRESSION_BODIED_BUILTINS) == 16


def test_every_builtin_body_type_checks_against_its_declared_result_sort() -> None:
    """Test the seeded registry sweeps clean.

    Each built-in body must synthesize a type compatible with its
    declared result sort. A failure names the offending built-in in the
    report, so the single assertion still points at the culprit.
    """
    report = check_all_registered_function_bodies()

    assert report.has_errors() is False, report.format()
