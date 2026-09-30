"""Tests for the Python interface over the Rust-backed function registry.

The entry classes are thin Python subclasses of ``fhy_core._rs`` classes,
and the registry is the Rust core's owned ``FunctionRegistry``, which the
extension keeps for Python's process-wide API. These tests cover what the
binding adds around the core: the class structure, argument checks and
their messages, object identity across registration and lookups, the
built-ins seen first, constants and their identifiers, binder equivalence,
the screen and the inliner reading the Rust registry without calling
Python, threads, and the built-ins' bodies, pinned as data.
"""

import math
import pickle
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.symbolic.expression import (
    BUILTIN_CONSTANTS,
    BUILTIN_FUNCTIONS,
    EntryLookupError,
    EntryRegistrationError,
    Expression,
    FunctionArityError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeConstant,
    NativeFunction,
    NonBooleanLogicalOperandError,
    RegisteredEntry,
    RegisteredFunction,
    call,
    get_native_constant_identifier,
    get_registered_entries,
    get_registered_entry,
    inline_functions,
    is_entry_registered,
    is_python_value_compatible_with_sort,
    logical_and,
    piecewise,
    register_function,
    register_native_constant,
    register_native_function,
    try_get_native_constant_for_identifier,
    try_get_registered_result_sort,
    validate_logical_operands,
    validate_predicate,
)
from fhy_core.symbolic.expression.passes.inline import FunctionInliner
from fhy_core.term import AlphaRenaming
from fhy_core.testing_patches import set_function_registry_state
from fhy_core.traits import FrozenMixin
from fhy_core.traits.frozen import FrozenMutationError

_ENTRY_CLASSES = [
    (RegisteredFunction, _rs.RegisteredFunction),
    (NativeFunction, _rs.NativeFunction),
    (NativeConstant, _rs.NativeConstant),
]


def _build_increment(name: str = "f") -> RegisteredFunction:
    """Return the unregistered entry ``name(x) = x + 1`` over the reals."""
    x = Identifier("x")
    return RegisteredFunction(
        name, (x,), (FunctionSort.REAL,), FunctionSort.REAL, IdentifierExpression(x) + 1
    )


def _register_increment(name: str) -> RegisteredFunction:
    """Register ``name(x) = x + 1`` over the reals."""
    x = Identifier("x")
    return register_function(
        name, [x], [FunctionSort.REAL], FunctionSort.REAL, IdentifierExpression(x) + 1
    )


def _build_entry_of_each_class() -> list[RegisteredEntry]:
    """Return an unregistered entry of each class."""
    return [
        _build_increment(),
        NativeFunction("g", (FunctionSort.REAL,), FunctionSort.REAL, math.sqrt),
        NativeConstant("c", FunctionSort.INT, 3),
    ]


# =============================================================================
# Class structure
# =============================================================================


@pytest.mark.parametrize(("public_class", "rust_class"), _ENTRY_CLASSES)
def test_public_entry_class_subclasses_its_rust_class_and_is_frozen(
    public_class: type, rust_class: type
) -> None:
    """Test each public entry class extends its ``_rs`` class and is frozen."""
    assert issubclass(public_class, rust_class)
    assert issubclass(public_class, FrozenMixin)


@pytest.mark.parametrize("entry", _build_entry_of_each_class(), ids=type)
def test_entry_refuses_mutation_with_frozen_mutation_error(entry: Any) -> None:
    """Test setting or deleting a field raises ``FrozenMutationError``."""
    assert entry.is_frozen
    entry.freeze()
    entry.assert_frozen()

    with pytest.raises(FrozenMutationError, match='Cannot modify "name"'):
        entry.name = "renamed"
    with pytest.raises(FrozenMutationError, match='Cannot delete "name"'):
        del entry.name
    with pytest.raises(FrozenMutationError):
        entry.extra = 1


def test_entry_classes_set_match_args_to_their_fields() -> None:
    """Test each entry class matches positionally by its fields, in order."""
    assert RegisteredFunction.__match_args__ == (
        "name",
        "parameters",
        "parameter_sorts",
        "result_sort",
        "body",
    )
    assert NativeFunction.__match_args__ == (
        "name",
        "parameter_sorts",
        "result_sort",
        "implementation",
    )
    assert NativeConstant.__match_args__ == ("name", "sort", "value")
    match NativeConstant("c", FunctionSort.INT, 3):
        case NativeConstant(name, sort, value):
            assert (name, sort, value) == ("c", FunctionSort.INT, 3)


def test_entry_reprs_follow_the_dataclass_form() -> None:
    """Test each entry prints as its class with its fields."""
    x = Identifier("x")
    function = RegisteredFunction(
        "f", (x,), (FunctionSort.REAL,), FunctionSort.REAL, IdentifierExpression(x)
    )

    assert repr(function) == (
        f"RegisteredFunction(name='f', parameters=({x!r},), "
        f"parameter_sorts=({FunctionSort.REAL!r},), "
        f"result_sort={FunctionSort.REAL!r}, body={IdentifierExpression(x)!r})"
    )
    assert repr(NativeConstant("c", FunctionSort.INT, 3)) == (
        f"NativeConstant(name='c', sort={FunctionSort.INT!r}, value=3)"
    )
    assert repr(NativeFunction("g", (), FunctionSort.REAL, math.pi.__float__)) == (
        f"NativeFunction(name='g', parameter_sorts=(), result_sort="
        f"{FunctionSort.REAL!r}, implementation={math.pi.__float__!r})"
    )


def test_user_entries_pickle_as_a_call_of_their_class() -> None:
    """Test a user entry round-trips through pickle to an equal entry."""
    for entry in _build_entry_of_each_class():
        if isinstance(entry, NativeFunction):
            continue
        restored = pickle.loads(pickle.dumps(entry))

        assert restored == entry
        assert type(restored) is type(entry)


def test_native_function_with_a_lambda_does_not_pickle() -> None:
    """Test a native entry pickles only if its implementation does."""
    entry = NativeFunction("g", (FunctionSort.REAL,), FunctionSort.REAL, lambda x: x)

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(entry)


@pytest.mark.parametrize("name", ["pi", "max", "exp"])
def test_builtin_entries_unpickle_as_the_builtin_itself(name: str) -> None:
    """Test a built-in's entry pickles as the built-in, by name."""
    entry = get_registered_entry(name)

    assert pickle.loads(pickle.dumps(entry)) is entry


# =============================================================================
# Arguments and their messages
# =============================================================================


# The argument checks are called with wrong types on purpose.
_UNTYPED_REGISTERED_FUNCTION: Any = RegisteredFunction
_UNTYPED_NATIVE_FUNCTION: Any = NativeFunction
_UNTYPED_NATIVE_CONSTANT: Any = NativeConstant


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: _UNTYPED_REGISTERED_FUNCTION(
                1, (), (), FunctionSort.REAL, LiteralExpression(1)
            ),
            "RegisteredFunction name must be a str, got int.",
        ),
        (
            lambda: _UNTYPED_REGISTERED_FUNCTION(
                "f",
                ("x",),
                (FunctionSort.REAL,),
                FunctionSort.REAL,
                LiteralExpression(1),
            ),
            "RegisteredFunction parameters must be an Identifier, got str.",
        ),
        (
            lambda: _UNTYPED_REGISTERED_FUNCTION(
                "f", (), ("real",), FunctionSort.REAL, LiteralExpression(1)
            ),
            "RegisteredFunction parameter_sorts must be a FunctionSort, got str.",
        ),
        (
            lambda: _UNTYPED_REGISTERED_FUNCTION(
                "f", (), (), "real", LiteralExpression(1)
            ),
            "RegisteredFunction result_sort must be a FunctionSort, got str.",
        ),
        (
            lambda: _UNTYPED_REGISTERED_FUNCTION("f", (), (), FunctionSort.REAL, 1),
            "RegisteredFunction body must be an Expression, got int.",
        ),
        (
            lambda: _UNTYPED_NATIVE_FUNCTION("g", (), FunctionSort.REAL, 1.0),
            "NativeFunction implementation must be callable, got float.",
        ),
        (
            lambda: _UNTYPED_NATIVE_CONSTANT("c", FunctionSort.REAL, "1.0"),
            "NativeConstant value must be a bool, an int or a float, got str.",
        ),
        (
            lambda: _UNTYPED_REGISTERED_FUNCTION("f", (), ()),
            "RegisteredFunction() missing required argument: 'result_sort'",
        ),
    ],
    ids=[
        "name",
        "parameter",
        "parameter_sort",
        "result_sort",
        "body",
        "implementation",
        "value",
        "missing",
    ],
)
def test_entry_construction_checks_argument_types(
    build: Callable[[], object], message: str
) -> None:
    """Test an argument of the wrong type raises ``TypeError`` naming it."""
    with pytest.raises(TypeError) as exc_info:
        build()

    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _build_increment(""), "function name is empty"),
        (lambda: _build_increment("max"), "function name `max` is a built-in function"),
        (
            lambda: RegisteredFunction(
                "f", (Identifier("x"),), (), FunctionSort.REAL, LiteralExpression(1)
            ),
            'function "f" has 1 parameter but 0 parameter sorts',
        ),
        (
            lambda: NativeConstant("n", FunctionSort.NAT, -1),
            'constant "n" of sort nat cannot hold -1',
        ),
        (
            lambda: NativeConstant("b", FunctionSort.INT, True),
            'constant "b" of sort int cannot hold true',
        ),
        (
            lambda: NativeFunction("exp", (FunctionSort.REAL,), FunctionSort.REAL, abs),
            "function name `exp` is a built-in function",
        ),
    ],
    ids=["empty", "builtin", "sort_count", "nat", "bool_as_int", "native_builtin"],
)
def test_entry_construction_raises_the_cores_value_errors(
    build: Callable[[], object], message: str
) -> None:
    """Test direct construction raises ``ValueError`` with the core's text."""
    with pytest.raises(ValueError) as exc_info:
        build()

    assert str(exc_info.value) == message


def test_native_function_keeps_the_python_arity_check() -> None:
    """Test the ``inspect`` arity check of an implementation stays Python's."""
    with pytest.raises(ValueError, match="does not match the implementation"):
        NativeFunction(
            "g", (FunctionSort.REAL, FunctionSort.REAL), FunctionSort.REAL, math.sqrt
        )


@pytest.mark.parametrize(
    ("register", "message"),
    [
        (
            lambda: _register_increment("max"),
            "function name `max` is a built-in function",
        ),
        (lambda: _register_increment("pi"), '"pi" is the name of a built-in constant'),
        (
            lambda: register_native_constant("e", FunctionSort.REAL, 2.0),
            '"e" is the name of a built-in constant',
        ),
        (
            lambda: register_native_constant("n", FunctionSort.NAT, -2),
            'constant "n" of sort nat cannot hold -2',
        ),
    ],
    ids=["builtin_function", "builtin_constant", "constant_named_e", "value"],
)
def test_registration_raises_entry_registration_error_with_the_cores_text(
    function_registry_snapshot: None, register: Callable[[], object], message: str
) -> None:
    """Test a refused registration raises ``EntryRegistrationError``."""
    with pytest.raises(EntryRegistrationError) as exc_info:
        register()

    assert str(exc_info.value) == message


def test_registration_of_a_taken_name_names_it(
    function_registry_snapshot: None,
) -> None:
    """Test registering a taken name raises with the core's text."""
    _register_increment("test_binding_taken")

    with pytest.raises(EntryRegistrationError) as exc_info:
        register_native_function(
            "test_binding_taken", [FunctionSort.REAL], FunctionSort.REAL, math.exp
        )

    assert str(exc_info.value) == '"test_binding_taken" is already registered'


def test_registration_wraps_a_construction_value_error_as_its_cause(
    function_registry_snapshot: None,
) -> None:
    """Test a construction ``ValueError`` is the registration error's cause."""
    with pytest.raises(EntryRegistrationError) as exc_info:
        _register_increment("")

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value) == "function name is empty"


def test_registration_passes_a_type_error_through(
    function_registry_snapshot: None,
) -> None:
    """Test a construction ``TypeError`` propagates unwrapped."""
    with pytest.raises(TypeError, match="body must be an Expression"):
        register_function("test_binding_type", [], [], FunctionSort.REAL, 3)  # type: ignore[arg-type]


def test_lookup_misses_keep_their_python_text() -> None:
    """Test a lookup miss raises ``EntryLookupError`` with Python's text."""
    with pytest.raises(EntryLookupError) as exc_info:
        get_registered_entry("test_binding_missing")
    assert exc_info.value.args == (
        "No entry is registered under the name 'test_binding_missing'.",
    )

    with pytest.raises(EntryLookupError) as exc_info:
        get_native_constant_identifier("max")
    assert exc_info.value.args == (
        "No native constant is registered under the name 'max'.",
    )


# =============================================================================
# Identity
# =============================================================================


def test_registration_returns_the_object_later_lookups_return(
    function_registry_snapshot: None,
) -> None:
    """Test each registration's entry is the object every lookup returns."""
    function = _register_increment("test_binding_identity")
    native = register_native_function(
        "test_binding_native", [FunctionSort.REAL], FunctionSort.REAL, math.exp
    )
    constant = register_native_constant("test_binding_constant", FunctionSort.INT, 7)

    assert get_registered_entry("test_binding_identity") is function
    assert get_registered_entry("test_binding_native") is native
    assert get_registered_entry("test_binding_constant") is constant
    assert get_registered_entries()["test_binding_identity"] is function
    identifier = get_native_constant_identifier("test_binding_constant")
    assert try_get_native_constant_for_identifier(identifier) is constant


def test_entry_fields_are_the_objects_given() -> None:
    """Test an entry keeps the very objects it was built from."""
    x = Identifier("x")
    body = IdentifierExpression(x) * 2
    function = RegisteredFunction(
        "f", [x], [FunctionSort.REAL], FunctionSort.REAL, body
    )
    value = 2**80
    constant = NativeConstant("big", FunctionSort.NAT, value)
    native = NativeFunction("g", [FunctionSort.REAL], FunctionSort.REAL, math.exp)

    assert function.body is body
    assert function.parameters == (x,)
    assert function.parameters[0] is x
    assert function.parameter_sorts == (FunctionSort.REAL,)
    assert function.result_sort is FunctionSort.REAL
    assert constant.value is value
    assert native.implementation is math.exp


@pytest.mark.parametrize("name", list(BUILTIN_FUNCTIONS))
def test_builtin_function_entry_is_one_object_with_its_body_built_once(
    name: str,
) -> None:
    """Test a built-in's entry is one object across lookups and the mapping."""
    entry = get_registered_entry(name)

    assert BUILTIN_FUNCTIONS[name] is entry  # type: ignore[literal-required]
    assert get_registered_entries()[name] is entry
    if isinstance(entry, RegisteredFunction):
        again = get_registered_entry(name)
        assert isinstance(again, RegisteredFunction)
        assert entry.body is again.body
    else:
        assert isinstance(entry, NativeFunction)
        assert entry.implementation is _rs.BuiltinNativeImplementation._of(name)


def test_registered_entries_list_the_builtins_in_catalogue_order_first(
    function_registry_snapshot: None,
) -> None:
    """Test the snapshot lists the constants, composed and native built-ins."""
    _register_increment("test_binding_listed")

    names = list(get_registered_entries())

    assert names[:4] == ["pi", "e", "inf", "nan"]
    assert names[4:20] == [
        name
        for name, entry in BUILTIN_FUNCTIONS.items()
        if isinstance(entry, RegisteredFunction)
    ]
    assert names[20:39] == [
        name
        for name, entry in BUILTIN_FUNCTIONS.items()
        if isinstance(entry, NativeFunction)
    ]
    assert names[-1] == "test_binding_listed"
    assert get_registered_entries() is get_registered_entries()


def test_builtin_result_sorts_are_the_catalogues() -> None:
    """Test the result-sort lookup answers for built-ins and none for constants."""
    assert try_get_registered_result_sort("xor") is FunctionSort.BOOL
    assert try_get_registered_result_sort("round") is FunctionSort.INT
    assert try_get_registered_result_sort("pi") is None
    assert try_get_registered_result_sort("") is None
    assert is_entry_registered("gelu")
    assert not is_entry_registered("")


# =============================================================================
# Constants
# =============================================================================


@pytest.mark.parametrize(
    ("name", "reserved_id"), [("pi", 48), ("e", 49), ("inf", 50), ("nan", 51)]
)
def test_builtin_constant_identifiers_are_pinned_and_resolve_by_id(
    name: str, reserved_id: int
) -> None:
    """Test each built-in constant's identifier has its reserved id.

    Resolution goes by the id: a restored identifier with the id resolves
    to the constant whatever its name hint, and one merely named like the
    constant does not.
    """
    identifier = get_native_constant_identifier(name)
    restored = Identifier.deserialize_from_dict({"id": reserved_id, "name_hint": "z"})

    assert (identifier.id, identifier.name_hint) == (reserved_id, name)
    assert get_native_constant_identifier(name) is identifier
    assert try_get_native_constant_for_identifier(identifier) is BUILTIN_CONSTANTS[name]  # type: ignore[literal-required]
    assert try_get_native_constant_for_identifier(restored) is BUILTIN_CONSTANTS[name]  # type: ignore[literal-required]
    assert try_get_native_constant_for_identifier(Identifier(name)) is None


def test_restoring_a_snapshot_prunes_a_constant_registered_after_it(
    function_registry_snapshot: None,
) -> None:
    """Test a constant not in the restored state stops resolving."""
    snapshot = dict(get_registered_entries())
    register_native_constant("test_binding_pruned", FunctionSort.REAL, 1.0)
    identifier = get_native_constant_identifier("test_binding_pruned")

    set_function_registry_state(snapshot)

    assert try_get_native_constant_for_identifier(identifier) is None
    assert not is_entry_registered("test_binding_pruned")


def test_restoring_a_snapshot_keeps_a_kept_constants_identifier(
    function_registry_snapshot: None,
) -> None:
    """Test a constant the restored state holds keeps its identifier object."""
    constant = register_native_constant("test_binding_kept", FunctionSort.REAL, 1.0)
    identifier = get_native_constant_identifier("test_binding_kept")
    snapshot = dict(get_registered_entries())
    _register_increment("test_binding_later")

    set_function_registry_state(snapshot)

    assert get_native_constant_identifier("test_binding_kept") is identifier
    assert try_get_native_constant_for_identifier(identifier) is constant
    assert not is_entry_registered("test_binding_later")


def test_restoring_a_state_registers_a_new_constant_with_a_new_identifier(
    function_registry_snapshot: None,
) -> None:
    """Test a constant object not registered under its name is registered anew."""
    constant = NativeConstant("test_binding_new", FunctionSort.INT, 1)

    set_function_registry_state({**get_registered_entries(), constant.name: constant})

    assert get_registered_entry("test_binding_new") is constant
    identifier = get_native_constant_identifier("test_binding_new")
    assert try_get_native_constant_for_identifier(identifier) is constant


def test_restoring_a_state_refuses_a_value_that_is_no_entry(
    function_registry_snapshot: None,
) -> None:
    """Test a state holding something other than an entry raises ``TypeError``."""
    with pytest.raises(TypeError, match="only a user RegisteredFunction"):
        set_function_registry_state({"test_binding_bad": 1})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 1, -1, 2**70, -(2**70), 0.0, -1.5, math.inf, math.nan],
    ids=repr,
)
@pytest.mark.parametrize("sort", list(FunctionSort), ids=str)
def test_constant_value_check_agrees_with_the_python_rule(
    value: bool | int | float, sort: FunctionSort
) -> None:
    """Test the core's value check agrees with the Python rule of the sorts."""
    try:
        NativeConstant("c", sort, value)
    except ValueError:
        is_accepted = False
    else:
        is_accepted = True

    assert is_accepted == is_python_value_compatible_with_sort(value, sort)


# =============================================================================
# Binder equivalence
# =============================================================================


def _build_affine(name: str, parameters: tuple[str, str]) -> RegisteredFunction:
    """Return ``name(p, q) = 2 * p + q`` over parameters named ``parameters``."""
    p, q = (Identifier(parameter) for parameter in parameters)
    return RegisteredFunction(
        name,
        (p, q),
        (FunctionSort.REAL, FunctionSort.REAL),
        FunctionSort.REAL,
        LiteralExpression(2) * p + q,
    )


def test_binder_equivalence_excludes_the_name_and_renames_the_parameters() -> None:
    """Test functions equal up to their names and a parameter rename are equivalent."""
    left = _build_affine("f", ("a", "b"))
    right = _build_affine("g", ("c", "d"))

    assert left.is_alpha_equivalent(right)
    assert left.is_alpha_equivalent_under(right, AlphaRenaming.empty())
    assert not left.is_structurally_equivalent(right)
    assert left.is_structurally_equivalent(left)
    assert left != right


def test_binder_equivalence_distinguishes_swapped_parameters() -> None:
    """Test swapping the parameters of a non-symmetric body is not a renaming."""
    a, b = Identifier("a"), Identifier("b")
    body = IdentifierExpression(a) - b
    left = RegisteredFunction(
        "f", (a, b), (FunctionSort.REAL,) * 2, FunctionSort.REAL, body
    )
    right = RegisteredFunction(
        "f", (b, a), (FunctionSort.REAL,) * 2, FunctionSort.REAL, body
    )

    assert not left.is_alpha_equivalent(right)


def test_binder_equivalence_compares_sorts_and_arity() -> None:
    """Test different sorts, result sort, or parameter count are not equivalent."""
    x, y = Identifier("x"), Identifier("y")
    base = RegisteredFunction(
        "f", (x,), (FunctionSort.REAL,), FunctionSort.REAL, LiteralExpression(1)
    )
    other_sort = RegisteredFunction(
        "f", (x,), (FunctionSort.INT,), FunctionSort.REAL, LiteralExpression(1)
    )
    other_result = RegisteredFunction(
        "f", (x,), (FunctionSort.REAL,), FunctionSort.INT, LiteralExpression(1)
    )
    other_arity = RegisteredFunction(
        "f", (x, y), (FunctionSort.REAL,) * 2, FunctionSort.REAL, LiteralExpression(1)
    )

    for other in (other_sort, other_result, other_arity):
        assert not base.is_alpha_equivalent(other)
        assert not base.is_structurally_equivalent(other)
    assert not base.is_alpha_equivalent(LiteralExpression(1))


def test_binder_equivalence_honors_a_given_free_renaming() -> None:
    """Test free identifiers of the bodies compare under the given renaming."""
    free_left, free_right = Identifier("u"), Identifier("v")
    x, y = Identifier("x"), Identifier("y")
    left = RegisteredFunction(
        "f",
        (x,),
        (FunctionSort.REAL,),
        FunctionSort.REAL,
        IdentifierExpression(x) + free_left,
    )
    right = RegisteredFunction(
        "g",
        (y,),
        (FunctionSort.REAL,),
        FunctionSort.REAL,
        IdentifierExpression(y) + free_right,
    )

    assert not left.is_alpha_equivalent(right)
    assert left.is_alpha_equivalent_under(
        right, AlphaRenaming.with_free_renaming({free_left: free_right})
    )


def test_binder_equivalence_refuses_a_renaming_that_is_no_alpha_renaming() -> None:
    """Test a renaming argument must be an ``AlphaRenaming``."""
    with pytest.raises(TypeError, match="renaming must be an AlphaRenaming"):
        _build_increment().is_alpha_equivalent_under(_build_increment(), {})  # type: ignore[arg-type]


def test_entries_compare_and_hash_by_their_fields() -> None:
    """Test entry equality and hashing follow the fields, as a dataclass's do."""
    x = Identifier("x")
    body = IdentifierExpression(x) + 1
    first = RegisteredFunction("f", (x,), (FunctionSort.REAL,), FunctionSort.REAL, body)
    second = RegisteredFunction(
        "f", (x,), (FunctionSort.REAL,), FunctionSort.REAL, IdentifierExpression(x) + 1
    )

    assert first == second
    assert hash(first) == hash(second)
    assert first != _build_increment("g")
    assert NativeConstant("c", FunctionSort.INT, 1) != NativeConstant(
        "c", FunctionSort.INT, 2
    )
    assert (first == 1) is False


# =============================================================================
# The screen reads the Rust registry
# =============================================================================


def test_the_screen_calls_no_python_lookup(
    function_registry_snapshot: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the screen judges user calls and constants without Python lookups."""
    register_function(
        "test_binding_real_valued",
        [Identifier("x")],
        [FunctionSort.REAL],
        FunctionSort.REAL,
        LiteralExpression(1),
    )
    register_native_constant("test_binding_real_constant", FunctionSort.REAL, 1.0)
    constant = IdentifierExpression(
        get_native_constant_identifier("test_binding_real_constant")
    )

    def refuse(*arguments: object) -> None:
        raise AssertionError(f"the screen called a Python lookup with {arguments}")

    import fhy_core.symbolic.expression.registry as registry_package  # noqa: PLC0415
    import fhy_core.symbolic.expression.registry.api as api_module  # noqa: PLC0415
    import fhy_core.symbolic.expression.registry.storage as storage_module  # noqa: PLC0415

    for module in (registry_package, api_module, storage_module):
        for name in (
            "try_get_native_constant_for_identifier",
            "try_get_registered_result_sort",
            "get_registered_entry",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, refuse)

    with pytest.raises(NonBooleanLogicalOperandError):
        validate_predicate(call("test_binding_real_valued", LiteralExpression(1)))
    with pytest.raises(NonBooleanLogicalOperandError):
        validate_logical_operands(logical_and(constant, LiteralExpression(True)))
    with pytest.raises(NonBooleanLogicalOperandError):
        validate_predicate(IdentifierExpression(get_native_constant_identifier("pi")))


# =============================================================================
# The inliner
# =============================================================================


def test_inliner_is_registered_and_create_builds_it() -> None:
    """Test the pass keeps its registry name."""
    inliner = CompilerPass.create("fhy_core.symbolic.expression.inline_functions")

    assert isinstance(inliner, FunctionInliner)
    assert not hasattr(FunctionInliner, "visit_call_expression")


def test_inliner_returns_its_input_and_reports_no_change_when_nothing_inlines() -> None:
    """Test ``changed`` is by identity, and the input comes back itself."""
    expression = call("exp", Identifier("x")) + 1

    result = FunctionInliner().execute(expression)

    assert result.output is expression
    assert not result.changed
    assert inline_functions(expression) is expression


def test_inliner_reports_a_change_when_it_inlines() -> None:
    """Test inlining a composed built-in changes the IR."""
    result = FunctionInliner().execute(call("relu", Identifier("x")))

    assert result.changed
    assert str(result.output) == "{x if ((x > 0) || (x != x)); 0 otherwise}"


@pytest.mark.parametrize(
    ("expression", "cause_class", "message"),
    [
        (call("test_binding_nothing"), EntryLookupError, "no function is registered"),
        (call("max", Identifier("x")), FunctionArityError, '"max" takes 2 arguments'),
        (call("pi"), FunctionArityError, '"pi" is a constant, not a function'),
    ],
    ids=["unknown", "arity", "constant"],
)
def test_inliner_errors_are_the_cause_of_the_pass_error(
    expression: Expression, cause_class: type[Exception], message: str
) -> None:
    """Test each refusal is the ``__cause__`` of ``PassExecutionError``."""
    with pytest.raises(PassExecutionError) as exc_info:
        inline_functions(expression)

    cause = exc_info.value.__cause__
    assert isinstance(cause, cause_class)
    assert message in str(cause)


def test_inliner_refuses_a_numeric_literal_in_a_piecewise_condition(
    function_registry_snapshot: None,
) -> None:
    """Test an argument that lands in a condition as a number raises ``ValueError``."""
    flag = Identifier("flag")
    register_function(
        "test_binding_choose",
        [flag],
        [FunctionSort.BOOL],
        FunctionSort.REAL,
        piecewise((IdentifierExpression(flag), 1), otherwise=0),
    )

    with pytest.raises(PassExecutionError) as exc_info:
        inline_functions(call("test_binding_choose", LiteralExpression(3)))

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause).startswith("inlining built an invalid piecewise: ")


def test_inliner_finishes_nested_builtins_a_hundred_deep() -> None:
    """Test ``relu`` nested 100 deep inlines quickly.

    An inliner that doubled its work per level would not finish this in
    five minutes.
    """
    tree: Expression = IdentifierExpression(Identifier("x"))
    for _ in range(100):
        tree = call("relu", tree)

    started = time.perf_counter()
    inlined = inline_functions(tree)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert inline_functions(inlined) is inlined


# =============================================================================
# Threads
# =============================================================================


def _run_threads(count: int, target: Callable[[int], object]) -> None:
    """Run ``target(index)`` on ``count`` threads started together."""
    barrier = threading.Barrier(count)

    def run(index: int) -> None:
        barrier.wait()
        target(index)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_concurrent_registrations_of_distinct_names_all_land(
    function_registry_snapshot: None,
) -> None:
    """Test registering distinct names from several threads loses none."""
    _run_threads(8, lambda index: _register_increment(f"test_binding_thread_{index}"))

    assert all(
        is_entry_registered(f"test_binding_thread_{index}") for index in range(8)
    )


def test_concurrent_registrations_of_one_name_let_exactly_one_win(
    function_registry_snapshot: None,
) -> None:
    """Test one of several racing registrations of a name wins."""
    winners: list[RegisteredFunction] = []
    losers: list[EntryRegistrationError] = []

    def register(index: int) -> None:
        del index
        try:
            winners.append(_register_increment("test_binding_race"))
        except EntryRegistrationError as error:
            losers.append(error)

    _run_threads(8, register)

    assert len(winners) == 1
    assert len(losers) == 7
    assert get_registered_entry("test_binding_race") is winners[0]


def test_lookups_and_screens_during_registrations_see_whole_states(
    function_registry_snapshot: None,
) -> None:
    """Test each snapshot seen during registrations is a whole registry state."""
    count = 200
    _register_increment("test_binding_growing_0")
    stop = threading.Event()
    views: list[int] = []
    failures: list[BaseException] = []

    def register() -> None:
        for index in range(1, count):
            _register_increment(f"test_binding_growing_{index}")
        stop.set()

    def read() -> None:
        try:
            while not stop.is_set():
                entries = get_registered_entries()
                names = [
                    name for name in entries if name.startswith("test_binding_growing_")
                ]
                expected = [
                    f"test_binding_growing_{index}" for index in range(len(names))
                ]
                assert names == expected
                assert all(
                    get_registered_entry(name) is entries[name] for name in names
                )
                views.append(len(names))
                validate_predicate(
                    call("test_binding_growing_0", LiteralExpression(1)) > 0
                )
        except BaseException as failure:
            failures.append(failure)

    writer = threading.Thread(target=register)
    reader = threading.Thread(target=read)
    writer.start()
    reader.start()
    writer.join()
    reader.join()

    assert not failures
    assert views == sorted(views)
    assert len(get_registered_entries()) >= count


# =============================================================================
# The built-ins' bodies, pinned as data
# =============================================================================

_BUILTIN_BODIES = {
    "max": "{a if ((a > b) || (a != a)); b otherwise}",
    "min": "{a if ((a < b) || (a != a)); b otherwise}",
    "abs": "{x if (x > 0); (0 - x) otherwise}",
    "sign": "{1 if (x > 0); -1 if (x < 0); 0 otherwise}",
    "clamp": "min(max(x, lo), hi)",
    "clamp_symmetric": "clamp(x, (-bound), bound)",
    "relu": "max(x, 0)",
    "leaky_relu": "{x if (x > 0); (x * slope) otherwise}",
    "xor": "((a || b) && (!(a && b)))",
    "nand": "(!(a && b))",
    "nor": "(!(a || b))",
    "implies": "((!a) || b)",
    "iff": "(a == b)",
    "sigmoid": "(1 / (1 + exp((-x))))",
    "silu": "(x * sigmoid(x))",
    "gelu": "((0.5 * x) * (1 + erf((x / sqrt(2)))))",
}


@pytest.mark.parametrize(("name", "body"), list(_BUILTIN_BODIES.items()))
def test_composed_builtin_body_prints_as_pinned(name: str, body: str) -> None:
    """Test each composed built-in's body is the catalogue's, as printed."""
    entry = get_registered_entry(name)

    assert isinstance(entry, RegisteredFunction)
    assert str(entry.body) == body
    assert entry.body.get_free_identifiers() == set(entry.parameters)


def test_every_composed_builtin_body_is_pinned() -> None:
    """Test the pinned bodies cover exactly the composed built-ins."""
    composed = {
        name
        for name, entry in BUILTIN_FUNCTIONS.items()
        if isinstance(entry, RegisteredFunction)
    }

    assert composed == set(_BUILTIN_BODIES)
