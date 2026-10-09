"""Benchmarks of the type system, the lattice and the partially ordered set.

They measure ``fhy_core.types`` (without ``checking``), ``fhy_core.lattice``
and ``fhy_core.utils.poset``, which the Rust core backs, through the public
API only. The array rows bind, substitute and unify the pattern
``T[N, M]`` against ``int32[4, 8]``; the Python-type row does the same
through a wrapper type defined here, with handlers registered on the
dispatchers, as a downstream package defines its own types.
"""

import itertools
import pickle
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.lattice import Lattice
from fhy_core.serialization import SerializedDict
from fhy_core.symbol_table import VariableSymbolTableFrame
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.traits import VerificationError
from fhy_core.types import (
    CoreDataType,
    DataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeQualifier,
    TypeUnificationEnvironment,
    bind_template,
    is_structurally_equivalent,
    promote_core_data_types,
    resolve_literal_core_data_type,
    substitute_template,
    unify,
    unify_expression,
)
from fhy_core.utils.override import override
from fhy_core.utils.poset import PartiallyOrderedSet

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="types")

# How many elements the benchmarked chain poset holds.
_CHAIN_LENGTH = 50
# How many identifiers the unification chain binds in a row.
_BINDING_CHAIN_LENGTH = 10
# The size of the set whose powerset the `is_lattice` row checks.
_POWERSET_SIZE = 3

_INTEGER_PROMOTION_ELEMENTS = (
    CoreDataType.UINT,
    CoreDataType.UINT8,
    CoreDataType.UINT16,
    CoreDataType.UINT32,
    CoreDataType.INT,
    CoreDataType.INT8,
    CoreDataType.INT16,
    CoreDataType.INT32,
    CoreDataType.INT64,
)
_INTEGER_PROMOTION_ORDERS = (
    (CoreDataType.UINT, CoreDataType.UINT8),
    (CoreDataType.UINT8, CoreDataType.UINT16),
    (CoreDataType.UINT16, CoreDataType.UINT32),
    (CoreDataType.INT, CoreDataType.INT8),
    (CoreDataType.INT8, CoreDataType.INT16),
    (CoreDataType.INT16, CoreDataType.INT32),
    (CoreDataType.INT32, CoreDataType.INT64),
    (CoreDataType.UINT, CoreDataType.INT),
    (CoreDataType.UINT8, CoreDataType.INT16),
    (CoreDataType.UINT16, CoreDataType.INT32),
    (CoreDataType.UINT32, CoreDataType.INT64),
)


# ---------------------------------------------------------------------------
# A type defined outside the package
# ---------------------------------------------------------------------------


class _TaggedType(Type):
    """A `Type` wrapping another `Type` under a string tag."""

    _tag: str
    _inner: Type

    def __init__(self, tag: str, inner: Type) -> None:
        super().__init__()
        self._tag = tag
        self._inner = inner

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def inner(self) -> Type:
        return self._inner

    @override
    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    @override
    def deserialize_data_from_dict(  # pragma: no cover
        cls, data: SerializedDict
    ) -> "_TaggedType":
        raise NotImplementedError


@is_structurally_equivalent.register
def _(left: _TaggedType, right: object) -> bool:
    return (
        isinstance(right, _TaggedType)
        and left.tag == right.tag
        and is_structurally_equivalent(left.inner, right.inner)
    )


@bind_template.register
def _(
    pattern: _TaggedType, actual: Type, environment: TypeUnificationEnvironment
) -> TypeUnificationEnvironment:
    if not isinstance(actual, _TaggedType) or pattern.tag != actual.tag:
        raise VerificationError("tag mismatch")
    return bind_template(pattern.inner, actual.inner, environment)


# ---------------------------------------------------------------------------
# Builders and helpers
# ---------------------------------------------------------------------------


def _build_templated_array() -> tuple[
    NumericalType, Identifier, Identifier, Identifier
]:
    """Return the pattern `T[N, M]` and its three placeholders."""
    t, n, m = Identifier("T"), Identifier("N"), Identifier("M")
    pattern = NumericalType(
        TemplateDataType(t), [IdentifierExpression(n), IdentifierExpression(m)]
    )
    return pattern, t, n, m


def _build_int32_array() -> NumericalType:
    """Return `int32[4, 8]`."""
    return NumericalType(
        PrimitiveDataType(CoreDataType.INT32),
        [LiteralExpression(4), LiteralExpression(8)],
    )


def _build_chain(length: int) -> PartiallyOrderedSet[int]:
    """Return the chain `0 < 1 < ... < length - 1`."""
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()
    for element in range(length):
        poset.add_element(element)
    for element in range(length - 1):
        poset.add_order(element, element + 1)
    return poset


def _build_integer_promotion_lattice() -> Lattice[CoreDataType]:
    """Return the integer promotion lattice `types.core` defines."""
    lattice: Lattice[CoreDataType] = Lattice()
    for element in _INTEGER_PROMOTION_ELEMENTS:
        lattice.add_element(element)
    for lower, upper in _INTEGER_PROMOTION_ORDERS:
        lattice.add_order(lower, upper)
    return lattice


def _build_powerset_lattice(size: int) -> Lattice[frozenset[int]]:
    """Return the powerset of `range(size)` ordered by inclusion."""
    lattice: Lattice[frozenset[int]] = Lattice()
    subsets = [
        frozenset(index for index in range(size) if mask >> index & 1)
        for mask in range(1 << size)
    ]
    for subset in subsets:
        lattice.add_element(subset)
    for lower in subsets:
        for upper in subsets:
            if len(upper) == len(lower) + 1 and lower < upper:
                lattice.add_order(lower, upper)
    return lattice


def _compare_types(left: Type, right: Type) -> bool:
    """Return `left == right`, which is structural."""
    return left == right


def _hash_type(value: Type) -> int:
    """Return `hash(value)`, which is structural."""
    return hash(value)


# ---------------------------------------------------------------------------
# Construction and attributes
# ---------------------------------------------------------------------------


def test_primitive_data_type_construction(benchmark: Benchmark) -> None:
    """Construct a primitive data type."""
    benchmark(PrimitiveDataType, CoreDataType.INT32)


def test_template_data_type_construction(benchmark: Benchmark) -> None:
    """Construct a template data type with a width constraint."""
    identifier = Identifier("T")
    benchmark(TemplateDataType, identifier, [8, 16])


@pytest.mark.parametrize("shape_length", [0, 2], ids=["scalar", "shape_2"])
def test_numerical_type_construction(benchmark: Benchmark, shape_length: int) -> None:
    """Construct a numerical type, scalar or with a two-dimension shape."""
    data_type = PrimitiveDataType(CoreDataType.INT32)
    shape: Sequence[Expression] = [LiteralExpression(4), LiteralExpression(8)][
        :shape_length
    ]
    benchmark(NumericalType, data_type, shape)


def test_index_type_construction(benchmark: Benchmark) -> None:
    """Construct an index type with the default stride."""
    lower, upper = LiteralExpression(0), IdentifierExpression(Identifier("N"))
    benchmark(IndexType, lower, upper)


def test_numerical_type_data_type_access(benchmark: Benchmark) -> None:
    """Read a numerical type's data type."""
    array = _build_int32_array()
    benchmark(lambda: array.data_type)


def test_numerical_type_shape_access(benchmark: Benchmark) -> None:
    """Read a numerical type's shape."""
    array = _build_int32_array()
    benchmark(lambda: array.shape)


def test_numerical_type_eq(benchmark: Benchmark) -> None:
    """Compare two separately built, equal numerical types with `==`."""
    left, right = _build_int32_array(), _build_int32_array()
    benchmark(_compare_types, left, right)


def test_numerical_type_hash(benchmark: Benchmark) -> None:
    """Hash a numerical type."""
    array = _build_int32_array()
    benchmark(_hash_type, array)


@pytest.mark.parametrize("kind", ["primitive", "numerical_2d", "index"])
def test_structural_equivalence(benchmark: Benchmark, kind: str) -> None:
    """Compare two separately built, equal types structurally."""
    builders: dict[str, Callable[[], Any]] = {
        "primitive": lambda: PrimitiveDataType(CoreDataType.INT32),
        "numerical_2d": _build_int32_array,
        "index": lambda: IndexType(LiteralExpression(0), LiteralExpression(9)),
    }
    left, right = builders[kind](), builders[kind]()
    benchmark(is_structurally_equivalent, left, right)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (CoreDataType.UINT16, CoreDataType.INT8),
        (CoreDataType.FLOAT16, CoreDataType.COMPLEX64),
    ],
    ids=["integer", "float_complex"],
)
def test_promote_core_data_types(
    benchmark: Benchmark, first: CoreDataType, second: CoreDataType
) -> None:
    """Promote two core data types within one family."""
    benchmark(promote_core_data_types, first, second)


def test_resolve_literal_core_data_type(benchmark: Benchmark) -> None:
    """Resolve an integer literal against the weak `INT` context."""
    benchmark(resolve_literal_core_data_type, 300, CoreDataType.INT)


# ---------------------------------------------------------------------------
# The environment and the dispatchers
# ---------------------------------------------------------------------------


def test_environment_empty(benchmark: Benchmark) -> None:
    """Build an empty environment."""
    benchmark(TypeUnificationEnvironment.empty)


def test_environment_with_binding(benchmark: Benchmark) -> None:
    """Extend an environment by one expression binding."""
    environment = TypeUnificationEnvironment.empty()
    identifier, value = Identifier("N"), LiteralExpression(4)
    benchmark(environment.with_expression_binding, identifier, value)


def test_environment_structural_equivalence(benchmark: Benchmark) -> None:
    """Compare two environments of three bindings structurally."""
    pattern, _, _, _ = _build_templated_array()
    left = bind_template(
        pattern, _build_int32_array(), TypeUnificationEnvironment.empty()
    )
    right = bind_template(
        pattern, _build_int32_array(), TypeUnificationEnvironment.empty()
    )
    benchmark(left.is_structurally_equivalent, right)


def test_bind_template_of_a_templated_array(benchmark: Benchmark) -> None:
    """Bind `T[N, M]` against `int32[4, 8]`."""
    pattern, _, _, _ = _build_templated_array()
    actual, environment = _build_int32_array(), TypeUnificationEnvironment.empty()
    benchmark(bind_template, pattern, actual, environment)


def test_substitute_template_of_a_templated_array(benchmark: Benchmark) -> None:
    """Substitute the bindings of `T[N, M]` into it."""
    pattern, _, _, _ = _build_templated_array()
    environment = bind_template(
        pattern, _build_int32_array(), TypeUnificationEnvironment.empty()
    )
    benchmark(substitute_template, pattern, environment)


def test_unify_of_a_templated_array(benchmark: Benchmark) -> None:
    """Unify `T[N, M]` with `int32[4, 8]`."""
    pattern, _, _, _ = _build_templated_array()
    actual, environment = _build_int32_array(), TypeUnificationEnvironment.empty()
    benchmark(unify, pattern, actual, environment)


def test_unify_of_index_types(benchmark: Benchmark) -> None:
    """Unify `index(0:N:1)` with `index(0:9:1)`."""
    n = Identifier("N")
    expected = IndexType(LiteralExpression(0), IdentifierExpression(n))
    actual = IndexType(LiteralExpression(0), LiteralExpression(9))
    benchmark(unify, expected, actual, TypeUnificationEnvironment.empty())


def test_unify_expression_through_a_chain_of_10(benchmark: Benchmark) -> None:
    """Unify an identifier bound through a chain of ten with a literal."""
    identifiers = [Identifier(f"n{index}") for index in range(_BINDING_CHAIN_LENGTH)]
    environment = TypeUnificationEnvironment.empty()
    for current, following in itertools.pairwise(identifiers):
        environment = environment.with_expression_binding(
            current, IdentifierExpression(following)
        )
    head = IdentifierExpression(identifiers[0])
    benchmark(unify_expression, head, LiteralExpression(4), environment)


def test_bind_template_through_a_python_type(benchmark: Benchmark) -> None:
    """Bind through a Python-defined `Type` and its registered handlers."""
    pattern, _, _, _ = _build_templated_array()
    tagged_pattern = _TaggedType("dense", pattern)
    tagged_actual = _TaggedType("dense", _build_int32_array())
    environment = TypeUnificationEnvironment.empty()
    benchmark(bind_template, tagged_pattern, tagged_actual, environment)


# ---------------------------------------------------------------------------
# Serialization and printing
# ---------------------------------------------------------------------------


def test_type_serialize_to_dict(benchmark: Benchmark) -> None:
    """Serialize `int32[4, 8]` to its payload."""
    array = _build_int32_array()
    benchmark(array.serialize_to_dict)


def test_type_deserialize_from_dict(benchmark: Benchmark) -> None:
    """Deserialize `int32[4, 8]` through the family."""
    payload = _build_int32_array().serialize_to_dict()
    benchmark(Type.deserialize_from_dict, payload)


def test_type_pickle_round_trip(benchmark: Benchmark) -> None:
    """Pickle and unpickle `int32[4, 8]`."""
    array = _build_int32_array()
    benchmark(lambda: pickle.loads(pickle.dumps(array)))


def test_numerical_type_str(benchmark: Benchmark) -> None:
    """Print `int32[4, 8]`."""
    array = _build_int32_array()
    benchmark(str, array)


def test_data_type_is_a_data_type(benchmark: Benchmark) -> None:
    """Check a primitive data type against `DataType`."""
    data_type = PrimitiveDataType(CoreDataType.INT32)
    benchmark(isinstance, data_type, DataType)


# ---------------------------------------------------------------------------
# The consumer in `src`
# ---------------------------------------------------------------------------


def test_variable_symbol_table_frame_construction(benchmark: Benchmark) -> None:
    """Construct a symbol-table frame holding a type."""
    name, scalar = Identifier("x"), NumericalType(PrimitiveDataType(CoreDataType.INT32))
    benchmark(VariableSymbolTableFrame, name, scalar, TypeQualifier.STATE)


def test_variable_symbol_table_frame_hash(benchmark: Benchmark) -> None:
    """Hash a symbol-table frame, which hashes its type."""
    frame = VariableSymbolTableFrame(
        Identifier("x"),
        NumericalType(PrimitiveDataType(CoreDataType.INT32)),
        TypeQualifier.STATE,
    )
    benchmark(hash, frame)


# ---------------------------------------------------------------------------
# The partially ordered set and the lattice
# ---------------------------------------------------------------------------


def test_poset_construction_of_a_50_chain(benchmark: Benchmark) -> None:
    """Build a chain of 50 elements."""
    benchmark(_build_chain, _CHAIN_LENGTH)


def test_poset_is_less_than_across_a_50_chain(benchmark: Benchmark) -> None:
    """Ask for the order across the whole chain."""
    poset = _build_chain(_CHAIN_LENGTH)
    benchmark(poset.is_less_than, 0, _CHAIN_LENGTH - 1)


def test_poset_contains(benchmark: Benchmark) -> None:
    """Ask whether an element is a member."""
    poset = _build_chain(_CHAIN_LENGTH)
    benchmark(poset.__contains__, _CHAIN_LENGTH // 2)


def test_poset_iter_of_a_50_chain(benchmark: Benchmark) -> None:
    """Iterate the chain."""
    poset = _build_chain(_CHAIN_LENGTH)
    benchmark(lambda: list(poset))


def test_poset_iter_stable_of_a_50_chain(benchmark: Benchmark) -> None:
    """Iterate the chain in the stable order."""
    poset = _build_chain(_CHAIN_LENGTH)
    benchmark(lambda: list(poset.iter_stable()))


def test_lattice_construction_of_the_integer_promotion_order(
    benchmark: Benchmark,
) -> None:
    """Build the nine-element integer promotion lattice."""
    benchmark(_build_integer_promotion_lattice)


def test_lattice_join(benchmark: Benchmark) -> None:
    """Join two members of the integer promotion lattice."""
    lattice = _build_integer_promotion_lattice()
    benchmark(lattice.get_join, CoreDataType.UINT16, CoreDataType.INT8)


def test_lattice_meet(benchmark: Benchmark) -> None:
    """Meet two members of the integer promotion lattice."""
    lattice = _build_integer_promotion_lattice()
    benchmark(lattice.get_meet, CoreDataType.UINT32, CoreDataType.INT16)


def test_lattice_verify_of_the_integer_promotion_order(benchmark: Benchmark) -> None:
    """Verify the integer promotion lattice."""
    lattice = _build_integer_promotion_lattice()
    benchmark(lattice.verify)


def test_lattice_is_lattice_of_a_powerset(benchmark: Benchmark) -> None:
    """Check that the powerset of three elements is a lattice."""
    lattice = _build_powerset_lattice(_POWERSET_SIZE)
    benchmark(lattice.is_lattice)
