"""Benchmarks of `Identifier`'s hot paths."""

import operator
import pickle

import pytest

from fhy_core.identifier import Identifier

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="identifier")


def _pickle_round_trip(identifier: Identifier) -> Identifier:
    restored: Identifier = pickle.loads(pickle.dumps(identifier))
    return restored


def test_identifier_construction(benchmark: Benchmark) -> None:
    """Benchmark constructing an identifier, which issues a new id."""
    benchmark(Identifier, "x")


def test_identifier_id_access(benchmark: Benchmark, identifier: Identifier) -> None:
    """Benchmark reading an identifier's id."""
    benchmark(operator.attrgetter("id"), identifier)


def test_identifier_name_hint_access(
    benchmark: Benchmark, identifier: Identifier
) -> None:
    """Benchmark reading an identifier's name hint."""
    benchmark(operator.attrgetter("name_hint"), identifier)


def test_identifier_eq(
    benchmark: Benchmark, identifier: Identifier, identifier_copy: Identifier
) -> None:
    """Benchmark comparing two distinct, equal identifier objects."""
    assert benchmark(operator.eq, identifier, identifier_copy)


def test_identifier_hash(benchmark: Benchmark, identifier: Identifier) -> None:
    """Benchmark hashing an identifier."""
    benchmark(hash, identifier)


def test_identifier_dict_lookup(
    benchmark: Benchmark,
    identifier_table: dict[Identifier, int],
    identifier: Identifier,
) -> None:
    """Benchmark looking up an identifier in an identifier-keyed dict."""
    benchmark(identifier_table.__getitem__, identifier)


def test_identifier_deserialize_from_dict(
    benchmark: Benchmark, identifier: Identifier
) -> None:
    """Benchmark deserializing an identifier from its dict."""
    data = identifier.serialize_to_dict()
    assert benchmark(Identifier.deserialize_from_dict, data) == identifier


def test_identifier_pickle_round_trip(
    benchmark: Benchmark, identifier: Identifier
) -> None:
    """Benchmark pickling an identifier and loading it back."""
    assert benchmark(_pickle_round_trip, identifier) == identifier
