"""Benchmarks of the interned tags' hot paths.

Each tag benchmark runs on `OpAttribute`, `NoteKind` and `ValueDomain`,
through a canonical instance each module ships.
"""

import operator

import pytest

from fhy_core.diagnostic import REMARK_NOTE_KIND, NoteKind
from fhy_core.op_attribute import COMMUTATIVE, OpAttribute
from fhy_core.value_domain import DATA_DOMAIN, ValueDomain

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="interned_tags")

_InternedTag = OpAttribute | NoteKind | ValueDomain

_CANONICAL_TAGS = pytest.mark.parametrize(
    "canonical",
    [COMMUTATIVE, REMARK_NOTE_KIND, DATA_DOMAIN],
    ids=["OpAttribute", "NoteKind", "ValueDomain"],
)


def _construct_with_same_key(canonical: _InternedTag) -> _InternedTag:
    return type(canonical)(canonical.name, canonical.description)


@_CANONICAL_TAGS
def test_interned_tag_construction_of_existing_key(
    benchmark: Benchmark, canonical: _InternedTag
) -> None:
    """Benchmark constructing a tag whose key already has a canonical tag."""
    benchmark(_construct_with_same_key, canonical)


@_CANONICAL_TAGS
def test_interned_tag_lookup(benchmark: Benchmark, canonical: _InternedTag) -> None:
    """Benchmark looking up the canonical tag for a key."""
    found = benchmark(type(canonical).require_interned, canonical.name)
    assert found is canonical


@_CANONICAL_TAGS
def test_interned_tag_eq(benchmark: Benchmark, canonical: _InternedTag) -> None:
    """Benchmark comparing a canonical tag with a distinct, equal tag."""
    assert benchmark(operator.eq, canonical, _construct_with_same_key(canonical))


@_CANONICAL_TAGS
def test_interned_tag_hash(benchmark: Benchmark, canonical: _InternedTag) -> None:
    """Benchmark hashing a tag."""
    benchmark(hash, canonical)


def test_value_domain_is_subdomain_of_root(
    benchmark: Benchmark, value_domain_chain: list[ValueDomain]
) -> None:
    """Benchmark a leaf domain walking its whole chain up to the root."""
    leaf, root = value_domain_chain[-1], value_domain_chain[0]
    assert benchmark(leaf.is_subdomain_of, root)


def test_value_domain_is_subdomain_of_unrelated(
    benchmark: Benchmark, value_domain_chain: list[ValueDomain]
) -> None:
    """Benchmark a leaf domain walking its whole chain without a match."""
    assert not benchmark(value_domain_chain[-1].is_subdomain_of, DATA_DOMAIN)
