"""Hypothesis property tests for `Provenance.fuse`.

`flatten_provenance_sources` is a test-side reimplementation of `fuse`'s
documented reduction (drop `UnknownProvenance`, splice the sources of any
metadata-less `FusedProvenance` in order, keep everything else opaque),
kept independent of `fuse`'s own loop. The properties check that `fuse`'s
result contains no `UnknownProvenance` and no metadata-less nested
`FusedProvenance`, equals the flattened input in order (collapsing to a
bare source or `UnknownProvenance` exactly as `fuse` documents for one or
zero remaining sources), and that fusing `fuse`'s own output alone is a
no-op.
"""

from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.provenance import (
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Provenance,
    UnknownProvenance,
)

pytestmark = pytest.mark.property

_FILE_PATHS: tuple[Path, ...] = (Path("a.fhy"), Path("b.fhy"), Path("c.fhy"))
_METADATA_VALUES: tuple[str, ...] = ("cse", "loop-fusion")

_A = FileProvenance(Path("a.fhy"))
_B = FileProvenance(Path("b.fhy"))
_C = FileProvenance(Path("c.fhy"))


def flatten_provenance_sources(
    provenances: tuple[Provenance, ...],
) -> list[Provenance]:
    """Return the flattened source list `Provenance.fuse` would build.

    Test-side reference for fuse's reduction rule: drop `UnknownProvenance`,
    and splice in the sources of any `FusedProvenance` whose metadata is
    `None` (recursively), in order. Any other provenance (`FileProvenance`,
    `NamedProvenance`, `CallSiteProvenance`, or a metadata-bearing
    `FusedProvenance`) is kept as one opaque source.
    """
    flat: list[Provenance] = []
    pending: list[Provenance] = list(reversed(provenances))
    while pending:
        provenance = pending.pop()
        if isinstance(provenance, UnknownProvenance):
            continue
        if isinstance(provenance, FusedProvenance) and provenance.metadata is None:
            pending.extend(reversed(provenance.sources))
            continue
        flat.append(provenance)
    return flat


def build_provenance_tree_strategy(max_depth: int = 2) -> st.SearchStrategy[Provenance]:
    """Build a strategy for provenance trees up to `max_depth` fused levels deep.

    Leaves are a file provenance, a named wrapper around one, or the
    unknown sentinel. Each recursive level nests 1 to 3 shallower trees in a
    `FusedProvenance`, with or without metadata.
    """
    leaf = st.one_of(
        st.sampled_from(_FILE_PATHS).map(FileProvenance),
        st.sampled_from(_FILE_PATHS).map(
            lambda path: NamedProvenance("n", FileProvenance(path))
        ),
        st.just(UnknownProvenance()),
    )
    if max_depth <= 0:
        return leaf

    inner = build_provenance_tree_strategy(max_depth - 1)
    metadata_strategy = st.one_of(st.none(), st.sampled_from(_METADATA_VALUES))
    fused = st.tuples(st.lists(inner, min_size=1, max_size=3), metadata_strategy).map(
        lambda pair: FusedProvenance(sources=tuple(pair[0]), metadata=pair[1])
    )
    return st.one_of(leaf, fused)


@st.composite
def draw_fuse_inputs(draw: st.DrawFn) -> tuple[Provenance, ...]:
    """Draw 0 to 4 provenance trees as positional inputs to Provenance.fuse."""
    tree_strategy = build_provenance_tree_strategy(max_depth=2)
    count = draw(st.integers(min_value=0, max_value=4))
    return tuple(draw(tree_strategy) for _ in range(count))


@example(inputs=(), metadata=None)
@example(inputs=(_A,), metadata=None)
@example(inputs=(_A,), metadata="cse")
@example(inputs=(_A, UnknownProvenance(), _B), metadata=None)
@example(
    inputs=(
        UnknownProvenance(),
        FusedProvenance(sources=(_A, _B)),
        UnknownProvenance(),
        FusedProvenance(sources=(_B, _C), metadata="x"),
        _C,
    ),
    metadata=None,
)
@given(
    inputs=draw_fuse_inputs(),
    metadata=st.one_of(st.none(), st.sampled_from(_METADATA_VALUES)),
)
def test_fuse_result_equals_flattened_input_with_no_unknown_or_bare_fused(
    inputs: tuple[Provenance, ...], metadata: str | None
) -> None:
    """Test fuse's result mirrors flatten_provenance_sources(inputs).

    Also checks fuse's documented collapse rule for zero and one remaining
    sources: zero sources collapse to `UnknownProvenance`, and exactly one
    source with no metadata is returned unwrapped.

    Oracle: flatten_provenance_sources, a reimplementation of the documented
    reduction independent of Provenance.fuse's own loop.
    """
    flat = flatten_provenance_sources(inputs)
    for source in flat:
        assert not isinstance(source, UnknownProvenance)
        assert not (isinstance(source, FusedProvenance) and source.metadata is None)

    result = Provenance.fuse(*inputs, metadata=metadata)

    if not flat:
        assert result == UnknownProvenance()
    elif len(flat) == 1 and metadata is None:
        assert result == flat[0]
    else:
        assert isinstance(result, FusedProvenance)
        assert result.sources == tuple(flat)
        assert result.metadata == metadata


@given(
    inputs=draw_fuse_inputs(),
    metadata=st.one_of(st.none(), st.sampled_from(_METADATA_VALUES)),
)
def test_fuse_is_idempotent_on_its_own_output(
    inputs: tuple[Provenance, ...], metadata: str | None
) -> None:
    """Test re-fusing fuse's own output (alone, no metadata) returns an equal value.

    Oracle: fuse's documented fixed point -- an already-reduced tree (no
    `UnknownProvenance`, no bare nested `FusedProvenance`) has nothing left
    to flatten, so wrapping it alone must be a no-op.
    """
    result = Provenance.fuse(*inputs, metadata=metadata)
    assert Provenance.fuse(result) == result
