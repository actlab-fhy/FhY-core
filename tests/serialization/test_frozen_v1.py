"""Pins the V1 wire format and old pickles against corpora frozen before S17.

``data/pickles_v1.json`` holds pickles (protocols 2 and 5) and
``data/v1_payloads.json`` the V1 payloads of the objects
``frozen_fixtures.build_fixtures`` builds, both written by the code before
slice S17 (``docs/design/python-switch.md``, D-S17-19 and D-S17-20). They
pin that old pickles load, that V1 payloads decode, and that V1 writing
is unchanged, until V1 is removed (D-S17-16), and are deleted with it.
"""

import base64
import contextlib
import json
import pickle
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fhy_core import serialization
from fhy_core.serialization import Serializable

from .frozen_fixtures import build_fixtures, is_equivalent

_DATA = Path(__file__).parent / "data"
_PICKLES: dict[str, dict[str, str]] = json.loads(
    (_DATA / "pickles_v1.json").read_text(encoding="utf-8")
)
_PAYLOADS: dict[str, Any] = json.loads(
    (_DATA / "v1_payloads.json").read_text(encoding="utf-8")
)
_FIXTURES = build_fixtures()


@contextlib.contextmanager
def _v1() -> Iterator[None]:
    """Write and read V1 without its deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wire_version = getattr(serialization, "wire_version", None)
        if wire_version is None:
            yield
        else:
            with wire_version(serialization.WireVersion.V1):
                yield


def test_the_corpora_cover_exactly_the_fixtures() -> None:
    """Test each corpus has one entry per fixture."""
    assert set(_PICKLES) == set(_FIXTURES)
    assert set(_PAYLOADS) == set(_FIXTURES)


@pytest.mark.parametrize("protocol", ["2", "5"])
@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_a_frozen_pickle_loads_to_the_same_value(name: str, protocol: str) -> None:
    """Test a pickle written before S17 loads to an equivalent object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        loaded = pickle.loads(base64.b64decode(_PICKLES[name][protocol]))
    assert is_equivalent(loaded, _FIXTURES[name])


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_a_frozen_v1_payload_decodes_to_the_same_value(name: str) -> None:
    """Test a V1 payload written before S17 decodes to an equivalent object."""
    expected = _FIXTURES[name]
    with _v1():
        rebuilt = type(expected).deserialize_from_dict(_PAYLOADS[name])
    assert is_equivalent(rebuilt, expected)


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_v1_writing_reproduces_the_frozen_payload(name: str) -> None:
    """Test V1 writing is the V1 writing of the code before S17."""
    instance: Serializable = _FIXTURES[name]
    with _v1():
        assert instance.serialize_to_dict() == _PAYLOADS[name]
