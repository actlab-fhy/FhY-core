"""Tests of the deprecated V1 wire format and its upgrade path.

V1 is written only inside ``wire_version(WireVersion.V1)``, reading or
writing it warns, readers tell the versions apart at each payload's root,
and `upgrade_v1_payload` and ``python -m fhy_core.serialization_upgrade``
convert stored V1 payloads (slice S17 of ``docs/design/python-switch.md``,
D-S17-14, D-S17-16 and D-S17-25). The tests are deleted with V1.
"""

import json
import warnings
from pathlib import Path

import pytest

from fhy_core import serialization_upgrade
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    Serializable,
    SerializationError,
    WireVersion,
    upgrade_v1_payload,
    wire_version,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import Param, create_ordinal_param

from ..v1 import writing_v1

_X = Identifier("x")


def _expression() -> Expression:
    return IdentifierExpression(_X) + LiteralExpression(1.5)


def test_writing_v1_warns_once_per_block() -> None:
    """Test entering a V1 block warns, and writing inside it does not again."""
    expression = _expression()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with wire_version(WireVersion.V1):
            expression.serialize_to_dict()
            expression.to_json()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "Writing the V1 wire format is deprecated" in str(deprecations[0].message)


def test_reading_v1_warns_once_per_payload_naming_the_upgrade() -> None:
    """Test reading a V1 payload warns once, at its root, naming the upgrade path."""
    with writing_v1():
        payload = _expression().serialize_to_dict()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rebuilt = Expression.deserialize_from_dict(payload)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "upgrade_v1_payload" in str(deprecations[0].message)
    assert rebuilt.is_structurally_equivalent(_expression())


def test_the_v1_warnings_name_the_release_that_removes_v1() -> None:
    """Test writing and reading V1 both warn that 0.3.0 removes it (R2-N2)."""
    with (
        pytest.warns(DeprecationWarning, match="removed in 0.3.0"),
        wire_version(WireVersion.V1),
    ):
        payload = _expression().serialize_to_dict()

    with pytest.warns(DeprecationWarning, match="removed in 0.3.0"):
        Expression.deserialize_from_dict(payload)


def test_an_unmarked_v1_read_fails_the_test() -> None:
    """Test the suite turns an unmarked V1 warning into an error (R2-N4).

    ``pyproject.toml`` filters the V1 warnings as errors, so a test that
    reads V1 without saying so fails instead of burying the warning.
    """
    with writing_v1():
        payload = _expression().serialize_to_dict()

    with pytest.raises(DeprecationWarning, match="V1 wire format"):
        Expression.deserialize_from_dict(payload)


def test_the_version_is_detected_at_the_root() -> None:
    """Test one reader reads a V1 envelope and a V2 table alike."""
    expression = _expression()
    with writing_v1():
        v1 = expression.serialize_to_dict()
    v2 = expression.serialize_to_dict()

    with pytest.warns(DeprecationWarning):
        from_v1 = Expression.deserialize_from_dict(v1)

    assert "__type__" in v1
    assert set(v2) == {"nodes"}
    assert from_v1.is_structurally_equivalent(Expression.deserialize_from_dict(v2))


def test_a_v1_param_is_detected_by_its_domain_envelope() -> None:
    """Test a plain class's V1 payload is told apart by the envelopes it nests."""
    param = create_ordinal_param([1, 2])
    with writing_v1():
        v1 = param.serialize_to_dict()

    with pytest.warns(DeprecationWarning):
        rebuilt: Param[int] = Param.deserialize_from_dict(v1)

    domain = v1["domain"]
    assert isinstance(domain, dict)
    assert "__type__" in domain
    assert rebuilt.is_structurally_equivalent(param)


def test_a_version_1_blob_decodes() -> None:
    """Test a binary blob of envelope version 1 still decodes."""
    with writing_v1():
        blob = _expression().to_bytes()

    with pytest.warns(DeprecationWarning):
        rebuilt = Serializable.from_bytes(blob)

    assert blob[4] == 1
    assert isinstance(rebuilt, Expression)
    assert rebuilt.is_structurally_equivalent(_expression())


def test_the_upgrade_converts_each_format() -> None:
    """Test `upgrade_v1_payload` returns V2 in the format it is given."""
    expression = _expression()
    param = create_ordinal_param([1, 2])
    with writing_v1():
        as_dict = expression.serialize_to_dict()
        as_text = param.to_json()
        as_blob = expression.to_bytes()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        upgraded_dict = upgrade_v1_payload(as_dict)
        upgraded_text = upgrade_v1_payload(as_text, Param)
        upgraded_blob = upgrade_v1_payload(as_blob)

    assert upgraded_dict == expression.serialize_to_dict()
    assert upgraded_text == param.to_json()
    assert isinstance(upgraded_blob, bytes)
    assert upgraded_blob[4] == 2


def test_the_upgrade_needs_the_class_of_a_plain_payload() -> None:
    """Test a payload that names no class of its own needs `cls`."""
    with writing_v1():
        payload = create_ordinal_param([1]).serialize_to_dict()

    with pytest.raises(SerializationError, match="cls"):
        upgrade_v1_payload(payload)


def test_the_entry_point_converts_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test ``python -m fhy_core.serialization_upgrade`` writes the V2 payload."""
    param = create_ordinal_param([1, 2])
    with writing_v1():
        (tmp_path / "param.json").write_text(param.to_json(), encoding="utf-8")
        (tmp_path / "expression.bin").write_bytes(_expression().to_bytes())

    assert (
        serialization_upgrade.main(["--type", "param", str(tmp_path / "param.json")])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == json.loads(param.to_json())
    assert (
        serialization_upgrade.main(
            [str(tmp_path / "expression.bin"), str(tmp_path / "out.bin")]
        )
        == 0
    )
    assert (tmp_path / "out.bin").read_bytes()[4] == 2
    assert (
        serialization_upgrade.main(["--type", "nowhere", str(tmp_path / "param.json")])
        == 2
    )
