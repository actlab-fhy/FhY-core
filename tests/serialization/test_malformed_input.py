"""Tests that malformed deserialization input stays within ``SerializationError``.

The module documents its default deserialization paths as safe against
untrusted input, and the JSON/binary entry points advertise
``Raises: SerializationError``. These tests pin that contract: invalid UTF-8,
invalid JSON, and a constructor that raises ``TypeError`` must all surface as a
``SerializationError`` rather than a raw ``UnicodeDecodeError`` /
``json.JSONDecodeError`` / ``TypeError``.
"""

import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

import pytest

from fhy_core.serialization import (
    _HEADER_STRUCT,
    BinaryPayloadCodec,
    DeserializationValueError,
    MalformedPayloadError,
    Serializable,
    SerializationError,
    register_serializable,
)
from fhy_core.traits.frozen import FrozenMixin


@register_serializable(type_id="_test_malformed_point")
@dataclass(frozen=True)
class _Point(Serializable, FrozenMixin):
    x: int
    y: int


@register_serializable(type_id="_test_malformed_typeerror")
@dataclass(frozen=True)
class _RaisesTypeError(Serializable, FrozenMixin):
    x: int

    def __post_init__(self) -> None:
        raise TypeError("constructor rejects this payload")


@register_serializable(type_id="_test_malformed_serialerror")
@dataclass(frozen=True)
class _RaisesSerializationError(Serializable, FrozenMixin):
    x: int

    def __post_init__(self) -> None:
        raise DeserializationValueError("constructor raised a serialization error")


# ============================================================================
# Malformed payload error type
# ============================================================================


def test_malformed_payload_error_is_a_serialization_error() -> None:
    """Test ``MalformedPayloadError`` is catchable as ``SerializationError``."""
    assert issubclass(MalformedPayloadError, SerializationError)


# ============================================================================
# from_json
# ============================================================================


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param("{not valid json", "not valid JSON", id="invalid-json-text"),
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="invalid-utf8-bytes"),
    ],
)
def test_from_json_wraps_malformed_input(payload: str | bytes, match: str) -> None:
    """Test ``from_json`` reports malformed input as ``MalformedPayloadError``."""
    with pytest.raises(MalformedPayloadError, match=match):
        _Point.from_json(payload)


# ============================================================================
# deserialize_from_binary
# ============================================================================


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="invalid-utf8-payload"),
        pytest.param(b"{not json", "not valid JSON", id="invalid-json-payload"),
    ],
)
def test_deserialize_from_binary_wraps_malformed_payload(
    payload: bytes, match: str
) -> None:
    """Test ``deserialize_from_binary`` reports a malformed JSON payload."""
    with pytest.raises(MalformedPayloadError, match=match):
        _Point.deserialize_from_binary(payload, codec=BinaryPayloadCodec.JSON)


# ============================================================================
# from_bytes (binary envelope)
# ============================================================================


def test_from_bytes_wraps_non_utf8_type_id() -> None:
    """Test a binary envelope with a non-UTF-8 type_id stays a serialization error."""
    blob = bytearray(_Point(1, 2).to_bytes())
    # The type_id bytes immediately follow the fixed-size header; 0xFF is never
    # a valid UTF-8 leading byte, so decoding the type_id must fail.
    blob[_HEADER_STRUCT.size] = 0xFF

    with pytest.raises(SerializationError):
        Serializable.from_bytes(bytes(blob))


# ============================================================================
# Constructor errors surface through the SerializationError hierarchy
# ============================================================================


def test_constructor_type_error_surfaces_as_serialization_error() -> None:
    """Test a ``TypeError`` from a constructor surfaces as a serialization error."""
    # The constructor raises unconditionally, so the payload is a structurally
    # valid literal rather than a round-tripped instance.
    with pytest.raises(DeserializationValueError):
        _RaisesTypeError.deserialize_from_dict({"x": 0})


def test_constructor_serialization_error_propagates_unwrapped() -> None:
    """Test a constructor's ``SerializationError`` propagates without re-wrapping."""
    with pytest.raises(
        DeserializationValueError, match="constructor raised a serialization error"
    ) as exc_info:
        _RaisesSerializationError.deserialize_from_dict({"x": 0})

    assert str(exc_info.value) == "constructor raised a serialization error"


# =============================================================================
# Deep payloads (R2-013c)
#
# Each runs in a child process, so a regression that overflows the stack
# kills the child, not the run.
# =============================================================================


def _run_child(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


_DEEP_DICT_PROGRAM = """
    from fhy_core.serialization import SerializationError
    from fhy_core.symbol_table import SymbolTable
    from fhy_core.symbolic.expression import Expression
    from fhy_core.symbolic.param import Param
    from fhy_core.types import NumericalType

    classes = {{
        "Expression": Expression,
        "SymbolTable": SymbolTable,
        "NumericalType": NumericalType,
        "Param": Param,
    }}
    nested = []
    for _ in range({depth}):
        nested = [nested]
    try:
        classes["{cls}"].deserialize_from_dict({{"{key}": nested}})
    except SerializationError as error:
        print("SerializationError", type(error).__name__, str(error)[-60:])
    except Exception as error:
        print(type(error).__name__, str(error)[:200])
"""


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("cls", "key", "depth"),
    [
        pytest.param("Expression", "nodes", 30_000, id="expression_30000"),
        pytest.param("SymbolTable", "x", 40_000, id="symbol_table_40000"),
        pytest.param("NumericalType", "x", 40_000, id="numerical_type_40000"),
        pytest.param("Param", "x", 40_000, id="param_40000"),
    ],
)
def test_a_deep_payload_dict_raises_instead_of_crashing(
    cls: str, key: str, depth: int
) -> None:
    """Test a V2 payload dict nested far too deep is refused, not a crash.

    The binding's reader of a Python payload refuses more than 128 levels
    with `DeserializationValueError`; a shape Python checks first is refused
    by that check (probes `p04` and `p40`). Either is a `SerializationError`.
    """
    completed = _run_child(_DEEP_DICT_PROGRAM.format(cls=cls, key=key, depth=depth))

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.startswith("SerializationError"), completed.stdout


def test_a_payload_dict_128_levels_deep_is_read() -> None:
    """Test the reader's limit refuses nesting beyond 128 levels only.

    The payload's dict and 127 lists inside it are 128 levels: they are
    read, and then refused by the payload's shape, with serde's text; one
    more list is refused by the depth limit.
    """
    from fhy_core.symbolic.expression import Expression  # noqa: PLC0415

    nested: Any = []
    for _ in range(126):
        nested = [nested]

    with pytest.raises(DeserializationValueError) as shallow:
        Expression.deserialize_from_dict({"nodes": nested})
    with pytest.raises(DeserializationValueError) as deep:
        Expression.deserialize_from_dict({"nodes": [nested]})

    assert "128 levels" not in str(shallow.value)
    assert "the payload nests more than 128 levels" in str(deep.value)


_DEEP_MEMBER_PROGRAM = """
    from fhy_core.identifier import Identifier
    from fhy_core.symbolic.constraint import InSetConstraint

    member = 1
    for _ in range({depth}):
        member = (member,)
    x = Identifier("x")
    try:
        {action}
    except RecursionError as error:
        print("RecursionError", str(error)[:120])
    except Exception as error:
        print(type(error).__name__, str(error)[:200])
"""


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize(
    "action",
    [
        pytest.param("InSetConstraint(x, [member])", id="build"),
        pytest.param(
            "InSetConstraint(x, [1]).evaluate_with_bindings({x: member})", id="bind"
        ),
    ],
)
def test_a_deep_member_raises_recursion_error(action: str) -> None:
    """Test a member nested past the recursion limit raises `RecursionError`.

    The member reader counts its depth against `sys.getrecursionlimit()`, as
    the provenance binding does (probe `p26`).
    """
    completed = _run_child(_DEEP_MEMBER_PROGRAM.format(depth=20_000, action=action))

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.startswith("RecursionError"), completed.stdout
