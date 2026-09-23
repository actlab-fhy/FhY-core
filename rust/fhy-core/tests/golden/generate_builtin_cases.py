"""Generate the golden built-in catalogue from the Python built-in oracle.

Records the content of ``fhy_core.symbolic.expression.builtins``, the
catalogue seeded into the registry at import:

* ``composed_functions``: every expression-bodied built-in, in
  ``BUILTIN_FUNCTIONS`` order, with its parameter name hints, parameter
  sorts, result sort, and body. The body is its wire dict
  (``serialize_to_dict``) with every identifier written as
  ``{"parameter": index}``, the position of that identifier in the
  function's parameter list, instead of ``{"id": ..., "name_hint": ...}``.
  JSON keeps each literal's type: ``0`` is an integer token and ``0.0`` a
  float token. The symbolic and functional printed forms (without ids) ride
  along for review;
* ``native_functions``: every natively computed built-in, in
  ``BUILTIN_FUNCTIONS`` order, with its parameter sorts and result sort;
* ``native_constants``: every built-in constant, in ``BUILTIN_CONSTANTS``
  order, with its sort, its value's ``repr``, and its value's IEEE 754 bits
  as a hexadecimal string (JSON cannot carry ``inf`` or ``nan``);
* ``registration_order``: the names in the order the registry holds them
  after import (constants, then natives, then composed functions).

Canonical constant identifiers and parameter ids are left out: the Rust
catalogue mints its own, and a replay compares identifiers by position.

The corpus is the whole catalogue, so it has no random part and takes no
seed. The Rust replay is ``rust/fhy-core/tests/builtins_equivalence.rs``.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_builtin_cases.py

This overwrites ``rust/fhy-core/tests/golden/builtin_cases.json``. The
ignored replay test reads a corpus written elsewhere with ``--output`` from
the file named in ``FHY_BUILTIN_CORPUS``.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

from _golden_support import build_provenance, write_document

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BUILTIN_CONSTANTS,
    BUILTIN_FUNCTIONS,
    CallExpression,
    Expression,
    pformat_expression,
)
from fhy_core.symbolic.expression.registry import (
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
    get_registered_entries,
)

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_builtin_cases.py"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = Path(__file__).resolve().with_name("builtin_cases.json")


def _replace_parameters(wire: Any, parameter_indices: dict[int, int]) -> Any:
    """Return ``wire`` with every identifier dict replaced by its parameter slot.

    Args:
        wire: A wire dict, list, or scalar of a composed body.
        parameter_indices: Parameter position by identifier id.

    Returns:
        The same structure with each ``{"id", "name_hint"}`` dict replaced by
        ``{"parameter": index}``.

    Raises:
        ValueError: If an identifier in ``wire`` is not a parameter.

    """
    if isinstance(wire, dict):
        if set(wire) == {"id", "name_hint"}:
            identifier_id = wire["id"]
            if identifier_id not in parameter_indices:
                raise ValueError(f"body identifier {wire!r} is not a parameter")
            return {"parameter": parameter_indices[identifier_id]}
        return {
            key: _replace_parameters(value, parameter_indices)
            for key, value in wire.items()
        }
    if isinstance(wire, list):
        return [_replace_parameters(item, parameter_indices) for item in wire]
    return wire


def _collect_call_names(expression: Expression) -> list[str]:
    """Return the function names of every call node in ``expression``."""
    names: list[str] = []
    pending = [expression]
    while pending:
        node = pending.pop()
        if isinstance(node, CallExpression):
            names.append(node.function_name)
        pending.extend(node.get_visit_children())
    return names


def _record_composed_function(entry: RegisteredFunction) -> dict[str, Any]:
    """Return the golden record of one composed built-in."""
    parameters: tuple[Identifier, ...] = tuple(entry.parameters)
    parameter_indices = {
        parameter.id: index for index, parameter in enumerate(parameters)
    }
    if len(parameter_indices) != len(parameters):
        raise ValueError(f"{entry.name} repeats a parameter")
    return {
        "name": entry.name,
        "parameter_names": [parameter.name_hint for parameter in parameters],
        "parameter_sorts": [sort.value for sort in entry.parameter_sorts],
        "result_sort": entry.result_sort.value,
        "body": _replace_parameters(entry.body.serialize_to_dict(), parameter_indices),
        "symbolic_print": pformat_expression(entry.body),
        "functional_print": pformat_expression(entry.body, functional=True),
    }


def _record_native_function(entry: NativeFunction) -> dict[str, Any]:
    """Return the golden record of one native built-in's signature."""
    return {
        "name": entry.name,
        "parameter_sorts": [sort.value for sort in entry.parameter_sorts],
        "result_sort": entry.result_sort.value,
    }


def _format_float_bits(value: float) -> str:
    """Return the IEEE 754 binary64 bits of ``value`` as ``0x`` + 16 hex digits."""
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return f"0x{bits:016x}"


def _record_catalogue() -> dict[str, Any]:
    """Return the catalogue sections of the golden document."""
    composed: list[dict[str, Any]] = []
    natives: list[dict[str, Any]] = []
    call_names: set[str] = set()
    for name, entry in BUILTIN_FUNCTIONS.items():
        if isinstance(entry, RegisteredFunction):
            composed.append(_record_composed_function(entry))
            call_names.update(_collect_call_names(entry.body))
        elif isinstance(entry, NativeFunction):
            natives.append(_record_native_function(entry))
        else:
            raise TypeError(f"{name} is neither composed nor native: {entry!r}")
        if entry.name != name:
            raise ValueError(f"{name} holds the entry of {entry.name}")
    constants: list[dict[str, Any]] = []
    for name, constant in BUILTIN_CONSTANTS.items():
        if not isinstance(constant, NativeConstant) or constant.name != name:
            raise TypeError(f"{name} holds {constant!r}")
        constants.append(
            {
                "name": constant.name,
                "sort": constant.sort.value,
                "value_repr": repr(constant.value),
                "value_bits": _format_float_bits(float(constant.value)),
            }
        )
    unknown = call_names - {record["name"] for record in composed + natives}
    if unknown:
        raise ValueError(f"composed bodies call non-built-in functions {unknown}")
    return {
        "composed_functions": composed,
        "native_functions": natives,
        "native_constants": constants,
        "registration_order": list(get_registered_entries()),
    }


def main() -> None:
    """Parse the options, read the catalogue, and write the corpus."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    document = {
        "provenance": build_provenance(_REPOSITORY_ROOT, GENERATOR_COMMAND),
        **_record_catalogue(),
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
