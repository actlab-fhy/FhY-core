"""Convert stored payloads of the deprecated V1 wire format to V2.

V1, the ``{"__type__": .., "__data__": ..}`` envelope format of the package
before 0.3, is deprecated, and its reader is removed in a later release
(0.4.0 is proposed). Convert stored payloads before then::

    python -m fhy_core.serialization_upgrade old.json > new.json
    python -m fhy_core.serialization_upgrade --type param old.json new.json
    python -m fhy_core.serialization_upgrade --import my_package blob.bin blob2.bin

A file starting with the binary envelope's ``FhYS`` magic is read and
written as a binary blob, and any other file as JSON text. A family
envelope and a binary blob name their own class; any other JSON payload,
such as a ``Param``'s or a ``SymbolTable``'s, needs ``--type`` with the
class's registered type id. Classes are looked up in the registry only, so
``--import`` names each module that registers a payload's classes; the
package's own classes are always registered. The function behind it is
`fhy_core.serialization.upgrade_v1_payload`.
"""

__all__ = ["main"]

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import fhy_core  # registers the package's classes
import fhy_core.symbol_table
import fhy_core.symbolic.param
import fhy_core.types  # noqa: F401
from fhy_core.serialization import (
    _MAGIC,
    _TYPE_REGISTRY,
    SerializationError,
    upgrade_v1_payload,
)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m fhy_core.serialization_upgrade",
        description="Convert a stored V1 payload (JSON text or a binary blob) to V2.",
    )
    parser.add_argument("input", type=Path, help="the V1 payload file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="where to write the V2 payload; standard output when omitted",
    )
    parser.add_argument(
        "--type",
        dest="type_id",
        help="the registered type id of a payload that is not a family envelope",
    )
    parser.add_argument(
        "--import",
        dest="modules",
        action="append",
        default=[],
        help="a module to import first, registering a payload's classes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Convert the file the arguments name, and return the exit status."""
    arguments = _parse_arguments(argv)
    for module in arguments.modules:
        importlib.import_module(module)
    cls = None
    if arguments.type_id is not None:
        cls = _TYPE_REGISTRY.get(arguments.type_id)
        if cls is None:
            print(f"unknown type id {arguments.type_id!r}", file=sys.stderr)
            return 2
    raw = arguments.input.read_bytes()
    try:
        if raw.startswith(_MAGIC):
            upgraded: str | bytes = upgrade_v1_payload(raw)  # type: ignore[assignment]
        else:
            upgraded = upgrade_v1_payload(raw.decode("utf-8"), cls)  # type: ignore[assignment]
    except (SerializationError, UnicodeDecodeError) as error:
        print(f"cannot upgrade {arguments.input}: {error}", file=sys.stderr)
        return 1
    if arguments.output is None:
        if isinstance(upgraded, bytes):
            sys.stdout.buffer.write(upgraded)
        else:
            sys.stdout.write(upgraded + "\n")
    elif isinstance(upgraded, bytes):
        arguments.output.write_bytes(upgraded)
    else:
        arguments.output.write_text(upgraded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
