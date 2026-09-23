"""Generate golden operation scripts from the `OpAttribute`/`ValueDomain` oracle.

Drives the real `fhy_core.op_attribute` and `fhy_core.value_domain` oracles
(the process-global `OpAttribute` and `ValueDomain` registries, including
their shipped default instances) through hand-picked and randomly generated
operation scripts, recording the observed outcome of each operation. The Rust
equivalence test (`rust/fhy-core/tests/tag_type_equivalence.rs`) replays these scripts
against `fhy_core::op_attribute` and `fhy_core::value_domain` and compares
every observation.

Identifiers are recorded as slots rather than ids, since concrete ids are
process-local in both runtimes: each script names a slot, and this generator
mints one `Identifier` per fresh slot the first time the script refers to it.
A handful of slot names are reserved as defaults, referring to the shipped
constants (`commutative`, `associative`, `pure`, `elementwise` for
`OpAttribute`; `data`, `address` for `ValueDomain`) instead of a freshly
minted identifier. A `bind_ahead` operation instead binds a slot to an id well
ahead of the id counter without restoring it, so a later `is_counter_past`
operation can observe whether a decode restored that id.

Run from the repository root:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_tag_type_cases.py

This overwrites `rust/fhy-core/tests/golden/tag_type_cases.json`. Options select a
larger random corpus written elsewhere, for the ignored expanded-corpus
equivalence test:

    uv run --no-sync python rust/fhy-core/tests/golden/generate_tag_type_cases.py \
        --seed 7 --random-count 2000 --max-ops 40 \
        --slots s0,s1,s2,s3,s4 --output /tmp/tag_type_corpus.json

then replay it by naming the file in `FHY_TAG_TYPE_CORPUS`:

    FHY_TAG_TYPE_CORPUS=/tmp/tag_type_corpus.json \
        cargo test --test tag_type_equivalence -- --ignored

`uv run nox -s golden_expanded` does both for every generator.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fhy_core.identifier import Identifier
from fhy_core.op_attribute import (
    ASSOCIATIVE,
    COMMUTATIVE,
    ELEMENTWISE,
    PURE,
    OpAttribute,
)
from fhy_core.serialization import DeserializationValueError, SerializationError
from fhy_core.value_domain import ADDRESS_DOMAIN, DATA_DOMAIN, ValueDomain

GENERATOR_COMMAND = (
    "uv run --no-sync python rust/fhy-core/tests/golden/generate_tag_type_cases.py"
)

_OP_ATTRIBUTE_DEFAULTS: dict[str, OpAttribute] = {
    "commutative": COMMUTATIVE,
    "associative": ASSOCIATIVE,
    "pure": PURE,
    "elementwise": ELEMENTWISE,
}
_VALUE_DOMAIN_DEFAULTS: dict[str, ValueDomain] = {
    "data": DATA_DOMAIN,
    "address": ADDRESS_DOMAIN,
}

_RANDOM_SLOTS = ["s0", "s1", "s2", "s3"]
_RANDOM_SEED = 20260919
_RANDOM_SCRIPT_COUNT = 110
_RANDOM_MAX_OPS = 26
_RANDOM_DOMAIN_PAYLOAD_DEPTH = 2

# Distance above a freshly minted id at which `bind_ahead` binds a slot, and
# between the ids of successive ahead slots in one script: far more ids than a
# script mints, so observing one ahead slot never passes the next.
_AHEAD_ID_SPACING = 1_000_000

_OP_ATTRIBUTE_KIND = "op_attribute"
_VALUE_DOMAIN_KIND = "value_domain"

_description_counter = 0


def _next_description() -> str:
    """Return a description text no other operation in this document uses."""
    global _description_counter  # noqa: PLW0603
    description = f"d{_description_counter}"
    _description_counter += 1
    return description


class _ScriptContext:
    """Per-script identifier-slot bookkeeping for one generated script.

    Every fresh (non-default) slot gets exactly one `Identifier`, minted the
    first time the script refers to it; a default slot resolves to the
    shipped constant's identifier instead. `registered_slots` tracks which
    slots currently hold the canonical instance for their identifier, so
    script construction can restrict operations such as `encode` or a
    `parent_slot` reference to slots that are actually interned. `held`
    keeps instances a `hold` operation set aside under a label, so a later
    operation can compare them with instances built after a clear.
    `ahead_slots` holds the slots `bind_ahead` bound to an id ahead of the
    counter.
    """

    def __init__(
        self,
        kind: str,
        cls: type[OpAttribute] | type[ValueDomain],
        defaults: dict[str, Any],
    ) -> None:
        self.kind = kind
        self.cls = cls
        self.default_names = tuple(defaults)
        self.slot_ids: dict[str, int] = {}
        self.id_to_slot: dict[int, str] = {}
        # Keyed by slot, values unused: an insertion-ordered set. A plain
        # `set` would iterate in an order that depends on the process's
        # string hash seed, so a seeded run would not reproduce.
        self.registered_slots: dict[str, None] = {}
        self.held: dict[str, Any] = {}
        self.ahead_slots: dict[str, None] = {}
        for slot, instance in defaults.items():
            self.bind_slot(slot, instance.name.id)
            self.mark_registered(slot)

    def bind_slot(self, slot: str, identifier_id: int) -> None:
        self.slot_ids[slot] = identifier_id
        self.id_to_slot[identifier_id] = slot

    def resolve_identifier(self, slot: str) -> Identifier:
        """Return `slot`'s identifier, minting a fresh one on first use."""
        if slot in self.slot_ids:
            return Identifier.deserialize_from_dict(
                {"id": self.slot_ids[slot], "name_hint": slot}
            )
        identifier = Identifier(slot)
        self.bind_slot(slot, identifier.id)
        return identifier

    def resolve_id(self, slot: str) -> int:
        """Return `slot`'s id, minting an identifier on first use.

        Unlike `resolve_identifier`, this leaves an ahead slot's id unrestored.
        """
        if slot in self.slot_ids:
            return self.slot_ids[slot]
        return self.resolve_identifier(slot).id

    def bind_ahead(self, slot: str) -> None:
        """Bind the fresh `slot` to an id ahead of the counter, unrestored."""
        if slot in self.slot_ids:
            raise ValueError(f"slot {slot!r} is already bound")
        spacing = _AHEAD_ID_SPACING * (len(self.ahead_slots) + 1)
        self.bind_slot(slot, Identifier(slot).id + spacing)
        self.ahead_slots[slot] = None

    def find_slot_for_id(self, identifier_id: int) -> str:
        return self.id_to_slot[identifier_id]

    def mark_registered(self, slot: str) -> None:
        self.registered_slots[slot] = None

    def reset_to_defaults(self) -> None:
        self.registered_slots = dict.fromkeys(self.default_names)


def _create_context(kind: str) -> _ScriptContext:
    if kind == _OP_ATTRIBUTE_KIND:
        return _ScriptContext(kind, OpAttribute, _OP_ATTRIBUTE_DEFAULTS)
    return _ScriptContext(kind, ValueDomain, _VALUE_DOMAIN_DEFAULTS)


# =============================================================================
# Encoded-value normalization (real dict, with a numeric id, <-> normalized
# dict, with the id replaced by the slot name it belongs to)
# =============================================================================


def _normalize_identifier_dict(
    raw: dict[str, Any], ctx: _ScriptContext
) -> dict[str, Any]:
    return {**raw, "id": ctx.find_slot_for_id(raw["id"])}


def _normalize_attribute_dict(
    raw: dict[str, Any], ctx: _ScriptContext
) -> dict[str, Any]:
    return {**raw, "name": _normalize_identifier_dict(raw["name"], ctx)}


def _normalize_domain_dict(raw: dict[str, Any], ctx: _ScriptContext) -> dict[str, Any]:
    parent = raw["parent"]
    return {
        **raw,
        "name": _normalize_identifier_dict(raw["name"], ctx),
        "parent": None if parent is None else _normalize_domain_dict(parent, ctx),
    }


def _denormalize_identifier_dict(
    norm: dict[str, Any], ctx: _ScriptContext
) -> dict[str, Any]:
    return {"id": ctx.resolve_id(norm["id"]), "name_hint": norm["name_hint"]}


def _denormalize_attribute_dict(
    norm: dict[str, Any], ctx: _ScriptContext
) -> dict[str, Any]:
    return {
        "name": _denormalize_identifier_dict(norm["name"], ctx),
        "description": norm["description"],
    }


def _denormalize_domain_dict(
    norm: dict[str, Any], ctx: _ScriptContext
) -> dict[str, Any]:
    parent = norm["parent"]
    return {
        "name": _denormalize_identifier_dict(norm["name"], ctx),
        "description": norm["description"],
        "parent": None if parent is None else _denormalize_domain_dict(parent, ctx),
    }


def _collect_domain_payload_slots(payload: dict[str, Any]) -> list[str]:
    slots = [payload["name"]["id"]]
    parent = payload["parent"]
    if parent is not None:
        slots.extend(_collect_domain_payload_slots(parent))
    return slots


# =============================================================================
# Op builders (the input half of an operation, before it is run)
# =============================================================================


def _build_new_attribute_op(slot: str, description: str) -> dict[str, Any]:
    return {"op": "new", "slot": slot, "description": description}


def _build_new_domain_op(
    slot: str, description: str, parent_slot: str | None = None
) -> dict[str, Any]:
    return {
        "op": "new",
        "slot": slot,
        "description": description,
        "parent_slot": parent_slot,
    }


def _build_get_op(slot: str) -> dict[str, Any]:
    return {"op": "get", "slot": slot}


def _build_require_op(slot: str) -> dict[str, Any]:
    return {"op": "require", "slot": slot}


def _build_clear_op() -> dict[str, Any]:
    return {"op": "clear"}


def _build_encode_op(slot: str) -> dict[str, Any]:
    return {"op": "encode", "slot": slot}


def _build_decode_op(
    payload: dict[str, Any], defect: dict[str, Any] | None = None
) -> dict[str, Any]:
    if defect is None:
        return {"op": "decode", "payload": payload}
    return {"op": "decode", "payload": payload, "defect": defect}


def _build_remove_defect(*path: str) -> dict[str, Any]:
    return {"kind": "remove", "path": list(path)}


def _build_set_defect(path: Sequence[str], value: Any) -> dict[str, Any]:
    return {"kind": "set", "path": list(path), "value": value}


def _build_wrap_defect(path: Sequence[str], key: str) -> dict[str, Any]:
    return {"kind": "wrap", "path": list(path), "key": key}


def _build_lift_defect(path: Sequence[str], key: str) -> dict[str, Any]:
    return {"kind": "lift", "path": list(path), "key": key}


def _build_is_subdomain_of_op(slot_a: str, slot_b: str) -> dict[str, Any]:
    return {"op": "is_subdomain_of", "slot_a": slot_a, "slot_b": slot_b}


def _build_eq_op(slot_a: str, slot_b: str) -> dict[str, Any]:
    return {"op": "eq", "slot_a": slot_a, "slot_b": slot_b}


def _build_eq_with_duplicate_attribute_op(
    slot: str, other_description: str
) -> dict[str, Any]:
    return {
        "op": "eq_with_duplicate",
        "slot": slot,
        "other_description": other_description,
    }


def _build_eq_with_duplicate_domain_op(
    slot: str, other_description: str, other_parent_slot: str | None
) -> dict[str, Any]:
    return {
        "op": "eq_with_duplicate",
        "slot": slot,
        "other_description": other_description,
        "other_parent_slot": other_parent_slot,
    }


def _build_is_subdomain_of_with_duplicate_op(
    self_slot: str,
    other_slot: str,
    other_description: str,
    other_parent_slot: str | None,
) -> dict[str, Any]:
    return {
        "op": "is_subdomain_of_with_duplicate",
        "self_slot": self_slot,
        "other_slot": other_slot,
        "other_description": other_description,
        "other_parent_slot": other_parent_slot,
    }


def _build_hold_op(slot: str, label: str) -> dict[str, Any]:
    return {"op": "hold", "slot": slot, "label": label}


def _build_eq_held_op(label: str, slot: str) -> dict[str, Any]:
    return {"op": "eq_held", "label": label, "slot": slot}


def _build_bind_ahead_op(slot: str) -> dict[str, Any]:
    return {"op": "bind_ahead", "slot": slot}


def _build_is_counter_past_op(slot: str) -> dict[str, Any]:
    return {"op": "is_counter_past", "slot": slot}


def _build_identifier_payload(slot: str) -> dict[str, Any]:
    return {"id": slot, "name_hint": slot}


def _build_attribute_payload(slot: str, description: str) -> dict[str, Any]:
    return {"name": _build_identifier_payload(slot), "description": description}


def _build_domain_payload(
    slot: str, description: str, parent: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "name": _build_identifier_payload(slot),
        "description": description,
        "parent": parent,
    }


# =============================================================================
# Op execution (records the oracle's observation for one operation)
# =============================================================================


def _describe_canonical(
    ctx: _ScriptContext, canonical: OpAttribute | ValueDomain
) -> dict[str, Any]:
    """Return the fields every op records about a canonical instance.

    A domain also records its parent's slot, or `None` for a root domain.
    """
    if isinstance(canonical, OpAttribute):
        return {"canonical_description": canonical.description}
    parent_slot = (
        ctx.find_slot_for_id(canonical.parent.name.id)
        if canonical.parent is not None
        else None
    )
    return {
        "canonical_description": canonical.description,
        "canonical_parent_slot": parent_slot,
    }


def _run_new(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    slot = op["slot"]
    description = op["description"]
    identifier = ctx.resolve_identifier(slot)

    instance: OpAttribute | ValueDomain
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        instance = OpAttribute(identifier, description)
    else:
        parent_slot = op["parent_slot"]
        parent = None
        if parent_slot is not None:
            parent = ValueDomain.require_interned(ctx.resolve_identifier(parent_slot))
        instance = ValueDomain(identifier, description, parent)
    canonical = ctx.cls.get_interned(identifier)
    if canonical is None:
        raise RuntimeError("new must register a canonical instance")
    ctx.mark_registered(slot)
    return {"registered": canonical is instance, **_describe_canonical(ctx, canonical)}


def _run_get(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    identifier = ctx.resolve_identifier(op["slot"])
    canonical = ctx.cls.get_interned(identifier)
    if canonical is not None:
        return _describe_canonical(ctx, canonical)
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        return {"canonical_description": None}
    return {"canonical_description": None, "canonical_parent_slot": None}


def _run_require(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    identifier = ctx.resolve_identifier(op["slot"])
    try:
        canonical = ctx.cls.require_interned(identifier)
    except KeyError:
        return {"error": "KeyError"}
    return _describe_canonical(ctx, canonical)


def _run_clear(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    del op  # clear takes no arguments; kept for a uniform dispatch signature
    ctx.cls.clear_interned_registry()
    ctx.cls.register_default_instances()
    ctx.reset_to_defaults()
    return {}


def _run_encode(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    identifier = ctx.resolve_identifier(op["slot"])
    canonical = ctx.cls.require_interned(identifier)
    raw = canonical.serialize_to_dict()
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        return {"encoding": _normalize_attribute_dict(raw, ctx)}
    return {"encoding": _normalize_domain_dict(raw, ctx)}


def _mark_registered_payload_slots(ctx: _ScriptContext, slots: list[str]) -> None:
    """Mark each of `slots` that the registry now holds as registered.

    A rejected payload may still have registered the fresh parents nested in
    it, which decode before the value holding them. An ahead slot stays
    unmarked: looking it up would restore its id, and the `is_counter_past`
    operation that follows the decode must see the counter as the decode left
    it.
    """
    for slot in slots:
        if slot in ctx.ahead_slots:
            continue
        if ctx.cls.get_interned(ctx.resolve_identifier(slot)) is not None:
            ctx.mark_registered(slot)


def _apply_defect(payload: dict[str, Any], defect: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a real (denormalized) payload with one structural defect.

    `defect["path"]` names the field to damage, as the keys leading to it
    from the payload's root. `remove` deletes that field, `set` replaces or
    adds it with the literal `defect["value"]`, `wrap` nests it one level
    deeper under `defect["key"]` (an empty path wraps the whole payload), and
    `lift` replaces it with its own `defect["key"]` member.
    """
    damaged = copy.deepcopy(payload)
    kind = defect["kind"]
    path = defect["path"]
    if not path:
        if kind != "wrap":
            raise ValueError(f"a {kind!r} defect needs a non-empty path")
        return {defect["key"]: damaged}
    container = damaged
    for key in path[:-1]:
        container = container[key]
    field_name = path[-1]
    if kind == "remove":
        del container[field_name]
    elif kind == "set":
        container[field_name] = copy.deepcopy(defect["value"])
    elif kind == "wrap":
        container[field_name] = {defect["key"]: container[field_name]}
    elif kind == "lift":
        container[field_name] = container[field_name][defect["key"]]
    else:
        raise ValueError(f"unknown defect kind {kind!r}")
    return damaged


# Text both runtimes put in the error for a payload that conflicts with the
# canonical instance for its key.
_CANONICAL_CONFLICT_MESSAGE = "conflicts with the canonical instance"


def _run_decode(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    """Decode a payload, recording the canonical it yields or its rejection.

    A payload whose key is already canonical must equal that canonical: one
    that names a different parent is rejected with a
    `DeserializationValueError`, recorded as the op's `error`. An op with a
    `defect` damages the payload's structure before decoding it (see
    `_apply_defect`); the oracle's rejection of the damaged payload is
    recorded as the op's `error`, named by the raised error's class. Every
    rejection also records as `conflict` whether it reported a conflict with
    the canonical instance, since both kinds of rejection raise the same
    class.
    """
    payload = op["payload"]
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        real = _denormalize_attribute_dict(payload, ctx)
        payload_slots = [payload["name"]["id"]]
    else:
        real = _denormalize_domain_dict(payload, ctx)
        payload_slots = _collect_domain_payload_slots(payload)

    defect = op.get("defect")
    # A well-formed payload can only be rejected as a canonical conflict; a
    # damaged one may be rejected with any serialization error.
    rejection_type: type[SerializationError] = SerializationError
    if defect is None:
        rejection_type = DeserializationValueError
    else:
        real = _apply_defect(real, defect)
    try:
        canonical = ctx.cls.deserialize_from_dict(real)
    except rejection_type as error:
        _mark_registered_payload_slots(ctx, payload_slots)
        is_conflict = _CANONICAL_CONFLICT_MESSAGE in str(error)
        if defect is None and not is_conflict:
            raise RuntimeError(
                f"a well-formed payload was rejected for another reason: {error}"
            ) from error
        return {"error": type(error).__name__, "conflict": is_conflict}

    _mark_registered_payload_slots(ctx, payload_slots)
    slot = ctx.find_slot_for_id(canonical.name.id)
    return {"canonical_slot": slot, **_describe_canonical(ctx, canonical)}


def _run_is_subdomain_of(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    a = ValueDomain.require_interned(ctx.resolve_identifier(op["slot_a"]))
    b = ValueDomain.require_interned(ctx.resolve_identifier(op["slot_b"]))
    return {"result": a.is_subdomain_of(b)}


def _run_eq(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    a = ctx.cls.require_interned(ctx.resolve_identifier(op["slot_a"]))
    b = ctx.cls.require_interned(ctx.resolve_identifier(op["slot_b"]))
    return {"result": a == b}


def _run_eq_with_duplicate(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    """Compare a slot's canonical instance against a fresh, non-canonical duplicate.

    Builds a second instance for `op["slot"]`'s already-registered identifier.
    Interning it loses the registration race, so the constructor hands back a
    plain, unregistered value (the canonical stays put); comparing the two
    exercises `==` the way a discarded duplicate would, which the ordinary
    `eq` op (always comparing two already-canonical, differently-named
    instances) never reaches.
    """
    slot = op["slot"]
    identifier = ctx.resolve_identifier(slot)
    canonical = ctx.cls.require_interned(identifier)
    other_description = op["other_description"]
    fresh: OpAttribute | ValueDomain
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        fresh = OpAttribute(identifier, other_description)
    else:
        other_parent_slot = op["other_parent_slot"]
        other_parent = None
        if other_parent_slot is not None:
            other_parent = ValueDomain.require_interned(
                ctx.resolve_identifier(other_parent_slot)
            )
        fresh = ValueDomain(identifier, other_description, other_parent)
    return {"result": canonical == fresh}


def _run_is_subdomain_of_with_duplicate(
    ctx: _ScriptContext, op: dict[str, Any]
) -> dict[str, Any]:
    """Test `is_subdomain_of` against a fresh duplicate rather than a canonical.

    `is_subdomain_of` is defined in terms of `==`; passing a duplicate (built
    the same way as `_run_eq_with_duplicate`) as `other` forces the walk to
    compare against a value that shares a registered slot's name but not
    necessarily its parent, which two canonical handles never exercise.
    """
    self_domain = ValueDomain.require_interned(ctx.resolve_identifier(op["self_slot"]))
    other_identifier = ctx.resolve_identifier(op["other_slot"])
    other_parent_slot = op["other_parent_slot"]
    other_parent = None
    if other_parent_slot is not None:
        other_parent = ValueDomain.require_interned(
            ctx.resolve_identifier(other_parent_slot)
        )
    fresh_other = ValueDomain(other_identifier, op["other_description"], other_parent)
    return {"result": self_domain.is_subdomain_of(fresh_other)}


def _run_hold(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    """Set a slot's current canonical instance aside under `op["label"]`."""
    identifier = ctx.resolve_identifier(op["slot"])
    ctx.held[op["label"]] = ctx.cls.require_interned(identifier)
    return {}


def _run_eq_held(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    """Compare a held instance with a slot's current canonical instance.

    A held instance may predate a clear, so this reaches `==` between two
    instances built independently for one identifier, whose parents are
    distinct instances that compare by value.
    """
    held = ctx.held[op["label"]]
    canonical = ctx.cls.require_interned(ctx.resolve_identifier(op["slot"]))
    return {"result": held == canonical}


def _run_bind_ahead(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    """Bind a fresh slot to an id ahead of the counter; see `bind_ahead`."""
    ctx.bind_ahead(op["slot"])
    return {}


def _run_is_counter_past(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    """Record whether the id counter has passed a slot's id.

    Mints one identifier to find out, and leaves the slot's id unrestored.
    """
    return {"result": Identifier("counter-probe").id > ctx.resolve_id(op["slot"])}


_OP_RUNNERS: dict[str, Callable[[_ScriptContext, dict[str, Any]], dict[str, Any]]] = {
    "new": _run_new,
    "get": _run_get,
    "require": _run_require,
    "clear": _run_clear,
    "encode": _run_encode,
    "decode": _run_decode,
    "is_subdomain_of": _run_is_subdomain_of,
    "eq": _run_eq,
    "eq_with_duplicate": _run_eq_with_duplicate,
    "is_subdomain_of_with_duplicate": _run_is_subdomain_of_with_duplicate,
    "hold": _run_hold,
    "eq_held": _run_eq_held,
    "bind_ahead": _run_bind_ahead,
    "is_counter_past": _run_is_counter_past,
}


def _run_op(ctx: _ScriptContext, op: dict[str, Any]) -> dict[str, Any]:
    runner = _OP_RUNNERS.get(op["op"])
    if runner is None:
        raise ValueError(f"unknown op {op['op']!r}")
    return runner(ctx, op)


def _run_fixed_script(
    name: str, kind: str, ops: list[dict[str, Any]]
) -> dict[str, Any]:
    ctx = _create_context(kind)
    recorded_ops = []
    for op in ops:
        expected = _run_op(ctx, op)
        recorded_ops.append({**op, "expected": expected})
    return {"name": name, "type": kind, "ops": recorded_ops}


# =============================================================================
# Hand-picked scripts
# =============================================================================


def _list_hand_picked_attribute_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    duplicate_shared_description = _next_description()
    return [
        (
            "eq_with_duplicate_ignores_description_whether_same_or_different",
            [
                _build_new_attribute_op("dupattr", duplicate_shared_description),
                _build_eq_with_duplicate_attribute_op(
                    "dupattr", duplicate_shared_description
                ),
                _build_eq_with_duplicate_attribute_op("dupattr", _next_description()),
            ],
        ),
        (
            "duplicate_new_keeps_the_first_canonical",
            [
                _build_new_attribute_op("dup", _next_description()),
                _build_new_attribute_op("dup", _next_description()),
                _build_get_op("dup"),
                _build_require_op("dup"),
            ],
        ),
        (
            "get_and_require_before_any_new",
            [_build_get_op("s0"), _build_require_op("s0")],
        ),
        (
            "new_clear_get_renew_get",
            [
                _build_new_attribute_op("s0", _next_description()),
                _build_clear_op(),
                _build_get_op("s0"),
                _build_new_attribute_op("s0", _next_description()),
                _build_get_op("s0"),
            ],
        ),
        (
            "interning_a_default_slot",
            [
                _build_get_op("commutative"),
                _build_new_attribute_op("commutative", _next_description()),
                _build_require_op("commutative"),
            ],
        ),
        (
            "clear_as_the_first_operation",
            [
                _build_clear_op(),
                _build_require_op("s0"),
                _build_require_op("commutative"),
            ],
        ),
        (
            "decode_of_a_registered_slot_ignores_the_payload_description",
            [
                _build_new_attribute_op("s0", _next_description()),
                _build_decode_op(_build_attribute_payload("s0", _next_description())),
            ],
        ),
        (
            "decode_of_an_unregistered_slot_registers_it",
            [_build_decode_op(_build_attribute_payload("s1", _next_description()))],
        ),
        (
            "encode_then_decode_round_trip",
            [
                _build_new_attribute_op("s0", _next_description()),
                _build_encode_op("s0"),
                _build_decode_op(_build_attribute_payload("s0", _next_description())),
                _build_eq_op("s0", "s0"),
            ],
        ),
    ]


def _list_hand_picked_domain_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    duplicate_shared_description = _next_description()
    return [
        (
            "eq_with_duplicate_ignores_description_whether_same_or_different",
            [
                _build_new_domain_op(
                    "dupdom", duplicate_shared_description, parent_slot="data"
                ),
                _build_eq_with_duplicate_domain_op(
                    "dupdom", duplicate_shared_description, "data"
                ),
                _build_eq_with_duplicate_domain_op(
                    "dupdom", _next_description(), "data"
                ),
            ],
        ),
        (
            "eq_with_duplicate_is_unequal_when_the_parent_differs",
            [
                _build_new_domain_op(
                    "parented", _next_description(), parent_slot="data"
                ),
                _build_eq_with_duplicate_domain_op(
                    "parented", _next_description(), "data"
                ),
                _build_eq_with_duplicate_domain_op(
                    "parented", _next_description(), "address"
                ),
                _build_eq_with_duplicate_domain_op(
                    "parented", _next_description(), None
                ),
            ],
        ),
        (
            "is_subdomain_of_with_duplicate_other",
            [
                _build_new_domain_op("mid4", _next_description(), parent_slot="data"),
                _build_new_domain_op("leaf4", _next_description(), parent_slot="mid4"),
                _build_is_subdomain_of_with_duplicate_op(
                    "leaf4", "mid4", _next_description(), "data"
                ),
                _build_is_subdomain_of_with_duplicate_op(
                    "leaf4", "mid4", _next_description(), "address"
                ),
                _build_is_subdomain_of_with_duplicate_op(
                    "leaf4", "mid4", _next_description(), None
                ),
            ],
        ),
        (
            "duplicate_new_keeps_the_first_canonical",
            [
                _build_new_domain_op("dup", _next_description()),
                _build_new_domain_op("dup", _next_description()),
                _build_get_op("dup"),
                _build_require_op("dup"),
            ],
        ),
        (
            "get_and_require_before_any_new",
            [_build_get_op("s0"), _build_require_op("s0")],
        ),
        (
            "a_domain_parented_on_a_default",
            [
                _build_new_domain_op("s0", _next_description(), parent_slot="data"),
                _build_get_op("s0"),
                _build_is_subdomain_of_op("s0", "data"),
                _build_is_subdomain_of_op("data", "s0"),
            ],
        ),
        (
            "a_three_deep_chain",
            [
                _build_new_domain_op("root", _next_description()),
                _build_new_domain_op("mid", _next_description(), parent_slot="root"),
                _build_new_domain_op("leaf", _next_description(), parent_slot="mid"),
                _build_is_subdomain_of_op("leaf", "root"),
                _build_is_subdomain_of_op("leaf", "mid"),
                _build_is_subdomain_of_op("root", "leaf"),
                _build_is_subdomain_of_op("leaf", "leaf"),
                _build_encode_op("leaf"),
            ],
        ),
        (
            "clear_as_the_first_operation",
            [_build_clear_op(), _build_require_op("s0"), _build_require_op("data")],
        ),
        (
            "interning_a_default_slot",
            [
                _build_get_op("address"),
                _build_new_domain_op("address", _next_description()),
                _build_require_op("address"),
            ],
        ),
        (
            "decode_of_a_registered_slot_ignores_the_payload_description",
            [
                _build_new_domain_op("s0", _next_description(), parent_slot="data"),
                _build_decode_op(
                    _build_domain_payload(
                        "s0",
                        _next_description(),
                        _build_domain_payload("data", _next_description(), None),
                    )
                ),
            ],
        ),
        (
            "decode_of_an_unregistered_slot_registers_it_with_a_default_parent",
            [
                _build_decode_op(
                    _build_domain_payload(
                        "s1",
                        _next_description(),
                        _build_domain_payload("address", _next_description(), None),
                    )
                )
            ],
        ),
        (
            "decode_with_a_divergent_parent_is_rejected",
            [
                _build_new_domain_op("child", _next_description(), parent_slot="data"),
                _build_decode_op(
                    _build_domain_payload(
                        "child",
                        _next_description(),
                        _build_domain_payload("address", _next_description(), None),
                    )
                ),
                _build_require_op("child"),
            ],
        ),
        (
            "decode_with_a_dropped_parent_is_rejected",
            [
                _build_new_domain_op("child", _next_description(), parent_slot="data"),
                _build_decode_op(
                    _build_domain_payload("child", _next_description(), None)
                ),
                _build_require_op("child"),
            ],
        ),
        (
            "a_rejected_decode_keeps_a_fresh_nested_parent_registered",
            [
                _build_new_domain_op("child", _next_description(), parent_slot="data"),
                _build_decode_op(
                    _build_domain_payload(
                        "child",
                        _next_description(),
                        _build_domain_payload("fresh7", _next_description(), None),
                    )
                ),
                _build_require_op("fresh7"),
                _build_require_op("child"),
            ],
        ),
        (
            "decode_after_a_clear_accepts_a_parent_equal_by_value",
            [
                _build_new_domain_op("root8", _next_description()),
                _build_new_domain_op("leaf8", _next_description(), parent_slot="root8"),
                _build_clear_op(),
                _build_new_domain_op("root8", _next_description()),
                _build_new_domain_op("leaf8", _next_description(), parent_slot="root8"),
                _build_decode_op(
                    _build_domain_payload(
                        "leaf8",
                        _next_description(),
                        _build_domain_payload("root8", _next_description(), None),
                    )
                ),
            ],
        ),
        (
            "decode_registers_a_fresh_nested_parent_too",
            [
                _build_decode_op(
                    _build_domain_payload(
                        "leaf3",
                        _next_description(),
                        _build_domain_payload("mid3", _next_description(), None),
                    )
                ),
                _build_require_op("mid3"),
                _build_is_subdomain_of_op("leaf3", "mid3"),
            ],
        ),
        (
            "a_chain_rebuilt_after_a_clear_equals_the_chain_built_before_it",
            [
                _build_new_domain_op("root5", _next_description()),
                _build_new_domain_op("mid5", _next_description(), parent_slot="root5"),
                _build_new_domain_op("leaf5", _next_description(), parent_slot="mid5"),
                _build_hold_op("mid5", "mid5_before"),
                _build_hold_op("leaf5", "leaf5_before"),
                _build_clear_op(),
                _build_new_domain_op("root5", _next_description()),
                _build_new_domain_op("mid5", _next_description(), parent_slot="root5"),
                _build_new_domain_op("leaf5", _next_description(), parent_slot="mid5"),
                _build_eq_held_op("mid5_before", "mid5"),
                _build_eq_held_op("leaf5_before", "leaf5"),
            ],
        ),
        (
            "a_chain_regrafted_after_a_clear_differs_from_the_chain_built_before_it",
            [
                _build_new_domain_op("root6", _next_description()),
                _build_new_domain_op("mid6", _next_description(), parent_slot="root6"),
                _build_hold_op("mid6", "mid6_before"),
                _build_clear_op(),
                _build_new_domain_op("root6", _next_description(), parent_slot="data"),
                _build_new_domain_op("mid6", _next_description(), parent_slot="root6"),
                _build_eq_held_op("mid6_before", "mid6"),
            ],
        ),
    ]


# =============================================================================
# Defective-payload scripts
#
# Each decodes payloads damaged by one structural defect: an unknown or
# missing field, a field of the wrong type, or an extra level of nesting.
# These run after the random scripts, so the descriptions they draw leave
# every earlier script's recording unchanged.
# =============================================================================

# Defects to an identifier payload, as paths relative to the identifier.
_IDENTIFIER_DEFECT_SPECS: list[tuple[str, tuple[str, ...], Any]] = [
    ("set", ("id",), -1),
    ("set", ("id",), "1"),
    ("set", ("id",), 1.0),
    ("set", ("id",), True),
    ("set", ("id",), None),
    ("set", ("id",), 2**64 - 1),
    ("set", ("id",), 2**64),
    ("remove", ("id",), None),
    ("remove", ("name_hint",), None),
    ("set", ("name_hint",), 3),
    ("set", ("extra",), 1),
]


def _list_identifier_defects(prefix: Sequence[str]) -> list[dict[str, Any]]:
    """Return defects to the identifier payload found at `prefix`."""
    defects = []
    for kind, path, value in _IDENTIFIER_DEFECT_SPECS:
        full_path = [*prefix, *path]
        if kind == "remove":
            defects.append(_build_remove_defect(*full_path))
        else:
            defects.append(_build_set_defect(full_path, value))
    defects.append(_build_wrap_defect(prefix, "name"))
    return defects


def _list_attribute_defects() -> list[dict[str, Any]]:
    return [
        _build_set_defect(["extra"], 1),
        _build_set_defect(["parent"], None),
        _build_remove_defect("name"),
        _build_remove_defect("description"),
        _build_set_defect(["name"], "bad"),
        _build_set_defect(["name"], None),
        _build_set_defect(["description"], 3),
        _build_set_defect(["description"], None),
        *_list_identifier_defects(["name"]),
        _build_wrap_defect([], "op_attribute"),
    ]


def _list_domain_defects() -> list[dict[str, Any]]:
    return [
        _build_remove_defect("parent"),
        _build_set_defect(["extra"], 1),
        _build_set_defect(["zzz"], 1),
        _build_remove_defect("name"),
        _build_remove_defect("description"),
        _build_set_defect(["name"], "bad"),
        _build_set_defect(["description"], None),
        _build_set_defect(["parent"], "data"),
        _build_set_defect(["parent"], []),
        _build_set_defect(["parent"], False),
        _build_lift_defect(["parent"], "name"),
        _build_wrap_defect(["parent"], "parent"),
        _build_set_defect(["parent", "extra"], 1),
        _build_remove_defect("parent", "parent"),
        _build_remove_defect("parent", "description"),
        *_list_identifier_defects(["name"]),
        *_list_identifier_defects(["parent", "name"]),
        _build_wrap_defect([], "value_domain"),
    ]


def _list_defective_attribute_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "decode_rejects_every_defective_payload_for_an_unregistered_slot",
            [
                *(
                    _build_decode_op(
                        _build_attribute_payload("bad", _next_description()), defect
                    )
                    for defect in _list_attribute_defects()
                ),
                _build_get_op("bad"),
                _build_decode_op(_build_attribute_payload("bad", _next_description())),
            ],
        ),
        (
            "decode_rejects_every_defective_payload_for_a_registered_slot",
            [
                _build_new_attribute_op("reg", _next_description()),
                *(
                    _build_decode_op(
                        _build_attribute_payload("reg", _next_description()), defect
                    )
                    for defect in _list_attribute_defects()
                ),
                _build_require_op("reg"),
            ],
        ),
        (
            "decode_accepts_a_name_hint_that_differs_from_the_slot",
            [
                _build_decode_op(
                    _build_attribute_payload("hinted", _next_description()),
                    _build_set_defect(["name", "name_hint"], "renamed"),
                ),
                _build_require_op("hinted"),
            ],
        ),
    ]


def _build_default_parented_domain_payload(slot: str) -> dict[str, Any]:
    return _build_domain_payload(
        slot,
        _next_description(),
        _build_domain_payload("data", _next_description(), None),
    )


def _list_defective_domain_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "decode_rejects_every_defective_payload_for_an_unregistered_slot",
            [
                *(
                    _build_decode_op(
                        _build_default_parented_domain_payload("bad"), defect
                    )
                    for defect in _list_domain_defects()
                ),
                _build_get_op("bad"),
                _build_decode_op(_build_default_parented_domain_payload("bad")),
            ],
        ),
        (
            "decode_rejects_every_defective_payload_for_a_registered_slot",
            [
                _build_new_domain_op("reg", _next_description(), parent_slot="data"),
                *(
                    _build_decode_op(
                        _build_default_parented_domain_payload("reg"), defect
                    )
                    for defect in _list_domain_defects()
                ),
                _build_require_op("reg"),
            ],
        ),
        (
            "decode_accepts_a_name_hint_that_differs_from_the_slot",
            [
                _build_decode_op(
                    _build_default_parented_domain_payload("hinted"),
                    _build_set_defect(["name", "name_hint"], "renamed"),
                ),
                _build_require_op("hinted"),
            ],
        ),
    ]


# =============================================================================
# Rejected-decode side-effect scripts
#
# Each decodes payloads naming slots bound ahead of the id counter, then
# records which of those ids the decode restored and which of their domains it
# registered, checking the counter before any lookup restores an id. These
# run last, so the descriptions they draw leave every earlier script's
# recording unchanged.
# =============================================================================


def _build_chain_payload(*slots: str) -> dict[str, Any]:
    """Return a domain payload naming `slots`, outermost first, each fresh."""
    payload = None
    for slot in reversed(slots):
        payload = _build_domain_payload(slot, _next_description(), payload)
    if payload is None:
        raise ValueError("a chain payload needs at least one slot")
    return payload


def _build_observe_chain_ops(slots: Sequence[str]) -> list[dict[str, Any]]:
    """Return ops recording the counter against, then the registry for, `slots`."""
    return [
        *(_build_is_counter_past_op(slot) for slot in slots),
        *(_build_get_op(slot) for slot in reversed(slots)),
    ]


def _build_rejected_chain_script(
    name: str, slots: Sequence[str], defect: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Return a script decoding a fresh chain over `slots` damaged by `defect`."""
    return (
        name,
        [
            *(_build_bind_ahead_op(slot) for slot in slots),
            _build_decode_op(_build_chain_payload(*slots), defect),
            *_build_observe_chain_ops(slots),
        ],
    )


def _list_side_effect_attribute_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "a_decode_rejected_for_a_trailing_unknown_key_restores_no_name",
            [
                _build_bind_ahead_op("ahead"),
                _build_decode_op(
                    _build_attribute_payload("ahead", _next_description()),
                    _build_set_defect(["zzz"], 1),
                ),
                _build_is_counter_past_op("ahead"),
                _build_decode_op(
                    _build_attribute_payload("ahead", _next_description()),
                    _build_set_defect(["description"], 3),
                ),
                _build_is_counter_past_op("ahead"),
                _build_decode_op(
                    _build_attribute_payload("ahead", _next_description())
                ),
                _build_is_counter_past_op("ahead"),
            ],
        ),
    ]


def _list_side_effect_domain_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    scripts = [
        _build_rejected_chain_script(
            "a_decode_rejected_for_a_trailing_unknown_key_registers_no_fresh_parent",
            ["outer", "parent"],
            _build_set_defect(["zzz"], 1),
        ),
        _build_rejected_chain_script(
            "a_decode_rejected_for_a_leading_unknown_key_registers_no_fresh_parent",
            ["outer", "parent"],
            _build_set_defect(["extra"], 1),
        ),
        _build_rejected_chain_script(
            "a_decode_rejected_for_a_trailing_unknown_key_registers_no_fresh_ancestor",
            ["outer", "parent", "grand"],
            _build_set_defect(["zzz"], 1),
        ),
        _build_rejected_chain_script(
            "a_decode_rejected_inside_its_parent_restores_only_the_outer_name",
            ["outer", "parent", "grand"],
            _build_set_defect(["parent", "zzz"], 1),
        ),
        _build_rejected_chain_script(
            "a_decode_rejected_inside_its_grandparent_restores_the_names_above_it",
            ["outer", "parent", "grand"],
            _build_set_defect(["parent", "parent", "zzz"], 1),
        ),
        _build_rejected_chain_script(
            "a_decode_rejected_for_a_missing_nested_parent_restores_the_names_above_it",
            ["outer", "parent", "grand"],
            _build_remove_defect("parent", "parent", "parent"),
        ),
        _build_rejected_chain_script(
            "a_decode_whose_parent_is_not_a_map_restores_nothing",
            ["outer"],
            _build_set_defect(["parent"], "data"),
        ),
        _build_rejected_chain_script(
            "a_decode_whose_parent_holds_an_out_of_range_id_restores_the_outer_name",
            ["outer", "parent"],
            _build_set_defect(["parent", "name", "id"], 2**64 - 1),
        ),
    ]
    trailing_script_name, trailing_ops = scripts[0]
    scripts[0] = (
        trailing_script_name,
        [*trailing_ops, _build_new_domain_op("parent", _next_description())],
    )
    scripts += [
        (
            "a_conflicting_decode_registers_its_fresh_parent",
            [
                _build_new_domain_op("child", _next_description(), parent_slot="data"),
                _build_bind_ahead_op("parent"),
                _build_decode_op(
                    _build_domain_payload(
                        "child",
                        _next_description(),
                        _build_chain_payload("parent"),
                    )
                ),
                _build_is_counter_past_op("parent"),
                _build_get_op("parent"),
                _build_require_op("child"),
            ],
        ),
        (
            "a_decode_whose_parent_conflicts_registers_only_the_fresh_grandparent",
            [
                _build_new_domain_op("middle", _next_description(), parent_slot="data"),
                _build_bind_ahead_op("outer"),
                _build_bind_ahead_op("grand"),
                _build_decode_op(
                    _build_domain_payload(
                        "outer",
                        _next_description(),
                        _build_domain_payload(
                            "middle", _next_description(), _build_chain_payload("grand")
                        ),
                    )
                ),
                *_build_observe_chain_ops(["outer", "grand"]),
                _build_require_op("middle"),
            ],
        ),
    ]
    return scripts


# =============================================================================
# Random scripts
# =============================================================================

_OP_WEIGHTS: dict[str, int] = {
    "new": 35,
    "get": 15,
    "require": 15,
    "clear": 8,
    "encode": 10,
    "decode": 12,
    "eq": 5,
    "is_subdomain_of": 8,
    "eq_with_duplicate": 6,
    "is_subdomain_of_with_duplicate": 6,
}

# Chance a random `new` domain, or a random decode payload's nesting level,
# attaches a parent rather than leaving it `None`.
_RANDOM_PARENT_PROBABILITY = 0.6

# Chance a random duplicate (for `eq_with_duplicate` and
# `is_subdomain_of_with_duplicate`) reuses the canonical's actual current
# description/parent rather than a value guaranteed to differ from it. Kept
# away from the extremes so a run exercises both the "same" and "different"
# branches of the two equality rules: a description never affects equality,
# and a parent always does.
_DUPLICATE_SAME_VALUE_PROBABILITY = 0.5


def _list_feasible_op_kinds(ctx: _ScriptContext) -> list[str]:
    kinds = ["new", "get", "require", "clear", "decode"]
    if ctx.registered_slots:
        kinds.append("encode")
        kinds.append("eq")
        kinds.append("eq_with_duplicate")
        if ctx.kind == _VALUE_DOMAIN_KIND:
            kinds.append("is_subdomain_of")
            kinds.append("is_subdomain_of_with_duplicate")
    return kinds


def _build_random_attribute_payload(
    rng: random.Random, slots: Sequence[str]
) -> dict[str, Any]:
    return _build_attribute_payload(rng.choice(slots), _next_description())


def _build_random_domain_payload(
    rng: random.Random, slots: Sequence[str], depth_budget: int
) -> dict[str, Any]:
    slot = rng.choice(slots)
    description = _next_description()
    parent = None
    if depth_budget > 0 and rng.random() < _RANDOM_PARENT_PROBABILITY:
        parent = _build_random_domain_payload(rng, slots, depth_budget - 1)
    return _build_domain_payload(slot, description, parent)


def _choose_random_new_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    slot = rng.choice(list(alphabet) + list(ctx.default_names))
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        return _build_new_attribute_op(slot, _next_description())
    parent_slot = None
    if rng.random() < _RANDOM_PARENT_PROBABILITY:
        parent_slot = rng.choice(list(ctx.registered_slots))
    return _build_new_domain_op(slot, _next_description(), parent_slot)


def _choose_random_get_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    return _build_get_op(rng.choice(list(alphabet) + list(ctx.default_names)))


def _choose_random_require_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    return _build_require_op(rng.choice(list(alphabet) + list(ctx.default_names)))


def _choose_random_clear_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    del rng, ctx, alphabet  # clear takes no arguments; kept for a uniform signature
    return _build_clear_op()


def _choose_random_encode_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    del alphabet
    return _build_encode_op(rng.choice(list(ctx.registered_slots)))


def _choose_random_decode_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    slots_and_defaults = list(alphabet) + list(ctx.default_names)
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        return _build_decode_op(
            _build_random_attribute_payload(rng, slots_and_defaults)
        )
    return _build_decode_op(
        _build_random_domain_payload(
            rng, slots_and_defaults, _RANDOM_DOMAIN_PAYLOAD_DEPTH
        )
    )


def _choose_random_eq_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    del alphabet
    registered = list(ctx.registered_slots)
    return _build_eq_op(rng.choice(registered), rng.choice(registered))


def _choose_random_is_subdomain_of_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    del alphabet
    registered = list(ctx.registered_slots)
    return _build_is_subdomain_of_op(rng.choice(registered), rng.choice(registered))


def _choose_random_duplicate_description(
    rng: random.Random, ctx: _ScriptContext, slot: str
) -> str:
    """Return a description for a duplicate of `slot`'s canonical instance.

    Reuses the canonical's actual current description about half the time
    (the "same" branch), and a fresh, guaranteed-different one the rest of
    the time (the "different" branch, which equality must ignore).
    """
    if rng.random() < _DUPLICATE_SAME_VALUE_PROBABILITY:
        canonical = ctx.cls.require_interned(ctx.resolve_identifier(slot))
        return canonical.description
    return _next_description()


def _choose_random_duplicate_parent_slot(
    rng: random.Random, ctx: _ScriptContext, slot: str
) -> str | None:
    """Return a parent slot for a duplicate of `slot`'s canonical domain.

    Candidates are the canonical's actual current parent slot (`None` for a
    root domain), an explicit `None`, and every other registered slot, so
    both the "same parent" and "different parent" branches of the rule that
    equality compares parents get exercised, including forcing a real parent
    down to `None`.
    """
    canonical = ValueDomain.require_interned(ctx.resolve_identifier(slot))
    same_parent_slot = (
        None
        if canonical.parent is None
        else ctx.find_slot_for_id(canonical.parent.name.id)
    )
    candidates: list[str | None] = [same_parent_slot, None, *ctx.registered_slots]
    return rng.choice(candidates)


def _choose_random_eq_with_duplicate_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    del alphabet
    slot = rng.choice(list(ctx.registered_slots))
    other_description = _choose_random_duplicate_description(rng, ctx, slot)
    if ctx.kind == _OP_ATTRIBUTE_KIND:
        return _build_eq_with_duplicate_attribute_op(slot, other_description)
    other_parent_slot = _choose_random_duplicate_parent_slot(rng, ctx, slot)
    return _build_eq_with_duplicate_domain_op(
        slot, other_description, other_parent_slot
    )


def _choose_random_is_subdomain_of_with_duplicate_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    del alphabet
    registered = list(ctx.registered_slots)
    self_slot = rng.choice(registered)
    other_slot = rng.choice(registered)
    other_description = _choose_random_duplicate_description(rng, ctx, other_slot)
    other_parent_slot = _choose_random_duplicate_parent_slot(rng, ctx, other_slot)
    return _build_is_subdomain_of_with_duplicate_op(
        self_slot, other_slot, other_description, other_parent_slot
    )


_RANDOM_OP_CHOOSERS: dict[
    str, Callable[[random.Random, _ScriptContext, Sequence[str]], dict[str, Any]]
] = {
    "new": _choose_random_new_op,
    "get": _choose_random_get_op,
    "require": _choose_random_require_op,
    "clear": _choose_random_clear_op,
    "encode": _choose_random_encode_op,
    "decode": _choose_random_decode_op,
    "eq": _choose_random_eq_op,
    "is_subdomain_of": _choose_random_is_subdomain_of_op,
    "eq_with_duplicate": _choose_random_eq_with_duplicate_op,
    "is_subdomain_of_with_duplicate": _choose_random_is_subdomain_of_with_duplicate_op,
}


def _build_random_op(
    rng: random.Random, ctx: _ScriptContext, alphabet: Sequence[str]
) -> dict[str, Any]:
    feasible = _list_feasible_op_kinds(ctx)
    kind = rng.choices(feasible, weights=[_OP_WEIGHTS[k] for k in feasible], k=1)[0]
    return _RANDOM_OP_CHOOSERS[kind](rng, ctx, alphabet)


def _run_random_script(
    rng: random.Random, index: int, kind: str, alphabet: Sequence[str], max_ops: int
) -> dict[str, Any]:
    ctx = _create_context(kind)
    num_ops = rng.randint(5, max_ops)
    recorded_ops = []
    for _ in range(num_ops):
        op = _build_random_op(rng, ctx, alphabet)
        expected = _run_op(ctx, op)
        recorded_ops.append({**op, "expected": expected})
    return {"name": f"random_{kind}_{index:03d}", "type": kind, "ops": recorded_ops}


# =============================================================================
# Document assembly
# =============================================================================


def _build_default_slots_json() -> dict[str, list[str]]:
    return {
        _OP_ATTRIBUTE_KIND: list(_OP_ATTRIBUTE_DEFAULTS),
        _VALUE_DOMAIN_KIND: list(_VALUE_DOMAIN_DEFAULTS),
    }


def _build_provenance(repository_root: Path) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    return {
        "package": "fhy_core",
        "package_version": version("fhy_core"),
        "git_commit": git_commit,
        "python_version": sys.version,
        "generator_command": GENERATOR_COMMAND,
    }


def _parse_arguments(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--seed", type=int, default=_RANDOM_SEED)
    parser.add_argument("--random-count", type=int, default=_RANDOM_SCRIPT_COUNT)
    parser.add_argument("--max-ops", type=int, default=_RANDOM_MAX_OPS)
    parser.add_argument(
        "--slots",
        default=",".join(_RANDOM_SLOTS),
        help="comma-separated alphabet of fresh (non-default) identifier slots",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    """Run every script through the oracle and write the golden document."""
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[3]
    arguments = _parse_arguments(script_path.parent / "tag_type_cases.json")
    output_path = arguments.output
    alphabet = arguments.slots.split(",")

    cases = [
        _run_fixed_script(name, _OP_ATTRIBUTE_KIND, ops)
        for name, ops in _list_hand_picked_attribute_scripts()
    ]
    cases += [
        _run_fixed_script(name, _VALUE_DOMAIN_KIND, ops)
        for name, ops in _list_hand_picked_domain_scripts()
    ]

    rng = random.Random(arguments.seed)
    for index in range(arguments.random_count):
        kind = _OP_ATTRIBUTE_KIND if index % 2 == 0 else _VALUE_DOMAIN_KIND
        cases.append(_run_random_script(rng, index, kind, alphabet, arguments.max_ops))

    cases += [
        _run_fixed_script(name, _OP_ATTRIBUTE_KIND, ops)
        for name, ops in _list_defective_attribute_scripts()
    ]
    cases += [
        _run_fixed_script(name, _VALUE_DOMAIN_KIND, ops)
        for name, ops in _list_defective_domain_scripts()
    ]
    cases += [
        _run_fixed_script(name, _OP_ATTRIBUTE_KIND, ops)
        for name, ops in _list_side_effect_attribute_scripts()
    ]
    cases += [
        _run_fixed_script(name, _VALUE_DOMAIN_KIND, ops)
        for name, ops in _list_side_effect_domain_scripts()
    ]

    document = {
        "provenance": _build_provenance(repository_root),
        "default_slots": _build_default_slots_json(),
        "cases": cases,
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, ensure_ascii=False, indent=1)
        output_file.write("\n")

    total_ops = sum(len(case["ops"]) for case in cases)
    print(f"wrote {len(cases)} cases, {total_ops} ops, to {output_path}")


if __name__ == "__main__":
    main()
