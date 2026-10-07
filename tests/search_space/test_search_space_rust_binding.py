"""Tests of what the binding adds around the Rust search space.

The class structure and freezing, identity `==` and `hash` beside
`ConfigurationKey`'s structural ones, `repr`, pickling, the V2 payloads and
their type ids, the refusals of the readers, the depth guard and cyclic
garbage collection.
"""

import contextlib
import copy
import gc
import json
import pickle
import textwrap
import weakref
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import Note
from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    Condition,
    Configuration,
    ConfigurationError,
    ConfigurationKey,
    DuplicateNameError,
    Forbidden,
    RandomOracle,
    SearchSpaceError,
    Space,
    Variable,
)
from fhy_core.serialization import (
    DeserializationValueError,
    MalformedPayloadError,
    Serializable,
    SerializationError,
    WrappedFamilySerializable,
    register_serializable,
    serialize_value,
)
from fhy_core.symbolic.constraint import NotInSetConstraint
from fhy_core.symbolic.param import create_natural_param_between
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override
from tests.native_modules import run_python
from tests.v1 import writing_v1

from .conftest import (
    Explosion,
    build_chain,
    build_complete_configuration,
    build_tiling_space,
    categorical,
    make_alternative,
    make_choice,
    make_variable,
)


def _canonical_text(payload: object) -> str:
    """Return the canonical JSON text of a payload dict."""
    return json.dumps(payload, separators=(",", ":"))


# ===========================================================================
# Class structure
# ===========================================================================


def test_public_classes_subclass_their_rust_classes() -> None:
    """Test each public class is a thin subclass of its `_rs` class."""
    assert issubclass(Variable, _rs.Variable)
    assert issubclass(Variable, WrappedFamilySerializable)
    assert issubclass(Alternative, _rs.Alternative)
    assert issubclass(Alternative, WrappedFamilySerializable)
    for public, native in (
        (Choice, _rs.Choice),
        (Space, _rs.Space),
        (Configuration, _rs.Configuration),
    ):
        assert issubclass(public, native)
        assert issubclass(public, Serializable)
    assert issubclass(Condition, _rs.Condition)
    assert issubclass(Forbidden, _rs.Forbidden)
    assert ConfigurationKey is _rs.ConfigurationKey
    assert issubclass(DuplicateNameError, SearchSpaceError)
    assert issubclass(ConfigurationError, SearchSpaceError)
    assert issubclass(SearchSpaceError, ValueError)


def test_values_are_frozen_mixin_instances() -> None:
    """Test every object the package builds counts as a `FrozenMixin`."""
    tiling = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True)
    configuration = Configuration(tiling.space)

    for value in (
        tiling.unroll,
        tiling.tiled,
        tiling.layout,
        tiling.conditions[0],
        tiling.forbidden[0],
        tiling.space,
        configuration,
        configuration.key(),
    ):
        assert isinstance(value, FrozenMixin)


@pytest.mark.parametrize(
    ("value", "type_id"),
    [
        (lambda: make_variable("k"), "search_space.variable"),
        (lambda: make_alternative("a"), "search_space.alternative"),
        (lambda: make_choice("c", make_alternative("a")), "search_space.choice"),
        (lambda: build_tiling_space().space, "search_space.space"),
        (
            lambda: Configuration(build_tiling_space().space),
            "search_space.configuration",
        ),
    ],
    ids=["variable", "alternative", "choice", "space", "configuration"],
)
def test_classes_serialize_under_their_type_ids(
    value: Callable[[], Serializable], type_id: str
) -> None:
    """Test each serializable class is registered under its `search_space.*` id."""
    assert value().get_serialization_class_type_id() == type_id


# ===========================================================================
# Freezing
# ===========================================================================


def _frozen_values() -> list[tuple[str, Any]]:
    """Return one value of each class, with its class name."""
    tiling = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True)
    return [
        ("Variable", tiling.unroll),
        ("Alternative", tiling.tiled),
        ("Choice", tiling.layout),
        ("Condition", tiling.conditions[0]),
        ("Forbidden", tiling.forbidden[0]),
        ("Space", tiling.space),
        ("Configuration", Configuration(tiling.space)),
    ]


def test_assigning_an_attribute_raises() -> None:
    """Test every class refuses a new attribute with `FrozenMutationError`."""
    for class_name, value in _frozen_values():
        with pytest.raises(FrozenMutationError) as excinfo:
            value.extra = 1
        assert str(excinfo.value) == f'Cannot modify "extra" on frozen {class_name}.'


def test_assigning_a_field_raises() -> None:
    """Test a field cannot be replaced either."""
    for class_name, value in _frozen_values():
        with pytest.raises(FrozenMutationError) as excinfo:
            value.name = Identifier("other")
        assert str(excinfo.value) == f'Cannot modify "name" on frozen {class_name}.'


def test_deleting_an_attribute_raises() -> None:
    """Test every class refuses to delete an attribute."""
    for class_name, value in _frozen_values():
        with pytest.raises(FrozenMutationError) as excinfo:
            del value.name
        assert str(excinfo.value) == f'Cannot delete "name" on frozen {class_name}.'


def test_variable_and_alternative_report_being_frozen() -> None:
    """Test the open bases report the frozen state of `FrozenMixin`."""
    variable, alternative = make_variable("k"), make_alternative("a")

    assert variable.is_frozen
    assert alternative.is_frozen
    variable.assert_frozen()


def test_initializing_a_variable_twice_is_refused() -> None:
    """Test the base fields are set once: a second `__init__` raises."""
    variable = make_variable("k")

    with pytest.raises(RuntimeError, match="initialized already"):
        _rs.Variable._initialize(variable, categorical(3))

    domain: Any = variable.param.domain
    assert domain.categories == (1, 2)


# ===========================================================================
# Equality and hashing
# ===========================================================================


def test_equality_and_hashing_are_identity() -> None:
    """Test two structurally equivalent values are distinct under `==`."""
    name = Identifier("k")
    param = categorical()
    left, right = Variable(param=param, name=name), Variable(param=param, name=name)
    tiling = build_tiling_space()
    first, second = Configuration(tiling.space), Configuration(tiling.space)

    assert left.is_structurally_equivalent(right)
    assert left != right
    assert left == left  # noqa: PLR0124
    assert len({left, right}) == 2
    assert first != second
    assert len({first, second, tiling.space}) == 3


def test_configuration_keys_compare_structurally() -> None:
    """Test equal configurations' keys are equal, hash alike and key a dict."""
    tiling = build_tiling_space()
    left, right = (
        build_complete_configuration(tiling),
        build_complete_configuration(tiling),
    )
    other = Configuration(tiling.space)

    assert left.key() is not right.key()
    assert left.key() == right.key()
    assert not left.key() != right.key()
    assert hash(left.key()) == hash(right.key())
    assert left.key() != other.key()
    assert len({left.key(), right.key(), other.key()}) == 2


def test_configuration_key_is_not_equal_to_another_type() -> None:
    """Test a key compared with something else is `NotImplemented`, so unequal."""
    key = Configuration(build_tiling_space().space).key()

    assert key.__eq__(3) is NotImplemented
    assert key != 3
    assert key != Configuration(build_tiling_space().space)


def test_configuration_key_has_no_constructor() -> None:
    """Test a key is only made by `Configuration.key`."""
    with pytest.raises(TypeError):
        ConfigurationKey()


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_configuration_key_pickles_to_an_equal_key(protocol: int) -> None:
    """Test a key pickles, under every protocol, to an equal key of one hash."""
    key = build_complete_configuration(build_tiling_space()).key()

    restored = pickle.loads(pickle.dumps(key, protocol=protocol))

    assert type(restored) is ConfigurationKey
    assert restored is not key
    assert restored == key
    assert hash(restored) == hash(key)


def test_pickled_key_keeps_telling_configurations_apart() -> None:
    """Test a restored key equals its own configuration's key only."""
    tiling = build_tiling_space()
    key = build_complete_configuration(tiling).key()
    other = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name}).key()

    restored = pickle.loads(pickle.dumps(key))

    assert restored != other
    assert {restored: "measured"}[key] == "measured"


def test_keys_of_relabeled_configurations_stay_equal_after_pickling() -> None:
    """Test corresponding configurations' keys stay equal across a pickle."""
    left = build_complete_configuration(build_tiling_space()).key()
    right = build_complete_configuration(build_tiling_space()).key()

    restored_left = pickle.loads(pickle.dumps(left))
    restored_right = pickle.loads(pickle.dumps(right))

    assert restored_left == right
    assert restored_left == restored_right
    assert hash(restored_left) == hash(right)


def test_pickled_key_with_a_bound_identifier_value_round_trips() -> None:
    """Test a key whose value is a name the space binds survives a pickle."""

    def build() -> ConfigurationKey:
        tiled, flat = make_alternative("tiled"), make_alternative("flat")
        mirror = Variable(
            param=categorical(tiled.name, flat.name), name=Identifier("m")
        )
        space = Space(variables=(mirror,), choices=(make_choice("c", tiled, flat),))
        return Configuration(space, {mirror.name: flat.name}).key()

    left, right = build(), build()

    assert pickle.loads(pickle.dumps(left)) == right


@register_serializable(type_id="tests.search_space.touchy_value")
class _Touchy(Serializable):
    """A `Serializable` value whose `==` raises while `explodes` is set."""

    explodes: ClassVar[bool] = False

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __eq__(self, other: object) -> bool:
        if _Touchy.explodes:
            raise Explosion("touchy")
        return isinstance(other, _Touchy) and self.value == other.value

    @override
    def __hash__(self) -> int:
        return hash(self.value)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_Touchy":
        return cls(int(data["value"]))


def _touchy_keys() -> tuple[ConfigurationKey, ConfigurationKey]:
    """Return the keys of two configurations holding equal `_Touchy` values."""
    variable = Variable(
        param=categorical(_Touchy(1), _Touchy(2)), name=Identifier("touchy")
    )
    space = Space(variables=(variable,))
    left = Configuration(space, {variable.name: _Touchy(1)}).key()
    right = Configuration(space, {variable.name: _Touchy(1)}).key()
    return left, right


def test_a_value_comparison_that_raises_raises_from_the_key_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a value's raising `==` reaches the caller of the keys' `==` and `!=`."""
    left, right = _touchy_keys()
    monkeypatch.setattr(_Touchy, "explodes", True)

    with pytest.raises(Explosion, match="touchy"):
        _ = left == right
    with pytest.raises(Explosion, match="touchy"):
        _ = left != right


def test_keys_compare_again_once_a_value_comparison_stopped_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a raised comparison leaves no exception behind on the thread."""
    left, right = _touchy_keys()
    monkeypatch.setattr(_Touchy, "explodes", True)
    with contextlib.suppress(Explosion):
        _ = left == right

    monkeypatch.setattr(_Touchy, "explodes", False)

    assert left == right
    assert not left != right
    assert {left: "measured"}.get(right) == "measured"


class _CountingVariable(Variable[Any]):
    """A variable counting the calls of its search-domain hook."""

    calls: ClassVar[int] = 0

    @override
    def extension_search_domain(self) -> None:
        _CountingVariable.calls += 1


def test_no_hook_is_called_once_a_value_comparison_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a subclass's hook is not called while an exception is pending.

    Sampling `counted` first evaluates its condition, whose value comparison
    raises: the exception is pending from then on, and the condition, which
    answers as if the values differed, leaves `counted` active.
    """
    touchy = Variable(param=categorical(_Touchy(1)), name=Identifier("touchy"))
    counted = _CountingVariable(param=categorical(), name=Identifier("counted"))
    space = Space(
        variables=(touchy, counted),
        conditions=(
            Condition(counted.name, (NotInSetConstraint(touchy.name, {_Touchy(1)}),)),
        ),
    )
    monkeypatch.setattr(_Touchy, "explodes", True)
    monkeypatch.setattr(_CountingVariable, "calls", 0)

    with pytest.raises(Explosion, match="touchy"):
        space.sample(RandomOracle(seed=0))

    assert _CountingVariable.calls == 0


def test_key_is_restored_from_its_wire_text_only() -> None:
    """Test the restoring function refuses a text of another shape."""
    with pytest.raises(DeserializationValueError):
        ConfigurationKey._from_wire('{"entries": [{"chosen": {}}]}')


# ===========================================================================
# repr
# ===========================================================================


def test_variable_repr_lists_its_fields() -> None:
    """Test a variable's `repr` is dataclass-like."""
    note = Note("hot")
    variable = Variable(param=categorical(), name=Identifier("k"), notes=(note,))

    assert repr(variable) == (
        f"Variable(name={variable.name!r}, param={variable.param!r}, notes=({note!r},))"
    )


def test_alternative_and_choice_reprs_nest() -> None:
    """Test an alternative's and a choice's `repr` hold their parts' reprs."""
    tile = make_variable("tile")
    tiled = make_alternative("tiled", (tile,))
    layout = make_choice("layout", tiled)

    assert repr(tiled) == (
        f"Alternative(name={tiled.name!r}, variables=({tile!r},), choices=(), notes=())"
    )
    assert repr(layout) == (
        f"Choice(name={layout.name!r}, alternatives=({tiled!r},), notes=())"
    )


def test_condition_and_forbidden_reprs() -> None:
    """Test a condition's and a clause's `repr` show their fields."""
    tiling = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True)
    condition, clause = tiling.conditions[0], tiling.forbidden[0]

    assert repr(condition) == (
        f"Condition(target={condition.target!r}, when={condition.when!r})"
    )
    assert repr(clause) == f"Forbidden(when={clause.when!r})"


def test_space_repr_lists_its_fields() -> None:
    """Test a space's `repr` lists its parts."""
    tiling = build_tiling_space(forbid_unroll_four=True)
    space = tiling.space

    assert repr(space) == (
        f"Space(name={space.name!r}, variables=({tiling.unroll!r},), "
        f"choices=({tiling.layout!r},), conditions=(), "
        f"forbidden=({tiling.forbidden[0]!r},), notes=())"
    )


def test_configuration_repr_names_its_space() -> None:
    """Test a configuration shows its space by name and its entries."""
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space, {tiling.unroll.name: 2})

    assert repr(configuration) == (
        f"Configuration(space={tiling.space.name!r}, "
        f"entries=(({tiling.unroll.name!r}, 2),))"
    )


def test_configuration_key_repr() -> None:
    """Test a key's `repr` names its class."""
    key = Configuration(build_tiling_space().space).key()

    assert repr(key).startswith("ConfigurationKey(")


# ===========================================================================
# Pickling
# ===========================================================================


def _round_trip_pickle(value: Any) -> Any:
    """Return `value` pickled and unpickled."""
    return pickle.loads(pickle.dumps(value))


def test_variable_pickles_as_a_call() -> None:
    """Test a variable pickles to a frozen equivalent variable with the same name."""
    variable = Variable(param=categorical(), name=Identifier("k"), notes=(Note("n"),))

    restored = _round_trip_pickle(variable)

    assert type(restored) is Variable
    assert restored.is_structurally_equivalent(variable)
    assert restored.name == variable.name
    assert restored.is_frozen


@pytest.mark.parametrize(
    "build",
    [
        lambda: make_alternative("a", (make_variable("v"),)),
        lambda: build_tiling_space().layout,
        lambda: (
            build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True).space
        ),
    ],
    ids=["alternative", "choice", "space"],
)
def test_containers_pickle_to_structurally_equivalent_copies(
    build: Callable[[], Any],
) -> None:
    """Test an alternative, a choice and a space survive pickling."""
    value = build()

    restored = _round_trip_pickle(value)

    assert type(restored) is type(value)
    assert restored is not value
    assert restored.is_structurally_equivalent(value)


def test_condition_and_forbidden_pickle() -> None:
    """Test a condition and a clause pickle to their fields."""
    tiling = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True)

    condition = _round_trip_pickle(tiling.conditions[0])
    clause = _round_trip_pickle(tiling.forbidden[0])

    assert type(condition) is Condition
    assert condition.target == tiling.unroll.name
    assert condition.when.is_structurally_equivalent(tiling.conditions[0].when)
    assert type(clause) is Forbidden
    assert clause.when.is_structurally_equivalent(tiling.forbidden[0].when)


def test_configuration_pickles_with_its_space() -> None:
    """Test a configuration pickles to one with an equal key and its values."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)

    restored = _round_trip_pickle(configuration)

    assert restored.is_structurally_equivalent(configuration)
    assert restored.key() == configuration.key()
    assert restored.value(tiling.tile.name) == 4


def test_deep_copy_builds_an_equivalent_space() -> None:
    """Test `copy.deepcopy` works through pickling."""
    space = build_tiling_space(forbid_unroll_four=True).space

    copied = copy.deepcopy(space)

    assert copied is not space
    assert copied.is_structurally_equivalent(space)


# ===========================================================================
# V2 payloads
# ===========================================================================


def test_variable_payload_is_the_tagged_plain_part() -> None:
    """Test a plain variable's V2 dict is `{"plain": {identifier, param, notes}}`."""
    note = Note("n")
    variable = Variable(param=categorical(1, 2), name=Identifier("k"), notes=(note,))

    payload = variable.serialize_to_dict()

    assert payload == {
        "plain": {
            "identifier": variable.name.serialize_to_dict(),
            "param": variable.param.serialize_to_dict(),
            "notes": [note.serialize_to_dict()],
        }
    }
    assert variable.serialize_data_to_dict() == payload["plain"]
    assert variable.to_json() == _canonical_text(payload)


def test_alternative_payload_nests_its_variables_tagged() -> None:
    """Test an alternative's V2 dict tags its plain variables."""
    tile = make_variable("tile")
    inner = make_choice("inner", make_alternative("only"))
    tiled = make_alternative("tiled", (tile,), (inner,))

    payload = tiled.serialize_to_dict()

    assert payload == {
        "plain": {
            "identifier": tiled.name.serialize_to_dict(),
            "variables": [tile.serialize_to_dict()],
            "choices": [inner.serialize_to_dict()],
            "notes": [],
        }
    }


def test_choice_payload() -> None:
    """Test a choice's V2 dict holds its tagged alternatives."""
    tiling = build_tiling_space()

    payload = tiling.layout.serialize_to_dict()

    assert payload == {
        "identifier": tiling.layout.name.serialize_to_dict(),
        "alternatives": [
            tiling.tiled.serialize_to_dict(),
            tiling.flat.serialize_to_dict(),
        ],
        "notes": [],
    }
    assert tiling.layout.to_json() == _canonical_text(payload)


def test_space_payload() -> None:
    """Test a space's V2 dict holds its parts, conditions and clauses."""
    tiling = build_tiling_space(unroll_on_tiled_only=True, forbid_unroll_four=True)

    payload = tiling.space.serialize_to_dict()

    assert payload == {
        "identifier": tiling.space.name.serialize_to_dict(),
        "variables": [tiling.unroll.serialize_to_dict()],
        "choices": [tiling.layout.serialize_to_dict()],
        "conditions": [
            {
                "target": tiling.unroll.name.serialize_to_dict(),
                "when": tiling.conditions[0].when.serialize_to_dict(),
            }
        ],
        "forbidden": [{"when": tiling.forbidden[0].when.serialize_to_dict()}],
        "notes": [],
    }
    assert tiling.space.to_json() == _canonical_text(payload)


def test_configuration_payload_lists_entries_in_canonical_order() -> None:
    """Test a configuration's V2 dict holds its space and its entries in order."""
    tiling = build_tiling_space()
    configuration = Configuration(
        tiling.space,
        [(tiling.tile.name, 8), (tiling.layout.name, tiling.tiled.name)],
    )

    payload = configuration.serialize_to_dict()

    assert payload == {
        "space": tiling.space.serialize_to_dict(),
        "entries": [
            {
                "name": tiling.layout.name.serialize_to_dict(),
                "value": serialize_value(tiling.tiled.name),
            },
            {
                "name": tiling.tile.name.serialize_to_dict(),
                "value": serialize_value(8),
            },
        ],
    }
    assert configuration.to_json() == _canonical_text(payload)


def test_to_json_reformats_for_indent_and_sorted_keys() -> None:
    """Test `indent` and `sort_keys` re-format the same payload."""
    space = build_tiling_space().space

    indented = space.to_json(indent=2, sort_keys=True)

    assert "\n" in indented
    assert json.loads(indented) == space.serialize_to_dict()


def _round_trips() -> list[tuple[type[Any], Callable[[], Any]]]:
    """Return each class with a builder of one of its values."""
    return [
        (Variable, lambda: make_variable("k", 1, 2)),
        (Alternative, lambda: make_alternative("a", (make_variable("v"),))),
        (Choice, lambda: build_tiling_space().layout),
        (
            Space,
            lambda: (
                build_tiling_space(
                    unroll_on_tiled_only=True, forbid_unroll_four=True
                ).space
            ),
        ),
        (Configuration, lambda: build_complete_configuration(build_tiling_space())),
    ]


@pytest.mark.parametrize(
    ("cls", "build"),
    _round_trips(),
    ids=["variable", "alternative", "choice", "space", "configuration"],
)
def test_payload_dict_round_trips(cls: type[Any], build: Callable[[], Any]) -> None:
    """Test decoding a value's V2 dict gives an equivalent value of its class."""
    value = build()

    decoded = cls.deserialize_from_dict(value.serialize_to_dict())

    assert type(decoded) is cls
    assert decoded.is_structurally_equivalent(value)
    assert decoded.serialize_to_dict() == value.serialize_to_dict()


@pytest.mark.parametrize(
    ("cls", "build"),
    _round_trips(),
    ids=["variable", "alternative", "choice", "space", "configuration"],
)
def test_payload_text_round_trips(cls: type[Any], build: Callable[[], Any]) -> None:
    """Test decoding a value's JSON text writes the same text back."""
    value = build()
    text = value.to_json()

    decoded = cls.from_json(text)

    assert type(decoded) is cls
    assert decoded.to_json() == text
    assert cls.from_json(text.encode()).to_json() == text


def test_decoded_configuration_keeps_its_key() -> None:
    """Test a decoded configuration has the key of the one written."""
    configuration = build_complete_configuration(build_tiling_space())

    decoded = Configuration.from_json(configuration.to_json())

    assert decoded.key() == configuration.key()
    assert decoded.is_complete()


def test_reader_refuses_a_payload_of_another_shape() -> None:
    """Test a payload missing a field names the class it was read for."""
    payload = build_tiling_space().space.serialize_to_dict()
    del payload["forbidden"]

    with pytest.raises(
        DeserializationValueError, match='Invalid V2 payload for "Space"'
    ):
        Space.deserialize_from_dict(payload)


def test_reader_refuses_a_payload_of_another_class() -> None:
    """Test a choice's payload is not a space's."""
    choice = build_tiling_space().layout

    with pytest.raises(
        DeserializationValueError, match='Invalid V2 payload for "Space"'
    ):
        Space.deserialize_from_dict(choice.serialize_to_dict())


def test_reader_refuses_a_payload_the_constructor_refuses() -> None:
    """Test a payload with an empty choice fails as the constructor would."""
    payload = build_tiling_space().layout.serialize_to_dict()
    payload["alternatives"] = []

    with pytest.raises(DeserializationValueError, match="has no alternative"):
        Choice.deserialize_from_dict(payload)


def test_reader_refuses_a_configuration_the_space_refuses() -> None:
    """Test a decoded entry naming no alternative of its choice is refused."""
    tiling = build_tiling_space()
    payload: Any = Configuration(
        tiling.space, {tiling.layout.name: tiling.flat.name}
    ).serialize_to_dict()
    payload["entries"][0]["value"] = serialize_value(Identifier("ghost"))

    with pytest.raises(DeserializationValueError, match="has no alternative ghost"):
        Configuration.deserialize_from_dict(payload)


def test_reader_refuses_text_that_is_no_json() -> None:
    """Test a text that is not JSON raises `MalformedPayloadError`."""
    with pytest.raises(MalformedPayloadError):
        Space.from_json("{not json")


def test_variable_reader_refuses_an_alternative_payload() -> None:
    """Test a variable is not read from an alternative's payload."""
    with pytest.raises(DeserializationValueError):
        Variable.deserialize_from_dict(make_alternative("a").serialize_to_dict())


def test_variable_data_reader_builds_a_variable() -> None:
    """Test `deserialize_data_from_dict` reads the base fields' data."""
    variable = make_variable("k", 1, 2)

    decoded: Variable[Any] = Variable.deserialize_data_from_dict(
        variable.serialize_data_to_dict()
    )

    assert type(decoded) is Variable
    assert decoded.is_structurally_equivalent(variable)


def test_variable_data_reader_refuses_another_key() -> None:
    """Test the base fields' data reader refuses a key it does not know."""
    data = make_variable("k").serialize_data_to_dict()
    data["extra"] = 1

    with pytest.raises(DeserializationValueError):
        Variable.deserialize_data_from_dict(data)


def test_writing_v1_is_refused() -> None:
    """Test the classes have no V1 form: writing one raises."""
    space = build_tiling_space().space

    with writing_v1(), pytest.raises(SerializationError, match="V1"):
        space.serialize_to_dict()
    with writing_v1(), pytest.raises(SerializationError, match="V1"):
        make_variable("k").serialize_to_dict()


# ===========================================================================
# Depth
# ===========================================================================


MAX_CHOICE_DEPTH = 16
"""The core's `fhy_core::search_space::MAX_CHOICE_DEPTH`."""


def _build_deepest_space() -> Space:
    """Return a space whose choices nest as deep as the core allows.

    The innermost alternative holds a variable over a bounded integer
    param, so the payload nests as deep as a realistic space of this
    depth.
    """
    choice = make_choice(
        "level_1",
        make_alternative(
            "leaf",
            variables=(
                Variable(
                    param=create_natural_param_between(1, 8), name=Identifier("tile")
                ),
            ),
        ),
    )
    for level in range(2, MAX_CHOICE_DEPTH + 1):
        choice = make_choice(
            f"level_{level}", make_alternative(f"holder_{level}", choices=(choice,))
        )
    return Space(choices=(choice,))


def test_choices_nested_to_the_cap_round_trip_as_text_and_as_a_dict() -> None:
    """Test the deepest space the core builds reads back through both paths.

    Its payload nests within the limits of both readers: serde_json's for
    JSON text, and the dict reader's for a payload dict.
    """
    space = _build_deepest_space()
    configuration = Configuration(space, {})
    holder = space.choices[0].alternatives[0]

    assert space.is_structurally_equivalent(space)
    for value, cls in (
        (space, Space),
        (space.choices[0], Choice),
        (holder, Alternative),
        (configuration, Configuration),
    ):
        text = value.to_json()
        from_text = cls.from_json(text)
        from_dict = cls.deserialize_from_dict(value.serialize_to_dict())
        assert from_text.to_json() == text
        assert from_dict.to_json() == text


def test_payload_nesting_choices_past_the_cap_is_refused_on_both_paths() -> None:
    """Test a payload one choice deeper than the core allows is a value error.

    The payload is valid JSON within both readers' limits, so neither path
    calls it malformed: the core's decoder refuses it before building.
    """
    payload = _build_deepest_space().serialize_to_dict()
    holder_id = Identifier("holder")
    deeper: dict[str, Any] = {
        "identifier": Identifier("level_17").serialize_to_dict(),
        "alternatives": [
            {
                "plain": {
                    "identifier": holder_id.serialize_to_dict(),
                    "variables": [],
                    "choices": payload["choices"],
                    "notes": [],
                }
            }
        ],
        "notes": [],
    }
    payload["choices"] = [deeper]
    message = f"choice nesting exceeds {MAX_CHOICE_DEPTH} levels"

    with pytest.raises(DeserializationValueError, match=message):
        Space.deserialize_from_dict(payload)
    with pytest.raises(DeserializationValueError, match=message):
        Space.from_json(_canonical_text(payload))


def test_choices_nested_past_the_cap_are_refused() -> None:
    """Test a choice one level deeper than the core allows is refused."""
    chain = build_chain(MAX_CHOICE_DEPTH)
    holder = make_alternative("holder", choices=(chain,))
    top = Identifier("top")

    with pytest.raises(SearchSpaceError) as excinfo:
        Choice((holder,), name=top)

    assert str(excinfo.value) == (
        f"the choice {top!r} nests choices more than {MAX_CHOICE_DEPTH} levels deep"
    )


@pytest.mark.subprocess
def test_choices_nested_deeper_than_the_recursion_limit_are_refused() -> None:
    """Test a choice deeper than a lowered recursion limit raises `RecursionError`.

    The default limit is far above the core's cap, so the limit is lowered
    in a fresh interpreter.
    """
    program = textwrap.dedent(
        """
        import sys

        from fhy_core.identifier import Identifier
        from fhy_core.search_space import Alternative, Choice

        choice = Choice((Alternative(name=Identifier("leaf")),))
        for _ in range(12):
            choice = Choice((Alternative(choices=(choice,)),))
        sys.setrecursionlimit(13)
        try:
            Choice((Alternative(choices=(choice,)),))
        except RecursionError as error:
            print("refused:", error)
        """
    )

    completed = run_python(program)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "refused: maximum recursion depth exceeded: the choice is 14 levels deep",
    ]


@pytest.mark.slow
@pytest.mark.subprocess
def test_payload_deeper_than_the_recursion_limit_is_refused() -> None:
    """Test decoding a payload nesting more choices than the limit allows raises.

    The payload is written under the default limit and read under a lower
    one, so the reader refuses it before it reaches the core; a payload
    within the lower limit still decodes.
    """
    program = textwrap.dedent(
        """
        import sys

        from fhy_core.identifier import Identifier
        from fhy_core.search_space import Alternative, Choice, Space

        def chain(depth):
            choice = Choice((Alternative(name=Identifier("leaf")),))
            for _ in range(depth - 1):
                choice = Choice((Alternative(choices=(choice,)),))
            return Space(choices=(choice,))

        deep, shallow = chain(14), chain(8)
        deep_dict, deep_text = deep.serialize_to_dict(), deep.to_json()
        shallow_text = shallow.to_json()
        sys.setrecursionlimit(13)
        for read in (
            lambda: Space.deserialize_from_dict(deep_dict),
            lambda: Space.from_json(deep_text),
        ):
            try:
                read()
            except RecursionError as error:
                print("refused:", error)
        print("decoded:", Space.from_json(shallow_text).to_json() == shallow_text)
        """
    )

    completed = run_python(program)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "refused: maximum recursion depth exceeded: the payload nests choices "
        "14 levels deep",
        "refused: maximum recursion depth exceeded: the payload nests choices "
        "14 levels deep",
        "decoded: True",
    ]


# ===========================================================================
# Cyclic garbage collection
# ===========================================================================


class _Holder(Alternative):
    """An alternative holding a mutable box, through which a cycle runs."""

    def __init__(self, *, box: list[Any], **fields: Any) -> None:
        super().__init__(**fields)
        self.box = box


class _HeldVariable(Variable[Any]):
    """A variable holding a mutable box, through which a cycle runs."""

    def __init__(self, *, box: list[Any], **fields: Any) -> None:
        super().__init__(**fields)
        self.box = box


def _collects(build: Callable[[], object]) -> bool:
    """Return whether the cycle `build` makes is freed by `gc.collect()`."""
    watched = weakref.ref(build())
    gc.collect()
    return watched() is None


def test_cycle_through_a_choice_and_a_subclass_alternative_is_collected() -> None:
    """Test a choice holding an alternative whose box holds the choice is freed."""

    def build() -> object:
        box: list[Any] = []
        alternative = _Holder(box=box, name=Identifier("held"))
        box.append(Choice((alternative,)))
        return alternative

    assert _collects(build)


def test_cycle_through_a_space_and_a_subclass_variable_is_collected() -> None:
    """Test a space holding a variable whose box holds the space is freed."""

    def build() -> object:
        box: list[Any] = []
        variable = _HeldVariable(box=box, param=categorical(), name=Identifier("v"))
        space = Space(variables=(variable,))
        box.append(Configuration(space, {variable.name: 1}))
        return variable

    assert _collects(build)


def test_cycle_through_an_alternative_holding_a_subclass_variable_is_collected() -> (
    None
):
    """Test an alternative holding a variable whose box holds it is freed."""

    def build() -> object:
        box: list[Any] = []
        variable = _HeldVariable(box=box, param=categorical(), name=Identifier("v"))
        box.append(Alternative(variables=(variable,)))
        return variable

    assert _collects(build)
