"""Python subclasses of `Variable` and `Alternative`, through their hooks.

The subclasses are modelled on MOGA-VM's `ArrayTileKnob`, `PortBoundKnob`,
a marker knob such as `NamespaceKnob`, and `RealizationOption`, as
`docs/design/search-space.md`, "Implementors", shows them. The core compares
their base fields itself and calls their `extension_*` hooks for their own
data, once per pair of nodes of one kind; their own type ids are their
kinds, and they round-trip through them.
"""

import json
import pickle
from typing import Any, ClassVar

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    Configuration,
    DuplicateNameError,
    Space,
    Variable,
)
from fhy_core.serialization import (
    SerializedDict,
    UnknownTypeIdError,
    register_serializable,
)
from fhy_core.term import AlphaRenaming
from fhy_core.traits import FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override

from .conftest import (
    Explosion,
    categorical,
    make_alternative,
    make_choice,
    make_variable,
)

_BASE_VARIABLE_KEYS = ("identifier", "param", "notes")
_BASE_ALTERNATIVE_KEYS = ("identifier", "variables", "choices", "notes")


def _base_fields(base: type[Any], data: Any, keys: tuple[str, ...]) -> Any:
    """Return the plain value of the base keys of a subclass's data."""
    return base.deserialize_data_from_dict({key: data[key] for key in keys})


@register_serializable(type_id="tests.search_space.array_tile_knob")
class ArrayTileKnob(Variable[Any]):
    """A tile size of an array, over the array's index symbols (references)."""

    calls: ClassVar[list[str]] = []

    def __init__(self, *, index_symbols: tuple[Identifier, ...], **fields: Any) -> None:
        super().__init__(**fields)
        self.index_symbols = tuple(index_symbols)

    @override
    def extension_is_structurally_equivalent(self, other: "ArrayTileKnob") -> bool:
        ArrayTileKnob.calls.append("structural")
        return self.index_symbols == other.index_symbols

    @override
    def extension_is_alpha_equivalent_under(
        self, other: "ArrayTileKnob", renaming: AlphaRenaming
    ) -> bool:
        ArrayTileKnob.calls.append("alpha")
        return len(self.index_symbols) == len(other.index_symbols) and all(
            renaming.are_identifiers_alpha_equivalent(left, right)
            for left, right in zip(self.index_symbols, other.index_symbols, strict=True)
        )

    @override
    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            **super().serialize_data_to_dict(),
            "index_symbols": [
                symbol.serialize_to_dict() for symbol in self.index_symbols
            ],
        }

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "ArrayTileKnob":
        fields: Any = data
        base = _base_fields(Variable, data, _BASE_VARIABLE_KEYS)
        return cls(
            index_symbols=tuple(
                Identifier.deserialize_from_dict(symbol)
                for symbol in fields["index_symbols"]
            ),
            param=base.param,
            name=base.name,
            notes=base.notes,
        )


@register_serializable(type_id="tests.search_space.port_bound_knob")
class PortBoundKnob(Variable[Any]):
    """A knob bound to a port, compared by its role and index."""

    def __init__(self, *, port_role: str, port_index: int, **fields: Any) -> None:
        super().__init__(**fields)
        self.port_role = port_role
        self.port_index = port_index

    @override
    def extension_is_structurally_equivalent(self, other: "PortBoundKnob") -> bool:
        return (self.port_role, self.port_index) == (other.port_role, other.port_index)

    @override
    def extension_is_alpha_equivalent_under(
        self, other: "PortBoundKnob", renaming: AlphaRenaming
    ) -> bool:
        return (self.port_role, self.port_index) == (other.port_role, other.port_index)


@register_serializable(type_id="tests.search_space.namespace_knob")
class NamespaceKnob(Variable[Any]):
    """A marker knob: no data of its own, told apart by its kind."""


@register_serializable(type_id="tests.search_space.realization_option")
class RealizationOption(Alternative):
    """An option realized as a walk: it binds its axes, and has a shape."""

    calls: ClassVar[list[str]] = []

    def __init__(
        self, *, axes: tuple[Identifier, ...], shape: tuple[int, ...], **fields: Any
    ) -> None:
        super().__init__(**fields)
        self.axes = tuple(axes)
        self.shape = tuple(shape)

    @override
    def extension_bound_identifiers(self) -> tuple[Identifier, ...]:
        RealizationOption.calls.append("bound")
        return self.axes

    @override
    def extension_is_structurally_equivalent(self, other: "RealizationOption") -> bool:
        RealizationOption.calls.append("structural")
        return self.axes == other.axes and self.shape == other.shape

    @override
    def extension_is_alpha_equivalent_under(
        self, other: "RealizationOption", renaming: AlphaRenaming
    ) -> bool:
        RealizationOption.calls.append("alpha")
        return self.shape == other.shape and all(
            renaming.are_identifiers_alpha_equivalent(left, right)
            for left, right in zip(self.axes, other.axes, strict=True)
        )

    @override
    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            **super().serialize_data_to_dict(),
            "axes": [axis.serialize_to_dict() for axis in self.axes],
            "shape": list(self.shape),
        }

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "RealizationOption":
        fields: Any = data
        base = _base_fields(Alternative, data, _BASE_ALTERNATIVE_KEYS)
        return cls(
            axes=tuple(
                Identifier.deserialize_from_dict(axis) for axis in fields["axes"]
            ),
            shape=tuple(fields["shape"]),
            variables=base.variables,
            choices=base.choices,
            name=base.name,
            notes=base.notes,
        )


class _Uninitialized(Variable[Any]):
    """A subclass whose `__init__` forgets to call `Variable.__init__`."""

    def __init__(self) -> None:
        self.data = 1


class _Answering(Variable[Any]):
    """A knob whose structural hook answers `answer`, or raises it."""

    def __init__(self, *, answer: Any, **fields: Any) -> None:
        super().__init__(**fields)
        self.answer = answer

    @override
    def extension_is_structurally_equivalent(self, other: Any) -> bool:
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer  # type: ignore[no-any-return]

    @override
    def extension_is_alpha_equivalent_under(
        self, other: Any, renaming: AlphaRenaming
    ) -> bool:
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer  # type: ignore[no-any-return]


class _Binding(Alternative):
    """An alternative whose bound identifiers are `bound`, or raise it."""

    def __init__(self, *, bound: Any, **fields: Any) -> None:
        super().__init__(**fields)
        self.bound = bound

    @override
    def extension_bound_identifiers(self) -> Any:
        if isinstance(self.bound, BaseException):
            raise self.bound
        return self.bound


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    """Forget the hook calls an earlier test recorded."""
    ArrayTileKnob.calls.clear()
    RealizationOption.calls.clear()


def _tile_knob(
    name: Identifier, index_symbols: tuple[Identifier, ...], *values: int
) -> ArrayTileKnob:
    return ArrayTileKnob(
        index_symbols=index_symbols, param=categorical(*values), name=name
    )


def _realized_space(
    *, shape: tuple[int, ...] = (4,)
) -> tuple[Space, RealizationOption]:
    """Return a space whose one choice holds a realization with a tile knob.

    The knob's index symbols are the realization's axes, which the
    realization binds, so a relabeled copy corresponds name for name (C-1).
    """
    axis = Identifier("i")
    knob = _tile_knob(Identifier("tile"), (axis,), 4, 8)
    option = RealizationOption(
        axes=(axis,), shape=shape, variables=(knob,), name=Identifier("realized")
    )
    choice = Choice((option, make_alternative("plain")), name=Identifier("layout"))
    return Space(choices=(choice,), name=Identifier("program")), option


# ===========================================================================
# Construction and the base fields
# ===========================================================================


def test_subclass_constructs_with_its_own_fields() -> None:
    """Test a subclass keeps its base fields and its own attributes."""
    name, axis = Identifier("tile"), Identifier("i")
    param = categorical(4, 8)

    knob = ArrayTileKnob(index_symbols=(axis,), param=param, name=name)

    assert knob.name is name
    assert knob.param is param
    assert knob.index_symbols == (axis,)
    assert knob.kind == "tests.search_space.array_tile_knob"
    assert isinstance(knob, Variable)
    assert isinstance(knob, FrozenMixin)


def test_subclass_is_frozen_after_its_init() -> None:
    """Test a subclass refuses a mutation once its `__init__` returned."""
    knob = _tile_knob(Identifier("tile"), (), 4)

    with pytest.raises(FrozenMutationError):
        knob.index_symbols = ()
    assert knob.is_frozen


def test_marker_subclass_has_its_own_kind() -> None:
    """Test a subclass without data of its own is told apart by its kind."""
    knob = NamespaceKnob(param=categorical(), name=Identifier("port0"))

    assert knob.kind == "tests.search_space.namespace_knob"


def test_alternative_subclass_keeps_its_fields() -> None:
    """Test an alternative subclass keeps its base fields and its kind."""
    _, option = _realized_space()

    assert option.kind == "tests.search_space.realization_option"
    assert option.variables[0].name.name_hint == "tile"
    assert option.axes[0].name_hint == "i"


def test_uninitialized_subclass_is_refused_in_a_container() -> None:
    """Test an instance whose `__init__` skipped `Variable.__init__` is refused."""
    knob = _Uninitialized()

    with pytest.raises(RuntimeError, match="_Uninitialized"):
        Alternative(variables=(knob,))
    with pytest.raises(RuntimeError, match="_Uninitialized"):
        knob.name  # noqa: B018


def test_default_hooks_answer_true_and_bind_nothing() -> None:
    """Test the hooks a subclass does not override keep the defaults."""
    left = NamespaceKnob(param=categorical(), name=Identifier("a"))
    right = NamespaceKnob(param=categorical(), name=Identifier("b"))

    assert left.extension_is_structurally_equivalent(right) is True
    assert (
        left.extension_is_alpha_equivalent_under(right, AlphaRenaming.empty()) is True
    )
    assert tuple(make_alternative("a").extension_bound_identifiers()) == ()


# ===========================================================================
# Equivalence through the hooks
# ===========================================================================


def test_tile_knobs_with_equal_data_are_structurally_equivalent() -> None:
    """Test two knobs sharing name, param and index symbols are equivalent."""
    name, axis = Identifier("tile"), Identifier("i")
    param = categorical(4)
    left = ArrayTileKnob(index_symbols=(axis,), param=param, name=name)
    right = ArrayTileKnob(index_symbols=(axis,), param=param, name=name)

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)
    assert ArrayTileKnob.calls == ["structural", "structural"]


def test_tile_knobs_with_other_index_symbols_are_not_equivalent() -> None:
    """Test the hook discriminates on the subclass's own data."""
    name = Identifier("tile")
    param = categorical(4)
    left = ArrayTileKnob(index_symbols=(Identifier("i"),), param=param, name=name)
    right = ArrayTileKnob(index_symbols=(Identifier("j"),), param=param, name=name)

    assert not left.is_structurally_equivalent(right)
    assert not left.is_alpha_equivalent(right)


def test_tile_knob_references_correspond_under_a_renaming() -> None:
    """Test index symbols are references: they correspond under a renaming."""
    left_axis, right_axis = Identifier("i"), Identifier("j")
    left = _tile_knob(Identifier("tile"), (left_axis,), 4)
    right = _tile_knob(Identifier("tile"), (right_axis,), 4)

    renaming = AlphaRenaming.empty().extend({left_axis: right_axis})

    assert left.is_alpha_equivalent_under(right, renaming)
    assert not left.is_alpha_equivalent(right)


def test_hooks_are_not_called_for_different_kinds() -> None:
    """Test a kind mismatch answers `False` before any hook runs."""
    name = Identifier("tile")
    param = categorical(4)
    knob = ArrayTileKnob(index_symbols=(), param=param, name=name)
    plain = Variable(param=param, name=name)

    assert not knob.is_structurally_equivalent(plain)
    assert not plain.is_structurally_equivalent(knob)
    assert ArrayTileKnob.calls == []


def test_variables_of_different_kinds_are_not_structurally_equivalent() -> None:
    """Test a marker subclass and a plain variable differ by kind alone.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_distinct_knob_kinds_not_structurally_equivalent`.
    """
    name = Identifier("port0")
    param = categorical(Identifier("affine"))
    namespace = NamespaceKnob(param=param, name=name)
    generic = Variable(param=param, name=name)

    assert not namespace.is_structurally_equivalent(generic)
    assert not generic.is_structurally_equivalent(namespace)


def test_variables_of_one_subclass_kind_sharing_parts_are_structurally_equivalent() -> (
    None
):
    """Test two marker knobs sharing name and param are equivalent.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_same_knob_kind_structurally_equivalent_for_shared_name_and_param`.
    """
    name = Identifier("port0")
    param = categorical(Identifier("affine"))

    assert NamespaceKnob(param=param, name=name).is_structurally_equivalent(
        NamespaceKnob(param=param, name=name)
    )


def test_port_bound_knobs_compare_their_role_and_index() -> None:
    """Test a `PortBoundKnob`-shaped subclass compares its own fields by `==`."""
    name, param = Identifier("port"), categorical()

    def build(role: str, index: int) -> PortBoundKnob:
        return PortBoundKnob(port_role=role, port_index=index, param=param, name=name)

    assert build("input", 0).is_structurally_equivalent(build("input", 0))
    assert not build("input", 0).is_structurally_equivalent(build("input", 1))
    assert not build("input", 0).is_alpha_equivalent(build("output", 0))


def test_alternatives_with_different_own_data_are_not_structurally_equivalent() -> None:
    """Test two realizations differing only in their own data differ.

    Ported from MOGA-VM
    `test_structural_equivalence.py::test_options_not_structurally_equivalent_when_realization_domain_differs`.
    """
    name, axis = Identifier("o"), Identifier("i")
    knob = make_variable("k")
    left = RealizationOption(axes=(axis,), shape=(4,), variables=(knob,), name=name)
    rehomed = RealizationOption(axes=(axis,), shape=(8,), variables=(knob,), name=name)
    same = RealizationOption(axes=(axis,), shape=(4,), variables=(knob,), name=name)

    assert not left.is_structurally_equivalent(rehomed)
    assert left.is_structurally_equivalent(same)


def test_realized_spaces_are_alpha_equivalent_through_the_bound_axes() -> None:
    """Test a relabeled realized space corresponds, its knob's axes included.

    The realization binds its axes, so the knob's index symbols, references
    to them, correspond in the space's frame (C-1).
    """
    left, _ = _realized_space()
    right, _ = _realized_space()

    assert not left.is_structurally_equivalent(right)
    assert left.is_alpha_equivalent(right)
    assert right.is_alpha_equivalent(left)


def test_realized_spaces_differing_in_shape_are_not_alpha_equivalent() -> None:
    """Test the realization's hook still discriminates inside a space."""
    left, _ = _realized_space(shape=(4,))
    right, _ = _realized_space(shape=(8,))

    assert not left.is_alpha_equivalent(right)


def test_realizations_are_alpha_equivalent_standalone() -> None:
    """Test a realization compared on its own binds its axes itself."""
    _, left = _realized_space()
    _, right = _realized_space()

    assert left.is_alpha_equivalent(right)


def test_hook_is_called_once_per_pair_in_a_space() -> None:
    """Test comparing two spaces calls each hook once for the one pair."""
    left, _ = _realized_space()
    right, _ = _realized_space()
    ArrayTileKnob.calls.clear()
    RealizationOption.calls.clear()

    assert left.is_alpha_equivalent(right)

    assert ArrayTileKnob.calls == ["alpha"]
    assert RealizationOption.calls == ["alpha"]


def test_bound_identifiers_are_read_when_the_choice_is_built() -> None:
    """Test the bound identifiers are read once, by the choice holding them."""
    axis = Identifier("i")
    option = RealizationOption(axes=(axis,), shape=(), name=Identifier("o"))
    RealizationOption.calls.clear()

    Choice((option,))

    assert RealizationOption.calls == ["bound"]


def test_bound_identifier_clashing_with_a_name_is_refused() -> None:
    """Test an axis named like a decision of the space is a duplicate name."""
    shared = Identifier("shared")
    option = RealizationOption(axes=(shared,), shape=(), name=Identifier("o"))

    with pytest.raises(DuplicateNameError):
        Space(
            variables=(Variable(param=categorical(), name=shared),),
            choices=(Choice((option,)),),
        )


# ===========================================================================
# Hook failures
# ===========================================================================


def test_hook_exception_propagates_as_itself() -> None:
    """Test an exception a hook raises reaches the caller as the same object."""
    explosion = Explosion("hook")
    name, param = Identifier("k"), categorical()
    left = _Answering(answer=explosion, param=param, name=name)
    right = _Answering(answer=True, param=param, name=name)

    with pytest.raises(Explosion) as excinfo:
        left.is_structurally_equivalent(right)
    assert excinfo.value is explosion

    with pytest.raises(Explosion) as excinfo:
        Space(variables=(left,)).is_alpha_equivalent(Space(variables=(right,)))
    assert excinfo.value is explosion


def test_keyboard_interrupt_in_a_hook_passes_through() -> None:
    """Test a `KeyboardInterrupt` a hook raises is not wrapped."""
    interrupt = KeyboardInterrupt()
    name, param = Identifier("k"), categorical()
    left = _Answering(answer=interrupt, param=param, name=name)
    right = _Answering(answer=True, param=param, name=name)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        left.is_structurally_equivalent(right)

    assert excinfo.value is interrupt


def test_hook_answering_a_non_bool_raises_type_error() -> None:
    """Test a hook's result must be a `bool`."""
    name, param = Identifier("k"), categorical()
    left = _Answering(answer=1, param=param, name=name)
    right = _Answering(answer=1, param=param, name=name)

    with pytest.raises(TypeError) as excinfo:
        left.is_structurally_equivalent(right)

    assert str(excinfo.value) == (
        "_Answering.extension_is_structurally_equivalent must return a bool, got int."
    )


def test_bound_identifiers_hook_exception_fails_the_choice() -> None:
    """Test the exception `extension_bound_identifiers` raises is the choice's."""
    explosion = Explosion("bound")

    with pytest.raises(Explosion) as excinfo:
        Choice((_Binding(bound=explosion, name=Identifier("o")),))

    assert excinfo.value is explosion


def test_bound_identifiers_of_another_type_raise_type_error() -> None:
    """Test bound identifiers must be `Identifier`s."""
    with pytest.raises(TypeError) as excinfo:
        Choice((_Binding(bound=("i",), name=Identifier("o")),))

    assert str(excinfo.value) == (
        "_Binding.extension_bound_identifiers must return Identifiers, got str."
    )


# ===========================================================================
# Containers, configurations and kept objects
# ===========================================================================


def test_containers_return_the_subclass_instances_given() -> None:
    """Test a subclass instance comes back as itself from its containers."""
    space, option = _realized_space()
    (knob,) = option.variables

    assert space.choices[0].alternatives[0] is option
    assert space.decision(knob.name) is knob
    assert knob in space.decisions


def test_configuration_chooses_a_subclass_alternative() -> None:
    """Test a configuration's alternative is the subclass instance itself."""
    space, option = _realized_space()
    (choice,) = space.choices
    (knob,) = option.variables

    configuration = Configuration(space, {choice.name: option.name, knob.name: 8})

    assert configuration.alternative(choice.name) is option
    assert configuration.is_complete()


# ===========================================================================
# Serialization and pickling
# ===========================================================================


def test_subclass_payload_data_is_its_data_dict() -> None:
    """Test the foreign part's data is the text of `serialize_data_to_dict`."""
    knob = _tile_knob(Identifier("tile"), (Identifier("i"),), 4)

    payload: Any = knob.serialize_to_dict()
    (foreign,) = payload.values()

    assert foreign["type_id"] == "tests.search_space.array_tile_knob"
    assert json.loads(foreign["data"]) == knob.serialize_data_to_dict()


def test_subclass_round_trips_through_its_type_id() -> None:
    """Test decoding a subclass's payload builds an instance of the subclass."""
    knob = _tile_knob(Identifier("tile"), (Identifier("i"),), 4, 8)

    decoded: Any = Variable.deserialize_from_dict(knob.serialize_to_dict())

    assert type(decoded) is ArrayTileKnob
    assert decoded.index_symbols == knob.index_symbols
    assert decoded.is_structurally_equivalent(knob)


def test_space_of_subclasses_round_trips() -> None:
    """Test a space holding subclass parts decodes with them in place."""
    space, option = _realized_space()

    decoded = Space.from_json(space.to_json())

    decoded_option = decoded.choices[0].alternatives[0]
    assert type(decoded_option) is RealizationOption
    assert decoded_option.axes == option.axes
    assert type(decoded_option.variables[0]) is ArrayTileKnob
    assert decoded.is_structurally_equivalent(space)
    assert decoded.to_json() == space.to_json()


def test_unregistered_type_id_fails_to_decode() -> None:
    """Test a foreign part of an unknown type id raises the registry's error."""
    payload: Any = {
        "foreign": {"type_id": "tests.search_space.nobody", "data": "{}"},
    }

    with pytest.raises(UnknownTypeIdError):
        Variable.deserialize_from_dict(payload)


def test_subclass_hook_exception_while_writing_propagates() -> None:
    """Test an exception `serialize_data_to_dict` raises reaches the writer."""
    explosion = Explosion("write")

    class _Unwritable(Variable[Any]):
        @override
        def serialize_data_to_dict(self) -> SerializedDict:
            raise explosion

    space = Space(variables=(_Unwritable(param=categorical(), name=Identifier("v")),))

    with pytest.raises(Explosion) as excinfo:
        space.to_json()

    assert excinfo.value is explosion


def test_subclass_pickles_with_its_data() -> None:
    """Test a subclass instance pickles to a frozen copy with its data."""
    knob = _tile_knob(Identifier("tile"), (Identifier("i"),), 4)

    restored = pickle.loads(pickle.dumps(knob))

    assert type(restored) is ArrayTileKnob
    assert restored.index_symbols == knob.index_symbols
    assert restored.is_structurally_equivalent(knob)
    assert restored.is_frozen


def test_subclass_repr_names_its_class() -> None:
    """Test a subclass's `repr` uses its own class name over the base fields."""
    knob = _tile_knob(Identifier("tile"), (), 4)

    assert repr(knob) == (
        f"ArrayTileKnob(name={knob.name!r}, param={knob.param!r}, notes=())"
    )


def test_subclass_in_a_choice_compares_inside_a_configuration() -> None:
    """Test configurations of relabeled realized spaces share a key."""
    keys = []
    for _ in range(2):
        space, option = _realized_space()
        (choice,) = space.choices
        (knob,) = option.variables
        keys.append(
            Configuration(space, {choice.name: option.name, knob.name: 4}).key()
        )

    assert keys[0] == keys[1]


def test_choice_mixes_subclass_and_plain_alternatives() -> None:
    """Test one choice holds a subclass alternative beside a plain one."""
    _, option = _realized_space()

    choice = make_choice("mixed", option, make_alternative("plain"))

    assert choice.alternatives[0] is option
