"""Tests of downstream Rust kinds of `Variable` and `Alternative`.

`rust/example-aggregate` registers two kinds from its `#[pymodule]`:
`TiledVariable` (`example.tiled_variable`), a variable with index symbols
compared by its Rust hooks, and `AxisAlternative`
(`example.axis_alternative`), an alternative binding its axes. Each test
runs in a fresh interpreter whose native module is the aggregate, and
checks that the binding reads, compares, writes and decodes the kinds as it
does its own classes, and that the registry refuses a second registration.
"""

import pathlib
import textwrap

import pytest

from tests.native_modules import run_python
from tests.test_composed_extension import site  # noqa: F401  # the fixture

pytestmark = [pytest.mark.slow, pytest.mark.subprocess]

_PRELUDE = """
import fhy_core
import _fhy_example_aggregate as aggregate
from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Alternative,
    Choice,
    Configuration,
    Space,
    Variable,
)
from fhy_core.symbolic.param import create_categorical_param


def realized_space(free_symbol=None):
    axis = Identifier("i")
    tiled = aggregate.TiledVariable(
        create_categorical_param(frozenset({4, 8})),
        (free_symbol or axis,),
        Identifier("tile"),
    )
    realized = aggregate.AxisAlternative((axis,), (tiled,), Identifier("realized"))
    plain = Alternative(name=Identifier("plain"))
    choice = Choice((realized, plain), name=Identifier("layout"))
    space = Space(choices=(choice,), name=Identifier("program"))
    return space, choice, realized, tiled
"""


def _run(site: pathlib.Path, program: str) -> list[str]:  # noqa: F811
    """Run the prelude and `program` in a fresh interpreter; return its lines."""
    completed = run_python(_PRELUDE + textwrap.dedent(program), python_path=[site])
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


def test_kinds_are_virtual_subclasses_of_the_public_classes(
    site: pathlib.Path,  # noqa: F811
) -> None:
    """Test each registered class counts as a `Variable` or `Alternative`."""
    output = _run(
        site,
        """
        print(issubclass(aggregate.TiledVariable, Variable))
        print(issubclass(aggregate.AxisAlternative, Alternative))
        print(issubclass(aggregate.Tagger, Variable))
        """,
    )

    assert output == ["True", "True", "False"]


def test_kinds_are_read_into_containers(site: pathlib.Path) -> None:  # noqa: F811
    """Test a space holds the kinds, and returns the objects it was given."""
    output = _run(
        site,
        """
        space, choice, realized, tiled = realized_space()
        print(space.choices[0].alternatives[0] is realized)
        found = space.decision(tiled.name)
        same_symbols = found.index_symbols == tiled.index_symbols
        print(type(found).__name__, found.kind, same_symbols)
        print(",".join(decision.name.name_hint for decision in space.decisions))
        """,
    )

    assert output == [
        "True",
        "TiledVariable",
        "example.tiled_variable",
        "True",
        "layout,tile",
    ]


def test_kinds_compare_through_their_rust_hooks(site: pathlib.Path) -> None:  # noqa: F811
    """Test relabeled spaces of the kinds correspond, and a free symbol does not."""
    output = _run(
        site,
        """
        left, *_ = realized_space()
        right, *_ = realized_space()
        stray, *_ = realized_space(Identifier("stray"))
        print(left.is_alpha_equivalent(right), right.is_alpha_equivalent(left))
        print(left.is_structurally_equivalent(right))
        print(left.is_alpha_equivalent(stray))
        """,
    )

    assert output == ["True", "True", "False", "False"]


def test_kinds_round_trip_through_their_type_ids(site: pathlib.Path) -> None:  # noqa: F811
    """Test a space of the kinds writes their foreign parts and decodes them."""
    output = _run(
        site,
        """
        space, choice, realized, tiled = realized_space()
        text = space.to_json()
        print('"example.axis_alternative"' in text)
        decoded = Space.from_json(text)
        print(decoded.to_json() == text, decoded.is_structurally_equivalent(space))
        (decoded_realized, _) = decoded.choices[0].alternatives
        print(type(decoded_realized).__name__, decoded_realized.axes == realized.axes)
        (decoded_tiled,) = decoded_realized.variables
        print(type(decoded_tiled).__name__, decoded_tiled.kind)
        print(decoded_tiled.name == tiled.name)
        print(decoded_tiled.index_symbols == tiled.index_symbols)
        """,
    )

    assert output == [
        "True",
        "True",
        "True",
        "AxisAlternative",
        "True",
        "TiledVariable",
        "example.tiled_variable",
        "True",
        "True",
    ]


def test_a_foreign_kind_decodes_as_a_variable(site: pathlib.Path) -> None:  # noqa: F811
    """Test `Variable.deserialize_from_dict` decodes a registered kind's part."""
    output = _run(
        site,
        """
        space, choice, realized, tiled = realized_space()
        holder = Alternative(variables=(tiled,), name=Identifier("holder"))
        (payload,) = holder.serialize_to_dict()["plain"]["variables"]
        print(sorted(payload), payload["foreign"]["type_id"])
        decoded = Variable.deserialize_from_dict(payload)
        print(type(decoded).__name__, isinstance(decoded, Variable))
        print(decoded.index_symbols == tiled.index_symbols)
        """,
    )

    assert output == [
        "['foreign']",
        "example.tiled_variable",
        "TiledVariable",
        "True",
        "True",
    ]


def test_configuration_chooses_a_kind(site: pathlib.Path) -> None:  # noqa: F811
    """Test a configuration's alternative is the kind's object, its key shared."""
    output = _run(
        site,
        """
        keys = []
        for _ in range(2):
            space, choice, realized, tiled = realized_space()
            entries = {choice.name: realized.name, tiled.name: 8}
            configuration = Configuration(space, entries)
            keys.append(configuration.key())
        print(configuration.alternative(choice.name) is realized)
        print(configuration.is_complete(), keys[0] == keys[1])
        """,
    )

    assert output == ["True", "True", "True"]


def test_registry_refusal_messages(site: pathlib.Path) -> None:  # noqa: F811
    """Test the registry's refusals say why."""
    completed = run_python(
        _PRELUDE
        + textwrap.dedent(
            """
            import types

            variable = aggregate.KindRegistrar.register_variable
            alternative = aggregate.KindRegistrar.register_alternative
            tiled_class = aggregate.TiledVariable
            bare = types.ModuleType("bare")

            class Other:
                pass

            for attempt in (
                lambda: variable(_rs, "example.tiled_variable", Other),
                lambda: variable(_rs, "example.other", tiled_class),
                lambda: variable(_rs, "search_space.variable", Other),
                lambda: alternative(_rs, "search_space.alternative", Other),
                lambda: variable(bare, "example.bare", Other),
            ):
                try:
                    attempt()
                except (ValueError, RuntimeError) as error:
                    print(error)
            """
        ),
        python_path=[site],
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        'the Variable kind "example.tiled_variable" cannot be registered: '
        "it is registered already",
        'the Variable kind "example.other" cannot be registered: '
        "the class TiledVariable is registered for a kind already",
        'the Variable kind "search_space.variable" cannot be registered: '
        "it is the plain variable's kind",
        'the Alternative kind "search_space.alternative" cannot be registered: '
        "it is the plain alternative's kind",
        "the module bare holds no fhy_core binding: register fhy_core's classes "
        "into it first",
    ]


def test_a_kind_registered_after_import_is_a_virtual_subclass(
    site: pathlib.Path,  # noqa: F811
) -> None:
    """Test a kind registered once the public class exists subclasses it at once."""
    output = _run(
        site,
        """
        class Late:
            pass

        aggregate.KindRegistrar.register_variable(_rs, "example.late", Late)
        print(issubclass(Late, Variable))
        """,
    )

    assert output == ["True"]
