"""Python oracles and the `extension_search_domain` hook.

Any object with a callable `decide` is an oracle: its answers must be
coordinates, an exception it raises stops the run as the same object, and
the `PendingStep` it receives is a snapshot that still answers after the
call. A `Variable` subclass offers its own step domain through
`extension_search_domain`, called once per step built.
"""

from typing import Any, ClassVar

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    ChoiceDomain,
    OrderDomain,
    PendingStep,
    RandomOracle,
    Recorder,
    Rng,
    Space,
    StridedDomain,
    StridedRun,
    Variable,
)

from .conftest import Explosion, build_tiling_space, categorical, make_variable

_SUBJECT = Identifier("subject")
_OPTION = "moga.cir.option"


class _Constant:
    """An oracle answering `answer` to every step."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def decide(self, step: Any) -> Any:
        return self._answer


class _Capturing:
    """An oracle answering coordinate 0 and keeping the steps it was asked."""

    def __init__(self) -> None:
        self.steps: list[PendingStep] = []

    def decide(self, step: PendingStep) -> int:
        self.steps.append(step)
        return 0


class _Raising:
    """An oracle raising `error` when asked."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def decide(self, step: Any) -> int:
        raise self._error


# ===========================================================================
# Answers
# ===========================================================================


def test_python_oracle_answers_a_choice_with_an_int() -> None:
    """Test an `int` answer picks the domain's value at that coordinate."""
    recorder = Recorder(_Constant(2))

    value = recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b", "c")))

    assert value == "c"


def test_python_oracle_answers_an_order_with_a_tuple() -> None:
    """Test a tuple answer orders the elements by its positions."""
    elements = tuple(Identifier(name) for name in "abc")
    recorder = Recorder(_Constant((2, 0, 1)))

    value = recorder.decide_dynamic(
        "moga.cir.walk_order", _SUBJECT, OrderDomain(elements)
    )

    assert value == (elements[2], elements[0], elements[1])


def test_python_oracle_answers_a_strided_step_with_an_index() -> None:
    """Test an index answer picks the integer at that position of the runs."""
    domain = StridedDomain((StridedRun(0, 3), StridedRun(100, 105)))
    recorder = Recorder(_Constant(4))

    value = recorder.decide_dynamic("moga.cir.address", _SUBJECT, domain)

    assert value == 101


@pytest.mark.parametrize(
    "answer",
    ["0", 0.0, True, [0], None],
    ids=["str", "float", "bool", "list", "none"],
)
def test_python_oracle_answering_a_non_coordinate_raises_type_error(
    answer: Any,
) -> None:
    """Test an answer that is no `int` or tuple of `int`s is a `TypeError`."""
    recorder = Recorder(_Constant(answer))

    with pytest.raises(TypeError, match="must return an int or a tuple of ints"):
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((1, 2)))


@pytest.mark.parametrize(
    "answer", [(True, 0), (0, "1")], ids=["bool_member", "str_member"]
)
def test_python_oracle_answering_a_tuple_of_non_ints_raises_type_error(
    answer: tuple[Any, ...],
) -> None:
    """Test every member of a tuple answer must be an `int`."""
    recorder = Recorder(_Constant(answer))

    with pytest.raises(TypeError):
        recorder.decide_dynamic(
            "moga.cir.walk_order", _SUBJECT, OrderDomain(("a", "b"))
        )


def test_a_refused_answer_type_records_no_step() -> None:
    """Test a `TypeError` for an answer leaves the trace empty."""
    recorder = Recorder(_Constant("0"))

    with pytest.raises(TypeError):
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((1, 2)))

    assert len(recorder.trace) == 0


@pytest.mark.parametrize("oracle", [object(), None], ids=["no_decide", "none"])
def test_an_object_without_decide_is_no_oracle(oracle: Any) -> None:
    """Test a recorder asked to run over a non-oracle raises `TypeError`."""
    recorder = Recorder(oracle)

    with pytest.raises(TypeError):
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((1, 2)))


# ===========================================================================
# Exceptions
# ===========================================================================


def test_oracle_exception_propagates_as_itself() -> None:
    """Test an exception `decide` raises reaches the caller as the same object."""
    error = Explosion("oracle")
    recorder = Recorder(_Raising(error))

    with pytest.raises(Explosion) as excinfo:
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((1, 2)))

    assert excinfo.value is error


def test_oracle_exception_propagates_through_space_sample() -> None:
    """Test `Space.sample` raises the oracle's own exception object."""
    error = Explosion("sample")

    with pytest.raises(Explosion) as excinfo:
        build_tiling_space().space.sample(_Raising(error))

    assert excinfo.value is error


def test_keyboard_interrupt_in_an_oracle_passes_through() -> None:
    """Test a `KeyboardInterrupt` an oracle raises is not wrapped."""
    interrupt = KeyboardInterrupt()
    recorder = Recorder(_Raising(interrupt))

    with pytest.raises(KeyboardInterrupt) as excinfo:
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((1, 2)))

    assert excinfo.value is interrupt


def test_an_oracle_delegating_to_draw_uniform_stays_on_the_rng_stream() -> None:
    """Test `step.draw_uniform(rng)` draws as a `RandomOracle` of that seed does."""

    class Delegating:
        def __init__(self) -> None:
            self.rng = Rng(42)

        def decide(self, step: PendingStep) -> Any:
            return step.draw_uniform(self.rng)

    domain = ChoiceDomain(("a", "b", "c"))
    delegating, reference = Recorder(Delegating()), Recorder(RandomOracle(seed=42))
    for recorder in (delegating, reference):
        for _ in range(8):
            recorder.decide_dynamic(_OPTION, _SUBJECT, domain)

    assert delegating.trace.coordinates == reference.trace.coordinates


# ===========================================================================
# The step an oracle sees
# ===========================================================================


def test_a_dynamic_step_shows_its_kind_subject_and_domain() -> None:
    """Test `decide` sees the kind, the subject and the domain object given."""
    oracle = _Capturing()
    domain = ChoiceDomain(("a", "b"))

    Recorder(oracle).decide_dynamic(_OPTION, _SUBJECT, domain)

    (step,) = oracle.steps
    assert step.kind == _OPTION
    assert step.subject is _SUBJECT
    assert step.domain is domain


def test_a_dynamic_step_has_no_decision_or_configuration() -> None:
    """Test a dynamic step names no `Variable`, `Choice` or configuration."""
    oracle = _Capturing()

    Recorder(oracle).decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b")))

    (step,) = oracle.steps
    assert step.decision is None
    assert step.configuration is None


def test_step_positions_count_from_zero() -> None:
    """Test the steps of one run are numbered 0, 1, 2."""
    oracle = _Capturing()
    recorder = Recorder(oracle)
    for _ in range(3):
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b")))

    assert [step.position for step in oracle.steps] == [0, 1, 2]


def test_a_static_step_shows_its_variable_or_choice_object() -> None:
    """Test `decision` is the `Variable` or `Choice` object the space holds."""
    tiling = build_tiling_space()
    oracle = _Capturing()

    tiling.space.sample(oracle)

    unroll_step, layout_step, tile_step = oracle.steps
    assert unroll_step.decision is tiling.unroll
    assert layout_step.decision is tiling.layout
    assert tile_step.decision is tiling.tile
    assert unroll_step.subject == tiling.unroll.name
    assert layout_step.kind == "search_space.choice"


def test_a_static_step_shows_the_configuration_so_far() -> None:
    """Test `configuration` holds the entries answered before the step."""
    tiling = build_tiling_space()
    oracle = _Capturing()

    tiling.space.sample(oracle)

    unroll_step, layout_step, _ = oracle.steps
    assert unroll_step.configuration is not None
    assert unroll_step.configuration.entries == ()
    assert layout_step.configuration is not None
    assert [name for name, _ in layout_step.configuration.entries] == [
        tiling.unroll.name
    ]


def test_a_static_steps_domain_holds_its_alternatives() -> None:
    """Test a choice step's domain has one coordinate per alternative."""
    tiling = build_tiling_space()
    oracle = _Capturing()

    tiling.space.sample(oracle)

    layout_step = oracle.steps[1]
    assert layout_step.domain.cardinality == 2


def test_a_step_still_answers_admits_after_the_call() -> None:
    """Test the snapshot kept past `decide` judges coordinates of its domain."""
    oracle = _Capturing()
    Recorder(oracle).decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b", "c")))
    (step,) = oracle.steps

    assert step.admits(2)
    assert not step.admits(3)


def test_a_static_step_still_answers_admits_after_the_call() -> None:
    """Test a snapshot of a static step judges coordinates after the run."""
    tiling = build_tiling_space()
    oracle = _Capturing()
    tiling.space.sample(oracle)
    layout_step = oracle.steps[1]

    assert layout_step.admits(1)
    assert not layout_step.admits(2)


def test_step_admits_refuses_a_coordinate_that_is_no_int() -> None:
    """Test `admits("0")` raises `TypeError`."""
    oracle = _Capturing()
    Recorder(oracle).decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b")))
    (step,) = oracle.steps

    with pytest.raises(TypeError):
        step.admits("0")  # type: ignore[arg-type]


def test_step_coordinate_of_gives_the_coordinate_of_a_value() -> None:
    """Test `coordinate_of(value)` is the value's coordinate in the domain."""
    oracle = _Capturing()
    Recorder(oracle).decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b", "c")))
    (step,) = oracle.steps

    assert step.coordinate_of("c") == 2


# ===========================================================================
# extension_search_domain
# ===========================================================================


class _Answering(Variable[Any]):
    """A variable whose hook returns `answer`, or raises it if an exception."""

    calls: ClassVar[list[Identifier]] = []

    def __init__(self, *, answer: Any, **fields: Any) -> None:
        super().__init__(**fields)
        self.answer = answer

    def extension_search_domain(self) -> Any:
        _Answering.calls.append(self.name)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer() if callable(self.answer) else self.answer


def _answering(answer: Any) -> _Answering:
    return _Answering(answer=answer, param=categorical(1, 2, 3), name=Identifier("k"))


def test_the_hook_defaults_to_none() -> None:
    """Test a plain variable offers no domain of its own."""
    assert make_variable("k").extension_search_domain() is None


def test_a_hook_domain_gives_the_steps_over_the_variable() -> None:
    """Test the step over a subclass takes its coordinates from the hook's domain."""
    variable = _answering(lambda: ChoiceDomain((3, 2, 1)))

    configuration, _ = Space(variables=(variable,)).sample(_Constant(0))

    assert configuration.value(variable.name) == 3


def test_the_hook_is_called_once_per_step_built() -> None:
    """Test sampling a space with the variable calls its hook once."""
    variable = _answering(lambda: ChoiceDomain((3, 2, 1)))
    _Answering.calls.clear()

    Space(variables=(variable,)).sample(_Constant(0))

    assert _Answering.calls == [variable.name]


@pytest.mark.parametrize("answer", ["text", 3, (1, 2)], ids=["str", "int", "tuple"])
def test_a_hook_answering_a_wrong_type_raises_type_error(answer: Any) -> None:
    """Test a hook's result must be a domain or `None`."""
    variable = _answering(answer)

    with pytest.raises(TypeError):
        Space(variables=(variable,)).sample(_Constant(0))


def test_hook_exception_propagates_as_itself() -> None:
    """Test an exception the hook raises reaches the caller as the same object."""
    error = Explosion("domain")
    variable = _answering(error)

    with pytest.raises(Explosion) as excinfo:
        Space(variables=(variable,)).sample(_Constant(0))

    assert excinfo.value is error


def test_keyboard_interrupt_in_the_hook_passes_through() -> None:
    """Test a `KeyboardInterrupt` the hook raises is not wrapped."""
    interrupt = KeyboardInterrupt()
    variable = _answering(interrupt)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        Space(variables=(variable,)).sample(_Constant(0))

    assert excinfo.value is interrupt
