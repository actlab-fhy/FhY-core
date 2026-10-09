"""Shared helpers for the `tests/symbolic` sub-package.

The child conftests under `constraint`, `expression`, and `param` reach
these names as `..conftest`, which resolves here rather than to the root
`tests/conftest.py`. Removing this module breaks all three.
"""

from collections.abc import Callable, Iterator

import pytest

from fhy_core.symbolic.solver import (
    SatResult,
    SmtScript,
    SmtSolver,
    Solver,
    get_default_solver,
    set_default_solver,
)
from fhy_core.utils.override import override

from ..conftest import (  # re-exported below
    SerializableEqualHashable,
    mock_identifier,
)

__all__ = [
    "RecordingSmtSolver",
    "SerializableEqualHashable",
    "mock_identifier",
    "plug_smt_solver",
]


class RecordingSmtSolver(SmtSolver):
    """An SMT backend answering every check alike and recording each check.

    Plugged into the default solver by ``plug_smt_solver``, it stands in for
    z3 behind every question the constraints and params ask, as the tests
    once patched the solver's module functions for.
    """

    def __init__(self, answer: SatResult) -> None:
        super().__init__()
        self.answer = answer
        self.checks: list[tuple[str, int | None]] = []

    @override
    def check(
        self, script: SmtScript, *, timeout_milliseconds: int | None
    ) -> SatResult:
        self.checks.append((script.text, timeout_milliseconds))
        return self.answer


@pytest.fixture
def plug_smt_solver() -> Iterator[Callable[[SatResult], RecordingSmtSolver]]:
    """Yield a function making a recording backend the default SMT backend.

    The default solver keeps its simplifier, and is restored afterwards.
    """
    original = get_default_solver()

    def plug(answer: SatResult) -> RecordingSmtSolver:
        backend = RecordingSmtSolver(answer)
        set_default_solver(Solver(smt_solver=backend, simplifier=original.simplifier))
        return backend

    yield plug
    set_default_solver(original)
