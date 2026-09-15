"""Hypothesis property tests for the `Stack` utility.

A stateful model (`RuleBasedStateMachine`) drives every public `Stack`
method against a plain Python `list` model, using the list itself as the
oracle for LIFO order, length, iteration order, and the empty-stack error
cases.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from fhy_core.utils.stack import Stack

pytestmark = pytest.mark.property


class StackStateMachine(RuleBasedStateMachine):
    """Drive a `Stack[int]` against a plain `list[int]` model.

    A fresh stack and a fresh model both start empty. Every rule performs
    the same operation on both; `length_matches_model` and
    `iteration_order_matches_model` hold as invariants after every rule.
    """

    def __init__(self) -> None:
        super().__init__()
        self._stack: Stack[int] = Stack()
        self._model: list[int] = []

    @rule(value=st.integers(min_value=-1000, max_value=1000))
    def push(self, value: int) -> None:
        """Test push appends value to both the stack and the model."""
        self._stack.push(value)
        self._model.append(value)

    @rule()
    def pop(self) -> None:
        """Test pop removes the top element, or raises IndexError when empty.

        Oracle: the model's own emptiness and its last element.
        """
        if not self._model:
            with pytest.raises(IndexError):
                self._stack.pop()
        else:
            expected = self._model.pop()
            assert self._stack.pop() == expected

    @rule()
    def peek(self) -> None:
        """Test peek returns the top element without mutating, or raises IndexError."""
        if not self._model:
            with pytest.raises(IndexError):
                self._stack.peek()
        else:
            assert self._stack.peek() == self._model[-1]

    @rule()
    def clear(self) -> None:
        """Test clear empties both the stack and the model."""
        self._stack.clear()
        self._model.clear()

    @invariant()
    def length_matches_model(self) -> None:
        """Test len(stack) always equals len(model)."""
        assert len(self._stack) == len(self._model)

    @invariant()
    def iteration_order_matches_model(self) -> None:
        """Test iterating the stack yields the model's elements in the same order."""
        assert list(self._stack) == self._model


TestStackStateMachine = StackStateMachine.TestCase
# Capped like a tree-heavy property: each example already runs a whole
# sequence of rules, so the profile's example count would multiply out.
TestStackStateMachine.settings = settings(max_examples=50)
