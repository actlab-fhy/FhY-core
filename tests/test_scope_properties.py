"""Hypothesis property tests for the lexical ``Scope`` utility.

A stateful model (``RuleBasedStateMachine``) drives every public ``Scope``
method against a ``list[dict[str, int]]`` model: index 0 is the root frame,
the last element is the innermost frame. Each rule mutates or queries the
real ``Scope`` and the model the same way, then compares results, so the
model itself is the oracle.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from fhy_core.utils import Scope

from .strategies.settings import cap_max_examples

pytestmark = pytest.mark.property

_KEY_ALPHABET = "abcd"


def find_innermost_binding(model: list[dict[str, int]], key: str) -> int | None:
    """Return the value bound to key in the innermost frame that defines it.

    Searches ``model`` from the last (innermost) frame outward, mirroring
    ``Scope.lookup``'s shadowing search. Returns ``None`` when no frame
    binds ``key`` (values are always plain ints, so ``None`` is unambiguous
    as a not-found sentinel).
    """
    for frame in reversed(model):
        if key in frame:
            return frame[key]
    return None


class ScopeStateMachine(RuleBasedStateMachine):
    """Drive a `Scope[str, int]` against a `list[dict[str, int]]` frame-stack model.

    A fresh scope and a fresh model both start with exactly one (root)
    frame. Every rule below performs the same operation on both and checks
    that the real `Scope` agrees with the model; `depth_matches_model` holds
    as an invariant after every rule.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scope: Scope[str, int] = Scope()
        self._model: list[dict[str, int]] = [{}]

    @rule()
    def push(self) -> None:
        """Test push adds a new, empty innermost frame to both scope and model."""
        self._scope.push()
        self._model.append({})

    @rule()
    def pop(self) -> None:
        """Test pop removes the innermost frame, or raises IndexError at the root.

        Oracle: the model's own frame count; the root frame (depth 1) can
        never be popped, per Scope.pop's documented contract.
        """
        if len(self._model) <= 1:
            with pytest.raises(IndexError):
                self._scope.pop()
        else:
            self._scope.pop()
            self._model.pop()

    @rule(
        key=st.sampled_from(_KEY_ALPHABET),
        value=st.integers(min_value=-1000, max_value=1000),
    )
    def define(self, key: str, value: int) -> None:
        """Test define binds key to value in the innermost frame of both."""
        self._scope.define(key, value)
        self._model[-1][key] = value

    @rule(key=st.sampled_from(_KEY_ALPHABET))
    def lookup(self, key: str) -> None:
        """Test lookup returns the innermost binding, or raises KeyError if unbound.

        Oracle: find_innermost_binding, an independent innermost-to-outermost
        search over the model.
        """
        expected = find_innermost_binding(self._model, key)
        if expected is None:
            with pytest.raises(KeyError):
                self._scope.lookup(key)
        else:
            assert self._scope.lookup(key) == expected

    @rule(key=st.sampled_from(_KEY_ALPHABET))
    def lookup_local(self, key: str) -> None:
        """Test lookup_local only sees the innermost frame, or raises KeyError.

        Oracle: direct membership check against the model's last frame.
        """
        innermost = self._model[-1]
        if key in innermost:
            assert self._scope.lookup_local(key) == innermost[key]
        else:
            with pytest.raises(KeyError):
                self._scope.lookup_local(key)

    @rule(key=st.sampled_from(_KEY_ALPHABET))
    def is_defined(self, key: str) -> None:
        """Test is_defined matches whether any model frame binds key."""
        expected = any(key in frame for frame in self._model)
        assert self._scope.is_defined(key) is expected

    @rule(key=st.sampled_from(_KEY_ALPHABET))
    def is_defined_local(self, key: str) -> None:
        """Test is_defined_local matches whether the innermost model frame binds key."""
        expected = key in self._model[-1]
        assert self._scope.is_defined_local(key) is expected

    @invariant()
    def depth_matches_model(self) -> None:
        """Test get_depth always equals the model's frame count."""
        assert self._scope.get_depth() == len(self._model)


TestScopeStateMachine = ScopeStateMachine.TestCase
# Capped like a tree-heavy property: each example already runs a whole
# sequence of rules, so the profile's example count would multiply out.
TestScopeStateMachine.settings = cap_max_examples(50)
