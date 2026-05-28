"""Pattern-driven rewriting over the expression IR.

This module ships the `RewriteRule` value object, the
`apply_rewrite_rule` and `apply_rewrite_rules` free functions, and
the `RewriteRuleApplier` pass. Together they let callers describe
local rewrites as ``(pattern, rewrite, optional guard, optional name)``
tuples and apply rule sets bottom-up over an expression tree.
"""

__all__ = [
    "RewriteRule",
    "RewriteRuleApplier",
    "apply_rewrite_rule",
    "apply_rewrite_rules",
]

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import final

from fhy_core.pass_infrastructure import RewritablePass, register_pass

from ..core import Expression
from .core import MatchBindings, Pattern


@final
@dataclass(frozen=True)
class RewriteRule:
    """A pattern paired with a rewrite that consumes the bindings.

    Attributes:
        pattern: The pattern to match.
        rewrite: Callable mapping the successful match's
            `MatchBindings` to a replacement `Expression`.
        guard: Optional callable mapping `MatchBindings` to ``bool``.
            When supplied, the rule fires only when
            ``guard(bindings)`` returns ``True``. ``None`` disables
            the guard.
        name: Optional human-readable rule name for diagnostics and
            pass logs. The empty string indicates an unnamed rule.

    """

    pattern: Pattern
    rewrite: Callable[[MatchBindings], Expression]
    guard: Callable[[MatchBindings], bool] | None = None
    name: str = ""


def apply_rewrite_rule(rule: RewriteRule, expression: Expression) -> Expression | None:
    """Attempt ``rule`` at the root of ``expression`` once.

    Args:
        rule: Rule to attempt.
        expression: Expression to rewrite at the root.

    Returns:
        The rewritten expression when the pattern matches and the
        guard (if any) returns ``True``; otherwise ``None``.

    Raises:
        Exception: Re-raises whatever the rule's ``guard`` or
            ``rewrite`` callable raised. The framework does not
            swallow caller-supplied exceptions.

    """
    bindings = rule.pattern.match(expression)
    if bindings is None:
        return None
    elif rule.guard is not None and not rule.guard(bindings):
        return None
    else:
        return rule.rewrite(bindings)


def apply_rewrite_rules(
    expression: Expression, rules: Sequence[RewriteRule]
) -> Expression:
    """Apply ``rules`` bottom-up over ``expression`` in a single pass.

    At each node (visited bottom-up), the rules are tried in order;
    the first rule whose pattern matches and whose guard returns
    ``True`` (or has no guard) fires. The resulting expression
    replaces the node and is *not* re-examined within the same walk.

    Args:
        expression: Expression tree to rewrite.
        rules: Rule sequence in priority (first-match) order.

    Returns:
        The rewritten expression tree. Identity-preserving: when
        no rule fires anywhere, the input ``expression`` is returned
        unchanged.

    Raises:
        PassExecutionError: When a rule's ``guard`` or ``rewrite``
            callable raises something other than
            ``PassExecutionError`` / ``PassValidationError``. The
            original is attached as ``__cause__``, matching the
            standard ``RewritablePass`` behavior.

    """
    return RewriteRuleApplier(rules)(expression)


@register_pass(
    "fhy_core.expression.apply_rewrite_rules",
    "Apply a sequence of rewrite rules bottom-up over an expression tree.",
)
class RewriteRuleApplier(RewritablePass[Expression]):
    """`RewritablePass` driver behind `apply_rewrite_rules`.

    The class exists so callers running rule application inside a
    `PassManager` can collect diagnostics and ``PassResult``
    metadata. Direct use of `apply_rewrite_rules` is the more
    common path; this class is the underlying pass type and is
    registered with the global pass registry.

    Attributes:
        rules: Rule sequence in priority (first-match) order.

    """

    _rules: tuple[RewriteRule, ...]

    def __init__(self, rules: Sequence[RewriteRule]) -> None:
        super().__init__()
        self._rules = tuple(rules)

    @property
    def rules(self) -> tuple[RewriteRule, ...]:
        """Return the rule sequence this applier was constructed with."""
        return self._rules

    def visit_unknown(self, node: Expression) -> Expression | None:
        for rule in self._rules:
            rewritten = apply_rewrite_rule(rule, node)
            if rewritten is not None:
                return rewritten
        return None
