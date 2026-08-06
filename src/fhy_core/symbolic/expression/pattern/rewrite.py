"""Pattern-driven rewriting over the expression IR.

This module ships the `RewriteRule` value object, the
`apply_rewrite_rule` and `apply_rewrite_rules` free functions, and
the `RewriteRuleApplier` pass. Together they let callers describe
local rewrites as ``(pattern, rewrite, optional guard, optional name)``
tuples and apply rule sets bottom-up over an expression tree.
"""

from fhy_core.utils.override import override

__all__ = [
    "RewriteRule",
    "RewriteRuleApplier",
    "apply_rewrite_rule",
    "apply_rewrite_rules",
]

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import final

from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.pass_infrastructure import RewritablePass, register_pass
from fhy_core.traits import FrozenMixin

from ..core import Expression
from .core import MatchBindings, Pattern


@final
@dataclass(frozen=True)
class RewriteRule(FrozenMixin):
    """A pattern paired with a rewrite that consumes the bindings.

    Attributes:
        pattern: The pattern to match.
        rewrite: Callable mapping the successful match's
            `MatchBindings` to a replacement `Expression`.
        guard: Optional callable mapping `MatchBindings` to ``bool``.
            When supplied, the rule fires only when
            ``guard(bindings)`` returns ``True``. ``None`` disables
            the guard.
        name: Optional human-readable rule name. When supplied and
            the rule fires inside a `RewriteRuleApplier`, an ``INFO``
            diagnostic naming the rule is emitted. ``None`` indicates
            an unnamed rule and suppresses the diagnostic.

    """

    pattern: Pattern
    rewrite: Callable[[MatchBindings], Expression]
    guard: Callable[[MatchBindings], bool] | None = None
    name: str | None = None


def apply_rewrite_rule(rule: RewriteRule, expression: Expression) -> Expression | None:
    """Attempt ``rule`` at the root of ``expression`` once.

    Args:
        rule: Rule to attempt.
        expression: Expression to rewrite at the root.

    Returns:
        The rewritten expression when the pattern matches and the
        guard (if any) returns ``True``; otherwise ``None``.

    Raises:
        TypeError: When ``rule.rewrite`` returns a non-`Expression`.
            The error message names the rule so the offending callback
            is identifiable in a stack trace.

    Notes:
        Exceptions raised by ``rule.guard`` or ``rule.rewrite``
        propagate to the caller unchanged.

    """
    bindings = rule.pattern.match(expression)
    if bindings is None:
        return None
    if rule.guard is not None and not rule.guard(bindings):
        return None
    result = rule.rewrite(bindings)
    if not isinstance(result, Expression):
        rule_label = rule.name if rule.name is not None else "<unnamed>"
        raise TypeError(
            f"RewriteRule {rule_label!r} rewrite callable must return an "
            f"Expression; got {type(result).__name__}."
        )
    return result


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
        The rewritten expression tree. Identity is preserved iff
        zero rules fire across the entire walk: even one fire at
        any depth causes the entire ancestor chain to be rebuilt
        as new objects.

    Raises:
        PassValidationError: When the underlying ``CompilerPass.execute``
            input or output validation rejects ``expression``.
        PassExecutionError: When a rule's ``guard`` or ``rewrite``
            callable raises any exception other than
            ``PassValidationError`` or ``PassExecutionError``. The
            original exception is attached as ``__cause__``. The
            two pass-framework exceptions are re-raised unchanged.
            Wrapping happens in ``CompilerPass.execute``.

    Notes:
        Traversal is bottom-up: a node's children are already in
        their post-rewrite form when its rules are tried. A rule
        whose pattern matches the *pre*-rewrite shape of a child
        will not fire at the parent level once that child has been
        rewritten.

    """
    return RewriteRuleApplier(rules)(expression)


@register_pass(
    "fhy_core.symbolic.expression.apply_rewrite_rules",
    "Apply a sequence of rewrite rules bottom-up over an expression tree.",
)
class RewriteRuleApplier(RewritablePass[Expression]):
    """`RewritablePass` driver behind `apply_rewrite_rules`.

    The class exists so callers running rule application inside a
    `PassManager` can collect diagnostics and ``PassResult``
    metadata. Direct use of `apply_rewrite_rules` is the more
    common path; this class is the underlying pass type and is
    registered with the global pass registry.
    """

    _rules: tuple[RewriteRule, ...]

    def __init__(self, rules: Sequence[RewriteRule]) -> None:
        super().__init__()
        self._rules = tuple(rules)

    @property
    def rules(self) -> tuple[RewriteRule, ...]:
        """Return the rule sequence this applier was constructed with."""
        return self._rules

    @override
    def visit_unknown(self, node: Expression) -> Expression | None:
        """Apply rules in order to ``node`` and return the first match.

        Returns ``None`` when no rule fires, signaling the framework
        to retain the node (possibly with already-rewritten children).
        Rewrites are *not* re-examined within the same walk. When a
        named rule fires, an ``INFO`` diagnostic is emitted naming
        the rule.
        """
        for rule in self._rules:
            rewritten = apply_rewrite_rule(rule, node)
            if rewritten is not None:
                if rule.name is not None:
                    self.report(
                        DiagnosticLevel.INFO,
                        f"Applied rewrite rule {rule.name!r}.",
                    )
                return rewritten
        return None
