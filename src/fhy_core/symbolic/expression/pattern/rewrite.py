"""Pattern-driven rewriting over the expression IR.

This module ships the `RewriteRule` pairing a pattern with a rewrite, the
`Rule` ABC for rules written in Python, the `apply_rewrite_rule` and
`apply_rewrite_rules` free functions, the `RewriteRuleApplier` pass, and
the errors of a rewrite walk.

The rules and the walk are backed by the Rust implementation
(``fhy_core._rs``), with the Rust core's semantics:

- A rule fires when its pattern matches, every guard allows it, and its
  rewrite returns a replacement other than the matched node itself;
  returning the node itself declines, so the next rule is tried.
- `apply_rewrite_rules` walks bottom-up once, on its own work stack, so a
  tree of any depth works. A node that occurs in several places is
  rewritten once, and a replacement is not rewritten again.
- A failing callback or a refused rebuild raises a `RewriteError` naming
  the rule.
"""

from fhy_core.utils.override import override

__all__ = [
    "FiredRule",
    "RewriteCallbackError",
    "RewriteError",
    "RewriteRebuildError",
    "RewriteRule",
    "RewriteRuleApplier",
    "Rule",
    "apply_rewrite_rule",
    "apply_rewrite_rules",
]

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.pass_infrastructure import CompilerPass, register_pass
from fhy_core.traits import FrozenMixin

from ..core import Expression


class RewriteError(RuntimeError):
    """A rewrite walk that failed, naming the rule responsible.

    The message is the core's text, such as ``rewrite rule 0 (name)
    failed``. It is a ``RuntimeError``, as the ``PassExecutionError`` the
    walk raised before is, so ``except RuntimeError`` still catches it.

    Attributes:
        rule_index: The position of the responsible rule in the rule list.
        rule_name: The name of the responsible rule, or ``None``.

    """

    rule_index: int
    rule_name: str | None

    def __init__(self, message: str, rule_index: int, rule_name: str | None) -> None:
        super().__init__(message)
        self.rule_index = rule_index
        self.rule_name = rule_name

    @override
    def __reduce__(self) -> tuple[type["RewriteError"], tuple[str, int, str | None]]:
        return (type(self), (str(self), self.rule_index, self.rule_name))


class RewriteCallbackError(RewriteError):
    """A predicate in a rule's pattern, one of its guards, or its rewrite raised.

    The callback's exception is the ``__cause__``. An exception that is
    not an ``Exception``, such as ``KeyboardInterrupt``, is never wrapped:
    it propagates from the walk unchanged.
    """


class RewriteRebuildError(RewriteError):
    """A node could not be rebuilt around its rewritten children.

    For example, a rule rewrote a piecewise case condition to a literal
    other than a Boolean. The rule named is the one that rewrote the
    refused child, and the rebuild's ``ValueError`` is the ``__cause__``.
    """


class Rule(_rs.RuleBase, ABC):
    """A rewrite tried at the root of one expression, written in Python.

    Subclasses implement `apply`, and may override `name`. A walk calls
    `apply` from Rust, once per node it tries the rule on, with the node's
    object, and reads `name` once per walk. A rule list may mix Python
    rules and `RewriteRule` objects, which are registered as virtual
    subclasses of this class.
    """

    @abstractmethod
    def apply(self, expression: Expression) -> Expression | None:
        """Return the replacement for ``expression``, or ``None`` to decline.

        Returning ``expression`` itself declines too: the walk records no
        firing and tries the next rule. Any other result than an
        `Expression` or ``None`` raises ``TypeError`` in a walk.

        Args:
            expression: The expression the rule is tried on.

        Returns:
            The replacement, or ``None``.

        """

    @property
    def name(self) -> str | None:
        """Return the rule's name, used in firings and errors, or ``None``."""
        return None


@final
class RewriteRule(_rs.RewriteRule):
    """A pattern paired with a rewrite of what it matches, optionally guarded and named.

    ``RewriteRule(pattern, rewrite, guard=None, name=None)`` builds a rule
    whose ``rewrite`` must return an `Expression`;
    ``RewriteRule.new_partial(...)`` builds one whose ``rewrite`` may return
    ``None`` to decline. ``with_guard(guard)`` and ``with_name(name)``
    return new rules.

    The rule fires on an expression when its pattern matches it at the
    root, every guard, in order, returns a true value for the match's
    `MatchBindings`, and the rewrite returns a replacement other than the
    expression itself; a replacement that is the expression itself
    declines. Any other rewrite result raises ``TypeError`` naming the
    rule. A pattern that is not a `Pattern`, a rewrite or guard that is not
    callable, or a name that is not a ``str`` raises ``TypeError``.

    Rules are immutable, and compare and hash by identity, since callables
    cannot be compared. A rule pickles only if its callables do.

    Attributes:
        pattern: The pattern.
        rewrite: The callable mapping the bindings to the replacement.
        guards: The guard callables, in order.
        name: The name, reported when the rule fires in a
            `RewriteRuleApplier`, or ``None``.

    """

    __slots__ = ()


FrozenMixin.register(RewriteRule)
Rule.register(RewriteRule)
RewriteRule._register_public_class()


@final
class FiredRule(_rs.FiredRule):
    """One firing of a rule during a rewrite walk.

    Firings compare and hash by their fields.

    Attributes:
        rule_index: The position of the rule in the rule list.
        name: The rule's name, or ``None``.

    """

    __slots__ = ()
    __match_args__ = ("rule_index", "name")


FrozenMixin.register(FiredRule)
FiredRule._register_public_class()


def apply_rewrite_rule(
    rule: "Rule | RewriteRule", expression: Expression
) -> Expression | None:
    """Try ``rule`` once at the root of ``expression``.

    The free-function form of ``rule.apply(expression)``.

    Args:
        rule: Rule to try.
        expression: Expression to rewrite at the root.

    Returns:
        The replacement when the rule fires; otherwise ``None``.

    Raises:
        TypeError: When the rewrite returns something other than an
            `Expression` (or ``None``, for a partial rule).

    Notes:
        Exceptions raised by a predicate, a guard or the rewrite propagate
        unchanged.

    """
    return rule.apply(expression)


def apply_rewrite_rules(
    expression: Expression, rules: Iterable["Rule | RewriteRule"]
) -> Expression:
    """Apply ``rules`` bottom-up over ``expression`` in a single pass.

    At each node, visited bottom-up, the rules are tried in order, and the
    first that fires replaces the node. The replacement is not re-examined
    in the same walk, and a node that occurs in several places is
    rewritten once.

    Args:
        expression: Expression tree to rewrite.
        rules: Rules in priority (first-match) order.

    Returns:
        The rewritten tree: ``expression`` itself when no rule fired;
        otherwise a tree sharing every node object the walk kept.

    Raises:
        TypeError: When ``expression`` is not an `Expression` or a rule is
            not a `Rule`.
        RewriteCallbackError: When a callback raises an ``Exception``,
            which is its ``__cause__``. Any other exception propagates
            unchanged.
        RewriteRebuildError: When a node cannot be rebuilt around its
            rewritten children.

    Notes:
        Traversal is bottom-up: a node's children are already in their
        rewritten form when its rules are tried.

    """
    return _rs.apply_rewrite_rules(expression, rules)


@register_pass(
    "fhy_core.symbolic.expression.apply_rewrite_rules",
    "Apply a sequence of rewrite rules bottom-up over an expression tree.",
)
class RewriteRuleApplier(CompilerPass[Expression, Expression]):
    """The compiler pass behind `apply_rewrite_rules`.

    A run is `apply_rewrite_rules` with the pass's rules. It changed the
    IR exactly when its output is not its input object. Each firing of a
    named rule reports the ``INFO`` diagnostic ``applied rewrite rule
    "name"``, and `fired` holds the last run's firings, or after a failed
    run the firings before the failure. A `RewriteError` fails the run
    with ``PassExecutionError``, whose ``__cause__`` it is.

    The pass is registered, so ``CompilerPass.create(name)`` builds one
    with no rules.
    """

    _rules: tuple[Rule | RewriteRule, ...]
    _fired: tuple[FiredRule, ...]

    def __init__(self, rules: Iterable[Rule | RewriteRule] = ()) -> None:
        super().__init__()
        self._rules = tuple(rules)
        self._fired = ()

    @property
    def rules(self) -> tuple[Rule | RewriteRule, ...]:
        """Return the rules, in the order they are tried."""
        return self._rules

    @property
    def fired(self) -> tuple[FiredRule, ...]:
        """Return the firings of the last run, in walk order."""
        return self._fired

    @override
    def get_noop_output(self, ir: Expression) -> Expression:
        return ir

    @override
    def run_pass(self, ir: Expression) -> Expression:
        fired: list[FiredRule] = []
        try:
            return _rs.apply_rewrite_rules(ir, self._rules, fired)
        finally:
            self._fired = tuple(fired)
            for firing in self._fired:
                message = firing._diagnostic_message()
                if message is not None:
                    self.report(DiagnosticLevel.INFO, message)

    @override
    def did_change(self, input_ir: Expression, output: Expression) -> bool:
        return output is not input_ir
