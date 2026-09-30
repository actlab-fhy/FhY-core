"""Pattern matching and rule-driven rewriting over the expression IR.

The sub-package exposes two layers:

- :mod:`.core` --- the `Capture` handle, the `Pattern` hierarchy, the
  `MatchBindings` a match returns, and the `match_pattern` /
  `does_pattern_match` free functions. These suffice for read-only
  structural queries over an expression tree.
- :mod:`.rewrite` --- the `RewriteRule` and the `Rule` ABC, the
  `apply_rewrite_rule` / `apply_rewrite_rules` free functions, the
  `RewriteRuleApplier` pass with its `FiredRule` records, and the
  `RewriteError` family. These build on the matching layer to express
  local rewrites and walk an expression tree bottom-up.

Both layers are backed by the Rust implementation, with its semantics;
see the modules' docstrings.

All public names are re-exported from this `__init__` so that
callers import from ``fhy_core.symbolic.expression.pattern`` without
descending into the sub-modules.
"""

__all__ = [
    "AlternativesPattern",
    "BinaryExpressionPattern",
    "CallExpressionPattern",
    "Capture",
    "CapturePattern",
    "FiredRule",
    "IdentifierPattern",
    "LiteralPattern",
    "LogicalExpressionPattern",
    "MatchBindings",
    "Pattern",
    "PiecewiseExpressionPattern",
    "PredicatePattern",
    "RewriteCallbackError",
    "RewriteError",
    "RewriteRebuildError",
    "RewriteRule",
    "RewriteRuleApplier",
    "Rule",
    "UnaryExpressionPattern",
    "WildcardPattern",
    "apply_rewrite_rule",
    "apply_rewrite_rules",
    "does_pattern_match",
    "match_pattern",
]

from .core import (
    AlternativesPattern,
    BinaryExpressionPattern,
    CallExpressionPattern,
    Capture,
    CapturePattern,
    IdentifierPattern,
    LiteralPattern,
    LogicalExpressionPattern,
    MatchBindings,
    Pattern,
    PiecewiseExpressionPattern,
    PredicatePattern,
    UnaryExpressionPattern,
    WildcardPattern,
    does_pattern_match,
    match_pattern,
)
from .rewrite import (
    FiredRule,
    RewriteCallbackError,
    RewriteError,
    RewriteRebuildError,
    RewriteRule,
    RewriteRuleApplier,
    Rule,
    apply_rewrite_rule,
    apply_rewrite_rules,
)
