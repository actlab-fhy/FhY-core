"""Pattern matching and rule-driven rewriting over the expression IR.

The sub-package exposes two layers:

- :mod:`.core` --- the `Pattern` hierarchy, `MatchBindings` value
  object, and the `match_pattern` / `does_pattern_match` free
  functions. These suffice for read-only structural queries over an
  expression tree.
- :mod:`.rewrite` --- the `RewriteRule` value object, the
  `apply_rewrite_rule` / `apply_rewrite_rules` free functions, and
  the `RewriteRuleApplier` pass. These build on the matching layer
  to express local rewrites and walk an expression tree bottom-up.

All public names are re-exported from this `__init__` so that
callers import from ``fhy_core.expression.pattern`` without
descending into the sub-modules.
"""

__all__ = [
    "AlternativesPattern",
    "BinaryExpressionPattern",
    "CallExpressionPattern",
    "CapturePattern",
    "IdentifierPattern",
    "LiteralPattern",
    "MatchBindings",
    "Pattern",
    "PiecewiseExpressionPattern",
    "PredicatePattern",
    "RewriteRule",
    "RewriteRuleApplier",
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
    CapturePattern,
    IdentifierPattern,
    LiteralPattern,
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
    RewriteRule,
    RewriteRuleApplier,
    apply_rewrite_rule,
    apply_rewrite_rules,
)
