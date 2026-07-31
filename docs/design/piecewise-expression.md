# Piecewise Expression (replaces TernaryExpression)

Design doc for FhY-core release feature 1. Target: `src/fhy_core/expression/` on branch `dev-new-feats` @ `ad9a311`. Doc location when committed: `docs/design/piecewise-expression.md`.

## Summary

Replace the 3-operand `TernaryExpression` with a mathematical n-case `PiecewiseExpression`: an ordered sequence of `(condition, value)` cases with first-match-wins semantics and a mandatory `otherwise` fallback that makes the expression a total function. `TernaryExpression`, `ternary()`, and `Expression.ternary` are deleted outright — no aliases, no deprecation shims. Every consumer (type checker, all backend lowerings, pretty-printer, pattern matching, serialization, builtins, exports, tests) is rewired to the new node. The old ternary is exactly the one-case piecewise: `ternary(c, t, f) ≡ piecewise((c, t), otherwise=f)`. The `CallExpression` "maybe" is **explicitly deferred** (see Non-goals).

## Motivation

`TernaryExpression` is a programming-language artifact (`cond ? a : b`), not a mathematical object. Multi-way selections today must be expressed as right-nested ternary chains — the builtin `sign` is `ternary(x > 0, 1, ternary(x < 0, -1, 0))`, and the SymPy lifter (`sympy.py:645-656`) reconstructs *every* multi-branch `sympy.Piecewise` as a nested ternary tree, losing the flat case structure SymPy natively carries. A first-class piecewise node:

- matches the mathematical notation users write (case analysis with an "otherwise" row);
- maps 1:1 onto `sympy.Piecewise` in both directions (no nesting/denesting loss);
- flattens builtin bodies (`sign` becomes one node with two cases);
- generalizes the numpy/z3 lowerings mechanically (where-chain / If-chain).

Deleting ternary rather than keeping both avoids two overlapping conditional nodes that every pass would have to handle, per the project's no-backward-compat rule.

## Key structural decision: parallel tuples, not a pairs field

The node stores `conditions: tuple[Expression, ...]`, `values: tuple[Expression, ...]`, `otherwise: Expression` — **not** `cases: tuple[tuple[Expression, Expression], ...]`. Rationale (verified against the code):

1. **Serialization derivation.** The codec-inference engine rejects fixed-length tuple element types: `_sequence_element_type` (`serialization.py:1197-1208`) only derives the homogeneous `tuple[T, ...]` form and raises `_CodecInferenceError` for `tuple[Expression, Expression]`. A pairs field would force either a hand-written `serialize_data_to_dict`/`deserialize_data_from_dict` (the `LiteralExpression` escape hatch, undesirable for a plain structural node) or a new `Serializable` helper class for the pair (extra registered type, extra equivalence plumbing). Parallel tuples derive fully automatically, exactly like `CallExpression.arguments` already does.
2. **Precedent.** `CallExpression` proves `tuple[Expression, ...]` fields work with derived equivalence, derived serialization, and alpha-equivalence.
3. **Ergonomics preserved.** The `piecewise(...)` builder and `Expression.piecewise` take `(condition, value)` **pairs** — the public construction surface reads mathematically. A `get_cases()` accessor returns the zipped pairs for consumers that want case-shaped iteration.

## Public interface (interface stubs)

All in `src/fhy_core/expression/core.py` unless noted. Docstrings abbreviated to one-liners here; the implementation carries full Google-style docstrings mirroring the existing `ternary`/`call` documentation style (coercion rules paragraph, Args/Returns/Raises).

```python
@register_serializable(type_id="piecewise_expression")
@dataclass(frozen=True, eq=False)
class PiecewiseExpression(Expression, HasOperands[Expression]):
    """Mathematical piecewise expression: ordered first-match cases with a total fallback.

    Attributes:
        conditions: Scalar boolean case conditions, in evaluation order.
        values: Case result values, positionally paired with ``conditions``.
        otherwise: Result when no condition holds; makes the function total.

    Raises:
        ValueError: If ``conditions`` and ``values`` differ in length, or if
            no case is supplied.
    """

    conditions: tuple[Expression, ...]
    values: tuple[Expression, ...]
    otherwise: Expression

    def __post_init__(self) -> None: ...  # explicit raises, no assert

    def get_cases(self) -> tuple[tuple[Expression, Expression], ...]:
        """Return the ``(condition, value)`` case pairs in evaluation order."""

    @override
    def get_operands(self) -> tuple[Expression, ...]:
        """Return interleaved case children then ``otherwise`` (see ordering below)."""

    @override
    def get_visit_children(self) -> tuple[Expression, ...]:
        """Return ``(c1, v1, c2, v2, ..., cn, vn, otherwise)``."""

    @override
    def rebuild_with_visit_children(
        self, new_children: Sequence[Expression]
    ) -> "PiecewiseExpression":
        """Rebuild from a flat child sequence produced by ``get_visit_children``."""


def piecewise(
    *cases: tuple[
        "Expression | Identifier | LiteralType",
        "Expression | Identifier | LiteralType",
    ],
    otherwise: "Expression | Identifier | LiteralType",
) -> "PiecewiseExpression":
    """Build a ``PiecewiseExpression`` from ``(condition, value)`` case pairs.

    Each element of each pair, and ``otherwise``, is coerced with the same
    rules as the operator dunders (Expression passthrough, Identifier ->
    IdentifierExpression, LiteralType -> LiteralExpression).

    Raises:
        ValueError: If no case is supplied, if a case is not a 2-tuple, or
            if an operand has an unsupported type.
    """


class Expression:  # existing class; new static method replacing Expression.ternary
    @staticmethod
    def piecewise(
        *cases: tuple[
            "Expression | Identifier | LiteralType",
            "Expression | Identifier | LiteralType",
        ],
        otherwise: "Expression | Identifier | LiteralType",
    ) -> "PiecewiseExpression":
        """Build a ``PiecewiseExpression``; delegates to the free function."""
```

`otherwise` follows `*cases`, so Python makes it keyword-only and required — a missing `otherwise` is a `TypeError` at the call site, guaranteeing totality syntactically.

In `src/fhy_core/expression/pattern/core.py` (replacing `TernaryExpressionPattern`):

```python
@final
@dataclass(frozen=True)
class PiecewiseExpressionPattern(Pattern):
    """Match a ``PiecewiseExpression``.

    Attributes:
        cases: ``(condition_pattern, value_pattern)`` pairs matched
            position-wise against the expression's cases; the case count
            must match exactly. ``None`` means "any cases."
        otherwise: Pattern the ``otherwise`` expression must match.
    """

    cases: tuple[tuple[Pattern, Pattern], ...] | None
    otherwise: Pattern

    @override
    def match_under(
        self, expression: Expression, bindings: MatchBindings
    ) -> MatchBindings | None:
        """Thread bindings condition-then-value per case in order, then otherwise."""
```

(`Pattern` subclasses are not `Serializable`, so the nested-pair tuple is unproblematic here.) The `None`-wildcard idiom for `cases` mirrors `CallExpressionPattern.arguments`; `otherwise` stays required — pass `WildcardPattern()` to ignore it.

New type-checker visitor (public pass surface, `passes/type_checker.py`):

```python
def visit_piecewise_expression(
    self, piecewise_expression: PiecewiseExpression
) -> tuple[Type, TypeQualifier]:
    """Infer and check the type of the piecewise expression."""
```

## Semantics

**Evaluation model.** Cases are ordered; the expression denotes the value of the first case whose condition holds, else `otherwise`. Overlapping conditions are legal — first match wins. Duplicate or unreachable cases are not detected or normalized (non-goal). The IR imposes no lazy-evaluation guarantee (same stance as ternary's docstring); backends document their own strictness.

**Edge cases, spelled out:**
- Zero cases → `ValueError` at construction (both the node's `__post_init__` and the builder). A piecewise that is only `otherwise` is meaningless; write the value directly.
- `len(conditions) != len(values)` → `ValueError` in `__post_init__`.
- A builder case that is not a 2-tuple → `ValueError` with a message naming the offending position.
- One case ≡ old ternary, and all typing rules below degenerate exactly to the old ternary rules.
- Non-boolean condition, or a non-`NumericalType` (e.g. `IndexType`) case value / otherwise → framed type error at type-check time, **not** at construction (same construction-vs-checking split as ternary and call).
- Missing `otherwise` → `TypeError` from Python's keyword-only enforcement.
- Nested piecewise anywhere (in conditions, values, otherwise) is legal.
- Structural/alpha equivalence, free identifiers, and substitution all derive generically from `get_visit_children`/`rebuild_with_visit_children` and the dataclass field schema — no overrides needed beyond the three methods above.

**Child ordering.** `get_visit_children()` and `get_operands()` return the interleaved sequence `(c1, v1, ..., cn, vn, otherwise)`. Interleaving keeps each case's condition adjacent to its value in traversal order (readable debug traces, matches the semantic reading order and SymPy's pair layout). `rebuild_with_visit_children` recovers the fields as `flat = tuple(new_children); conditions = flat[:-1][0::2]; values = flat[:-1][1::2]; otherwise = flat[-1]`, raising `ValueError` (not assert) if the child count is not odd and ≥ 3.

**Type checking** (`_infer_piecewise_expression`, generalizing `_infer_ternary_expression` at `type_checker.py:859-917`, which is deleted):
- Each condition is inferred against `_BOOLEAN_NUMERICAL_TYPE` and its value type must satisfy `_is_boolean_numerical_type`, else a framed type error: `"piecewise case {i} condition must be boolean, but got {t}"` (zero-based index).
- Branch expected-type propagation: helper `_ternary_branch_expected_type` is renamed `_piecewise_branch_expected_type`, logic unchanged — a scalar `NumericalType` expected type propagates into every case value **and** `otherwise`; anything else propagates `None`.
- Every case value and `otherwise` must synthesize `NumericalType`, else `"piecewise case values and otherwise must all be scalar numerical types, but got {t} for {which}"` where `{which}` is `"case {i}"` or `"otherwise"`.
- Result core data type = left fold of `promote_primitive_data_types` across all case values then `otherwise`.
- Result qualifier = left fold of `promote_type_qualifiers` across all condition qualifiers, all value qualifiers, and the `otherwise` qualifier.
- Wired into both dispatch paths: `visit_piecewise_expression` (bare-visit path, replacing `visit_ternary_expression` at `:509-513`) and a `case PiecewiseExpression():` arm in the `_infer` match (replacing the `TernaryExpression` arm at `:541-542`). Method-name dispatch (`visit_piecewise_expression` from CamelCase→snake_case) needs no infrastructure change.
- `body_type_checker.py`: **no change** — it delegates to the generic checker; piecewise-bodied builtins check exactly as ternary-bodied ones did.

## Backend translations (precise)

- **SymPy lowering** (`passes/sympy.py`, replaces `visit_ternary_expression` at `:216-224`): `visit_piecewise_expression` emits `sympy.Piecewise((v1, c1), ..., (vn, cn), (otherwise, sympy.true), evaluate=False)` — note SymPy's pair order is `(expr, cond)`, reversed from ours; children are visited in case order, then `otherwise`.
- **SymPy lifting** (`_convert_piecewise` at `:640-656`): rewritten to produce a single flat `PiecewiseExpression`; the recursive `_convert_piecewise_branches` helper is **deleted**. Preserved behaviors: empty `Piecewise` → `ValueError`; single-branch `Piecewise` degenerates to just the converted value (no piecewise node). For ≥ 2 branches: `branches[:-1]` become cases (each condition and value converted), and the **last branch's value** becomes `otherwise` with its condition dropped — exactly the current lifter's treatment of the final branch (see Open questions). Net improvement: a multi-branch `sympy.Piecewise` now lifts to one flat node instead of a right-nested ternary chain; `simplify_expression` outputs change shape accordingly.
- **NumPy** (`passes/numpy.py`, replaces `visit_ternary_expression` at `:256-268`): `visit_piecewise_expression` visits all conditions and values in case order, then `otherwise`, then right-folds `numpy.where`: `result = otherwise_array; for c, v in reversed(cases): result = np.where(c, v, result)`. The outermost `where` is the first case, so first-match-wins holds. **`np.select` was considered and rejected**: (a) the where-chain is bit-for-bit continuous with the nested-ternary lowering it replaces; (b) `np.select`'s `default` parameter is documented scalar-only, but `otherwise` is an arbitrary expression that may evaluate to an array; (c) under NEP-50 promotion the pairwise `result_type` fold of the where-chain equals `np.select`'s global promotion anyway. The "not lazy" caveat is preserved and generalized in the module docstring (`numpy.py:30-33`) and visitor docstring: **all** case values and `otherwise` are evaluated for every element; a domain error in an unselected case still emits its `nan`/`inf` and warning. Prose mentions of "ternary" at `numpy.py:30-33, 351, 359` are updated.
- **Z3** (`passes/z3.py`, replaces `visit_ternary_expression` at `:146-152`): `visit_piecewise_expression` visits conditions/values in case order then `otherwise` (deterministic left-to-right for the identifier cache side effects), then right-folds `z3.If`: `result = otherwise_z3; for c, v in reversed(cases): result = z3.If(c, v, result)`. Z3 has no n-ary conditional; nested `If` is the canonical encoding and is first-match-wins by construction.
- **Evaluator / inliner** (`passes/evaluate.py`, `passes/inline.py`): **no changes** — neither has a ternary visitor today; the `RewritablePass` `visit_unknown` default preserves the node and recurses into children, which works unchanged for piecewise. Constant-folding piecewise is a non-goal (ternary was never folded either).
- **Pretty-printer** (`pprint.py`, replaces `visit_ternary_expression` at `:80-86`): symbolic form uses mathematical case notation `{v1 if c1; v2 if c2; otherwise_v otherwise}` (e.g. abs body: `{x if (x >= 0.0); (-x) otherwise}`); functional form is the flat s-expression `(piecewise c1 v1 c2 v2 otherwise_v)` — odd argument count, last element is always `otherwise`, matching the minimal style of `(ternary c t f)`. `show_id` propagates into all children as before.

## Builtins (`builtins.py`)

Five bodies rewritten (the `ternary` import at `:50` becomes `piecewise`; module-docstring prose at `:21` updated):
- `max`: `piecewise((a_expr > b, a), otherwise=b)`
- `min`: `piecewise((a_expr < b, a), otherwise=b)`
- `abs`: `piecewise((x_expr >= 0.0, x), otherwise=-x_expr)`
- `sign`: `piecewise((x_expr > 0.0, 1), (x_expr < 0.0, -1), otherwise=0)` — the nested ternary flattens to one two-case node.
- `leaky_relu`: `piecewise((x_expr > 0.0, x), otherwise=x_expr * slope)`

Registered sorts/parameters are untouched; registration-time body checking passes through the generalized checker unchanged. Downstream builtins (`clamp`, `relu`, `sigmoid`, `silu`, `gelu`, boolean algebra) are call/operator-bodied and unaffected structurally, but inlining `max`/`min`/`sign`/`abs` now yields piecewise trees — the backend tests that observe inlined shapes change accordingly.

## Files created / modified / deleted

**Created:** `docs/design/piecewise-expression.md` (this document). No new source modules — the node lives in `core.py` beside its siblings.

**Modified (src):**
- `src/fhy_core/expression/core.py` — delete `TernaryExpression` (`:787-821`), `ternary()` (`:212-240`), `Expression.ternary` (`:475-499`); add `PiecewiseExpression`, `piecewise()`, `Expression.piecewise`; update `__all__` (`:17, :26`).
- `src/fhy_core/expression/__init__.py` — `__all__` and imports: remove `TernaryExpression`, `TernaryExpressionPattern`, `ternary`; add `PiecewiseExpression`, `PiecewiseExpressionPattern`, `piecewise`.
- `src/fhy_core/expression/builtins.py` — five bodies + import + docstring prose (`:21, :50, :177, :189, :201, :213, :262`).
- `src/fhy_core/expression/pprint.py` — import (`:17`), visitor (`:80-86`).
- `src/fhy_core/expression/passes/type_checker.py` — import (`:57`), visitor (`:509-513`), `_infer` arm (`:541-542`), `_infer_ternary_expression` → `_infer_piecewise_expression` (`:859-909`), `_ternary_branch_expected_type` → `_piecewise_branch_expected_type` (`:911-917`), error messages.
- `src/fhy_core/expression/passes/numpy.py` — import (`:68`), visitor (`:256-268`), docstring prose (`:30-33, :351, :359`).
- `src/fhy_core/expression/passes/sympy.py` — import (`:28`), lowering visitor (`:216-224`), lifting (`:640-656`, delete `_convert_piecewise_branches`).
- `src/fhy_core/expression/passes/z3.py` — import (`:28`), visitor (`:146-152`).
- `src/fhy_core/expression/pattern/core.py` — delete `TernaryExpressionPattern` (`:406-435`), add `PiecewiseExpressionPattern`; `__all__` (`:24`), import (`:49`).
- `src/fhy_core/expression/pattern/__init__.py` — export swap (`:31, :50`).

**Unchanged (verified):** `passes/evaluate.py`, `passes/inline.py`, `passes/body_type_checker.py`, `passes/native_lowering.py`, `errors.py`, `sort.py`, `registry/*`, `pattern/rewrite.py` (its `visit_unknown` rule application is node-agnostic), `src/fhy_core/__init__.py` (namespace-only exposure), `src/fhy_core/serialization.py`, and — verified by grep — **nothing outside `expression/`**: `constraint.py`, `param/`, and all other subsystems have zero ternary references, keeping this diff disjoint from features 2 and 3.

**Modified (tests):** `tests/expression/test_ternary_and_call.py` → renamed `test_piecewise_and_call.py` (piecewise construction/builder/protocol coverage replacing `:21-114`; call half `:122-237` kept); `test_pprint.py`; `test_functions_stories.py`; `passes/test_builtins.py`, `test_evaluator.py`, `test_inline_pass.py`, `test_numpy_evaluator.py`, `test_sympy_pass.py`, `test_type_checker_booleans.py`, `test_type_checker_sorts.py`, `test_z3_pass.py`; `pattern/test_core.py`, `pattern/test_rewrite.py`. (`passes/test_type_checker.py` and `test_core.py` contain no ternary references — verified — but `test_core.py` gains piecewise serialization round-trip coverage.)

**Deleted:** nothing at file granularity; the deletions are the symbols above.

## Serialization impact

- New registration: `@register_serializable(type_id="piecewise_expression")` — globally unique, snake_case, explicitly pinned (stable across the later `fhy_core.symbolic` module move). All three fields (`tuple[Expression, ...]` ×2, `Expression`) derive automatically via `_SequenceFieldCodec`/`_SerializableFieldCodec`; **no hand-written serialize methods**.
- Wire shape: `{"__type__": "piecewise_expression", "__data__": {"conditions": [...], "values": [...], "otherwise": {...}}}`.
- The `"ternary_expression"` type_id is removed with its class. Any previously serialized blob containing it raises `UnknownTypeIdError` on deserialize. **No `alias=True` bridge and no migration shim** — deliberate, per the no-backward-compat rule; the release notes for the merged release should state that persisted expression blobs containing ternary nodes must be regenerated.
- Malformed-payload behavior (wrong tuple lengths) surfaces at construction via `__post_init__`'s `ValueError` after field decoding — same layered validation as `CallExpression`'s non-empty-name check.

## Type-checking (mypy --strict) considerations

- `@override` (from `fhy_core.utils.override`) on `get_operands`, `get_visit_children`, `rebuild_with_visit_children`, `match_under`, and every replaced pass visitor — `explicit-override` is enforced repo-wide.
- `rebuild_with_visit_children` receives `Sequence[Expression]`; materialize `tuple(new_children)` before stride-slicing so slices type as `tuple[Expression, ...]` cleanly.
- `get_operands` returns variadic `tuple[Expression, ...]` (unlike ternary's fixed 3-tuple annotation) — still satisfies `HasOperands[Expression]`.
- The builder's `*cases: tuple[coercible, coercible]` variadic types each positional argument as a pair — mypy statically rejects non-pair arguments; the runtime `ValueError` covers untyped callers.
- The `_infer` `match` gains `case PiecewiseExpression():` — exhaustiveness still falls through to the existing `NotImplementedError` arm.
- `sympy.py`/`z3.py` keep their existing `# type: ignore` import pragmas; fold loops introduce no new `Any` beyond what those modules already carry. The numpy visitor stays `-> Any` like its siblings.
- `ty` remains advisory; no known ty-specific hazards in this shape.

## Test plan (outline — tests written in phase 2)

- **Unit — node & builder** (`test_piecewise_and_call.py`): construction happy path; frozen-mutation rejection; `HasOperands` protocol conformance; `get_cases`/`get_operands`/`get_visit_children` ordering; `rebuild_with_visit_children` round-trip and bad-child-count `ValueError`; `__post_init__` `ValueError`s (length mismatch, zero cases); builder coercion table (Expression passthrough, `mock_identifier`-based Identifier wrap, literal wrap, unsupported type `ValueError`, non-2-tuple case `ValueError`); missing-`otherwise` `TypeError`; `Expression.piecewise` delegation; single-case ≡ former ternary structural shape.
- **Unit — serialization** (`test_core.py`, using `tests/serialization/conftest.py` round-trip helpers): dict/all-formats round trip with structural equivalence; nested piecewise; adversarial: a hand-built `"ternary_expression"` blob raises `UnknownTypeIdError`; malformed length-mismatched payload raises `ValueError`.
- **Unit — type checker** (`test_type_checker_booleans.py` / `_sorts.py`): boolean-condition requirement per case (indexed error message); multi-case primitive promotion fold (int case + float otherwise → float64, etc.); qualifier fold across conditions+values+otherwise; expected-type propagation into all branches including otherwise; `IndexType` value rejection; single-case parity with all former ternary rules.
- **Unit — backends**: sympy lowering emits `Piecewise` with trailing `(otherwise, true)` pair and `evaluate=False`; sympy lift of multi-branch `Piecewise` → one flat node; single-branch degeneration; lower→lift structural round trip. numpy where-chain first-match-wins under overlapping conditions; non-laziness warning from unselected case; scalar-vs-array otherwise. z3 nested-`If` equivalence checks (marker `z3`), e.g. `does_expression_imply` parity between a piecewise and its hand-nested `If` encoding. evaluate/inline pass-through preservation with child recursion.
- **Unit — pattern** (`pattern/test_core.py`, `test_rewrite.py`): exact case matching with binding threading order; `cases=None` wildcard; case-count mismatch fail-fast; otherwise matching; rewrite rules firing on piecewise nodes via `RewriteRuleApplier`.
- **Unit — pprint**: symbolic and functional forms, multi-case, `show_id` propagation, nesting.
- **Integration** (`test_builtins.py`, `test_functions_stories.py`, `test_native_stories.py`): builtin bodies are piecewise (flat 2-case `sign`); inline→typecheck→evaluate/numpy/sympy/z3 stories for `max`/`abs`/`sign`/`leaky_relu`/clamp compositions.
- **Property** (marker `property`): serialize/deserialize identity under structural equivalence for random piecewise trees; numpy evaluation equals a pointwise Python first-match fold on random inputs; sympy round-trip preserves case count.
- **Adversarial/chaos**: 100+ cases through z3/numpy/pprint; duplicate identical conditions; piecewise-in-condition nesting; all-cases-false paths hitting otherwise.

## Non-goals

- **CallExpression reframing: explicitly deferred.** Recommendation is to ship piecewise only. `CallExpression` already *is* mathematical function application — a registry-resolved name applied to an ordered argument tuple; any recast would be a pure rename (e.g. `ApplyExpression`) with an enormous blast radius (registry api/entries/storage, all eight passes, builtins, pattern, serialization `type_id` churn, ~5,000 lines of tests) and zero semantic gain. It would also maximize conflict with features 2/3 (constraint and param both build on the expression API in parallel worktrees) and is better bundled with the post-merge `fhy_core.symbolic` reorganization if ever wanted. No renaming, no partial reframing, in this feature.
- No constant folding of piecewise in `evaluate_expression` (parity with ternary, which was never folded).
- No case normalization: no overlap detection, unreachable-case pruning, duplicate merging, or condition simplification.
- No partial piecewise (`otherwise` is always required); no `sympy.Piecewise`-style implicit-`nan` semantics.
- No lazy/short-circuit branch evaluation in the numpy backend.
- No serialization alias or migration path for `"ternary_expression"` blobs.
- No changes to `constraint.py`, `param/`, or any subsystem outside `src/fhy_core/expression/` and `tests/expression/`.

## Open questions

1. **SymPy lift of a non-total `Piecewise`:** the current lifter silently treats the last branch's value as the fallback even when its condition is not `True` (`sympy.py:649-651` ignores the final condition). This design preserves that behavior exactly for continuity. Alternative: raise `ValueError` when the final condition is not `sympy.true` (stricter, but a behavior change for user-supplied SymPy input). Confirm preference before test writing.
2. **Symbolic pprint syntax:** `{v1 if c1; v2 if c2; v3 otherwise}` is proposed; confirm against any downstream tooling that parses the symbolic form (none known in-repo).
3. **Test file naming:** `test_piecewise_and_call.py` keeps the two-node grouping; alternatively split into `test_piecewise.py` + moving call coverage elsewhere. Cosmetic; default is the rename.

# APPENDIX: files_to_modify
- docs/design/piecewise-expression.md
- src/fhy_core/expression/core.py
- src/fhy_core/expression/__init__.py
- src/fhy_core/expression/builtins.py
- src/fhy_core/expression/pprint.py
- src/fhy_core/expression/passes/type_checker.py
- src/fhy_core/expression/passes/numpy.py
- src/fhy_core/expression/passes/sympy.py
- src/fhy_core/expression/passes/z3.py
- src/fhy_core/expression/pattern/core.py
- src/fhy_core/expression/pattern/__init__.py
- tests/expression/test_ternary_and_call.py
- tests/expression/test_core.py
- tests/expression/test_pprint.py
- tests/expression/test_functions_stories.py
- tests/expression/passes/test_builtins.py
- tests/expression/passes/test_evaluator.py
- tests/expression/passes/test_inline_pass.py
- tests/expression/passes/test_numpy_evaluator.py
- tests/expression/passes/test_sympy_pass.py
- tests/expression/passes/test_type_checker_booleans.py
- tests/expression/passes/test_type_checker_sorts.py
- tests/expression/passes/test_z3_pass.py
- tests/expression/pattern/test_core.py
- tests/expression/pattern/test_rewrite.py

# APPENDIX: key_decisions
- Node structure is parallel tuples (conditions, values, otherwise), not a tuple of (condition, value) pairs: the serialization codec engine (serialization.py:1197-1208) rejects fixed-length tuple element types, so a pairs field would force hand-written serialization; parallel tuples derive fully automatically, following the proven CallExpression.arguments precedent. Pair-shaped construction is preserved at the API surface via piecewise((c, v), ..., otherwise=x) and get_cases().
- otherwise is a required keyword-only parameter (follows *cases), making totality syntactically guaranteed; zero cases is a construction-time ValueError.
- First-match-wins semantics across all backends: sympy lowers to native Piecewise((v,c)..., (otherwise, true), evaluate=False); z3 lowers to a right-folded nested z3.If chain; numpy lowers to a right-folded np.where chain (np.select rejected: scalar-only default parameter, and where-chain is bit-for-bit continuous with the nested-ternary lowering it replaces).
- SymPy lifter now produces one flat PiecewiseExpression from a multi-branch Piecewise (deleting the recursive nested-ternary reconstruction), preserving the existing empty-Piecewise ValueError, single-branch degeneration, and silent final-condition drop.
- TernaryExpression, ternary(), Expression.ternary, TernaryExpressionPattern, and the ternary_expression serialization type_id are deleted with no aliases; old blobs raise UnknownTypeIdError by design.
- CallExpression reframing is explicitly deferred: it already models mathematical function application, a rename is cosmetic with a huge blast radius (registry, 8 passes, builtins, patterns, serialization, ~5k test lines) and would maximize merge conflicts with the parallel constraint/param features; revisit, if ever, in the post-merge fhy_core.symbolic reorg.
- evaluate.py, inline.py, and body_type_checker.py need zero changes (RewritablePass visit_unknown default preserves-and-recurses, matching ternary's current treatment); grep-verified that no code outside src/fhy_core/expression/ and tests/expression/ references ternary, keeping the diff disjoint from features 2 and 3.
- Type checking generalizes the ternary rules as folds: per-case boolean condition check with indexed error messages, promote_primitive_data_types folded over all values plus otherwise, promote_type_qualifiers folded over all conditions and branches, and the scalar-NumericalType expected-type propagation rule renamed to _piecewise_branch_expected_type unchanged.
- Child ordering is interleaved (c1, v1, ..., cn, vn, otherwise) for get_visit_children/get_operands, keeping each case adjacent in traversal order; rebuild_with_visit_children validates odd child count >= 3 with an explicit raise (no asserts).
- Builtin bodies max/min/abs/leaky_relu become one-case piecewise and sign flattens to a single two-case node; pprint renders symbolic {v if c; ...; x otherwise} and functional (piecewise c1 v1 ... otherwise).

# APPENDIX: risks
- Serialized-blob break: any persisted expression containing a ternary_expression node becomes undeserializable (UnknownTypeIdError) with no migration path — intentional per the no-backward-compat rule, but must be called out in release notes.
- simplify_expression and the sympy round-trip change output shape: multi-branch sympy.Piecewise now lifts to one flat PiecewiseExpression instead of nested ternaries, so downstream code or tests that pattern-match simplification results will see different trees.
- The preserved SymPy-lift behavior of silently dropping a non-True final Piecewise condition (treating the last value as otherwise) can misrepresent a genuinely partial user-supplied Piecewise; flagged as open question 1 (alternative: raise ValueError).
- Inlined builtin shapes change (max/min/abs/sign/leaky_relu bodies are now piecewise), touching many backend and story tests at once; the large mechanical test diff raises the chance of a missed assertion update — mitigated by the fact that grep shows all ternary test references are enumerable in 13 files.
- numpy where-chain dtype promotion is pairwise-folded; on NumPy 1.x (pre-NEP-50 value-based casting) a fold can promote differently than np.select's global result_type — identical to today's nested-ternary behavior, but worth a pinned test if NumPy 1.x support matters.
- Merge-window risk with parallel features is low but nonzero: features 2/3 import the expression package's public API; if either worktree happens to construct ternary() (nothing in current constraint.py/param/ does), their branch breaks on merge — the implementing agents should be told ternary is gone.
- The interleaved visit-children ordering is a new invariant; any future generic pass that assumes 'children = dataclass field order' (grouped) would silently mis-rebuild — the rebuild_with_visit_children validation (explicit ValueError on bad counts) only catches arity, not transposition.
# DECISIONS (orchestrator review, final — these override any open question above)
1. SymPy lift of a non-total Piecewise: PRESERVE the current behavior (last branch's value becomes `otherwise`, its condition dropped). Document this rule explicitly in the lifter's docstring.
2. Symbolic pprint syntax `{v1 if c1; ...; v otherwise}`: approved as proposed.
3. Test file: rename to `test_piecewise_and_call.py` (keep call coverage in place).
4. CallExpression reframing: deferred, per the design's recommendation. Do not rename anything call-related.
