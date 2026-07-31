# `fhy_core.symbolic` Subpackage — Reorganization of expression, constraint, and param behind a backend-agnostic solver seam

Design doc for Feature 4 (final feature) of the next FhY-core release. Runs AFTER features 1 (piecewise), 2 (dependent constraints), and 3 (param ops) are merged into `dev-new-feats`; all line references are to `ad9a311` and remain structurally valid post-merge except where a feature's own design notes changes (called out inline). Doc location when committed: `docs/design/symbolic-subpackage.md`.

## Summary

Move the strictly-layered `expression → constraint → param` cluster (plus the cluster-private `symbol_type.py` leaf) into a new `fhy_core.symbolic` subpackage, and introduce `fhy_core.symbolic.solver` — a small backend-agnostic routing module that becomes the single documented entry point for solver queries (simplification/evaluation via SymPy, satisfiability/implication/universal validity via Z3), restructuring but not rewriting the existing bridge math. Old import paths are deleted outright: no aliases, no re-export shims, every in-repo importer updated. All serialized `type_id`s are explicitly pinned and survive unchanged, so persisted data round-trips. Callers: every FhY-stack repo that consumes the symbolic tower, and in-repo `types/`, which keeps its (verified) mutual edge with the expression IR.

## Motivation

- The three subsystems form one vertical stack (verified: `constraint.py:84` imports `.expression`; `param/core.py:19-20` and `param/domains.py:28-34` import `fhy_core.constraint` and `fhy_core.expression`; no back-edges within the cluster). The conceptual audit (stash `b3b1eac`, `docs/audit/fhy-core-conceptual-audit.md:17`) describes them as "a vertical FhY stack … the whole tower rides on hard, eagerly-imported SymPy/Z3 dependencies." Naming the stack `fhy_core.symbolic` makes the architecture legible and gives the three merged features one coherent home before downstream repos adopt them.
- Solver access is today scattered and duplicated: `simplify_expression` lives in `expression/passes/sympy.py:675`; `does_expression_imply`/`holds_for_all_free_assignments` in `expression/passes/z3.py:316/238`; the "is this constraint conjunction satisfiable?" idiom (lower conjunction, ask whether it implies `False`, invert) is hand-rolled in `param/domains.py:288-305` (`_numeric_has_feasible_value`), again in `param/domains.py:117-157` (`compute_constraint_implication_subset`), and a third time in Feature 2's `ConstraintSystem.check_satisfiability`. The package audit flags the solver bridges as the worst subsystem (Critical findings F-001, F-002, F-003; High F-008, F-009, F-013, F-014 — semantic divergence between the z3 bridge, the sympy bridge, and the type checker). A single seam does not fix that math (explicit non-goal) but gives it exactly one documented home, one capability table where the divergences are stated, and one choke point where the audit's systemic "Z3 solver call with no timeout/resource bound" finding is naturally remediated (see `timeout_milliseconds` below).
- The user's framing — "fall back to sympy or z3 for everything, but be generic in representation" — is satisfied by keeping the FhY `Expression` IR as the one representation and making backend selection an explicit, typed parameter with per-query capability declarations instead of an accident of which module you imported.

## Package layout and file-by-file move map (src)

All moves are `git mv` (history-preserving). Old paths cease to exist; there are no shims.

```
src/fhy_core/symbolic/
    __init__.py            NEW — family namespace + solver seam re-exports
    solver.py              NEW — backend-agnostic query routing (public interface below)
    symbol_type.py         MOVED from src/fhy_core/symbol_type.py (verbatim)
    constraint.py          MOVED from src/fhy_core/constraint.py
    expression/            MOVED wholesale from src/fhy_core/expression/
    param/                 MOVED wholesale from src/fhy_core/param/
```

Exhaustive move map (every file, old → new; content edits per file listed in "Import rewiring"):

| Old path (`src/fhy_core/`) | New path (`src/fhy_core/symbolic/`) |
|---|---|
| `symbol_type.py` | `symbol_type.py` |
| `constraint.py` | `constraint.py` |
| `expression/__init__.py` | `expression/__init__.py` |
| `expression/builtins.py` | `expression/builtins.py` |
| `expression/core.py` | `expression/core.py` |
| `expression/errors.py` | `expression/errors.py` |
| `expression/pprint.py` | `expression/pprint.py` |
| `expression/sort.py` | `expression/sort.py` |
| `expression/passes/__init__.py` | `expression/passes/__init__.py` |
| `expression/passes/body_type_checker.py` | `expression/passes/body_type_checker.py` |
| `expression/passes/evaluate.py` | `expression/passes/evaluate.py` |
| `expression/passes/inline.py` | `expression/passes/inline.py` |
| `expression/passes/native_lowering.py` | `expression/passes/native_lowering.py` |
| `expression/passes/numpy.py` | `expression/passes/numpy.py` |
| `expression/passes/sympy.py` | `expression/passes/sympy.py` |
| `expression/passes/type_checker.py` | `expression/passes/type_checker.py` |
| `expression/passes/z3.py` | `expression/passes/z3.py` |
| `expression/pattern/__init__.py` | `expression/pattern/__init__.py` |
| `expression/pattern/core.py` | `expression/pattern/core.py` |
| `expression/pattern/rewrite.py` | `expression/pattern/rewrite.py` |
| `expression/registry/__init__.py` | `expression/registry/__init__.py` |
| `expression/registry/api.py` | `expression/registry/api.py` |
| `expression/registry/entries.py` | `expression/registry/entries.py` |
| `expression/registry/storage.py` | `expression/registry/storage.py` |
| `param/__init__.py` | `param/__init__.py` |
| `param/core.py` | `param/core.py` |
| `param/domains.py` | `param/domains.py` |
| `param/values.py` | `param/values.py` |

**What stays top-level, and why:**

- `value_domain.py` — **stays, untouched.** Verified: zero code edges to/from expression/constraint/param (only `__init__.py` and a docstring mention in `diagnostic.py:72`). `ValueDomain` classifies IR operation value kinds (data vs. address) and is unrelated to the symbolic stack despite the name; the `ValueDomain`/`ParamDomain` vocabulary collision is noted but not addressed (non-goal).
- `symbol_type.py` — **moves into `symbolic/`.** Verified consumers: `expression/passes/z3.py:48`, `param/core.py`, `param/domains.py`, plus Feature 2's new `constraint.py` edge — all inside the cluster; the module itself imports nothing from fhy_core. It is the solver-sort vocabulary of the symbolic family, not an ownerless primitive, so per the curated-namespace philosophy it belongs in the family namespace. No cycle is possible (it remains a pure leaf).
- `types/` — **stays top-level, mutual edge preserved.** Correction to the cross-cutting map (which listed `types` as upstream-only): `types/core.py:43-44` imports `..expression.core` and `..expression.pprint`, and `types/dispatch.py:58` imports `..expression.core`, while `expression/passes/{type_checker,numpy,body_type_checker}.py` import `fhy_core.types`. This is a genuine bidirectional package dependency resolved today by submodule load order: the top-level `__init__` reaches the expression package first, `expression/__init__.py` fully loads `.core` (via its `.builtins` import) before any pass imports `fhy_core.types`, so when `types/core.py` executes mid-cycle, `expression.core` is already complete in `sys.modules`. The reorg preserves this mechanism one level deeper (`..expression.core` becomes `..symbolic.expression.core`); `symbolic/__init__.py` must import `expression` before anything else, and `symbolic/expression/__init__.py`'s existing import order (builtins → core → … → passes) is load-bearing and unchanged. A regression test pins the ordering (test plan).
- `pass_infrastructure`, `traits`, `identifier`, `serialization`, `utils`, `logger`, `error`, `diagnostic` — stay top-level; upstream-only, unchanged. The `identifier.py → traits.equality`/`traits.frozen` direct-submodule cycle-breaker (`identifier.py:20-21`) is untouched.

## Public interface — `fhy_core/symbolic/solver.py` (interface stubs)

The seam RESTRUCTURES the bridges: `passes/sympy.py` and `passes/z3.py` keep every line of conversion/decision math and their `CompilerPass` registrations, but the user-facing query functions move their public home to `solver.py`, which routes on an explicit backend parameter. The pass modules' same-named implementation functions remain module-public (no cross-class-private calls; plain module functions), but are no longer exported from `symbolic.expression`'s `__init__`.

```python
"""Backend-agnostic entry point for symbolic queries over expressions.

Simplification and evaluation are answered by the SymPy bridge;
satisfiability, implication, and universal validity by the Z3 bridge.
Backend selection is explicit; asking a backend for a query kind it
cannot answer raises ``SolverCapabilityError``.
"""

__all__ = [
    "SolverBackend",
    "SolverCapabilityError",
    "SolverQueryKind",
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
    "check_expression_satisfiability",
    "does_expression_imply",
    "get_backend_capabilities",
    "holds_for_all_free_assignments",
    "simplify_expression",
]


class SolverBackend(StrEnum):  # StrEnum from fhy_core.utils
    """Symbolic engine selectable for a solver query."""

    SYMPY = "sympy"
    Z3 = "z3"


class SolverQueryKind(StrEnum):
    """Kind of question a solver backend can be asked."""

    SIMPLIFICATION = "simplification"
    SATISFIABILITY = "satisfiability"
    IMPLICATION = "implication"
    UNIVERSAL_VALIDITY = "universal_validity"


@register_error
class SolverCapabilityError(ValueError):
    """Raised when the requested backend cannot answer the requested query kind."""


def get_backend_capabilities(backend: SolverBackend) -> frozenset[SolverQueryKind]:
    """Return the query kinds the given backend can answer.

    Returns:
        ``{SIMPLIFICATION}`` for SYMPY; ``{SATISFIABILITY, IMPLICATION,
        UNIVERSAL_VALIDITY}`` for Z3.
    """
    raise NotImplementedError


def simplify_expression(
    expression: Expression,
    environment: dict[Identifier, Expression] | None = None,
    *,
    backend: SolverBackend = SolverBackend.SYMPY,
) -> Expression:
    """Simplify an expression, optionally substituting an environment first.

    With an environment binding every free identifier, simplification is
    evaluation: the result is a ``LiteralExpression`` whenever the backend
    can decide the value. Delegates to the SymPy bridge; the math is
    byte-identical to the pre-seam behavior.

    Raises:
        SolverCapabilityError: If ``backend`` is not SIMPLIFICATION-capable
            (currently: any backend other than SYMPY).
    """
    raise NotImplementedError


def check_expression_satisfiability(
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether some assignment to the free identifiers satisfies the expression.

    True if a satisfying assignment provably exists; False if provably
    none exists; None if the solver returns unknown. Implemented as the
    inversion of ``does_expression_imply(expression, false)`` — the exact
    construction previously duplicated in ``param.domains`` and
    ``constraint.ConstraintSystem``.

    Raises:
        SolverCapabilityError: If ``backend`` is not SATISFIABILITY-capable.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
    """
    raise NotImplementedError


def does_expression_imply(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the antecedent logically implies the consequent.

    Contract, semantics, and math identical to the pre-seam function of
    the same name; None means the solver returned unknown.

    Raises:
        SolverCapabilityError: If ``backend`` is not IMPLICATION-capable.
        KeyError / RuntimeError: As before (propagated from the bridge).
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
    """
    raise NotImplementedError


def holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Check the forall/exists validity query; contract unchanged from the bridge.

    Raises:
        SolverCapabilityError: If ``backend`` is not UNIVERSAL_VALIDITY-capable.
    """
    raise NotImplementedError


def assert_holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Strict variant: raise ``UndecidableError`` instead of returning None."""
    raise NotImplementedError


def assert_expression_implies(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Strict variant of ``does_expression_imply``; raises ``UndecidableError`` on unknown."""
    raise NotImplementedError
```

**`fhy_core/symbolic/__init__.py`** exposes the five submodules as namespaces (`constraint`, `expression`, `param`, `solver`, `symbol_type`), re-exports the full `solver` public surface (the seam is "the one documented entry point": `fhy_core.symbolic.simplify_expression` works), and re-exports `SymbolType` (the family's shared sort vocabulary). Import order inside it is `expression` first (load-bearing for the `types` cycle), then `solver`, `constraint`, `param`. Precedent for family-level aggregation: `expression/__init__.py` already aggregates its whole subtree.

**`src/fhy_core/__init__.py`** after the move: the `from . import (...)` block and `__all__` drop `constraint`, `expression`, `param`, `symbol_type` and gain `symbolic`. `Identifier` remains the only re-exported symbol. Docstring examples updated (`fhy_core.symbolic.param.create_integer_param`).

## Semantics and edge cases

- **Routing is total and explicit.** Every seam function validates `backend` against the capability table before touching a bridge; an incapable pairing raises `SolverCapabilityError` naming both the backend and the query kind. Today each query kind has exactly one capable backend — the seam's honest value is uniform routing, deduplication, and a place where future capabilities (z3 simplification via tactics, sympy satisfiability via solveset) attach without new entry points. The doc and docstrings say this plainly; "interchangeable" is aspiration encoded as an extension point, not a claim about current behavior.
- **Math is unchanged.** `solver.simplify_expression(e, env)` produces the identical result object graph as the pre-seam `simplify_expression(e, env)`; the z3-backed functions delegate to the moved-in-place bridge functions with identical semantics, including `None`-on-unknown, `KeyError` on missing symbol types, and `UndecidableError` from the `assert_` variants (which stays defined in `symbolic/expression/errors.py`).
- **`timeout_milliseconds`** (new, optional, default `None` = today's unbounded behavior): threaded to `z3.Solver().set(timeout=...)` inside the bridge's single solver-invocation helper. `None` → no change; positive int → z3 may return unknown at the deadline, surfacing as `None`/`UndecidableError` per the existing unknown contract; zero/negative → `ValueError` at the seam. SymPy-backed queries do not accept the parameter (it does not appear in `simplify_expression`'s signature). This is the one audit-flagged systemic issue (unbounded Z3 solver calls, representative location `z3.py:290-300`) the reorganization naturally remediates.
- **`check_expression_satisfiability` truth table:** provable model exists → `True`; proven unsat → `False`; unknown → `None`. Empty-`symbol_types` with a closed expression is legal (no free identifiers → no lookups). Callers keep their own unknown policies: `param.domains` maps `None` → feasible (optimistic, documented, unchanged); Feature 2's `ConstraintSystem.check_satisfiability` maps to `ConstraintOutcome.UNDECIDED`.
- **Consumer rewiring through the seam** (all internal, behavior-preserving):
  - `symbolic/constraint.py`: `simplify_expression` import moves from `.expression` to `.solver` (used by `EquationConstraint.evaluate`/`evaluate_with_bindings`); Feature 2's `ConstraintSystem.check_satisfiability`/`check_satisfiability_with_bindings` call `solver.check_expression_satisfiability` instead of hand-rolling the implies-`False` inversion over `does_expression_imply`.
  - `symbolic/param/domains.py`: `_numeric_has_feasible_value` routes through `solver.check_expression_satisfiability`; `compute_constraint_implication_subset` routes through `solver.does_expression_imply`. Optimistic-unknown mapping stays local to `domains.py`.
  - `symbolic/expression/__init__.py`: stops exporting `simplify_expression`, `does_expression_imply`, `holds_for_all_free_assignments`, `assert_expression_implies`, `assert_holds_for_all_free_assignments` (public homes now `symbolic.solver`); keeps exporting the converters (`convert_expression_to_sympy_expression`, `substitute_sympy_expression_variables`, `convert_sympy_expression_to_expression`, `convert_expression_to_z3_expression`) and the IR-level evaluators (`evaluate_expression`, `evaluate_expression_with_numpy`, `inline_functions`, type-checker entries) — those are lowerings/folds over the IR, deliberately outside the solver seam, and the design doc says why (native constant folding and NumPy array evaluation are not sympy/z3 queries and have array-valued or IR-valued returns).
- **Registered pass names**: the nine `@register_pass` name strings embedding `fhy_core.expression.` (`to_sympy`, `from_sympy`, `to_z3`, `evaluate`, `evaluate_with_numpy`, `inline_functions`, `type_checker`, `check_registered_function_body`, `apply_rewrite_rules`) are renamed to the `fhy_core.symbolic.expression.` prefix — describing current state only, per the no-stale-references rule. `tests/.../test_cross_cutting.py` updates accordingly.
- **Import rewiring conventions** (stated once, applied mechanically): intra-`symbolic` edges use relative imports (`constraint.py`'s existing `from .expression import ...` survives verbatim since `expression` is still its sibling; `param/*` switches from absolute `fhy_core.constraint`/`fhy_core.expression` to `..constraint`/`..expression`; `passes/{sympy,z3,type_checker}.py` switch from absolute `fhy_core.expression.*` to `..core`/`..errors`/`..registry`/etc.; `symbol_type` consumers use `..symbol_type`/`...symbol_type`). Edges to modules outside `symbolic` use absolute `fhy_core.X` (existing single-dot relatives like `constraint.py`'s `from .identifier import` are rewritten absolute rather than to fragile `..` forms). Docstring prose referencing old dotted paths (e.g. `expression/passes/__init__.py:8`) is updated.
- **Audit findings, honestly stated:** none of the bridge Critical/High math defects (F-001 logical-not, F-002 Rational lift, F-003 integer-division divergence, F-008 inf/nan lift, F-009/F-013/F-014 floor-div/modulo Euclidean divergence) is fixed by this reorganization — restructure, not rewrite. What the reorg does deliver against the audit: (a) the systemic unbounded-solver-call finding gains a first-class `timeout_milliseconds` control at the seam; (b) the triplicated satisfiability construction collapses into one function; (c) the per-backend semantic divergences get one documented home (the `solver.py` module docstring carries a "known divergences" section citing the finding IDs) so follow-up fixes land at one choke point with conformance tests attachable per `SolverQueryKind`.

## Files created / modified / deleted

**Created:** `src/fhy_core/symbolic/__init__.py`; `src/fhy_core/symbolic/solver.py`; `docs/design/symbolic-subpackage.md` (this doc); `tests/symbolic/__init__.py`; `tests/symbolic/conftest.py`; `tests/symbolic/test_solver.py`.

**Moved (src, `git mv`):** the 30-file map above (28 cluster files + `constraint.py` + `symbol_type.py`).

**Modified in place (src, outside the moved tree):**
- `src/fhy_core/__init__.py` — namespace list, `__all__`, docstring.
- `src/fhy_core/types/core.py` (`:43-44`) and `src/fhy_core/types/dispatch.py` (`:58`) — `..expression.*` → `..symbolic.expression.*`.

**Modified within moved files:** import paths per the rewiring conventions; nine pass-name strings; `symbolic/expression/__init__.py` export removals; `constraint.py` and `param/domains.py` seam rewiring; `passes/sympy.py`/`passes/z3.py` restructure (public query functions rehomed to `solver.py`; conversion classes, converters, and decision math stay).

**Moved (tests, `git mv`), mirror structure:**
- `tests/expression/**` → `tests/symbolic/expression/**` (including `passes/`, `pattern/`, and the Feature 1-renamed `test_piecewise_and_call.py`)
- `tests/constraint/**` → `tests/symbolic/constraint/**` (including Feature 2's `test_bindings_evaluation.py`, `test_constraint_system.py`)
- `tests/param/**` → `tests/symbolic/param/**` (including Feature 3's `test_param_multiplication.py`, `test_param_union.py`, `test_param_intersection.py`)

Leaf conftests keep their `from ..conftest import mock_identifier, ...` chain: `..conftest` now resolves to the new `tests/symbolic/conftest.py`, which re-exports `mock_identifier` and `SerializableEqualHashable` from the (unmoved) root `tests/conftest.py`, exactly the documented re-export convention.

**Modified (tests, outside the moved tree):** `tests/types/test_core.py`, `tests/types/test_extension.py`, `tests/types/test_serialization.py`, `tests/types/test_unification.py` (`fhy_core.expression` → `fhy_core.symbolic.expression`); `tests/test_error.py` (`fhy_core.constraint`/`fhy_core.param` error imports); `tests/test_testing_patches.py` (`fhy_core.expression.core` import). Within moved test files, `fhy_core.expression/constraint/param/symbol_type` import paths are mass-updated, and tests that imported the five rehomed query functions from `fhy_core.expression` now import from `fhy_core.symbolic.solver`.

**Deleted:** the old paths themselves (consequence of `git mv`); the five query-function exports from the expression namespace. No file is deleted with its content discarded.

**Unchanged (verified):** `identifier.py` (and its traits-submodule cycle-breaker), `traits/*`, `serialization.py`, `pass_infrastructure/*`, `value_domain.py`, `lattice.py`, `symbol_table.py`, `op_attribute.py`, `diagnostic.py`, `provenance.py`, `error.py`, `logger.py`, `testing_patches.py`, `utils/*`; `noxfile.py` (whole-tree `src`/`tests` globs), `.pre-commit-config.yaml`; root `tests/conftest.py` (z3 auto-skip, `mock_identifier`, `SerializableEqualHashable` all stay put); `tests/serialization/`, `tests/pass_infrastructure/`, `tests/types/` directories stay in place (only import lines change in `tests/types/`).

**Modified (one addition, no packaging impact):** `pyproject.toml` gains the `import_order` pytest marker (`markers = [..., "import_order: pins module import-order for circular-import resolution"]`), registered because `addopts` enables `--strict-markers` and the new `tests/symbolic/test_import_order.py` applies `@pytest.mark.import_order`; without the registration, collection of that suite would fail outright. Packaging and coverage configuration are otherwise unaffected: uv_build's src-layout auto-discovers the subpackage, and coverage's `source=["fhy_core"]` and path maps need no changes.

## Serialization impact

**Wire format: zero change; all persisted data round-trips.** Verified class-by-class: every `Serializable` in the moved tree pins an explicit `type_id` via `@register_serializable(type_id=...)` — expression: `unary_expression`, `binary_expression`, `identifier_expression`, `literal_expression`, `call_expression`, plus Feature 1's `piecewise_expression` (`ternary_expression` already deleted by Feature 1); constraint: `equation_constraint`, `in_set_constraint`, `not_in_set_constraint`, plus Feature 2's `constraint_system`; param: `integer_domain`, `real_domain`, `interval_integer_domain`, `ordinal_domain`, `categorical_domain`, `permutation_domain`, `param`, `param_assignment`. Justification from the framework: `register_serializable` writes the pinned string onto `_SERIALIZATION_CLASS_TYPE_ID` and keys the process-global `_TYPE_REGISTRY` by that string (`serialization.py:659-715`); the module-qualified default (`_get_default_type_id`, `:390-391`) applies only when `type_id` is omitted — which no moved class does. The optional import-fallback resolver (`_resolve_type_id`, `:394-455`, `allowed_module_prefixes=("fhy_core.",)`) only matters for module-shaped ids, which none of these are; the `fhy_core.` prefix guard still admits `fhy_core.symbolic.*` regardless. `SymbolType` is a plain enum (not independently serialized). No `alias=True` registrations are added anywhere — nothing is renamed on the wire. A golden-blob regression test locks all 18 ids (test plan).

## Type-checking (mypy --strict) considerations

- `solver.py` is fully annotated with zero `Any`: `StrEnum` from `fhy_core.utils`; returns `Expression`, `bool | None`, `bool`, `frozenset[SolverQueryKind]`. No overrides, so no `@override` needed there; every moved file keeps its existing `@override` usage and passes unchanged.
- Parameter types are kept verbatim from the pre-seam signatures (`dict[Identifier, Expression] | None`, `dict[Identifier, SymbolType]`, `AbstractSet[Identifier]`) so no caller-side type churn beyond import paths.
- `passes/sympy.py`/`passes/z3.py` retain their `# type: ignore` import pragmas; the seam adds no new ignores. `SolverCapabilityError` guards are explicit `raise` (no `assert` in src). Capability dispatch uses `if backend is not SolverBackend.X: raise ...` rather than a `match` needing exhaustiveness gymnastics.
- The `types ↔ symbolic.expression` mutual edge type-checks as before (mypy resolves modules statically; the runtime import-order subtlety is invisible to it). `ty` remains advisory; the relocated tree carries no new ty hazards.
- Ruff: first-party prefix is still `fhy_core`, so isort grouping is stable; `uv run nox -s lint` and `-s type_check` run unmodified.

## Test plan (outline; tests written in phase 2, all identifiers via `mock_identifier`)

- **Unit — solver seam** (`tests/symbolic/test_solver.py`): capability table exact contents; `SolverCapabilityError` for every incapable (backend, query) pairing with message naming both; `simplify_expression` delegation equivalence against representative expressions/environments (structural equality with the sympy-bridge pipeline output); `check_expression_satisfiability` truth table (satisfiable / unsat / closed-expression / empty symbol_types) with `@pytest.mark.z3`; `does_expression_imply`/`holds_for_all_free_assignments`/`assert_*` parity with pre-seam contracts including `UndecidableError` and `KeyError` propagation; `timeout_milliseconds` validation (`ValueError` on 0/negative, accepted positive, absent from the sympy signature).
- **Unit — namespace shape**: `fhy_core` exposes `symbolic` and no longer exposes `expression`/`constraint`/`param`/`symbol_type` (`__all__` assertions); `fhy_core.symbolic.__all__` contents; rehomed query functions absent from `fhy_core.symbolic.expression`'s exports.
- **Integration — move integrity**: golden serialized-dict fixtures for one instance of each of the 18 pinned `type_id`s, deserialized post-move via `tests/serialization` round-trip helpers; pass-registry names test asserting the nine `fhy_core.symbolic.expression.*` names (updated `test_cross_cutting.py`); rewired-consumer parity — `Param.is_feasible`/`is_subset` and `ConstraintSystem.check_satisfiability` results unchanged on the existing scenario matrices (`@pytest.mark.z3` where solving).
- **Integration — import order / cycle**: subprocess tests (`@pytest.mark.subprocess`) importing, in fresh interpreters, `fhy_core.types` first, `fhy_core.symbolic.param` first, and `fhy_core.symbolic.expression.core` first — each must succeed, pinning the load-bearing ordering.
- **Relocated suites**: the entire existing `tests/{expression,constraint,param}` corpus moves and must pass with only import-path edits — the behavioral no-op guarantee for everything except the rehomed entry points.
- **Property** (`@pytest.mark.property`): existing relocated property suites; plus seam-routing identity — for randomly generated small expressions, `solver.simplify_expression` output is structurally equivalent to the direct bridge pipeline.
- **Adversarial**: unknown backend value (a raw string where the enum is expected → mypy rejects; runtime path raises via enum coercion failure); `timeout_milliseconds=1` on a large conjunction returning `None` without raising; blob with an unregistered type id still raising `UnknownTypeIdError` unchanged.

## Non-goals

- **No bridge math changes.** F-001, F-002, F-003, F-008, F-009, F-013, F-014 (division/modulo/logical-not/Rational/inf-nan defects) remain open; the seam documents them and is where their fixes will later land. Likewise F-004 (frozen deepcopy/pickle) is a traits issue outside this feature.
- **No optionality for sympy/z3.** Both remain hard, eagerly-imported runtime dependencies (`pyproject.toml:33-34`); the seam selects between them, it does not make them optional extras.
- **No new backend capabilities** (no z3-based simplification, no sympy-based satisfiability) and no additional backends (numpy/native evaluators stay outside the seam as IR-level tools).
- **No API redesign of the moved subsystems**: the three merged features' surfaces move verbatim; no `CallExpression` renaming (deferred by Feature 1, still deferred); no `ValueDomain`/`ParamDomain` naming reconciliation; no constraint-package split (`constraint.py` stays one module).
- **No compatibility machinery**: no re-export shims at old paths, no `alias=True` serialization aliases, no deprecation wording anywhere. Downstream repos update their imports in one breaking release.
- **No changes** to `pass_infrastructure`, `traits`, `serialization`, `identifier`, `value_domain`, tooling configs, or CI workflows.

## Open questions

1. **`symbol_type.py` placement**: moved into `symbolic/` here (family-owned vocabulary, cluster-only consumers). Confirm — the fallback is leaving it top-level with only import-path updates in the cluster, a strictly smaller diff.
2. **Registered pass-name strings**: renamed to `fhy_core.symbolic.expression.*` for current-state accuracy, but they are runtime registry keys — confirm no out-of-repo tooling keys on the old strings before committing to the rename.
3. **`timeout_milliseconds`**: kept because it directly remediates the audit's unbounded-solver-call finding with default-preserving semantics; drop it if the reorg should be a pure move with zero new parameters.
4. **`fhy_core.symbolic` re-exporting the solver surface** (so `fhy_core.symbolic.simplify_expression` works): chosen for "one documented entry point" ergonomics, precedented by `expression/__init__.py` aggregation. Confirm versus a stricter namespaces-only `symbolic/__init__.py`.
5. **Rehoming the five query functions out of `fhy_core.symbolic.expression`'s exports**: chosen so the seam is the only public home (no dual export). Confirm; the alternative (leave them also exported from the bridges' package) contradicts the single-entry-point goal.

# APPENDIX: files_to_modify
- src/fhy_core/symbolic/__init__.py (new)
- src/fhy_core/symbolic/solver.py (new)
- docs/design/symbolic-subpackage.md (new)
- src/fhy_core/symbol_type.py -> src/fhy_core/symbolic/symbol_type.py
- src/fhy_core/constraint.py -> src/fhy_core/symbolic/constraint.py
- src/fhy_core/expression/** (21 files) -> src/fhy_core/symbolic/expression/**
- src/fhy_core/param/** (4 files) -> src/fhy_core/symbolic/param/**
- src/fhy_core/__init__.py
- src/fhy_core/types/core.py
- src/fhy_core/types/dispatch.py
- pyproject.toml (`import_order` pytest marker registration)
- tests/symbolic/__init__.py (new)
- tests/symbolic/conftest.py (new)
- tests/symbolic/test_solver.py (new)
- tests/expression/** -> tests/symbolic/expression/**
- tests/constraint/** -> tests/symbolic/constraint/**
- tests/param/** -> tests/symbolic/param/**
- tests/types/test_core.py
- tests/types/test_extension.py
- tests/types/test_serialization.py
- tests/types/test_unification.py
- tests/test_error.py
- tests/test_testing_patches.py

# APPENDIX: key_decisions
- Move expression/, constraint.py, param/, AND symbol_type.py (cluster-only leaf enum, verified: consumed solely by z3 pass, param, and Feature 2's constraint) into fhy_core/symbolic/; value_domain.py stays top-level (verified zero code edges to the cluster); types/ stays top-level with its mutual edge to expression preserved.
- New fhy_core/symbolic/solver.py is the single documented query entry point: SolverBackend/SolverQueryKind StrEnums, SolverCapabilityError, get_backend_capabilities, plus rehomed simplify_expression (sympy), does_expression_imply / holds_for_all_free_assignments / assert_* (z3), and NEW check_expression_satisfiability that unifies the implies-False satisfiability idiom triplicated in param/domains.py:288-305, domains.py:117-157, and Feature 2's ConstraintSystem — bridges keep all math; only public homes move.
- Each query kind currently has exactly one capable backend; incapable (backend, query) pairings raise SolverCapabilityError — the seam is honest routing plus an extension point, not a claim of present interchangeability.
- Optional timeout_milliseconds keyword on z3-backed seam functions (default None = current unbounded behavior) is the one audit finding (systemic unbounded Z3 solver calls) the reorg naturally remediates; the bridge math Criticals (F-001/002/003/008/009) are explicitly NOT fixed.
- Serialization wire format is untouched: all 18 serializable classes in the cluster pin explicit type_ids (verified class-by-class), and register_serializable keys the registry by the pinned string, not the module path; golden-blob tests lock this.
- The cross-cutting map's claim that types/ is upstream-only is wrong: types/core.py:43-44 and types/dispatch.py:58 import ..expression.core/pprint while expression passes import fhy_core.types; the load-bearing submodule import order (expression/__init__ loads .core via .builtins before passes import types) is preserved one level deeper and pinned by subprocess import-order tests.
- Nine @register_pass name strings renamed from fhy_core.expression.* to fhy_core.symbolic.expression.* (describe-current-state rule); intra-symbolic imports go relative, upstream imports go absolute fhy_core.*; five query functions are removed from fhy_core.symbolic.expression's exports (solver is their only public home); no shims, no aliases, git mv throughout; tests mirror to tests/symbolic/ with the existing conftest re-export chain extended one level.

# APPENDIX: risks
- Downstream repos (FhY, fhy_llvm) break on every old import path (fhy_core.expression/constraint/param/symbol_type) and on the five rehomed query functions — intended breaking release, but adoption must be coordinated; grep-verified the in-repo importer set is closed (types/, tests/types/, tests/test_error.py, tests/test_testing_patches.py, top-level __init__).
- The types <-> symbolic.expression circular import is resolved only by submodule load order (expression/__init__ must load .core before any pass imports fhy_core.types, and symbolic/__init__ must import expression first); an innocent import reordering during the move could break interpreter startup — mitigated by dedicated subprocess import-order regression tests.
- Renaming the nine registered pass-name strings changes runtime PassRegistry keys; any out-of-repo tooling or persisted pass-pipeline config keyed on 'fhy_core.expression.*' breaks silently (flagged as open question 2).
- This feature rebases onto three concurrently-developed merged features; their final merged shapes (e.g. Feature 2's exact constraint.py imports, Feature 1's piecewise exports) may drift from the designs this move map assumes — the move map must be re-verified against the actual merged tree before implementation.
- timeout_milliseconds threads a new parameter into the z3 bridge's solver invocation; while default-preserving, it is the only non-pure-move behavior change and could be cut (open question 3) if reviewers want a zero-delta reorg.
- Audit docs exist only in stash b3b1eac (not on any branch); finding IDs cited in solver.py's known-divergences docstring section cannot link to committed files until the audit is committed.
# DECISIONS (orchestrator review, final — these override any open question above)
1. `symbol_type.py` moves into `symbolic/`.
2. The nine `@register_pass` name strings are renamed to `fhy_core.symbolic.expression.*` — intentional breaking release.
3. `timeout_milliseconds`: keep as designed.
4. `fhy_core.symbolic.__init__` re-exports the full solver surface.
5. `solver.py` is the sole public home of the five query functions (removed from `symbolic.expression` exports).
6. Cite audit finding IDs in solver.py's known-divergences docstring as plain IDs (no file links; the audit doc is not committed on any branch).
