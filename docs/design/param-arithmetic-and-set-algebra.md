# Param Arithmetic and Set Algebra

## Summary

Extends the `fhy_core.param` subsystem with (1) interval-integer **multiplication** (`__mul__`/`__rmul__`) alongside the existing `+`/`-`/unary `-`, with correct sign handling and unbounded-end propagation, and (2) **union and intersection** of parameters: union for the finite-set domains that can represent it (categorical, ordinal), and intersection for every domain kind (finite-set baking, permutation constraint conjunction, interval bound tightening, and integer/real constraint conjunction). Set operations are exposed as verb-phrase factory functions `create_union_param` / `create_intersection_param` plus delegating `__or__` / `__and__` dunders on `Param`. Callers are compiler passes and auto-tuning code that combine parameter spaces (e.g., merging candidate tile-size sets, intersecting feasibility regions from two analyses).

Scope is strictly `src/fhy_core/param/` and `tests/param/` (plus this doc). `expression/` and `constraint.py` are untouched; the design uses only their existing public surface (`Expression.substitute`, `EquationConstraint(variable, expression)`, `InSetConstraint(variable, members)` / `NotInSetConstraint(variable, members)`, the `.members` property).

## Motivation

Today arithmetic on params is limited to `+`, binary `-`, and unary `-` on `IntervalIntegerDomain` params (`core.py:390-437`). Multiplication is the single most common missing operation for compiler parameters (tile size × unroll factor, loop-extent products). Meanwhile there is **no** way to combine two parameters' value sets at all — grep confirms no union/intersection/merge operation exists anywhere in `param/`. Users who want "the categories of A plus the categories of B" or "an integer valid for both A and B" must manually re-enumerate domains and hand-copy constraints, which is error-prone (constraints reference the wrong variable, strict-kind uniqueness rules are easy to violate).

## Public interface

All new code follows existing conventions: verb-phrase names, keyword-only `name`, no `assert` in src, `@override` (from `fhy_core.utils`) on every override, frozen dataclasses untouched (no new dataclasses are introduced).

### `src/fhy_core/param/domains.py` — `ParamDomain` additions

```python
class ParamDomain(WrappedFamilySerializable, FrozenMixin, StructuralEquivalence, ABC):
    # ... existing abstract API unchanged ...

    def compute_union(
        self,
        own_constraints: Sequence[Constraint],
        other: "ParamDomain",
        other_constraints: Sequence[Constraint],
        variable: Identifier,
    ) -> tuple["ParamDomain", tuple[Constraint, ...]]:
        """Compute the domain and constraints representing the union of two value sets.

        The base implementation raises: union is representable only for
        finite-set domain kinds that can bake both operands' effective value
        sets into a new member set. ``OrdinalDomain`` and ``CategoricalDomain``
        override this.

        Args:
            own_constraints: Constraints carried by the parameter owning this
                domain (referencing that parameter's variable).
            other: Domain of the right operand.
            other_constraints: Constraints carried by the right operand.
            variable: Variable of the result parameter; every returned
                constraint is bound to it.

        Returns:
            A ``(domain, constraints)`` pair describing the union. For the
            finite-set overrides the returned constraint tuple is always empty
            (operand constraints are baked into the member set).

        Raises:
            TypeError: If this domain kind does not support union, or ``other``
                is a different domain kind. Ordinal unions whose merged values
                are not mutually comparable also raise ``TypeError`` (propagated
                from ``build_ordinal_domain``).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_intersection(
        self,
        own_constraints: Sequence[Constraint],
        other: "ParamDomain",
        other_constraints: Sequence[Constraint],
        variable: Identifier,
    ) -> tuple["ParamDomain", tuple[Constraint, ...]]:
        """Compute the domain and constraints representing the intersection of two value sets.

        Every domain kind implements intersection. Finite-set kinds bake the
        strict-kind intersection of both operands' effective value sets into a
        fresh member set with no constraints; permutation kinds keep the member
        set and return the conjunction of both operands' constraints rebound to
        ``variable``; numeric kinds merge domain attributes conservatively and
        return the rebound constraint conjunction.

        Args:
            own_constraints: Constraints carried by the parameter owning this
                domain.
            other: Domain of the right operand; must be the same domain kind.
            other_constraints: Constraints carried by the right operand.
            variable: Variable of the result parameter; every returned
                constraint is bound to it.

        Returns:
            A ``(domain, constraints)`` pair describing the intersection.

        Raises:
            TypeError: If ``other`` is a different domain kind, or a carried
                constraint kind cannot be rebound.
            ParamError: If the intersection is provably empty (finite-set kinds
                detect this here; numeric emptiness is detected by the calling
                factory).
        """
        raise NotImplementedError
```

Concrete `@override` implementations (leaf-by-leaf semantics in **Behavior**):

- `IntegerDomain.compute_intersection` — merged attributes + rebound conjunction.
- `RealDomain.compute_intersection` — `RealDomain()` + rebound conjunction.
- `IntervalIntegerDomain.compute_intersection` — merged attributes + rebound bound-constraint conjunction (bound tightening happens through the existing effective-min/max fold).
- `OrdinalDomain.compute_union` / `compute_intersection` — baked effective sets via `build_ordinal_domain`.
- `CategoricalDomain.compute_union` / `compute_intersection` — baked effective sets via `build_categorical_domain`.
- `PermutationDomain.compute_intersection` — equal-member-set gate + rebound conjunction.

Private module helpers in `domains.py` (implementation phase; listed for completeness, not part of the public contract): `_rebind_constraint_to_variable(constraint, variable)` (reconstructs `EquationConstraint` via `convert_to_expression().substitute(...)`, `InSetConstraint`/`NotInSetConstraint` via `.members`; unknown kinds raise `TypeError`), `_collect_effective_finite_values(domain, constraints)` (filters members through the existing `_is_value_valid_for`).

### `src/fhy_core/param/core.py` — factories and dunders

```python
def create_union_param(
    left: Param[_T],
    right: Param[_T],
    *,
    name: Identifier | None = None,
) -> Param[_T]:
    """Create a parameter admitting exactly the values valid for either operand.

    Both operands' constraints are folded into the result: each operand's
    member set is filtered by its own constraints before the sets are merged,
    so the result carries no constraints of its own.

    Args:
        left: Left operand; must have an ``OrdinalDomain`` or
            ``CategoricalDomain``.
        right: Right operand; must have the same domain kind as ``left``.
        name: Variable for the result; defaults to a fresh
            ``Identifier("param")``.

    Returns:
        A new parameter over the union of the operands' effective value sets.

    Raises:
        TypeError: If either operand's domain kind does not support union, the
            kinds differ, or merged ordinal values are not mutually comparable.
    """
    raise NotImplementedError


def create_intersection_param(
    left: Param[_T],
    right: Param[_T],
    *,
    name: Identifier | None = None,
) -> Param[_T]:
    """Create a parameter admitting exactly the values valid for both operands.

    Finite-set operands are intersected by baking both effective value sets;
    permutation and numeric operands keep their domain (attributes merged
    conservatively) and carry the conjunction of both operands' constraints
    rebound to the result variable. A mixed pair of one interval-integer
    parameter and one plain integer parameter whose constraints are all bound
    expressions is supported by first coercing the plain parameter through the
    existing interval coercion machinery.

    Args:
        left: Left operand.
        right: Right operand; must have the same domain kind as ``left``
            (modulo the interval/integer coercion above).
        name: Variable for the result; defaults to a fresh
            ``Identifier("param")``.

    Returns:
        A new parameter over the intersection of the operands' feasible sets.

    Raises:
        TypeError: If the domain kinds are incompatible or a carried constraint
            cannot be rebound.
        ParamError: If the intersection is provably empty (an empty finite set,
            an empty integer interval, or a numeric constraint conjunction the
            solver proves infeasible).
    """
    raise NotImplementedError
```

```python
class Param(...):
    # ... existing API unchanged ...

    def __mul__(self, other: Any) -> "Param[int]":
        """Multiply two interval-integer parameters with interval semantics."""
        raise NotImplementedError

    def __rmul__(self, other: Any) -> "Param[int]":
        """Multiply with a reflected operand (multiplication is commutative)."""
        raise NotImplementedError

    def __or__(self, other: Any) -> "Param[_T]":
        """Delegate to :func:`create_union_param` with a fresh result variable."""
        raise NotImplementedError

    def __and__(self, other: Any) -> "Param[_T]":
        """Delegate to :func:`create_intersection_param` with a fresh result variable."""
        raise NotImplementedError
```

Both set-op dunders return `NotImplemented` when `other` is not a `Param` (so Python raises the standard `TypeError`); domain-kind mismatches between two `Param`s raise `TypeError` from the delegated factory. `__mul__` mirrors `__add__` exactly, including the `_coerce_interval_operand` path for a non-interval `self` and `_coerce_to_interval_param` for the operand (`int` promoted, `bool` rejected, plain integer params with bound-only constraints rewrapped). No reflected `__ror__`/`__rand__` are needed: set operations accept `Param` operands only, and both operands then share the exact `Param` type, so the reflected slot is never consulted.

Private helpers in `core.py` (implementation phase): `_multiply_interval_params(left, other)` (body of `__mul__` after the coercion preamble), `_multiply_optional_bounds(self_min, self_max, other_min, other_max) -> tuple[int | None, int | None]` (pure-integer sign-case analysis over `None`-as-infinity bounds; no float sentinels), and an extension of the existing `_create_class_preserved_interval_param` with a keyword-only `zero_included: bool` argument so callers supply the operation-appropriate zero rule (the `__add__` call site passes its current template-derived value — behavior unchanged).

### `src/fhy_core/param/__init__.py`

Adds `create_union_param` and `create_intersection_param` to `__all__` and the `from .core import (...)` block (alphabetically sorted). Nothing changes in the top-level `fhy_core/__init__.py` — the curated namespace already exposes `param` as a namespace and these are family members, not ownerless primitives.

## Behavior

### Multiplication (`IntervalIntegerDomain` only)

Interval product: `[a,b] x [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]`, evaluated over extended integers where `None` means unbounded and the product of zero with an unbounded end is zero (set semantics: if a factor is identically 0, the product set is {0}).

- `[2,3] * [4,5] -> [8,15]`
- `[-2,3] * [4,5] -> [-10,15]` (sign handling: min comes from `-2*5`)
- `[-2,-1] * [-3,-1] -> [1,6]` (negative x negative flips)
- `[1,+inf) * [2,3] -> [2,+inf)`; `(-inf,+inf) * [2,3] -> (-inf,+inf)`
- `[0,0] * (-inf,+inf) -> [0,0]`; `[0,5] * (-inf,0] -> (-inf,0]`
- `3 * p` and `p * 3` promote the scalar via `create_interval_integer_param_exactly`; `bool` operands raise `TypeError`; real/finite-set/other operands raise `TypeError` (via `NotImplemented` or the coercion gate) — identical to `__add__`.
- Class preservation mirrors `__add__`: `non_negative` is kept only when **both** operand domains are non-negative interval domains. The zero rule is multiplication-specific: `zero_included = left.zero_included or right.zero_included` (sound: for non-negative integers `x>0 and y>0 => xy>0`, but `x>0, y>=0` admits `xy=0`). This is why `_create_class_preserved_interval_param` gains an explicit `zero_included` argument instead of reusing the template's.
- An operand whose bound constraints already describe an empty interval raises `ParamError` from the existing `_get_effective_min_max` fold — unchanged inherited behavior.
- Result bounds are rendered honoring `prefer_inclusive` of the left/template domain via the existing `_apply_interval_bounds`; the result reuses the template's variable, exactly as `__add__`/`__sub__` do today.

### Union

Supported: `CategoricalDomain x CategoricalDomain`, `OrdinalDomain x OrdinalDomain`. Everything else raises `TypeError` (base-class default), including permutation x permutation, all numeric kinds (a union of two constraint sets is a *disjunction*, which the conjunction-of-constraints model cannot represent), and any cross-kind pair.

Semantics: each operand's member set is first filtered by that operand's own constraints ("effective value set" — exact, because finite-set domains only admit `InSetConstraint`/`NotInSetConstraint`, which are always decidable), then merged with strict-kind matching (`do_param_values_match`: `True`/`1` and `1`/`1.0` stay distinct). The result is a fresh domain built through `build_categorical_domain`/`build_ordinal_domain` (so all invariants re-validate) and **carries no constraints** — per-operand constraints cannot be conjoined across a union, so they are baked instead.

- `{"a","b","c"} with NotInSet{"c"}  |  {"c","d"}` → categorical `{"a","b","c","d"}` ("c" is valid on the right, so it survives).
- Ordinal `{1,2} | {2,3}` → ordinal `{1,2,3}` (re-sorted). Ordinal union of mutually incomparable values (e.g. `{1,2} | {"a"}`) raises `TypeError` from `build_ordinal_domain`.
- Result variable: fresh `Identifier("param")` unless `name=` given. Dunder form `p | q` always uses the default.

### Intersection

Supported for every kind, same-kind operands only, with one coercion exception: `create_intersection_param` pre-coerces a mixed (interval-integer, plain-integer-with-bound-only-constraints) pair using the existing `_coerce_to_interval_param` machinery before delegating — extending, not forking, the coercion path.

- **Categorical / Ordinal**: strict-kind intersection of both effective value sets, baked into a fresh domain, no constraints. Empty intersection (e.g. `{1} & {True}`) raises `ParamError` with an explicit "intersection is empty" message from `compute_intersection` (not the confusing "Categories must be non-empty").
- **Permutation**: if the member sets are not equal as sets (mutual `is_value_set_subset`), the intersection is genuinely empty → `ParamError`. Otherwise the result keeps the **left** operand's `ordered_members` (member order is a representation detail) and carries the conjunction of both operands' constraints rebound to the result variable. The factory's final feasibility check enumerates permutations — exponential, consistent with every existing permutation-domain check.
- **Interval integer**: result domain merges attributes (`prefer_inclusive` = left's; `non_negative = left.non_negative or right.non_negative`; `zero_included = not ((left.non_negative and not left.zero_included) or (right.non_negative and not right.zero_included))`), constraints are the rebound conjunction of both bound sets — the existing effective-min/max fold then realizes bound tightening (`[0,10] & [5,20]` behaves as `[5,10]`). A conjunction whose effective interval is empty is caught by the factory's feasibility check → `ParamError`.
- **Integer / Real**: same attribute-merge rule for `IntegerDomain` (`RealDomain` has no attributes); constraints are the rebound conjunction. Constraint rebinding handles all three concrete constraint kinds (integer domains accept any constraint kind today); an unknown `Constraint` subclass raises `TypeError`.
- **Emptiness rule (uniform)**: `create_intersection_param` raises `ParamError` whenever the result is *provably* empty — finite-set kinds detect it during `compute_intersection`; numeric kinds via the factory calling `result.is_feasible()` (Z3-backed, optimistic on `unknown`: an undecided conjunction is returned, not rejected, matching the documented "not disproven" convention). Raising is forced, not optional: finite-set domains cannot represent an empty member set at all.
- Result variable: fresh `Identifier("param")` unless `name=` given; all carried constraints are rebound to it, so alpha-equivalence with any equivalent hand-built param holds.

## Files created / modified / deleted

- **Modified**: `src/fhy_core/param/domains.py` (two `ParamDomain` methods + leaf overrides + two private helpers), `src/fhy_core/param/core.py` (`__mul__`/`__rmul__`/`__or__`/`__and__`, two factories, private multiply helpers, `zero_included` keyword on `_create_class_preserved_interval_param`), `src/fhy_core/param/__init__.py` (two new exports).
- **Created**: `docs/design/param-arithmetic-and-set-algebra.md` (this document; `docs/` does not exist yet); test files in the test phase: `tests/param/test_param_multiplication.py`, `tests/param/test_param_union.py`, `tests/param/test_param_intersection.py`; two parametrize entries added to `tests/param/test_signatures.py`.
- **Deleted**: none.
- **Untouched (parallel-feature discipline)**: `src/fhy_core/expression/**`, `src/fhy_core/constraint.py`, `src/fhy_core/param/values.py`, `src/fhy_core/__init__.py`.

## Serialization impact

None on the wire format. No new serializable classes, no new fields on existing serializable classes, no `type_id` changes. Results of every operation are ordinary `Param` instances over existing registered domain kinds and constraint kinds, so round-tripping works through the existing machinery unchanged. The two new `ParamDomain` methods are behavior, not state. Test coverage still asserts round-trips of operation *results* to lock this in.

## Type-checking (mypy --strict) considerations

- `@override` (from `fhy_core.utils`) on every leaf override of `compute_union`/`compute_intersection`; the base `compute_union` is a concrete default (raises `TypeError`), `compute_intersection` is `@abstractmethod`.
- Dunders type `other: Any` and `return NotImplemented`, matching the existing `__add__`/`__sub__` pattern that already passes strict mode.
- Factories are generic in `_T`: `def create_union_param(left: Param[_T], right: Param[_T], *, name: Identifier | None = None) -> Param[_T]` — no `Any` in public signatures beyond the dunder-operand convention.
- `compute_*` return type is spelled `tuple["ParamDomain", tuple[Constraint, ...]]` exactly; no named-tuple indirection.
- `_multiply_optional_bounds` works entirely in `int | None` with explicit sign-case analysis — no `float("inf")` sentinels that would contaminate the integer bound types.
- No `assert` anywhere; all invariants via explicit `raise TypeError`/`raise ParamError`.
- `ty` remains advisory; mypy is the gate (`uv run nox -s type_check`).

## Test plan

- **Unit — multiplication** (`test_param_multiplication.py`): sign matrix (pos x pos, mixed, neg x neg), unbounded-end propagation, zero-width x unbounded, scalar `int` on both sides, `bool`/real/finite-set operand `TypeError`, non-interval-integer coercion in both operand orders, class preservation (`non_negative` kept iff both, `zero_included` or-rule), `prefer_inclusive` rendering of result bounds, empty-operand `ParamError`, result variable reuse. All identifiers via `mock_identifier`.
- **Unit — union** (`test_param_union.py`): categorical/ordinal member merge, constraint baking (`NotInSet` filtering per operand), strict-kind preservation (`True` vs `1`), ordinal re-sort and incomparable-values `TypeError`, cross-kind and unsupported-kind (`permutation`, all numeric) `TypeError`, result carries no constraints, `name=` vs default identifier, `|` delegation, non-`Param` operand `TypeError`.
- **Unit — intersection** (`test_param_intersection.py`): baked finite-set intersection, empty-set `ParamError`, permutation equal-member conjunction and differing-member `ParamError`, interval bound tightening through `&` and factory, mixed interval/integer coercion, integer/real conjunction with domain-attribute merge, constraint rebinding verified via alpha-equivalence against a hand-built param, Z3-proven-empty `ParamError` and unknown-optimistic acceptance (`@pytest.mark.z3` where the solver is required), unknown-constraint-kind `TypeError`.
- **Signatures**: two new entries in `test_signatures.py` asserting `name` is keyword-only on both factories.
- **Integration**: serialization round-trips of results in all formats (reusing `tests/serialization/conftest.py` helpers); results interoperating with `is_subset`/`is_feasible`/`assign`; chained operations (`(a + b) * c`, `(a | b) & c`).
- **Property** (`@pytest.mark.property`): soundness of interval multiplication — for random concrete `x in [a,b]`, `y in [c,d]`, `x*y` is valid for the result; finite-set membership law — `v` is valid for the union (intersection) iff valid for either (both) operand(s).

## Non-goals

- **Division** (`__truediv__`, `__floordiv__`): true division leaves the integer domain (no real-interval domain exists — `is_bound_expression` only recognizes integer literals); floor division requires divisor sign-splitting, is unsound for divisor intervals containing 0, and the committed audit notes floor-div/modulo semantic divergence across the solver bridges, making constraint-level verification of such results unreliable today. Excluded, not deferred-by-flag.
- **Real-valued arithmetic**: `RealDomain` params do not participate in `*` (or `+`/`-`) — there is no interval representation for reals in the domain model.
- **Union of numeric or permutation params**: disjunctions are not representable in the conjunction-of-constraints model; permutation union would require enumerating factorially many tuples into an `InSetConstraint`. Interval union as convex hull is deliberately excluded as a silent over-approximation.
- **Set operations with raw scalar operands** (`p & 5`, `p | "x"`): `Param`-to-`Param` only.
- **Refactoring existing arithmetic** (e.g., the `__sub__` widening asymmetry, or retrofitting named public helpers onto `__add__`/`__sub__`): left as-is.
- **`__pow__`, `__mod__`, n-ary variadic union/intersection helpers**: future work; binary forms compose.

## Open questions

1. Should a named public `create_product_param(left, right, *, name=...)` exist alongside `__mul__`? Chosen: no — `__add__`/`__sub__` expose no named counterparts and a mul-only named factory would be asymmetric; the set-op factories exist because `name=` control is essential there. Confirm.
2. Empty intersection: raise `ParamError` (chosen — forced anyway for finite-set kinds, which cannot represent empty) vs returning an infeasible param for numeric kinds. Confirm the uniform-raise rule.
3. Mixed interval-integer ∩ plain-integer via the existing coercion path: keep, or require exact kind match? Chosen: keep (extends existing machinery).
4. Permutation union via factorial enumeration into an `InSetConstraint`: worth a follow-up, or permanently out?


# APPENDIX: files_to_modify
- src/fhy_core/param/core.py
- src/fhy_core/param/domains.py
- src/fhy_core/param/__init__.py
- docs/design/param-arithmetic-and-set-algebra.md
- tests/param/test_param_multiplication.py
- tests/param/test_param_union.py
- tests/param/test_param_intersection.py
- tests/param/test_signatures.py

# APPENDIX: key_decisions
- Multiplication is interval-integer only, mirroring __add__ exactly (same _coerce_interval_operand/_coerce_to_interval_param machinery, extended not forked); bounds via four-candidate extended-integer product with 0*infinity=0 set semantics and pure int|None case analysis (no float sentinels).
- Mul preserves non_negative only when both operand domains are non-negative (same rule as __add__), but with a multiplication-specific zero rule zero_included = left or right; the private _create_class_preserved_interval_param gains an explicit keyword-only zero_included argument (add's call site passes its current value, behavior unchanged).
- Division (true and floor) and real-param arithmetic are explicit non-goals: no real-interval representation exists, divisor-contains-zero is unsound, and audited floor-div/modulo divergence across solver bridges makes results unverifiable.
- Set algebra ships both forms: verb-phrase factories create_union_param/create_intersection_param (keyword-only name, fresh Identifier('param') default) and delegating __or__/__and__ dunders; dunders return NotImplemented for non-Param operands, factories raise TypeError on kind mismatch.
- Union is supported only where representable: categorical and ordinal, by baking each operand's constraint-filtered effective value set (exact, since set constraints are always decidable) into a fresh domain carrying no constraints; numeric/permutation union raises TypeError because disjunction is not expressible in the conjunction-of-constraints model.
- Intersection is supported for all six domain kinds: finite-set baking (categorical/ordinal), constraint conjunction with rebinding for permutation (equal member sets), interval (bound tightening emerges from the existing min/max fold), integer, and real; constraint rebinding uses only public constraint/expression API (Expression.substitute, constructors, .members) so constraint.py stays untouched.
- Uniform emptiness rule: create_intersection_param raises ParamError when the result is provably empty (finite-set detection in compute_intersection; numeric via result.is_feasible(), optimistic on Z3 unknown) - forced because finite-set domains cannot represent an empty member set.
- Domain-kind logic lives on ParamDomain per the post-PR#89 composition design: compute_intersection is a new abstract method (all leaves implement), compute_union is a concrete base default raising TypeError overridden only by OrdinalDomain/CategoricalDomain, avoiding boilerplate on the four kinds that cannot support union.
- No serialization impact: no new serializable classes, fields, or type_ids; results are ordinary Params over existing registered kinds.
- Diff surface is disjoint from parallel features: only param/ sources, param/__init__.py exports, tests/param/, and a new docs file; expression/, constraint.py, and the top-level curated namespace are untouched.

# APPENDIX: risks
- Adding an abstract compute_intersection to ParamDomain breaks any out-of-tree ParamDomain subclass (none exist in-repo); if external subclasses matter, both methods could default to raising TypeError instead - flagged for reviewer sign-off.
- Permutation intersection's factory-level feasibility check enumerates itertools.permutations (factorial cost) - consistent with every existing permutation-domain check, but a large member set will be slow.
- Numeric intersection emptiness is optimistic on Z3 unknown: an undecidable-empty conjunction is returned as a live param and only fails later (matches the documented 'not disproven' convention, but callers may be surprised).
- Baked finite-set union/intersection discards constraint provenance (results carry no constraints); structurally equivalent to, but not field-identical with, a hand-built domain-plus-constraints param - tests must compare via structural/alpha equivalence, not repr.
- Parallel-feature merge risk is low but nonzero: feature 2 (dependent constraints) may add new Constraint kinds that _rebind_constraint_to_variable cannot rebind; the designed TypeError on unknown constraint kinds makes this fail loudly rather than silently, and the post-merge symbolic reorg should revisit.
- The chosen mul zero_included or-rule changes a shared private helper's signature; the __add__ call site must be updated in the same commit or construction of non-negative sums regresses.
- Mixed interval/plain-integer intersection reuses _coerce_to_interval_param from the factory in core.py (same module, so no cross-class private-call violation), but reviewers may prefer strict same-kind matching - listed as open question 3.
# DECISIONS (orchestrator review, final — these override any open question above)
1. No `create_product_param` named factory — dunders only, symmetric with `__add__`/`__sub__`.
2. Uniform rule confirmed: `create_intersection_param` raises `ParamError` on provably empty results.
3. Mixed interval-integer ∩ plain-integer coercion via `_coerce_to_interval_param`: keep.
4. Permutation union: permanently out of this release (keep as stated non-goal).
