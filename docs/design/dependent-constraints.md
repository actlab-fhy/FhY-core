# Dependent (Multi-Variable) Constraint Support

Design doc for Feature 2 of the next FhY-core release. Target branch: worktree off `dev-new-feats` (`ad9a311`). All source changes are confined to `src/fhy_core/constraint.py`.

## Summary

Today every `Constraint` binds exactly one `variable: Identifier`, and the entire checking API (`evaluate`, `is_satisfied`, `__call__`) accepts a single positional value bound to that one variable. An `EquationConstraint` whose expression mentions a second free identifier silently degrades to `UNDECIDED` (`constraint.py:534` builds a one-key substitution dict; the sympy simplifier cannot reduce the residual expression). This feature adds (a) a first-class *bindings* evaluation API on `Constraint` — evaluate a constraint under a `Mapping[Identifier, value]` — and (b) a `ConstraintSystem` value object for joint reasoning over a set of constraints spanning multiple variables, including z3-backed joint satisfiability that lowers through the existing `does_expression_imply` bridge (`expression/passes/z3.py:316`). Callers: parameter-space tooling that needs cross-parameter ("dependent") constraints, and any future symbolic-library consumer.

## Motivation

- `EquationConstraint` documents ("whose *only* free identifier is meant to be `self.variable`", `constraint.py:500-501`) but does not enforce single-variable expressions. Multi-variable constraints are constructible but unusable: `evaluate` can pass only one value, so `x + y < 10` can never be decided.
- Feasibility checking exists only inside `param/domains.py`, is per-parameter, and rewrites every constraint onto one common variable (`_convert_constraints_to_implication_expression`, `domains.py:100-114`) — it structurally *cannot* express "x and y are two different variables related by a constraint".
- The z3 bridge (`does_expression_imply`, `holds_for_all_free_assignments`) already reasons over expressions with many free identifiers; nothing in the constraint layer exposes it.
- Prior art followed: the tri-state `ConstraintOutcome` enum (`constraint.py:103-118`), the frozen-dataclass + `WrappedFamilySerializable` family pattern, `Param`'s canonical-ordering of constraint tuples (`param/core.py:135, 150-159`), and `Expression.substitute` multi-key substitution (`expression/core.py:326`).

## Design decisions (with justification)

1. **Bindings API on `Constraint` itself, not a new constraint kind.** A "multi-variable constraint" is not a new predicate shape — `EquationConstraint` already *is* one structurally; only the evaluation entry point is too narrow. Adding a leaf kind would duplicate `EquationConstraint` and force `Param`/domain code to learn a new sum-type arm. Instead: a concrete base-class method `evaluate_with_bindings` with sound single-variable default semantics (so `InSetConstraint`/`NotInSetConstraint` and any third-party leaf get correct behavior with zero code), overridden only by `EquationConstraint` to substitute every bound identifier.
2. **Plus a `ConstraintSystem` class for the set-level story.** Joint satisfiability is a property of a *collection* of constraints sharing identifiers; it has no home on any single `Constraint`. `ConstraintSystem` is a small frozen, serializable value object — not a `Constraint` subclass (it has no single `variable`, and must not satisfy the leaf contract).
3. **Partial bindings yield `UNDECIDED`, never raise.** This aligns exactly with the documented meaning of `UNDECIDED` ("the checker cannot decide") and with `is_satisfied`'s conservative-rejection contract (`constraint.py:455-470`). Errors are reserved for malformed inputs, not missing information.
4. **Satisfiability results reuse `ConstraintOutcome`.** `SATISFIED` = a joint assignment provably exists, `VIOLATED` = provably none exists, `UNDECIDED` = solver returned `unknown`. This avoids a second three-state enum and keeps one vocabulary in the module. Unlike the param layer's optimistic `unknown → feasible` convention (`domains.py:301-305`), `ConstraintSystem` reports `UNDECIDED` explicitly and lets the caller choose a policy.
5. **`ConstraintSystem` canonicalizes constraint order (repr-sort) at construction**, following `Param._build_canonical_constraints` precedent, so structurally identical systems built in different orders are structurally equivalent, serialize identically, and lower to identical conjunctions (matching the module's determinism contract, `constraint.py:30-32`).
6. **`param/` is not touched.** Dependent constraints across params are expressed by collecting `param.constraints` tuples (public field) plus linking `EquationConstraint`s into a `ConstraintSystem` keyed by the params' own variable identifiers. Deeper unification is deferred to the post-merge symbolic feature.

## Public interface (interface stubs)

All additions live in `src/fhy_core/constraint.py`. New `__all__` entries: `"ConstraintBindings"`, `"ConstraintSystem"`, `"create_constraint_system"`.

```python
# --- New imports (top of constraint.py) ---
# from collections.abc import Mapping, Sequence
# from .expression import does_expression_imply   (already imports from .expression)
# from .symbol_type import SymbolType             (new leaf-module edge; no cycle)

ConstraintBindings: TypeAlias = Mapping[Identifier, "Expression | LiteralType"]
"""Assignment of candidate values (literals or expressions) to identifiers."""


class Constraint(WrappedFamilySerializable, FrozenMixin, DerivedEquivalenceMixin, ABC):
    # ... existing members unchanged: __call__, evaluate, is_satisfied,
    #     convert_to_expression, __repr__, __str__ ...

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return every identifier this constraint constrains or references.

        The base implementation returns ``frozenset((self.variable,))``.
        ``EquationConstraint`` overrides it to also include the free
        identifiers of its expression.

        Returns:
            Non-empty frozen set of identifiers; always contains
            ``self.variable``.
        """
        raise NotImplementedError

    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Return the tri-state outcome of checking the constraint under bindings.

        The base implementation is sound for any single-variable leaf: it
        looks up ``self.variable`` in ``bindings``, unwraps a
        ``LiteralExpression`` binding to its raw value, and delegates to
        ``evaluate``. A missing binding for ``self.variable`` or a
        non-literal ``Expression`` binding yields ``UNDECIDED``.
        Identifiers in ``bindings`` that the constraint does not reference
        are ignored. ``EquationConstraint`` overrides this to substitute
        every bound identifier simultaneously.

        Args:
            bindings: Mapping from identifiers to candidate values. Raw
                ``LiteralType`` values and ``Expression`` values are both
                accepted; raw values behave identically to their
                ``LiteralExpression`` wrapping.

        Returns:
            ``SATISFIED``/``VIOLATED`` when decidable under the given
            (possibly partial) bindings; ``UNDECIDED`` otherwise.

        Raises:
            TypeError: Propagated from ``evaluate`` for leaves that reject
                the bound value's type (e.g. an unhashable value against a
                set constraint).
        """
        raise NotImplementedError

    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy the constraint.

        Derived from ``evaluate_with_bindings``; both ``VIOLATED`` and
        ``UNDECIDED`` map to ``False`` (conservative rejection), matching
        ``is_satisfied``.
        """
        raise NotImplementedError


@register_serializable(type_id="equation_constraint")
@dataclass(frozen=True, eq=False)
class EquationConstraint(Constraint):
    # ... existing fields and members unchanged ...

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the expression's free identifiers united with ``variable``."""
        raise NotImplementedError

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Substitute every bound identifier, simplify, and classify.

        Coerces each raw ``LiteralType`` binding value to a
        ``LiteralExpression`` (as ``evaluate`` does), substitutes the full
        multi-key environment through ``simplify_expression``, and reports
        ``SATISFIED`` for the ``bool`` literal ``True``, ``VIOLATED`` for
        any other literal, and ``UNDECIDED`` when no literal results. The
        designated ``variable`` has no special role here; it is bound like
        any other free identifier. Logging on ``UNDECIDED``: DEBUG when
        free identifiers remained unbound (expected partial evaluation),
        WARNING when every free identifier was bound and the simplifier
        still failed (matches ``evaluate``'s anomaly contract).
        """
        raise NotImplementedError


def create_constraint_system(*constraints: Constraint) -> "ConstraintSystem":
    """Create a constraint system from the given constraints.

    Args:
        constraints: Zero or more constraints; identifiers shared between
            constraints denote the same variable.

    Returns:
        A frozen ``ConstraintSystem`` holding the constraints in canonical
        (repr-sorted) order.

    Raises:
        ConstraintError: If any argument is not a ``Constraint``.
    """
    raise NotImplementedError


@register_serializable(type_id="constraint_system")
@dataclass(frozen=True, eq=False)
class ConstraintSystem(WrappedFamilySerializable, FrozenMixin, DerivedEquivalenceMixin):
    """An ordered conjunction of constraints over shared identifiers.

    Semantically the logical AND of its member constraints. Constraints
    are normalized into canonical (repr-sorted) order at construction, so
    structurally equivalent systems built from differently ordered inputs
    are structurally equivalent and serialize identically. Duplicate
    constraints are retained (conjunction is idempotent). Instances are
    frozen; mutation raises ``FrozenMutationError``.
    """

    constraints: tuple[Constraint, ...] = field(metadata=compared_as_value())

    def __post_init__(self) -> None:
        """Validate element types (raise ConstraintError) and canonicalize order."""
        raise NotImplementedError

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the union of every member constraint's free identifiers."""
        raise NotImplementedError

    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        """Return the conjunction outcome of all members under the bindings.

        ``VIOLATED`` if any member is ``VIOLATED`` (a definite violation
        dominates indeterminacy; members are checked in canonical order and
        checking stops at the first violation); ``SATISFIED`` if every
        member is ``SATISFIED``; ``UNDECIDED`` otherwise. The empty system
        is vacuously ``SATISFIED``.
        """
        raise NotImplementedError

    def is_satisfied_with_bindings(self, bindings: ConstraintBindings) -> bool:
        """Return whether the bindings provably satisfy every constraint."""
        raise NotImplementedError

    def convert_to_expression(self) -> Expression:
        """Return the conjunction of every member's expression form.

        Empty system yields ``LiteralExpression(True)``; a single member
        yields that member's expression unwrapped; otherwise a
        ``logical_and`` over members in canonical order.

        Raises:
            ConstraintError: If any member cannot be expressed.
        """
        raise NotImplementedError

    def check_satisfiability(
        self, symbol_types: Mapping[Identifier, SymbolType]
    ) -> ConstraintOutcome:
        """Return whether some joint assignment satisfies every constraint.

        Lowers ``convert_to_expression()`` to the z3 bridge and asks
        whether the conjunction implies ``False``
        (``does_expression_imply``): implication proven -> ``VIOLATED``
        (unsatisfiable); counterexample found -> ``SATISFIED`` (the
        counterexample is a satisfying assignment); solver ``unknown`` ->
        ``UNDECIDED``. The empty system returns ``SATISFIED`` without
        invoking the solver.

        Args:
            symbol_types: Z3 sort for each free identifier of the system.

        Raises:
            KeyError: If ``symbol_types`` lacks an entry for a free
                identifier of the lowered conjunction (propagated from the
                z3 bridge).
            ConstraintError: If a member cannot be converted to an
                expression.
        """
        raise NotImplementedError

    def check_satisfiability_with_bindings(
        self,
        bindings: ConstraintBindings,
        symbol_types: Mapping[Identifier, SymbolType],
    ) -> ConstraintOutcome:
        """Return whether the system is satisfiable given a partial assignment.

        Substitutes the bindings into the conjunction, then decides
        satisfiability of the residual over the remaining free identifiers
        via the z3 bridge. ``symbol_types`` needs entries only for the
        identifiers left free after substitution. Answers questions of the
        form "given x = 4, can y and z still be chosen?".
        """
        raise NotImplementedError

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "ConstraintSystem":
        """Route deserialized fields through the constructor for re-validation."""
        raise NotImplementedError

    @override
    def __repr__(self) -> str: ...
    @override
    def __str__(self) -> str: ...
```

## Semantics and edge cases

Let `x`, `y`, `z` be distinct `Identifier`s.

**Bindings evaluation, `EquationConstraint(x, x + y < 10)`:**
- `evaluate_with_bindings({x: 3, y: 5})` -> `SATISFIED`; `({x: 20, y: 1})` -> `VIOLATED`.
- `evaluate_with_bindings({x: 3})` -> `UNDECIDED`, DEBUG log (unbound `y` remains — expected partial case, not an anomaly).
- `evaluate_with_bindings({})` -> `UNDECIDED` (unless the expression is closed, e.g. `LiteralExpression(True)` -> `SATISFIED`).
- Symbolic bindings may still decide: `EquationConstraint(x, x > y)` under `{x: IdentifierExpression(y) + 1}` simplifies to `True` -> `SATISFIED`.
- All free identifiers bound but no literal after simplification -> `UNDECIDED` with WARNING (same anomaly contract as `evaluate`, tested at `tests/constraint/test_equation_constraint.py:253-286`).
- Extraneous binding keys are ignored. `evaluate(v)` is exactly equivalent to `evaluate_with_bindings({self.variable: v})`; the existing `evaluate` implementation and its always-WARNING contract are left byte-for-byte unchanged.

**Bindings evaluation, set constraints (base default):**
- `InSetConstraint(x, {1, 2, 3})`: `{x: 2}` -> `SATISFIED`; `{y: 2}` -> `UNDECIDED`; `{x: LiteralExpression(2)}` -> `SATISFIED` (literal unwrap; type-strictness preserved because the raw value is stored); `{x: IdentifierExpression(y)}` -> `UNDECIDED`.
- Type strictness unchanged: `InSetConstraint(x, {1})` under `{x: True}` -> `VIOLATED`.
- Unhashable bound value -> `TypeError` propagates (same as `evaluate`).

**ConstraintSystem:**
- `create_constraint_system()` (empty): `evaluate_with_bindings(anything)` -> `SATISFIED`; `check_satisfiability({})` -> `SATISFIED`; `convert_to_expression()` -> `LiteralExpression(True)`.
- Conjunction outcome table: any `VIOLATED` -> `VIOLATED` (dominates `UNDECIDED`); all `SATISFIED` -> `SATISFIED`; otherwise `UNDECIDED`.
- `create_constraint_system(EquationConstraint(x, x < y), EquationConstraint(y, y < z), EquationConstraint(z, z < x)).check_satisfiability({x: INT, y: INT, z: INT})` -> `VIOLATED` (strict cycle is unsat).
- `check_satisfiability_with_bindings({x: 5}, {y: SymbolType.INT})` on `{x < y, y < 3}` -> `VIOLATED`.
- Mixed kinds work: an `InSetConstraint` lowers via its existing `convert_to_expression` (repr-sorted equality disjunction) and conjoins with equation constraints; shared identifiers are *not* rewritten onto a common variable (deliberate contrast with `param/domains.py:_convert_constraints_to_implication_expression` — identity preservation is the whole point).
- Non-`Constraint` element (including a nested `ConstraintSystem`) -> `ConstraintError` at construction. Duplicates retained. Canonical repr-sort order governs equivalence, serialization, conjunction order, and evaluation order.
- String-valued bindings/members flow through the sympy path as today; if they reach the z3 lowering, the bridge's own rejection propagates (no new masking).
- `ConstraintBindings` is read once into an internal dict per call, so caller-side mapping mutation mid-call cannot corrupt evaluation.

**Param interaction (no param/ changes):** the documented recipe is `create_constraint_system(*param_a.constraints, *param_b.constraints, linking_constraint)` where the linking `EquationConstraint` references both params' `variable` identifiers, then `check_satisfiability` with per-variable `SymbolType`s. `Param.is_constraints_satisfied`/`validate_value`/`is_feasible` behavior is untouched.

## Files created / modified / deleted

- **Modified:** `src/fhy_core/constraint.py` — module docstring paragraph on bindings + systems; new imports (`Mapping`, `Sequence` if needed, `does_expression_imply` from `.expression`, `SymbolType` from `.symbol_type` — a new but acyclic leaf edge); `ConstraintBindings` alias; three methods on `Constraint`; two overrides on `EquationConstraint`; `ConstraintSystem`; `create_constraint_system`; `__all__` additions.
- **Created (this phase):** `docs/design/dependent-constraints.md` (this document).
- **Created (test phase):** `tests/constraint/test_bindings_evaluation.py`, `tests/constraint/test_constraint_system.py`; possible small fixture additions to `tests/constraint/conftest.py`.
- **Deleted:** none. **Untouched:** everything under `param/` and `expression/` (parallel worktrees), top-level `__init__.py` (curated namespace unchanged — new names live in `fhy_core.constraint`).

## Serialization impact

- One new pinned type id: `constraint_system` (explicit pin makes the wire identity survive the later `fhy_core.symbolic` move). No existing type ids or wire shapes change.
- Wire shape: `{"__type__": "constraint_system", "__data__": {"constraints": [<wrapped constraint envelope>, ...]}}` with members in canonical repr-sorted order (deterministic, matching the module's existing determinism contract).
- The `tuple[Constraint, ...]` field should be covered by the derived sequence codec over `_SerializableFieldCodec(Constraint)`, with polymorphic leaf resolution via the `WrappedFamilySerializable` envelope; **verify at implementation** — fallback is an explicit `FieldCodec` in the style of `_VALID_VALUES_CODEC` (`constraint.py:392-393`). `construct_from_fields` reruns `__post_init__` validation, matching the set-constraint precedent (`constraint.py:627-630`).
- `ConstraintBindings` is a transient evaluation input and is deliberately not serializable.

## Type-checking (mypy --strict)

- Every new signature fully annotated; `@override` (from `fhy_core.utils.override`) on `EquationConstraint.get_free_identifiers`/`evaluate_with_bindings`, `ConstraintSystem.construct_from_fields`/`__repr__`/`__str__`.
- Base-default `evaluate_with_bindings` reads `self.variable`, typed via the existing `TYPE_CHECKING`-only declaration (`constraint.py:427-433`); no runtime base state added.
- `ConstraintBindings = Mapping[...]` gives callers covariant-friendly input typing; internal copy to `dict[Identifier, Expression]` after literal coercion satisfies `simplify_expression`'s `dict` parameter; `dict(symbol_types)` bridges `Mapping` -> the z3 helpers' declared `dict[Identifier, SymbolType]`.
- `constraints` declared as `tuple[Constraint, ...]`; the variadic `create_constraint_system` factory is the ergonomic entry, so no `Sequence`-declared field or `cast` is needed; `__post_init__` still defensively validates and re-tuples via `object.__setattr__` with explicit `raise ConstraintError` (no asserts).
- No new `Any` beyond the pre-existing `evaluate(value: Any)` contract. `ty` remains advisory; no known ty-specific hazards.

## Test plan outline (test-writing phase; all tests use `mock_identifier`)

- **Unit — bindings on leaves** (`test_bindings_evaluation.py`): base-default paths via both set constraints (parametrized): bound/missing/literal-expression-unwrap/symbolic-expression/extraneous-key/type-strict (`True` vs `1`)/`TypeError` propagation; `EquationConstraint` full/partial/empty/closed-expression/symbolic-deciding bindings; DEBUG-vs-WARNING logging split via `caplog`; `evaluate(v) == evaluate_with_bindings({var: v})` equivalence; `is_satisfied_with_bindings` folding `UNDECIDED` to `False`; `get_free_identifiers` for all three kinds including a variable-absent-from-expression equation.
- **Unit — ConstraintSystem** (`test_constraint_system.py`): factory/empty/element-type rejection (`ConstraintError`, not bare `ValueError`); canonical-order equivalence across construction orders; frozen (`FrozenMutationError`); repr/str; conjunction outcome matrix incl. VIOLATED-dominates-UNDECIDED and stop-at-first-violation observability; `convert_to_expression` empty/singleton/multi/error-propagation; free-identifier union.
- **Serialization**: dict/JSON/binary round-trips via `tests/serialization` conftest helpers; mixed-kind member round-trip; malformed-payload rejection; canonical member order on the wire; empty system round-trip.
- **Integration (`@pytest.mark.z3`)**: `check_satisfiability` satisfiable/unsatisfiable multi-variable systems (incl. the strict-cycle case and mixed set+equation systems); empty-system short-circuit (no solver call); `KeyError` on missing symbol type; `check_satisfiability_with_bindings` deciding both ways after partial substitution. Deliberately no `Param` imports (param/ evolves in a parallel worktree); a param-interop user story lands with the post-merge symbolic feature.
- **Property (`@pytest.mark.property`, plus `z3` where solving)**: for random full bindings, system outcome equals the fold of individual outcomes; for small systems of finite `InSetConstraint`s, brute-force joint satisfiability agrees with `check_satisfiability`.
- **Adversarial**: mapping mutated between calls; NaN members; unhashable binding values; expression with zero free identifiers; duplicate constraints.

## Non-goals

- No new `Constraint` leaf kind, and no n-ary `variables` field on existing leaves.
- No changes to `evaluate`/`is_satisfied`/`__call__` semantics, signatures, or logging.
- No changes to `param/` (feasibility conventions, `Param` API, domains) or `expression/` (z3/sympy bridges used as-is, including their known division/modulo divergences and unbounded solver calls — pre-existing audit findings, out of scope).
- No deduplication or simplification of member constraints inside `ConstraintSystem`; no minimal-unsat-core / model-extraction ("which assignment satisfies it") API.
- No strict `assert_*` satisfiability variant raising `UndecidableError` (callers branch on `UNDECIDED`; can be added later if demanded).
- No serialization of bindings; no top-level namespace additions.

## Open questions

1. Is reusing `ConstraintOutcome` for satisfiability results acceptable, or is a dedicated enum (e.g. `SatisfiabilityOutcome`) preferred despite the duplication?
2. Is the DEBUG (partial bindings) vs WARNING (fully bound, still irreducible) logging split for `evaluate_with_bindings` the desired contract, given `evaluate` always WARNs?
3. Should `check_satisfiability_with_bindings` stay, or is `check_satisfiability` alone minimal enough? (Recommended: keep — it directly answers the user's "we can only pass one value" pain.)
4. Confirm at implementation that the derived codec chain handles `tuple[Constraint, ...]` with abstract element type; otherwise add an explicit field codec.

# APPENDIX: files_to_modify
- src/fhy_core/constraint.py
- docs/design/dependent-constraints.md
- tests/constraint/test_bindings_evaluation.py
- tests/constraint/test_constraint_system.py
- tests/constraint/conftest.py

# APPENDIX: key_decisions
- Add a bindings API (evaluate_with_bindings / is_satisfied_with_bindings / get_free_identifiers) to the existing Constraint base rather than a new constraint kind: a concrete, sound single-variable default on the base serves both set constraints and third-party leaves with zero code; only EquationConstraint overrides it to substitute every bound identifier through simplify_expression.
- Partial bindings never raise: missing or symbolic-undecidable information yields ConstraintOutcome.UNDECIDED, matching the existing tri-state semantics and is_satisfied's conservative rejection; in conjunctions a definite VIOLATED dominates UNDECIDED.
- Introduce ConstraintSystem (frozen, WrappedFamilySerializable, DerivedEquivalenceMixin, pinned type_id 'constraint_system') as the set-level object; it is NOT a Constraint subclass, canonicalizes members by repr-sort at construction (Param precedent) for order-insensitive equivalence and deterministic serialization, and is built via the verb-phrase factory create_constraint_system(*constraints).
- Joint satisfiability lowers to the existing z3 bridge via does_expression_imply(conjunction, False) with identifier identity preserved (no common-variable rewrite, unlike param/domains.py); solver 'unknown' maps to UNDECIDED explicitly instead of the param layer's optimistic convention; check_satisfiability_with_bindings supports partial-assignment queries.
- Existing evaluate/is_satisfied semantics, logging contract, and all serialized wire shapes are unchanged; the entire diff is confined to src/fhy_core/constraint.py plus tests/constraint/, keeping the worktree disjoint from the parallel expression/ and param/ features; ConstraintBindings is deliberately non-serializable.
- New UNDECIDED logging split: DEBUG when free identifiers remain unbound (expected partial evaluation), WARNING only when everything was bound and the simplifier still failed (preserving the anomaly signal evaluate already has).

# APPENDIX: risks
- Derived field-codec inference for the polymorphic tuple[Constraint, ...] field must be verified at implementation; if the generic sequence/_SerializableFieldCodec chain does not resolve abstract Constraint elements through the wrapped envelope, an explicit FieldCodec (like _VALID_VALUES_CODEC) is needed.
- check_satisfiability inherits pre-existing z3 bridge defects: no solver timeout/resource bound and the audited integer-division/floor/modulo semantic divergences (stash-only audit findings F-003/F-013/F-014); systems using those operators can get unsound answers until the bridge is fixed separately.
- constraint.py gains a new import edge to fhy_core.symbol_type (acyclic leaf, verified) and uses does_expression_imply re-exported from fhy_core.expression; the later fhy_core.symbolic reorg must carry both, though the pinned type_id 'constraint_system' protects the wire format across the move.
- Feature 3's parallel param worktree imports constraint.py; this design only adds names and never alters existing behavior, so merges should be clean, but both branches editing constraint.py's import block or __all__ could produce trivial textual conflicts.
- Canonical repr-sorting of ConstraintSystem members depends on constraint __repr__ stability; a future repr change would silently reorder serialized systems (determinism, not correctness, risk).
- Reusing ConstraintOutcome for satisfiability results overloads the enum's meaning (value-check outcome vs existence-of-assignment outcome); flagged as open question #1 for user sign-off before test writing.
# DECISIONS (orchestrator review, final — these override any open question above)
1. Reuse `ConstraintOutcome` for satisfiability results. No second enum.
2. The DEBUG (partial bindings) vs WARNING (fully bound, still irreducible) logging split: approved.
3. `check_satisfiability_with_bindings`: keep.
4. Verify the derived codec for `tuple[Constraint, ...]` at implementation; if inference fails, add an explicit FieldCodec in the `_VALID_VALUES_CODEC` style.
