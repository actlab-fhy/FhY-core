# `fhy_core.constraint`

## Summary

Three constraint kinds — `EquationConstraint`, `InSetConstraint`,
`NotInSetConstraint` — over a single `Identifier` variable. Each
constraint is frozen on construction, serializable through
`WrappedFamilySerializable`, callable as a predicate, structurally
comparable via singledispatch, and convertible to an `Expression`. The
module is the foundation of `fhy_core.param.*`'s parameter-domain
machinery.

## Motivation

The module has 100% test coverage and passes today, but its public
surface is under-documented, has small contract drift (e.g. the
docstring lies about the exception type), and inherits Python set
semantics that conflict with the rest of `fhy_core`'s strict-int
discipline. The hardening pass tightens contracts, makes implicit
behavior explicit, and removes a class of subclass-forgets-to-freeze
bugs by leaning on the new `FrozenMixin.freeze_on_init` mechanism (see
[freeze-on-init.md](docs/design/freeze-on-init.md)).

## Public interface

```python
# src/fhy_core/constraint.py — desired interface

__all__ = [
    "Constraint",
    "ConstraintError",
    "ConstraintMember",         # NEW: re-exported type alias
    "EquationConstraint",
    "InSetConstraint",
    "NotInSetConstraint",
]


class ConstraintError(ValueError):
    """Domain error for constraint construction, validation, and conversion.
    ...
    """


class Constraint(  # CHANGED: freeze_on_init kwargs declared on the base
    WrappedFamilySerializable,
    FrozenMixin,
    StructuralEquivalenceMixin,
    ABC,
    freeze_on_init=True,
    freeze_on_init_deep=True,
):
    """A named-variable predicate.

    Subclasses model the three constraint shapes:
    - ``EquationConstraint``: the value satisfies an arbitrary Boolean
      expression over the variable.
    - ``InSetConstraint``: the value is a member of a permitted set.
    - ``NotInSetConstraint``: the value is not a member of a forbidden
      set.

    Instances are frozen at the end of construction (via
    ``FrozenMixin.freeze_on_init=True``); attribute mutation post-init
    raises ``FrozenMutationError``.

    Subclassing contract:
        - Override ``is_satisfied`` to define the predicate.
        - Override ``convert_to_expression`` to produce an equivalent
          ``Expression`` over the variable.
        - Override ``serialize_data_to_dict`` /
          ``deserialize_data_from_dict`` to enable round-trip
          serialization.
        - Override ``__repr__`` and ``__str__`` so the textual form
          includes the kind and variable.
        - Register a handler with
          ``_is_constraint_structurally_equivalent.register`` —
          unregistered subclasses raise ``NotImplementedError`` when
          compared.   <-- F-003
    """

    @property
    def variable(self) -> Identifier: ...

    def __call__(self, value: Any) -> bool:
        """Alias for ``is_satisfied(value)``."""

    @abstractmethod
    def is_satisfied(self, value: Any) -> bool: ...

    @abstractmethod
    def convert_to_expression(self) -> Expression:
        """Return an expression equivalent to the constraint.

        Returns:
            An ``Expression`` whose free variable is ``self.variable``.

        Raises:
            ConstraintError: If the constraint cannot be expressed.   <-- F-001
        """

    def is_structurally_equivalent(self, other: object) -> bool: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...


ConstraintMember = (
    str | int | float | bool
    | Serializable
    | tuple["ConstraintMember", ...]
    | frozenset["ConstraintMember"]
)
"""Allowed constraint member kinds.   <-- F-008

The four primitive Python types, any ``Serializable+Hashable`` instance,
or a tuple/frozenset of valid members. Members are stored with
type-strict equality: ``int``, ``float``, ``bool`` are not interchangeable,
even at the leaves of nested containers.   <-- F-006
"""


class EquationConstraint(Constraint):
    """Boolean expression predicate over the variable.

    ``is_satisfied(value)`` substitutes ``self.variable = value`` and
    simplifies the resulting expression. Returns True iff the simplified
    expression is ``LiteralExpression(True)``.

    Indeterminate case:
        When the simplifier cannot reduce the substituted expression to a
        ``LiteralExpression(bool=...)`` (e.g. the expression references
        additional free identifiers), ``is_satisfied`` logs a WARNING
        through ``fhy_core.logger`` and returns ``False``. Callers that
        need to distinguish "rejects" from "cannot decide" should
        configure log capture or escalate the warning channel.   <-- F-004
    """

    def __init__(
        self, constrained_variable: Identifier, expression: Expression
    ) -> None: ...

    def is_satisfied(self, value: Expression | LiteralType) -> bool: ...

    def convert_to_expression(self) -> Expression:
        """Return the wrapped expression unchanged."""


class InSetConstraint(Constraint):  # CHANGED: drops Generic[_ConstraintMemberT]   <-- F-010
    """Permitted-set membership predicate.

    ``is_satisfied(value)`` returns True iff ``value`` is in the
    constrained value set, comparing by type-strict equality.

    Members are normalized to a frozen set of type-strict wrappers at
    construction time; the public API still accepts and returns raw
    values.

    Determinism:
        ``convert_to_expression()`` emits leaves in ``repr``-sorted order
        to match ``serialize_data_to_dict``.   <-- F-007
    """

    def __init__(
        self,
        constrained_variable: Identifier,
        valid_values: Collection[ConstraintMember],
    ) -> None: ...

    def is_satisfied(self, value: Any) -> bool:
        """Return whether ``value`` is in the permitted set.

        Raises:
            TypeError: If ``value`` is not hashable.   <-- F-009
        """

    def convert_to_expression(self) -> Expression:
        """Return ``False``, a single ``EQUAL`` comparison, or a
        repr-sorted ``logical_or`` over per-member ``EQUAL`` comparisons.

        Raises:
            ConstraintError: If any member is not a ``LiteralType``.   <-- F-001
        """


class NotInSetConstraint(Constraint):  # CHANGED: drops Generic   <-- F-010
    """Forbidden-set membership predicate. Symmetric to InSetConstraint."""

    def __init__(
        self,
        constrained_variable: Identifier,
        invalid_values: Collection[ConstraintMember],
    ) -> None: ...

    def is_satisfied(self, value: Any) -> bool:
        """Return whether ``value`` is NOT in the forbidden set.

        Raises:
            TypeError: If ``value`` is not hashable.   <-- F-009
        """

    def convert_to_expression(self) -> Expression:
        """Return ``True``, a single ``NOT_EQUAL`` comparison, or a
        repr-sorted ``logical_and`` over per-member ``NOT_EQUAL``
        comparisons.

        Raises:
            ConstraintError: If any member is not a ``LiteralType``.   <-- F-001
        """
```

## Interface delta

| Change | Symbol | Before | After | Breaking? |
| :----- | :----- | :----- | :---- | :-------- |
| Doc fix | `Constraint.convert_to_expression.__doc__` | "Raises: ValueError" | "Raises: ConstraintError" | No (subtype already raised) |
| Auto-freeze | `Constraint` | manual `self.freeze(deep=True)` in three subclass `__init__`s | declared `freeze_on_init=True, freeze_on_init_deep=True` on the base | No (same end state) |
| Dispatch default | `_is_constraint_structurally_equivalent` | returns `False` | raises `NotImplementedError` | Yes (visible to unregistered subclasses) |
| Predicate semantics | `EquationConstraint.is_satisfied` | silently returns `False` on non-literal reduction | logs WARNING, returns `False`; documented as the indeterminate case | No (return value unchanged) |
| Member typing | `InSetConstraint` | `Generic[_ConstraintMemberT]` | non-generic, `Collection[ConstraintMember]` | Minor (mypy-visible only) |
| Member typing | `NotInSetConstraint` | `Generic[_ConstraintMemberT]` | non-generic, `Collection[ConstraintMember]` | Minor (mypy-visible only) |
| Member semantics | `InSetConstraint`, `NotInSetConstraint` | `True == 1`, `1 == 1.0` collapse | type-strict equality; `True`, `1`, `1.0` are distinct members; applies inside nested tuples/frozensets too | Yes (observable behavior change) |
| Determinism | `InSetConstraint.convert_to_expression` | non-deterministic leaf order | repr-sorted leaf order | No (logically equivalent) |
| Determinism | `NotInSetConstraint.convert_to_expression` | non-deterministic leaf order | repr-sorted leaf order | No (logically equivalent) |
| Validation order | `_validate_constraint_member` | nested-members-first, then container hashability | container hashability first, then nested members | No (error attribution flips for combined failures; no current test triggers both) |
| Error attribution | `_normalize_constraint_member_collection` | TypeError catch with generic message | TypeError catch names the offending value | No |
| Repr | `EquationConstraint.__repr__` | `repr(self._expression)` | `"EquationConstraint(<variable>, <expression-repr>)"` | Yes (debug output) |
| Repr | `InSetConstraint.__repr__` | `repr(self._valid_values)` | `"InSetConstraint(<variable>, <values-repr>)"` | Yes (debug output) |
| Repr | `NotInSetConstraint.__repr__` | `repr(self._invalid_values)` | `"NotInSetConstraint(<variable>, <values-repr>)"` | Yes (debug output) |
| Add export | `ConstraintMember` type alias | private | re-exported in `__all__` | No (additive) |
| Remove | `_deserialize_constraint_member` defensive guard | `if not is_serialized_dict(value): raise ... # pragma: no cover` | guard removed; trust the upstream TypeGuard | No (unreachable from public API) |

## Behavior

### Type-strict membership

```python
c = InSetConstraint(x, {1})
c.is_satisfied(True)   # False (previously True)
c.is_satisfied(1.0)    # False (previously True)
c.is_satisfied(1)      # True

c = InSetConstraint(x, {True, 1})  # two distinct members stored
c.is_satisfied(True)   # True
c.is_satisfied(1)      # True
c.is_satisfied(False)  # False
c.is_satisfied(0)      # False

c = InSetConstraint(x, {(True, 1)})
c.is_satisfied((1, 1))    # False  (deep type strictness)
c.is_satisfied((True, 1)) # True
```

Internally, members are wrapped in a private `_TypedMember` that
overrides `__eq__` / `__hash__` to include `type(value)`. Nested
containers are wrapped recursively so equality within tuples and
frozensets is also type-strict.

### Indeterminate `is_satisfied`

```python
# EquationConstraint with a free identifier other than self.variable
x = Identifier("x")
y = Identifier("y")
c = EquationConstraint(x, IdentifierExpression(y))
c.is_satisfied(LiteralExpression(True))
# Emits via fhy_core.logger.get_logger(__name__):
#   WARNING: EquationConstraint.is_satisfied: substituted expression
#   <...> did not reduce to LiteralExpression(bool=...); returning False
# Returns False
```

### Singledispatch default raises

```python
class _Unregistered(Constraint): ...

a = _Unregistered(x); b = _Unregistered(x)
a.is_structurally_equivalent(b)
# raises NotImplementedError(
#   "is_structurally_equivalent is not registered for _Unregistered."
# )
```

### Deterministic conversion

```python
c1 = InSetConstraint(x, [3, 1, 2])
c2 = InSetConstraint(x, [2, 3, 1])
e1 = c1.convert_to_expression()
e2 = c2.convert_to_expression()
e1.is_structurally_equivalent(e2)  # True (was sometimes False before)
```

## Non-goals

The following audit findings are **not** addressed in this round:

- **F-013** (no variable-type validation in `Constraint.__init__`):
  rejected. The FhY style is to trust type hints at internal
  boundaries; adding a runtime `isinstance` check here would be
  off-pattern.

The following are **out of scope** as separate concerns:

- The `FrozenMixin.freeze_on_init` mechanism is its own design — see
  [freeze-on-init.md](docs/design/freeze-on-init.md). The constraint
  hardening *uses* that mechanism but does not own it.
- z3-based fallback for `EquationConstraint.is_satisfied` (the "try
  sympy, then z3, then log" path). The audit's F-004 discussion
  considered this; deferred because z3 requires per-identifier
  `SymbolType` info that the constraint module doesn't currently track.
  Documented as a future enhancement (P2/P3 in the F-004 spec
  discussion).

## Test plan

### Existing tests

The audit's 165 tests across 8 files are an excellent baseline. The
spec changes a handful of them; most stay.

#### Keep as-is

- `test_equation_constraint.py` — all 22 tests stay; predicate behavior
  is unchanged for the cases tested. Add one new test for the warning
  path (see below).
- `test_freezing.py` — both tests stay; mechanism changes but the
  observable contract (`isinstance(c, Frozen)`, `c.is_frozen`,
  `FrozenMutationError` on setattr) does not.
- `test_error_registration.py` — stays.
- `test_serialization.py` — every test stays; serialization contract
  is unchanged. May need a small fixture update if `_TypedMember` wraps
  pass through serialization (they should be transparent — wrappers
  unwrap on `serialize_data_to_dict`).

#### Modify

- `test_set_constraints.py:test_in_set_constraint_collapses_bool_and_int_per_python_set_semantics`
  — **flip polarity**. Rename to
  `test_in_set_constraint_treats_bool_and_int_as_distinct`. Assertions
  flip: `is_satisfied(True) is False` (not True) for
  `InSetConstraint(x, {1})`.
- `test_set_constraints.py` 4-field `_KINDS` — split per F-017 into
  shape-specific lists. Tests update to consume only the fields they
  need.
- `test_set_constraints.py`, `test_member_validation.py`,
  `test_convert_to_expression.py`, `test_structural_equivalence.py`,
  `test_serialization.py`, `test_freezing.py` — replace local `_KINDS`
  lists with shared `SET_KINDS` / `ALL_KINDS` from `conftest.py`
  (per F-016). Local rich-parametrize lists stay where they have
  test-specific data.
- `test_structural_equivalence.py:test_dispatch_default_returns_false_for_unregistered_constraint_subclass`
  — **flip polarity**. Rename to
  `test_dispatch_default_raises_for_unregistered_constraint_subclass`.
  Assertion changes from `is False` to `pytest.raises(NotImplementedError)`.
- `test_equation_constraint.py:test_equation_constraint_returns_false_when_unable_to_reduce_to_literal`
  — keep the assertion (return False), add `caplog` check that the
  WARNING was emitted. Rename to
  `test_equation_constraint_returns_false_and_logs_when_unable_to_reduce`.
- `test_convert_to_expression.py:test_multi_value_set_returns_combinator_of_leaves`
  — currently checks `is_structurally_equivalent` of the produced tree.
  With determinism fix, also assert that two constraints constructed
  with the same value set in different iteration orders produce
  structurally-equivalent expressions.
- All test docstrings with mutation-kill annotations — strip the
  mutation-kill mentions; keep behavioral docstrings (per F-015).
- `tests/constraint/conftest.py` helper class docstrings — replace
  line-number references (`line 138`, `lines 164-165`) with symbol
  references (e.g., "the `Hashable` check in
  `_validate_constraint_member`", per F-015).

#### Delete

None.

### New tests

#### F-006 — Type-strict membership

- `test_in_set_constraint_distinguishes_true_from_one`
- `test_in_set_constraint_distinguishes_one_from_one_float`
- `test_in_set_constraint_with_mixed_bool_and_int_stores_both`
- `test_in_set_constraint_with_nested_tuple_uses_strict_inner_equality` —
  `InSetConstraint(x, {(True, 1)})` does NOT match `(1, 1)`.
- `test_in_set_constraint_with_nested_frozenset_uses_strict_inner_equality`
- Same five tests for `NotInSetConstraint`.

#### F-007 — Determinism

- `test_in_set_constraint_convert_to_expression_is_deterministic` —
  Two constraints constructed with the same members in different
  insertion orders produce structurally-equivalent expressions.
- Same for `NotInSetConstraint`.

#### F-004 — Warning on indeterminate

- `test_equation_constraint_logs_warning_when_substitution_leaves_free_identifier`
- `test_equation_constraint_logs_warning_when_simplifier_does_not_reduce`

#### F-003 — Dispatch default

- `test_unregistered_constraint_subclass_raises_on_structural_equivalence`
  (replaces the deleted "returns False" test).

#### F-001 — `convert_to_expression` exception type

- `test_in_set_constraint_convert_to_expression_raises_constraint_error_for_non_literal_member`
  — already exists as `test_non_literal_member_rejected_by_conversion`;
  ensure it asserts `ConstraintError` specifically, not the general
  `ValueError`.

#### F-012 — Error attribution

- `test_unhashable_post_validation_member_error_names_offending_value` —
  the `ConstraintError` message contains the offending value's `repr`.

#### F-014 — `__repr__` kind marker

- `test_equation_constraint_repr_includes_class_name_and_variable`
- `test_in_set_constraint_repr_includes_class_name_and_variable`
- `test_not_in_set_constraint_repr_includes_class_name_and_variable`

#### F-005 — Validation order

- `test_unhashable_outer_container_rejected_before_recursing_into_nested_members`
  — pass a container whose hash fails AND whose nested member is
  invalid; ensure the outer-container error is raised first.

### Edge cases

- `_TypedMember` round-trips through serialization: the on-disk form
  preserves type info via the registry-wrapped envelope (`__type__:
  builtins.bool` vs. `builtins.int`); on deserialize, the right type
  is recovered. Pin this with a serialization round-trip test for
  `{True, 1, 1.0}`.
- `_TypedMember` does NOT leak into the public surface:
  `constraint._valid_values` may be `frozenset[_TypedMember]`
  internally, but `constraint.serialize_to_dict()`, `str(constraint)`,
  `repr(constraint)`, and `is_satisfied(value)` all unwrap.
- `is_satisfied` for `EquationConstraint` with a value that simplifies
  to a non-bool literal (e.g. `LiteralExpression(1)`) returns False
  *without* logging the warning (the simplifier *did* reduce; the
  result just isn't a bool). The warning is only for non-reduction.

### Adversarial / chaos

- Member sets with NaN: `InSetConstraint(x, {float("nan")})` — the
  existing test pins that a distinct NaN instance does not satisfy;
  keep it. Add `NotInSetConstraint` equivalent.
- Member sets that mix `Serializable+Hashable` instances and
  primitives: round-trip + `is_satisfied` for each.
- Empty constraint sets: `InSetConstraint(x, [])` →
  `convert_to_expression()` returns `LiteralExpression(False)` (kept);
  `is_satisfied(anything)` returns False; deterministic across
  constructions.

## Findings resolution

| F-ID | Severity | Title | Resolution | Notes |
| :--- | :------- | :---- | :--------- | :---- |
| F-001 | High | `convert_to_expression` docstring lies about exception type | **Fix** | Update abstract + subclass docstrings to `ConstraintError`. |
| F-002 | High | `Constraint` abstract `__init__` doesn't freeze | **Fix via FrozenMixin enhancement** | See [freeze-on-init.md](docs/design/freeze-on-init.md). |
| F-003 | Medium | Dispatch default silently returns `False` | **Fix** | Raise `NotImplementedError`, matching `_is_expression_structurally_equivalent`. |
| F-004 | Medium | `EquationConstraint.is_satisfied` overloads `False` | **Fix via P1** | Log WARNING when sympy can't reduce; document in class docstring. No z3 fallback. |
| F-005 | Medium | Member validation checks `Hashable` after recursing | **Fix** | Swap order: outer hashability first, then recurse. |
| F-006 | Medium | Set constraints inherit Python `True == 1` collapse | **Fix — full type strictness** | Wrap members in private `_TypedMember`; deep wrap into nested tuples/frozensets; `int`/`float`/`bool` all distinct. |
| F-007 | Medium | `convert_to_expression` is non-deterministic | **Fix** | Sort `_valid_values` by `repr` before mapping to leaves, matching `serialize_data_to_dict`. |
| F-008 | Medium | Module + class docstrings are placeholder-thin | **Fix** | Rewrite module + 4 class + key method docstrings per Google style. |
| F-009 | Low | `is_satisfied` raises `TypeError` undocumented | **Fix (per-subclass)** | Per-subclass `Raises:` on set-constraint `is_satisfied` only. |
| F-010 | Low | `Generic[_ConstraintMemberT]` not load-bearing | **Fix** | Drop `Generic`; type members as `Collection[ConstraintMember]`. |
| F-011 | Low | Stacked `cast(...)` in `__init__` | **Closed (subsumed by F-010)** | Casts disappear when `Generic` is dropped. |
| F-012 | Low | `TypeError` catch loses offending value | **Fix** | Per-item hash check; error message names the offending value. |
| F-013 | Low | No variable-type validation in `__init__` | **Reject** | FhY trusts type hints at internal boundaries; adding an `isinstance` check here is off-pattern. |
| F-014 | Low | Abstract `__repr__` with no kind marker | **Fix** | Each subclass's `__repr__` includes class name and variable. |
| F-015 | Low | Mutation-kill annotations clutter test docstrings | **Fix** | Delete mutation-kill notes (trust the cosmic-ray report); convert conftest line-number refs to symbol refs. |
| F-016 | Low | Duplicated `_KINDS` parametrize across test files | **Fix** | Centralize `SET_KINDS` + `ALL_KINDS` in `tests/constraint/conftest.py`. |
| F-017 | Low | 4-field `_KINDS` carries unused fields | **Fix** | Split per-shape: factory-only / outcomes / str-marker. |
| F-018 | Low | `# pragma: no cover` on a defensive branch | **Fix** | Remove the defensive `is_serialized_dict` check; trust the upstream TypeGuard. |
