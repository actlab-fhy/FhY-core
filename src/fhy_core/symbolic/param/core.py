"""Constrained parameters built by composing a value domain.

A :class:`Param` pairs a variable identifier and a
:class:`~fhy_core.symbolic.constraint.ConstraintSystem` with a
:class:`~fhy_core.symbolic.param.domains.ParamDomain` that supplies all kind-specific
behavior. There is a single concrete ``Param`` class; the common kinds are built
through the ``create_*`` factory functions.

A parameter serializes through the schema-derived ``Serializable`` engine; the
``domain`` field carries a wrapped family envelope identifying the concrete
domain.
"""

import operator
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    FieldCodec,
    Serializable,
    _SerializableFieldCodec,
    deserialize_registry_wrapped_value,
    make_field_codec,
    register_serializable,
    serialize_registry_wrapped_value,
)
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintError,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.term import (
    DerivedEquivalenceMixin,
    compared_as_binder,
    compared_as_value,
)
from fhy_core.traits import FrozenMixin
from fhy_core.utils.override import override

from .domains import (
    IntegerDomain,
    IntervalIntegerDomain,
    IntervalProfile,
    ParamDomain,
    RealDomain,
    build_categorical_domain,
    build_ordinal_domain,
    build_permutation_domain,
    evaluate_system_outcome,
    is_bound_expression,
)
from .values import (
    CategoricalValue,
    OrdinalValue,
    ParamError,
    PermutationMemberValue,
    SerializableEqualValue,
    SerializableOrderableValue,
    _CategoricalValueT,
    _OrdinalValueT,
    _PermutationMemberValueT,
)

__all__ = [
    "CategoricalValue",
    "OrdinalValue",
    "Param",
    "ParamAssignment",
    "ParamError",
    "PermutationMemberValue",
    "SerializableEqualValue",
    "SerializableOrderableValue",
    "create_categorical_param",
    "create_integer_param",
    "create_integer_param_between",
    "create_integer_param_with_lower_bound",
    "create_integer_param_with_upper_bound",
    "create_intersection_param",
    "create_interval_integer_param",
    "create_interval_integer_param_between",
    "create_interval_integer_param_exactly",
    "create_interval_integer_param_with_lower_bound",
    "create_interval_integer_param_with_upper_bound",
    "create_interval_natural_param",
    "create_natural_param",
    "create_ordinal_param",
    "create_permutation_param",
    "create_real_param",
    "create_real_param_between",
    "create_real_param_with_lower_bound",
    "create_real_param_with_upper_bound",
    "create_single_valid_value_param",
    "create_union_param",
]

_T = TypeVar("_T")


_WRAPPED_VALUE_CODEC: FieldCodec = make_field_codec(
    serialize_registry_wrapped_value, deserialize_registry_wrapped_value
)

# ``constraint_system`` is annotated ``ConstraintSystem``, a concrete
# ``Serializable`` the engine could infer automatically, but the codec is
# supplied explicitly (mirroring ``_PARAM_CODEC`` below) so the wire shape is
# declared in this module rather than left to inference.
_CONSTRAINT_SYSTEM_CODEC: FieldCodec = _SerializableFieldCodec(ConstraintSystem)


# ---------------------------------------------------------------------------
# Param container
# ---------------------------------------------------------------------------


@register_serializable(type_id="param")
@dataclass(frozen=True, eq=False)
class Param(Serializable, FrozenMixin, DerivedEquivalenceMixin, Generic[_T]):
    """A constrained parameter that composes a value domain.

    A parameter is defined by its variable, a set of constraints, and a
    :class:`~fhy_core.symbolic.param.domains.ParamDomain` that supplies admissibility,
    subset, set algebra, equivalence, and serialization behavior. Construct one
    directly with a domain, or use a ``create_*`` factory for the common kinds.

    The ``Param[_T]`` type parameter is an advisory hint for call-site inference
    only; the admissible value type is enforced by the domain at runtime, not by
    ``_T``.

    Two parameters are structurally equivalent when they have the same variable,
    domain, and constraints; under alpha comparison they are equivalent up to
    renaming of the bound variable. Construction validates and de-duplicates the
    constraints, appends the domain's implied constraints, and stores the result
    as a :class:`~fhy_core.symbolic.constraint.ConstraintSystem`, which orders its
    members canonically by construction, so that constraint-set order does not
    affect equivalence.
    """

    domain: ParamDomain
    variable: Identifier = field(
        default_factory=lambda: Identifier("param"),
        metadata=compared_as_binder(scopes_over=("constraint_system",)),
    )
    constraint_system: ConstraintSystem = field(
        default_factory=create_constraint_system,
        metadata={"serialize_codec": _CONSTRAINT_SYSTEM_CODEC},
    )

    def __post_init__(self) -> None:
        canonical = self._build_canonical_constraints(
            self.constraint_system.constraints
        )
        object.__setattr__(
            self, "constraint_system", create_constraint_system(*canonical)
        )

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return this parameter's constraints in canonical order."""
        return self.constraint_system.constraints

    def _build_canonical_constraints(
        self, constraints: Sequence[Constraint]
    ) -> tuple[Constraint, ...]:
        accumulated = self._validate_and_deduplicate_constraints(constraints)
        for implied in self.domain.get_implied_constraints(self.variable):
            if not any(
                existing.is_structurally_equivalent(implied) for existing in accumulated
            ):
                accumulated = (*accumulated, implied)
        return accumulated

    def _validate_and_deduplicate_constraints(
        self, constraints: Sequence[Constraint]
    ) -> tuple[Constraint, ...]:
        accumulated: list[Constraint] = []
        for constraint in constraints:
            self.validate_constraint(constraint)
            if any(
                existing.is_structurally_equivalent(constraint)
                for existing in accumulated
            ):
                continue
            accumulated.append(constraint)
        return tuple(accumulated)

    @property
    def variable_expression(self) -> IdentifierExpression:
        """Return the parameter's variable as an identifier expression."""
        return IdentifierExpression(self.variable)

    @property
    def symbol_type(self) -> SymbolType | None:
        """Return the domain's numeric symbol type, or ``None`` if non-numeric."""
        return self.domain.symbol_type

    def replace_constraints(self, constraints: Sequence[Constraint]) -> "Param[_T]":
        """Return a copy of this parameter with its constraints replaced.

        The domain and variable are preserved; only the constraint set is
        replaced, validated, de-duplicated by structural equivalence, and
        re-canonicalized.

        Args:
            constraints: Constraints for the returned parameter.

        Returns:
            A new parameter with the same domain and variable.

        """
        return Param(
            self.domain,
            variable=self.variable,
            constraint_system=create_constraint_system(*constraints),
        )

    def is_value_valid(
        self, value: Any, *, bindings: ConstraintBindings | None = None
    ) -> bool:
        """Return whether a value is admissible and satisfies all constraints.

        Conservative about indeterminacy: a constraint the checker cannot
        decide counts as not satisfied, so a ``False`` result means
        "not proven valid" rather than "proven invalid"; use
        ``validate_value`` to tell the two cases apart from the message it
        raises. A dependent constraint checked without the bindings it
        needs is undecided for every value, so this reports ``False`` for
        every value until those bindings are supplied. ``is_feasible`` and
        ``is_subset`` share this polarity: a ``True`` from any of them is a
        proof.

        Args:
            value: Candidate value for this parameter's own variable.
            bindings: Values for identifiers a dependent constraint
                references besides this parameter's own variable.

        Returns:
            Whether the value is admissible and every constraint is
            provably satisfied.

        Raises:
            ParamError: If ``bindings`` supplies an entry for this
                parameter's own variable.
            ConstraintError: If a value bound to an identifier a
                constraint references -- ``value`` itself or a
                ``bindings`` entry -- cannot be lifted into the
                substitution environment. Checked only for an admissible
                value.
            NonBooleanLogicalOperandError: If a constraint holds a
                provably numeric operand in a Boolean position -- under
                a logical connective or as a piecewise case condition --
                counting ``value`` or a ``bindings`` entry that puts a
                number there. Such a constraint is ill-typed rather than
                undecided, so it raises instead of reporting ``False``.
                Checked only for an admissible value.

        """
        self._validate_bindings(bindings)
        return self.is_value_admissible(value) and self.is_constraints_satisfied(
            value, bindings=bindings
        )

    def is_value_admissible(self, value: Any) -> bool:
        """Return whether a value lies in this parameter's value domain."""
        return self.domain.is_value_admissible(value)

    def is_constraints_satisfied(
        self, value: Any, *, bindings: ConstraintBindings | None = None
    ) -> bool:
        """Return whether the value provably satisfies all constraints.

        Conservative about indeterminacy in the same way as
        ``is_value_valid``: only a ``SATISFIED`` outcome from the whole
        constraint system reports ``True``.

        Args:
            value: Candidate value for this parameter's own variable.
            bindings: Values for identifiers a dependent constraint
                references besides this parameter's own variable.

        Returns:
            Whether every constraint is provably satisfied.

        Raises:
            ParamError: If ``bindings`` supplies an entry for this
                parameter's own variable.
            ConstraintError: If a value bound to an identifier a
                constraint references -- ``value`` itself or a
                ``bindings`` entry -- cannot be lifted into the
                substitution environment, for any value.
            NonBooleanLogicalOperandError: As :meth:`is_value_valid`
                raises it, for any value.

        """
        normalized = self.domain.normalize_value(value)
        environment = self._build_environment(normalized, bindings)
        return self._find_failing_constraint(environment)[0]

    def _validate_bindings(self, bindings: ConstraintBindings | None) -> None:
        """Raise if ``bindings`` supplies an entry for this parameter's own variable.

        Raises:
            ParamError: If ``bindings`` supplies an entry for this
                parameter's own variable; passing it as both ``value`` and
                ``bindings`` is an ambiguous call.

        """
        if bindings is not None and self.variable in bindings:
            raise ParamError(
                f"bindings must not include this parameter's own variable "
                f"{self.variable!r}; its value is already supplied as `value`."
            )

    def _build_environment(
        self, value: Any, bindings: ConstraintBindings | None
    ) -> ConstraintBindings:
        """Merge ``value`` for this parameter's own variable with ``bindings``.

        Raises:
            ParamError: If ``bindings`` supplies an entry for this
                parameter's own variable; passing it as both ``value`` and
                ``bindings`` is an ambiguous call.

        """
        self._validate_bindings(bindings)
        environment: dict[Identifier, Any] = {self.variable: value}
        if bindings is not None:
            environment.update(bindings)
        return environment

    def _find_failing_constraint(
        self, environment: ConstraintBindings
    ) -> tuple[bool, Constraint | None, ConstraintOutcome | None]:
        """Return the constraint accounting for a non-satisfied outcome.

        The outcome is the constraint system's own: a definite violation
        dominates indeterminacy, so a provably violated constraint is
        reported even when an earlier constraint in canonical order is
        merely undecided. The per-constraint scan then names the member
        the system's outcome came from, preferring the violated one.

        Returns:
            A ``(is_satisfied, constraint, outcome)`` triple. When the
            system reports ``SATISFIED``, this is ``(True, None, None)``.
            Otherwise ``is_satisfied`` is ``False``, ``outcome`` is the
            system's ``ConstraintOutcome`` (``VIOLATED`` or
            ``UNDECIDED``), and ``constraint`` is a member exhibiting it.

        """
        system_outcome = evaluate_system_outcome(self.constraint_system, environment)
        if system_outcome is ConstraintOutcome.SATISFIED:
            return True, None, None
        fallback: Constraint | None = None
        for constraint in self.constraints:
            outcome = evaluate_system_outcome(
                create_constraint_system(constraint), environment
            )
            if outcome is system_outcome:
                return False, constraint, system_outcome
            if outcome is not ConstraintOutcome.SATISFIED and fallback is None:
                fallback = constraint
        return False, fallback, system_outcome

    def validate_value(
        self, value: Any, *, bindings: ConstraintBindings | None = None
    ) -> None:
        """Raise if ``value`` is not a valid assignment for this parameter.

        Args:
            value: Candidate value for this parameter's own variable.
            bindings: Values for identifiers a dependent constraint
                references besides this parameter's own variable.

        Raises:
            ParamError: If ``bindings`` supplies an entry for this
                parameter's own variable, if the value is not admissible,
                if the value violates a constraint, or if a constraint
                could not be verified. The bindings check runs first, so
                a caller error is reported whatever the value is.
            ConstraintError: If a value bound to an identifier a
                constraint references -- ``value`` itself or a
                ``bindings`` entry -- cannot be lifted into the
                substitution environment: a binding outside
                ``Expression | LiteralType``, or a literal value
                ``LiteralExpression`` refuses, such as a ``str`` outside
                the integer and float grammars. Such a value is a caller
                error rather than an unverifiable constraint, so it is
                not reported as a ``ParamError``. Checked only for an
                admissible value, after the bindings check.
            NonBooleanLogicalOperandError: If a constraint holds a
                provably numeric operand in a Boolean position -- under
                a logical connective or as a piecewise case condition --
                counting ``value`` or a ``bindings`` entry that puts a
                number there. Such a constraint is ill-typed rather than
                unverifiable, so it is not reported as a ``ParamError``.
                Checked only for an admissible value.

        """
        self._validate_bindings(bindings)
        if not self.is_value_admissible(value):
            raise ParamError(
                f"Value {value!r} is not admissible for parameter {self!r}."
            )
        normalized = self.domain.normalize_value(value)
        environment = self._build_environment(normalized, bindings)
        _, failing_constraint, outcome = self._find_failing_constraint(environment)
        if failing_constraint is None:
            return
        if outcome is ConstraintOutcome.UNDECIDED:
            raise ParamError(
                f"Value {value!r} could not be verified against constraint "
                f"{failing_constraint!r} for parameter {self!r}."
            )
        raise ParamError(
            f"Value {value!r} violates constraint {failing_constraint!r} "
            f"for parameter {self!r}."
        )

    def is_value_set_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's value set is a subset of ``other``'s."""
        return self.domain.is_value_set_subset(other.domain)

    def check_subset(self, other: "Param[_T]") -> ConstraintOutcome:
        """Decide whether this parameter's feasible set is a subset of ``other``'s.

        Comparison is gated on value space: numeric parameters compare only with
        numeric parameters sharing the same numeric symbol type (integers with
        integers, reals with reals), and finite-set parameters compare only
        within their own family. Cross-space and cross-family queries decide
        ``VIOLATED``.

        A finite-set parameter (ordinal, categorical, permutation)
        enumerates its domain and always decides; the solver, screening,
        and ``UNDECIDED`` below apply only to numeric parameters.

        A numeric parameter whose admissible values an ``InSetConstraint``
        makes finite is decided by evaluating each candidate on both sides
        with it bound. When this parameter is the finite one, a candidate
        decided into it and decided out of ``other`` is a counterexample
        (``VIOLATED``), ``other`` deciding every candidate not decided out
        of this parameter proves the relation (``SATISFIED``), and
        anything else, such as a candidate a dependent constraint leaves
        undecided, reports ``UNDECIDED`` (logged at ``WARNING``). When only
        ``other`` is finite, the relation is ``VIOLATED`` if this parameter
        provably admits a value outside ``other``'s candidates and
        otherwise goes to the solver.

        Otherwise the relation goes to the solver. A constraint reaching
        outside either parameter's own variable is dropped before the
        question is posed (logged at ``WARNING``), which only widens that
        side, so a decided answer is kept exactly when the weakened
        question still proves it: ``SATISFIED`` when ``other``'s side is
        exact, ``VIOLATED`` when this parameter's side is exact. A
        ``VIOLATED`` resting on a counterexample this parameter's dropped
        constraints might forbid, and a ``SATISFIED`` into a consequent
        ``other``'s dropped constraints might narrow, report ``UNDECIDED``
        (logged at ``WARNING``), as does a solver that gives up.

        Returns:
            ``SATISFIED`` when the subset relation is decided to hold,
            ``VIOLATED`` when a counterexample is decided, and
            ``UNDECIDED`` when neither the solver nor the enumeration
            could decide, or the solver decided only a weakened question.

        Raises:
            NonBooleanLogicalOperandError: If a constraint of either
                parameter that the query evaluates holds a provably
                numeric operand in a Boolean position -- under a logical
                connective or as a piecewise case condition -- counting
                an in-set candidate bound to its variable, or the variable
                itself, which a numeric domain declares INT or REAL to the
                solver. Such a constraint is ill-typed rather than
                undecided, so it raises instead of reporting ``UNDECIDED``.

        """
        return self.domain.compute_feasibility_subset(
            self.constraints,
            self.variable,
            other.domain,
            other.constraints,
            other.variable,
        )

    def is_subset(self, other: "Param[_T]") -> bool:
        """Return whether this parameter's feasible set is proven within ``other``'s.

        A conservative wrapper over :meth:`check_subset`: only a
        ``SATISFIED`` outcome reports ``True``, so a ``True`` result means
        the subset relation is proven. ``False`` covers both a decided
        counterexample and an ``UNDECIDED`` relation; call
        :meth:`check_subset` to tell them apart.

        Raises:
            NonBooleanLogicalOperandError: As :meth:`check_subset`
                raises it.

        """
        return self.check_subset(other) is ConstraintOutcome.SATISFIED

    def check_feasibility(self) -> ConstraintOutcome:
        """Decide whether some value satisfies the domain and all constraints.

        The constraints already include the domain's implied constraints, so the
        domain reasons only about the constraints it is given.

        A finite-set parameter (ordinal, categorical, permutation)
        enumerates its domain and always decides; the solver, screening,
        and ``UNDECIDED`` below apply only to numeric parameters.

        A numeric parameter whose admissible values an ``InSetConstraint``
        makes finite is decided by evaluating each candidate against the
        constraints with it bound: one decided to satisfy them reports
        ``SATISFIED``, all decided to violate them report ``VIOLATED``, and
        otherwise, when a dependent constraint leaves a candidate undecided
        and none is decided feasible, ``UNDECIDED`` (logged at
        ``WARNING``). Otherwise the question goes to the solver. A
        constraint reaching outside this parameter's own variable is
        dropped before the question is posed (logged at ``WARNING``),
        which only widens the admissible set: a ``VIOLATED`` answer to
        the weakened question stands, while a ``SATISFIED`` one names a
        value the dropped constraint might forbid and is reported
        ``UNDECIDED`` (logged at ``WARNING``). A solver that cannot decide
        satisfiability reports ``UNDECIDED``.

        Returns:
            ``SATISFIED`` when a satisfying value is decided to exist,
            ``VIOLATED`` when none can exist, and ``UNDECIDED`` when
            neither the solver nor the enumeration could decide, or the
            solver decided only a weakened question.

        Raises:
            NonBooleanLogicalOperandError: If a constraint the query
                evaluates holds a provably numeric operand in a Boolean
                position -- under a logical connective or as a piecewise
                case condition -- counting an in-set candidate bound to
                this parameter's variable, or the variable itself, which
                a numeric domain declares INT or REAL to the solver. Such
                a constraint is ill-typed rather than undecided, so it
                raises instead of reporting ``UNDECIDED``.

        """
        return self.domain.has_feasible_value(self.constraints, self.variable)

    def is_feasible(self) -> bool:
        """Return whether this parameter is proven to admit some value.

        A conservative wrapper over :meth:`check_feasibility`: only a
        ``SATISFIED`` outcome reports ``True``, so a ``True`` result means
        a value satisfying the domain and every constraint is proven to
        exist. ``False`` covers both a parameter proven empty and one whose
        feasibility is ``UNDECIDED``, so it is not the claim
        :meth:`is_empty` makes; call :meth:`check_feasibility` to tell the
        two apart.

        Raises:
            NonBooleanLogicalOperandError: As :meth:`check_feasibility`
                raises it.

        """
        return self.check_feasibility() is ConstraintOutcome.SATISFIED

    def is_empty(self) -> bool:
        """Return whether this parameter is proven to admit no value.

        A conservative wrapper over :meth:`check_feasibility`: only a
        ``VIOLATED`` outcome reports ``True``, so a ``True`` result means
        no value can satisfy the domain and every constraint. It is not the
        complement of :meth:`is_feasible`: an ``UNDECIDED`` outcome proves
        neither that a value exists nor that none does, so both report
        ``False``. Call :meth:`check_feasibility` to tell an undecided
        parameter apart from one proven feasible.

        Raises:
            NonBooleanLogicalOperandError: As :meth:`check_feasibility`
                raises it.

        """
        return self.check_feasibility() is ConstraintOutcome.VIOLATED

    def assign(
        self, value: _T, *, bindings: ConstraintBindings | None = None
    ) -> "ParamAssignment[_T]":
        """Assign a value to the parameter, returning a parameter assignment.

        Args:
            value: Value to assign; normalized by the domain before binding.
            bindings: Values for identifiers a dependent constraint
                references besides this parameter's own variable.

        Returns:
            A parameter assignment with the normalized value.

        Raises:
            ParamError: If ``bindings`` supplies an entry for this
                parameter's own variable, if the value is not admissible,
                if the value violates a constraint, or if a constraint
                could not be verified.
            ConstraintError: As :meth:`validate_value` raises it.
            NonBooleanLogicalOperandError: As :meth:`validate_value`
                raises it.

        """
        self.validate_value(value, bindings=bindings)
        normalized = cast(_T, self.domain.normalize_value(value))
        return _construct_unchecked_assignment(self, normalized)

    def add_constraint(self, constraint: Constraint) -> "Param[_T]":
        """Return a new parameter with an additional constraint.

        Structurally-equivalent duplicates are dropped, returning ``self``
        unchanged when the constraint is already present.
        """
        self.validate_constraint(constraint)
        if any(
            existing.is_structurally_equivalent(constraint)
            for existing in self.constraints
        ):
            return self
        return self.replace_constraints((*self.constraints, constraint))

    def add_constraints(self, constraints: Collection[Constraint]) -> "Param[_T]":
        """Return a new parameter with multiple additional constraints."""
        result = self
        for constraint in constraints:
            result = result.add_constraint(constraint)
        return result

    def validate_constraint(self, constraint: Constraint) -> None:
        """Validate whether a constraint can be added to this parameter.

        A constraint attaches exactly when this parameter's variable is a
        member of the constraint's scope (``get_free_identifiers()``); a
        dependent constraint whose scope also reaches other identifiers is
        accepted, while a ground or foreign-only-scope constraint is not.

        Raises:
            ParamError: If this parameter's variable is not in the
                constraint's scope, or the domain rejects the constraint.
            TypeError: If the domain forbids the constraint's type.

        """
        if self.variable not in constraint.get_free_identifiers():
            raise ParamError(
                f"Constraint scope must include the parameter's variable "
                f"{self.variable!r}, but got constraint {constraint!r} with "
                f"scope {constraint.get_free_identifiers()!r}."
            )
        self.domain.validate_constraint(constraint, self.variable)

    def add_lower_bound_constraint(
        self, lower_bound: int | float | str, *, is_inclusive: bool = True
    ) -> "Param[_T]":
        """Return a new parameter with an added lower-bound constraint."""
        _validate_natural_bound(
            self.domain, lower_bound, is_lower=True, is_inclusive=is_inclusive
        )
        return self.add_constraint(
            _create_bound_constraint(
                self.variable, lower_bound, is_lower=True, is_inclusive=is_inclusive
            )
        )

    def add_upper_bound_constraint(
        self, upper_bound: int | float | str, *, is_inclusive: bool = True
    ) -> "Param[_T]":
        """Return a new parameter with an added upper-bound constraint."""
        _validate_natural_bound(
            self.domain, upper_bound, is_lower=False, is_inclusive=is_inclusive
        )
        return self.add_constraint(
            _create_bound_constraint(
                self.variable, upper_bound, is_lower=False, is_inclusive=is_inclusive
            )
        )

    # -- interval arithmetic ------------------------------------------------
    #
    # Operands are told apart by their domain's interval profile, never by
    # their domain's kind. An operand whose profile admits only bounds takes
    # part as it stands; one whose profile admits other constraints is
    # recast against such a partner (see ``_coerce_to_interval_param``).
    #
    # Every result is a fresh parameter over a fresh variable: a derived
    # interval denotes its own quantity, not either operand's, so sharing an
    # operand's identifier would conflate the two wherever both reach one
    # constraint system.

    def _coerce_interval_operand(self, other: Any) -> "Param[int] | None":
        # Both operands are the single ``Param`` type, and Python skips the
        # reflected dunder when operands share a type. A non-interval ``self``
        # must therefore coerce and handle ``non_interval OP interval`` itself
        # rather than relying on the interval operand's reflected method.
        if not isinstance(other, Param):
            return None
        other_profile = _get_interval_operand_profile(other)
        if other_profile is None:
            return None
        return _coerce_to_interval_param(other_profile, self)

    def __add__(self, other: Any) -> "Param[int]":
        profile = _get_interval_operand_profile(self)
        if profile is None:
            coerced_self = self._coerce_interval_operand(other)
            if coerced_self is None:
                return NotImplemented
            return coerced_self.__add__(other)
        coerced = _coerce_to_interval_param(profile, other)
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        other_min, other_max = _get_effective_min_max(
            coerced.constraints, coerced.variable
        )
        new_min = _combine_optional_bounds(self_min, other_min, operator.add)
        new_max = _combine_optional_bounds(self_max, other_max, operator.add)
        return _create_class_preserved_interval_param(
            _require_interval_operand_profile(coerced),
            new_min,
            new_max,
            profile,
            zero_included=profile.zero_included,
        )

    def __radd__(self, other: Any) -> "Param[int]":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Param[int]":
        profile = _get_interval_operand_profile(self)
        if profile is None:
            coerced_self = self._coerce_interval_operand(other)
            if coerced_self is None:
                return NotImplemented
            return coerced_self.__sub__(other)
        coerced = _coerce_to_interval_param(profile, other)
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        other_min, other_max = _get_effective_min_max(
            coerced.constraints, coerced.variable
        )
        new_min = _combine_optional_bounds(self_min, other_max, operator.sub)
        new_max = _combine_optional_bounds(self_max, other_min, operator.sub)
        return _create_widened_interval_param(new_min, new_max, profile)

    def __rsub__(self, other: Any) -> "Param[int]":
        profile = _get_interval_operand_profile(self)
        if profile is None:
            return NotImplemented
        return _coerce_to_interval_param(profile, other).__sub__(self)

    def __mul__(self, other: Any) -> "Param[int]":
        profile = _get_interval_operand_profile(self)
        if profile is None:
            coerced_self = self._coerce_interval_operand(other)
            if coerced_self is None:
                return NotImplemented
            return coerced_self.__mul__(other)
        coerced = _coerce_to_interval_param(profile, other)
        coerced_profile = _require_interval_operand_profile(coerced)
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        other_min, other_max = _get_effective_min_max(
            coerced.constraints, coerced.variable
        )
        new_min, new_max = _multiply_optional_bounds(
            self_min, self_max, other_min, other_max
        )
        # A product reaches zero as soon as *either* operand admits zero
        # (``x > 0`` times ``y >= 0`` admits ``0``), unlike a sum, which needs
        # both. So the result admits zero whenever either operand does.
        return _create_class_preserved_interval_param(
            coerced_profile,
            new_min,
            new_max,
            profile,
            zero_included=profile.zero_included or coerced_profile.zero_included,
        )

    def __rmul__(self, other: Any) -> "Param[int]":
        return self.__mul__(other)

    def __neg__(self) -> "Param[int]":
        profile = _require_interval_operand_profile(self)
        self_min, self_max = _get_effective_min_max(self.constraints, self.variable)
        new_min = None if self_max is None else -self_max
        new_max = None if self_min is None else -self_min
        return _create_widened_interval_param(new_min, new_max, profile)

    # -- set algebra --------------------------------------------------------

    def __or__(self, other: "Param[_T]") -> "Param[_T]":
        if not isinstance(other, Param):
            return NotImplemented
        return create_union_param(self, other)

    def __and__(self, other: "Param[_T]") -> "Param[_T]":
        if not isinstance(other, Param):
            return NotImplemented
        return create_intersection_param(self, other)

    @override
    def __repr__(self) -> str:
        set_repr = self.domain.render_set_repr()
        if set_repr:
            set_repr = f"{set_repr}, "
        return (
            f"{type(self).__name__}({self.variable!r}, {set_repr}"
            f"constraints={self.constraints!r})"
        )

    @override
    def __str__(self) -> str:
        land = " /\\ "
        return (
            "{" + f"{self.variable} in {self.domain.render_set_string()} | "
            f"{land.join(str(c) for c in self.constraints)}"
            "}"
        )


# ---------------------------------------------------------------------------
# Parameter assignment
# ---------------------------------------------------------------------------


# The ``param`` field is annotated ``Param[_T]``, a parameterized generic the
# engine cannot resolve to the ``Param`` class for codec inference, so supply the
# serializable-class codec explicitly.
_PARAM_CODEC: FieldCodec = _SerializableFieldCodec(Param)


def _construct_unchecked_assignment(
    param: "Param[_T]", value: _T
) -> "ParamAssignment[_T]":
    """Build a ``ParamAssignment`` for a value the caller already validated.

    ``ParamAssignment.__post_init__`` re-validates through
    ``Param.validate_value`` with no ``bindings``, so it cannot see
    bindings a caller already used to prove a dependent constraint
    satisfied. ``Param.assign`` validates with the caller's bindings first,
    then builds the assignment through this bypass instead of the
    bindings-blind constructor path, using the same manual-construction
    pattern (``cls.__new__``, direct attribute assignment, an explicit
    ``freeze()`` call) ``FrozenMixin`` documents for deserialization.
    ``ParamAssignment`` is a native frozen dataclass, so ``is_frozen``
    reads ``True`` as soon as ``__new__`` runs the mixin's one-time class
    setup; field assignment must therefore go through
    ``object.__setattr__`` even here, exactly as the dataclass-generated
    ``__init__`` this bypasses would do internally.

    """
    assignment: ParamAssignment[_T] = ParamAssignment.__new__(ParamAssignment)
    object.__setattr__(assignment, "param", param)
    object.__setattr__(assignment, "value", value)
    assignment.freeze()
    return assignment


def _raise_if_value_provably_invalid(param: "Param[_T]", value: _T) -> None:
    """Raise if the parameter provably cannot hold ``value``.

    The deserialization-side counterpart of ``Param.validate_value``: it
    rejects an inadmissible value and any constraint outcome that is
    ``VIOLATED``, but accepts ``UNDECIDED``. A dependent constraint is
    undecidable from the assignment's own state -- the bindings that proved
    it satisfied at ``Param.assign`` time are not part of the serialized
    payload -- so absence of a provable violation is accepted rather than
    demanding a proof of satisfaction that cannot exist here.

    Raises:
        ParamError: If ``value`` is not admissible in the parameter's
            domain, or a constraint provably rejects it.
        ConstraintError: If ``value``, bound to the parameter's
            variable, cannot be lifted into the substitution environment.
        NonBooleanLogicalOperandError: If a constraint holds a provably
            numeric operand in a Boolean position -- under a logical
            connective or as a piecewise case condition -- counting
            ``value`` bound to the parameter's variable. Such a
            constraint is ill-typed rather than undecided, so it is
            refused rather than accepted as an undecided remainder.

    """
    if not param.is_value_admissible(value):
        raise ParamError(f"Value {value!r} is not admissible for parameter {param!r}.")
    environment: dict[Identifier, Any] = {
        param.variable: param.domain.normalize_value(value)
    }
    for constraint in param.constraints:
        if constraint.evaluate_with_bindings(environment) is ConstraintOutcome.VIOLATED:
            raise ParamError(
                f"Value {value!r} violates constraint {constraint!r} "
                f"for parameter {param!r}."
            )


@register_serializable(type_id="param_assignment")
@dataclass(frozen=True, eq=False)
class ParamAssignment(Serializable, FrozenMixin, DerivedEquivalenceMixin, Generic[_T]):
    """Immutable binding of a parameter definition to a concrete value.

    Two assignments are structurally equivalent when their parameters are
    equivalent and their bound values compare equal.

    Raises:
        ParamError: If ``value`` is not a valid assignment for ``param``,
            as :meth:`Param.validate_value` decides it without bindings.
        ConstraintError: As :meth:`Param.validate_value` raises it;
            without bindings, only for ``value`` itself.
        NonBooleanLogicalOperandError: As :meth:`Param.validate_value`
            raises it.

    """

    param: "Param[_T]" = field(metadata={"serialize_codec": _PARAM_CODEC})
    value: _T = field(
        metadata={
            "serialize_codec": _WRAPPED_VALUE_CODEC,
            **compared_as_value(),
        }
    )

    def __post_init__(self) -> None:
        # Normalize here so a directly constructed assignment and the
        # ``Param.assign`` form of the same binding hold the same canonical
        # value: they compare structurally equivalent and both serialize.
        self.param.validate_value(self.value)
        object.__setattr__(self, "value", self.param.domain.normalize_value(self.value))

    @classmethod
    @override
    def construct_from_fields(cls, fields: dict[str, Any]) -> "ParamAssignment[Any]":
        """Rebuild an assignment, rejecting only provable invalidity.

        The constructor path demands full proof of satisfaction, which a
        dependent constraint can only receive through the ``bindings`` of
        the originating ``Param.assign`` call; those bindings are not part
        of the serialized state. Deserialization therefore re-checks what
        is decidable in isolation -- domain admissibility and every
        constraint decidable from this parameter's own variable -- and
        accepts an undecided remainder. The accepted value is stored in
        the domain's canonical form.

        Raises:
            ParamError: If the value is not admissible in the parameter's
                domain, or a constraint provably rejects it.
            ConstraintError: If the value, bound to the parameter's
                variable, cannot be lifted into the substitution
                environment.
            NonBooleanLogicalOperandError: If binding the value puts a
                number in a Boolean position of a constraint, which is
                ill-typed rather than undecided. Reached through
                deserialization, each of these errors surfaces as a
                ``DeserializationValueError``.

        """
        param: Param[Any] = fields["param"]
        value = fields["value"]
        _raise_if_value_provably_invalid(param, value)
        normalized = param.domain.normalize_value(value)
        return _construct_unchecked_assignment(param, normalized)

    def is_value_set(self) -> bool:
        """Return whether this assignment has a value."""
        return True


# ---------------------------------------------------------------------------
# Bound constraint helpers and natural-number gates
# ---------------------------------------------------------------------------


def _create_bound_constraint(
    variable: Identifier,
    bound: int | float | str,
    *,
    is_lower: bool,
    is_inclusive: bool,
) -> EquationConstraint:
    variable_expression = IdentifierExpression(variable)
    if is_lower:
        equation = (
            variable_expression >= bound
            if is_inclusive
            else variable_expression > bound
        )
    else:
        equation = (
            variable_expression <= bound
            if is_inclusive
            else variable_expression < bound
        )
    return EquationConstraint(equation)


def _is_valid_natural_lower_bound(
    bound: int, *, zero_included: bool, is_inclusive: bool
) -> bool:
    """Return whether ``bound`` is an admissible natural-domain lower-bound literal."""
    if zero_included:
        if bound < 0:
            return False
        return is_inclusive or bound >= 1
    if is_inclusive:
        return bound >= 1
    return bound >= 0


def _is_valid_natural_upper_bound(
    bound: int, *, zero_included: bool, is_inclusive: bool
) -> bool:
    """Return whether ``bound`` is an admissible natural-domain upper-bound literal."""
    if zero_included:
        if is_inclusive:
            return bound >= 0
        return bound >= 1
    if is_inclusive:
        return bound >= 1
    return bound >= 2  # noqa: PLR2004


def _validate_natural_lower_bound(
    bound: int, *, zero_included: bool, is_inclusive: bool
) -> None:
    if _is_valid_natural_lower_bound(
        bound, zero_included=zero_included, is_inclusive=is_inclusive
    ):
        return
    if zero_included:
        if bound < 0:
            raise ParamError("Lower bound must be non-negative.")
        raise ParamError(
            "Lower bound must be at least 1 if zero is included and bound is exclusive."
        )
    if is_inclusive:
        raise ParamError("Lower bound must be at least 1 when zero is not included.")
    raise ParamError(
        "Lower bound must be non-negative when zero is not included "
        "and bound is exclusive."
    )


def _validate_natural_upper_bound(
    bound: int, *, zero_included: bool, is_inclusive: bool
) -> None:
    if _is_valid_natural_upper_bound(
        bound, zero_included=zero_included, is_inclusive=is_inclusive
    ):
        return
    if zero_included:
        if is_inclusive:
            raise ParamError("Upper bound must be non-negative when zero is included.")
        raise ParamError(
            "Upper bound must be at least 1 if zero is included and bound is exclusive."
        )
    if is_inclusive:
        raise ParamError("Upper bound must be at least 1 when zero is not included.")
    raise ParamError(
        "Upper bound must be at least 2 when zero is not included "
        "and bound is exclusive."
    )


def _validate_natural_bound(
    domain: ParamDomain,
    bound: int | float | str,
    *,
    is_lower: bool,
    is_inclusive: bool,
) -> None:
    """Apply the natural-number bound gates for a non-negative integer domain.

    The gate reads the domain's interval profile, so it applies to every
    non-negative profile, whether or not the profile admits only bounds.
    """
    profile = domain.get_interval_profile()
    if profile is None or not profile.non_negative:
        return
    if not isinstance(bound, int):
        return
    zero_included = profile.zero_included
    if is_lower:
        _validate_natural_lower_bound(
            bound, zero_included=zero_included, is_inclusive=is_inclusive
        )
    else:
        _validate_natural_upper_bound(
            bound, zero_included=zero_included, is_inclusive=is_inclusive
        )


# ---------------------------------------------------------------------------
# Interval arithmetic helpers
# ---------------------------------------------------------------------------


def _get_interval_operand_profile(param: "Param[Any]") -> IntervalProfile | None:
    """Return ``param``'s interval profile if it is an interval operand as it stands.

    That holds when the profile admits only bounds. A parameter whose
    profile admits other constraints, or whose domain has no profile, is
    not an operand until coerced (see :func:`_coerce_to_interval_param`).
    """
    profile = param.domain.get_interval_profile()
    if profile is None or not profile.admits_only_bounds:
        return None
    return profile


def _require_interval_operand_profile(param: "Param[Any]") -> IntervalProfile:
    """Return ``param``'s interval profile, raising unless it is an interval operand.

    Raises:
        TypeError: If ``param`` is not an interval operand as it stands.

    """
    profile = _get_interval_operand_profile(param)
    if profile is None:
        raise TypeError("Arithmetic is only supported on interval-integer parameters.")
    return profile


def _invert_comparison(operation: BinaryOperation) -> BinaryOperation:
    inverses = {
        BinaryOperation.GREATER: BinaryOperation.LESS,
        BinaryOperation.GREATER_EQUAL: BinaryOperation.LESS_EQUAL,
        BinaryOperation.LESS: BinaryOperation.GREATER,
        BinaryOperation.LESS_EQUAL: BinaryOperation.GREATER_EQUAL,
    }
    if operation not in inverses:
        raise ValueError(f"Cannot invert non-comparison operation: {operation}")
    return inverses[operation]


def _bound_from_literal(
    literal: LiteralExpression, operation: BinaryOperation
) -> tuple[bool, int, bool]:
    value = literal.value
    if not isinstance(value, int):  # pragma: no cover
        raise RuntimeError("Bound expression literal is not an integer.")
    is_lower = operation in (BinaryOperation.GREATER, BinaryOperation.GREATER_EQUAL)
    is_inclusive = operation in (
        BinaryOperation.GREATER_EQUAL,
        BinaryOperation.LESS_EQUAL,
    )
    return is_lower, value, is_inclusive


def _bound_from_constraint(
    constraint: Constraint, variable: Identifier
) -> tuple[bool, int, bool]:
    """Decode a bound constraint's ``(is_lower, bound, is_inclusive)`` triple.

    ``variable`` interprets the expression's two sides: the identifier
    side must name ``variable`` itself, so a well-formed but
    unexpectedly-scoped bound expression is rejected rather than
    silently misread.
    """
    if not isinstance(constraint, EquationConstraint):
        raise RuntimeError(
            "Interval parameter has a non-EquationConstraint constraint: "
            f"{type(constraint)}"
        )
    if not is_bound_expression(constraint.convert_to_expression()):
        raise RuntimeError(
            f"Interval parameter has a non-bound constraint: {constraint!r}"
        )
    expression = constraint.convert_to_expression()
    if not isinstance(expression, BinaryExpression):  # pragma: no cover
        raise RuntimeError("Interval parameter has a non-bound constraint.")
    if (
        isinstance(expression.left, IdentifierExpression)
        and expression.left.identifier == variable
        and isinstance(expression.right, LiteralExpression)
    ):
        return _bound_from_literal(expression.right, expression.operation)
    if (
        isinstance(expression.right, IdentifierExpression)
        and expression.right.identifier == variable
        and isinstance(expression.left, LiteralExpression)
    ):
        return _bound_from_literal(
            expression.left, _invert_comparison(expression.operation)
        )
    raise RuntimeError("Interval bound expression is malformed.")  # pragma: no cover


def _iter_interval_bounds(
    constraints: Sequence[Constraint], variable: Identifier
) -> list[tuple[bool, int, bool]]:
    return [_bound_from_constraint(constraint, variable) for constraint in constraints]


def _get_effective_min_max(
    constraints: Sequence[Constraint], variable: Identifier
) -> tuple[int | None, int | None]:
    min_int: int | None = None
    max_int: int | None = None
    for is_lower, bound, inclusive in _iter_interval_bounds(constraints, variable):
        if is_lower:
            effective = bound if inclusive else bound + 1
            min_int = effective if min_int is None else max(min_int, effective)
        else:
            effective = bound if inclusive else bound - 1
            max_int = effective if max_int is None else min(max_int, effective)
    if min_int is not None and max_int is not None and min_int > max_int:
        raise ParamError(
            f"Empty integer interval represented by constraints for {variable}."
        )
    return min_int, max_int


def _combine_optional_bounds(
    left: int | None,
    right: int | None,
    combine: Callable[[int, int], int],
) -> int | None:
    """Combine two optional bounds, yielding ``None`` if either is unbounded."""
    if left is None or right is None:
        return None
    return combine(left, right)


# An extended-integer bound is a ``(bucket, value)`` pair: ``bucket`` is ``-1``
# for negative infinity, ``0`` for a finite value, or ``1`` for positive
# infinity. ``value`` carries the finite magnitude when ``bucket == 0`` and is
# an unused placeholder otherwise. Ordinary tuple comparison then gives a
# total order (``-inf < any finite < +inf``) without any float sentinel.
def _convert_to_extended_bound(value: int | None, *, is_lower: bool) -> tuple[int, int]:
    if value is not None:
        return (0, value)
    return (-1, 0) if is_lower else (1, 0)


def _convert_from_extended_bound(extended: tuple[int, int]) -> int | None:
    bucket, value = extended
    return value if bucket == 0 else None


def _multiply_extended_bounds(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[int, int]:
    """Multiply two extended-integer bounds, per interval-product set semantics.

    A finite zero operand forces the product to zero even against an
    unbounded operand, since the product set of ``{0}`` with any interval
    is ``{0}``.
    """
    left_bucket, left_value = left
    right_bucket, right_value = right
    if (left_bucket == 0 and left_value == 0) or (
        right_bucket == 0 and right_value == 0
    ):
        return (0, 0)
    if left_bucket == 0 and right_bucket == 0:
        return (0, left_value * right_value)
    left_sign = left_bucket if left_bucket != 0 else (1 if left_value > 0 else -1)
    right_sign = right_bucket if right_bucket != 0 else (1 if right_value > 0 else -1)
    return (left_sign * right_sign, 0)


def _multiply_optional_bounds(
    self_min: int | None,
    self_max: int | None,
    other_min: int | None,
    other_max: int | None,
) -> tuple[int | None, int | None]:
    """Multiply two extended-integer intervals via the four-candidate rule.

    ``[self_min, self_max] * [other_min, other_max]`` spans the minimum and
    maximum of the four pairwise endpoint products, where ``None`` denotes
    an unbounded end (negative infinity for a lower bound, positive
    infinity for an upper bound).

    Args:
        self_min: Left interval's lower bound, or ``None`` if unbounded.
        self_max: Left interval's upper bound, or ``None`` if unbounded.
        other_min: Right interval's lower bound, or ``None`` if unbounded.
        other_max: Right interval's upper bound, or ``None`` if unbounded.

    Returns:
        The product interval's ``(min, max)`` pair, each ``None`` when that
        end is unbounded.

    """
    candidates = tuple(
        _multiply_extended_bounds(left, right)
        for left in (
            _convert_to_extended_bound(self_min, is_lower=True),
            _convert_to_extended_bound(self_max, is_lower=False),
        )
        for right in (
            _convert_to_extended_bound(other_min, is_lower=True),
            _convert_to_extended_bound(other_max, is_lower=False),
        )
    )
    return (
        _convert_from_extended_bound(min(candidates)),
        _convert_from_extended_bound(max(candidates)),
    )


def _apply_interval_bounds(
    param: "Param[int]",
    profile: IntervalProfile,
    min_int: int | None,
    max_int: int | None,
) -> "Param[int]":
    """Bound ``param`` to ``[min_int, max_int]``, rendered as ``profile`` prefers.

    ``profile`` is the interval profile of ``param``'s domain; a ``None``
    bound leaves that end unbounded.
    """
    if min_int is not None:
        if _is_exclusive_lower_rendering_valid(profile, min_int):
            param = param.add_lower_bound_constraint(min_int - 1, is_inclusive=False)
        else:
            param = param.add_lower_bound_constraint(min_int, is_inclusive=True)
    if max_int is not None:
        if _is_exclusive_upper_rendering_valid(profile, max_int):
            param = param.add_upper_bound_constraint(max_int + 1, is_inclusive=False)
        else:
            param = param.add_upper_bound_constraint(max_int, is_inclusive=True)
    return param


def _is_exclusive_lower_rendering_valid(profile: IntervalProfile, min_int: int) -> bool:
    """Return whether ``min_int`` may be rendered as ``> min_int - 1``.

    ``> min_int - 1`` admits exactly what ``>= min_int`` admits over the
    integers, but on a non-negative domain the natural-number gate judges
    the literal rather than what it admits, and rejects the shifted one
    (``> -1`` on a zero-included natural domain is exactly ``>= 0``, yet
    ``-1`` is not an admissible natural literal). Rendering falls back to
    the inclusive form in that case instead of tripping the gate.
    """
    if profile.prefer_inclusive:
        return False
    return not profile.non_negative or _is_valid_natural_lower_bound(
        min_int - 1, zero_included=profile.zero_included, is_inclusive=False
    )


def _is_exclusive_upper_rendering_valid(profile: IntervalProfile, max_int: int) -> bool:
    """Return whether ``max_int`` may be rendered as ``< max_int + 1``.

    The upper-bound mirror of :func:`_is_exclusive_lower_rendering_valid`.
    """
    if profile.prefer_inclusive:
        return False
    return not profile.non_negative or _is_valid_natural_upper_bound(
        max_int + 1, zero_included=profile.zero_included, is_inclusive=False
    )


def _create_widened_interval_param(
    min_int: int | None,
    max_int: int | None,
    template: IntervalProfile,
) -> "Param[int]":
    domain = IntervalIntegerDomain(prefer_inclusive=template.prefer_inclusive)
    param: Param[int] = Param(domain)
    return _apply_interval_bounds(
        param, domain.get_interval_profile(), min_int, max_int
    )


def _create_class_preserved_interval_param(
    other: IntervalProfile,
    min_int: int | None,
    max_int: int | None,
    template: IntervalProfile,
    *,
    zero_included: bool,
) -> "Param[int]":
    """Build a result that stays natural when both operands' profiles are.

    ``template`` is the left operand's profile and ``other`` the coerced
    right operand's; the result renders as ``template`` prefers.
    """
    if template.non_negative and other.non_negative:
        domain = IntervalIntegerDomain(
            prefer_inclusive=template.prefer_inclusive,
            non_negative=True,
            zero_included=zero_included,
        )
        param: Param[int] = Param(domain)
        return _apply_interval_bounds(
            param, domain.get_interval_profile(), min_int, max_int
        )
    return _create_widened_interval_param(min_int, max_int, template)


def _require_bound_constraint(constraint: Constraint) -> None:
    """Raise ``TypeError`` unless ``constraint`` lifts to a bound expression.

    A constraint that does not lift to an expression at all (its
    ``convert_to_expression`` raises ``ConstraintError``) is not a bound
    either; that error is chained as the cause.
    """
    message = (
        "Cannot coerce an integer parameter with non-bound constraints to an "
        "interval parameter."
    )
    try:
        expression = constraint.convert_to_expression()
    except ConstraintError as error:
        raise TypeError(message) from error
    if not is_bound_expression(expression):
        raise TypeError(message)


def _coerce_to_interval_param(template: IntervalProfile, other: Any) -> "Param[int]":
    """Return ``other`` as an interval operand that renders as ``template`` prefers.

    An ``int`` becomes the exact interval it denotes. A parameter that is
    an interval operand as it stands is returned unchanged. A parameter
    whose interval profile admits other constraints is recast over an
    interval domain, provided every constraint it carries is a bound; its
    sign restriction travels as the carried bound constraint rather than as
    a domain attribute.

    Raises:
        TypeError: If ``other`` is a ``bool``, a parameter whose domain has
            no interval profile, a parameter carrying a non-bound
            constraint, or a value of any other type.

    """
    if isinstance(other, bool):
        raise TypeError(f"Unsupported operand type: {type(other)}")
    if isinstance(other, int):
        return create_interval_integer_param_exactly(
            other, prefer_inclusive=template.prefer_inclusive
        )
    if isinstance(other, Param):
        profile = other.domain.get_interval_profile()
        if profile is not None:
            if profile.admits_only_bounds:
                return other
            for constraint in other.constraints:
                _require_bound_constraint(constraint)
            return Param(
                IntervalIntegerDomain(prefer_inclusive=template.prefer_inclusive),
                variable=other.variable,
                constraint_system=create_constraint_system(*other.constraints),
            )
    raise TypeError(f"Unsupported operand type: {type(other)}")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_integer_param(
    *, name: Identifier | None = None, constraints: Sequence[Constraint] = ()
) -> Param[int]:
    """Create an integer-valued parameter."""
    return Param(
        IntegerDomain(),
        variable=name or Identifier("param"),
        constraint_system=create_constraint_system(*constraints),
    )


def create_natural_param(
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    constraints: Sequence[Constraint] = (),
) -> Param[int]:
    """Create a natural-number (non-negative integer) parameter."""
    return Param(
        IntegerDomain(non_negative=True, zero_included=zero_included),
        variable=name or Identifier("param"),
        constraint_system=create_constraint_system(*constraints),
    )


def create_real_param(
    *, name: Identifier | None = None, constraints: Sequence[Constraint] = ()
) -> Param[str | float]:
    """Create a real-valued parameter."""
    return Param(
        RealDomain(),
        variable=name or Identifier("param"),
        constraint_system=create_constraint_system(*constraints),
    )


def create_integer_param_between(
    lower_bound: int,
    upper_bound: int,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> Param[int]:
    """Create an integer parameter bounded to ``[lower_bound, upper_bound]``."""
    if lower_bound > upper_bound or (
        lower_bound == upper_bound and not (is_lower_inclusive and is_upper_inclusive)
    ):
        raise ParamError("Lower bound must be less than or equal to upper bound.")
    param = create_integer_param(name=name)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    return param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )


def create_integer_param_with_lower_bound(
    lower_bound: int, *, name: Identifier | None = None, is_inclusive: bool = True
) -> Param[int]:
    """Create an integer parameter with a lower bound."""
    return create_integer_param(name=name).add_lower_bound_constraint(
        lower_bound, is_inclusive=is_inclusive
    )


def create_integer_param_with_upper_bound(
    upper_bound: int, *, name: Identifier | None = None, is_inclusive: bool = True
) -> Param[int]:
    """Create an integer parameter with an upper bound."""
    return create_integer_param(name=name).add_upper_bound_constraint(
        upper_bound, is_inclusive=is_inclusive
    )


def create_real_param_between(
    lower_bound: float | str,
    upper_bound: float | str,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter bounded to ``[lower_bound, upper_bound]``."""
    if float(lower_bound) > float(upper_bound) or (
        float(lower_bound) == float(upper_bound)
        and not (is_lower_inclusive and is_upper_inclusive)
    ):
        raise ParamError("Lower bound must be less than or equal to upper bound.")
    param = create_real_param(name=name)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    return param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )


def create_real_param_with_lower_bound(
    lower_bound: float | str,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter with a lower bound."""
    return create_real_param(name=name).add_lower_bound_constraint(
        lower_bound, is_inclusive=is_inclusive
    )


def create_real_param_with_upper_bound(
    upper_bound: float | str,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
) -> Param[str | float]:
    """Create a real parameter with an upper bound."""
    return create_real_param(name=name).add_upper_bound_constraint(
        upper_bound, is_inclusive=is_inclusive
    )


def create_interval_integer_param(
    *,
    name: Identifier | None = None,
    prefer_inclusive: bool = True,
    non_negative: bool = False,
    zero_included: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter (supports interval arithmetic)."""
    return Param(
        IntervalIntegerDomain(
            prefer_inclusive=prefer_inclusive,
            non_negative=non_negative,
            zero_included=zero_included,
        ),
        variable=name or Identifier("param"),
    )


def create_interval_integer_param_between(
    lower_bound: int,
    upper_bound: int,
    *,
    name: Identifier | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter bounded to ``[lower, upper]``."""
    param = create_interval_integer_param(name=name, prefer_inclusive=prefer_inclusive)
    param = param.add_lower_bound_constraint(
        lower_bound, is_inclusive=is_lower_inclusive
    )
    param = param.add_upper_bound_constraint(
        upper_bound, is_inclusive=is_upper_inclusive
    )
    _get_effective_min_max(param.constraints, param.variable)
    return param


def create_interval_integer_param_with_lower_bound(
    lower_bound: int,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter with a lower bound."""
    return create_interval_integer_param(
        name=name, prefer_inclusive=prefer_inclusive
    ).add_lower_bound_constraint(lower_bound, is_inclusive=is_inclusive)


def create_interval_integer_param_with_upper_bound(
    upper_bound: int,
    *,
    name: Identifier | None = None,
    is_inclusive: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create an interval-integer parameter with an upper bound."""
    return create_interval_integer_param(
        name=name, prefer_inclusive=prefer_inclusive
    ).add_upper_bound_constraint(upper_bound, is_inclusive=is_inclusive)


def create_interval_integer_param_exactly(
    value: int, *, name: Identifier | None = None, prefer_inclusive: bool = True
) -> Param[int]:
    """Create an interval-integer parameter bounded to exactly ``value``."""
    param = create_interval_integer_param(name=name, prefer_inclusive=prefer_inclusive)
    param = param.add_lower_bound_constraint(value, is_inclusive=True)
    return param.add_upper_bound_constraint(value, is_inclusive=True)


def create_interval_natural_param(
    *,
    name: Identifier | None = None,
    zero_included: bool = True,
    prefer_inclusive: bool = True,
) -> Param[int]:
    """Create a non-negative interval-integer parameter."""
    return create_interval_integer_param(
        name=name,
        prefer_inclusive=prefer_inclusive,
        non_negative=True,
        zero_included=zero_included,
    )


def create_ordinal_param(
    values: Sequence[_OrdinalValueT], *, name: Identifier | None = None
) -> Param[_OrdinalValueT]:
    """Create an ordinal parameter over a finite, ordered value set."""
    return Param(build_ordinal_domain(values), variable=name or Identifier("param"))


def create_categorical_param(
    categories: Collection[_CategoricalValueT], *, name: Identifier | None = None
) -> Param[_CategoricalValueT]:
    """Create a categorical parameter over a finite, unordered value set."""
    return Param(
        build_categorical_domain(tuple(categories)),
        variable=name or Identifier("param"),
    )


def create_permutation_param(
    members: Sequence[_PermutationMemberValueT], *, name: Identifier | None = None
) -> Param[tuple[_PermutationMemberValueT, ...]]:
    """Create a permutation parameter over a fixed, ordered set of members."""
    return Param(
        build_permutation_domain(members), variable=name or Identifier("param")
    )


def create_single_valid_value_param(
    value: _CategoricalValueT, *, name: Identifier | None = None
) -> Param[_CategoricalValueT]:
    """Create a parameter that admits only a single value."""
    return create_categorical_param([value], name=name)


# ---------------------------------------------------------------------------
# Set algebra
# ---------------------------------------------------------------------------


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
        left: Left operand; must have an ordinal or categorical domain.
        right: Right operand; must have the same domain kind as ``left``.
        name: Variable for the result; defaults to a fresh
            ``Identifier("param")``.

    Returns:
        A new parameter over the union of the operands' effective value sets.

    Raises:
        TypeError: If either operand's domain kind does not support union, the
            kinds differ, or merged ordinal values are not mutually comparable.
        ParamError: If both operands' effective value sets are empty, so the
            union would be empty.

    """
    variable = name or Identifier("param")
    union = left.domain.compute_union(
        left.constraints,
        left.variable,
        right.domain,
        right.constraints,
        right.variable,
        variable,
    )
    if union is None:
        raise TypeError(
            f"Union is not supported for domain kind {type(left.domain).__name__}."
        )
    domain, constraints = union
    return Param(
        domain,
        variable=variable,
        constraint_system=create_constraint_system(*constraints),
    )


def _coerce_intersection_operands(
    left: "Param[Any]", right: "Param[Any]"
) -> tuple["Param[Any]", "Param[Any]"]:
    """Coerce a pair with exactly one interval operand to one domain kind.

    When one operand is an interval operand as it stands and the other's
    interval profile admits other constraints (a plain integer parameter),
    the latter is recast through the interval-arithmetic coercion, over an
    ``IntervalIntegerDomain``, so both operands share a domain kind before
    dispatching to ``compute_intersection``. Any other pairing (already one
    kind, or an unsupported mix) is returned unchanged, leaving the
    delegated ``compute_intersection`` to report a kind mismatch.

    Raises:
        TypeError: If the plain integer operand carries a non-bound
            constraint, so it has no interval form (propagated from
            :func:`_coerce_to_interval_param`).

    """
    left_profile = left.domain.get_interval_profile()
    right_profile = right.domain.get_interval_profile()
    if left_profile is None or right_profile is None:
        return left, right
    if left_profile.admits_only_bounds and not right_profile.admits_only_bounds:
        return left, _coerce_to_interval_param(left_profile, right)
    if right_profile.admits_only_bounds and not left_profile.admits_only_bounds:
        return _coerce_to_interval_param(right_profile, left), right
    return left, right


def _is_intersection_provably_empty(
    result: "Param[Any]", left: "Param[Any]", right: "Param[Any]"
) -> bool:
    """Return whether the intersection ``result`` of two operands is proven empty.

    A ``VIOLATED`` conjunction is empty outright. An ``UNDECIDED`` one is
    empty when either operand is itself proven infeasible, since an
    intersection with an empty set is empty; the operands are not
    consulted for a ``SATISFIED`` conjunction.
    """
    outcome = result.check_feasibility()
    if outcome is ConstraintOutcome.VIOLATED:
        return True
    elif outcome is ConstraintOutcome.SATISFIED:
        return False
    else:
        return any(
            operand.check_feasibility() is ConstraintOutcome.VIOLATED
            for operand in (left, right)
        )


def create_intersection_param(
    left: Param[_T],
    right: Param[_T],
    *,
    name: Identifier | None = None,
) -> Param[_T]:
    """Create a parameter admitting exactly the values valid for both operands.

    Finite-set operands are intersected by baking both effective value sets;
    permutation operands keep their member set, and numeric operands merge
    domain attributes conservatively -- both of these kinds carry the
    conjunction of both operands' constraints with both operands' variables
    renamed to the result variable, so a constraint relating the two
    operands becomes a constraint on the result alone. A mixed pair of one
    interval-integer parameter and one plain integer parameter whose
    constraints are all bound expressions is supported by coercing the
    plain parameter to interval form first; the coerced operand contributes
    any sign bound as a carried constraint rather than as a domain
    attribute.

    Args:
        left: Left operand.
        right: Right operand; must have the same domain kind as ``left``
            (modulo the interval/integer coercion above).
        name: Variable for the result; defaults to a fresh
            ``Identifier("param")``.

    Returns:
        A new parameter over the intersection of the operands' feasible
        sets. A conjunction the solver leaves undecided is returned live;
        its :meth:`Param.check_feasibility` reports ``UNDECIDED``, which
        tells it apart from one decided feasible.

    Raises:
        TypeError: If the domain kinds are incompatible, or a mixed
            interval-integer/plain-integer pair's plain operand carries a
            non-bound constraint, so it has no interval form.
        ParamError: If the intersection is provably empty: an empty
            finite-set intersection, permutation operands over different
            member sets, a numeric conjunction the enumeration or the
            solver proves infeasible, or an operand that is itself proven
            infeasible.
        NonBooleanLogicalOperandError: If a carried constraint holds a
            provably numeric operand in a Boolean position, as the
            emptiness check through :meth:`Param.check_feasibility`
            raises it.

    """
    coerced_left, coerced_right = _coerce_intersection_operands(left, right)
    variable = name or Identifier("param")
    domain, constraints = coerced_left.domain.compute_intersection(
        coerced_left.constraints,
        coerced_left.variable,
        coerced_right.domain,
        coerced_right.constraints,
        coerced_right.variable,
        variable,
    )
    result: Param[_T] = Param(
        domain,
        variable=variable,
        constraint_system=create_constraint_system(*constraints),
    )
    if _is_intersection_provably_empty(result, coerced_left, coerced_right):
        raise ParamError("Intersection of parameters is empty.")
    return result
