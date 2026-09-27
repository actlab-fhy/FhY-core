//! [`ConstraintSystem`]: the conjunction of constraints, and the questions
//! the solver answers about it.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::expression::{BooleanScreen, Expression, ExpressionKind, SymbolTypes};
use crate::identifier::Identifier;
use crate::solver::{
    Answer, CheckLimits, Hazard, QueryContext, QueryKind, Question, UnknownReason,
};
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::binding::{Binding, Bindings};
use super::context::{ConstraintContext, ConstraintEvent, ConstraintObserver};
use super::error::ConstraintError;
use super::{Constraint, Outcome};

/// The conjunction of constraints, in canonical order: sorted by their
/// ordering keys, the order [`Constraint`]'s `Ord` defines, duplicates
/// kept. Cloning one shares it.
///
/// Keys are equal exactly when constraints are equivalent, for every
/// conforming [`CustomConstraint`](super::CustomConstraint) and
/// [`OpaqueValue`](super::OpaqueValue); an implementation whose keys collide
/// breaks that contract, as a `Hash` that disagrees with `==` does. A run
/// of members with one key, a tie run, is still grouped: its equivalent
/// members are next to one another, the groups in the order of their first
/// member in the input. That residual order is the one thing a colliding
/// key leaves to the input; equivalence and hashing compare each tie run as
/// a multiset, so they do not depend on it.
#[derive(Debug, Clone, Default)]
pub struct ConstraintSystem {
    constraints: Arc<[Constraint]>,
    /// The length of each tie run, in order; they sum to the member count.
    tie_runs: Arc<[usize]>,
}

impl ConstraintSystem {
    /// Return the system of `constraints`, reading each member's ordering
    /// key once and sorting by the keys, as [`Constraint`]'s `Ord` compares
    /// them.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::Custom`] for a custom member whose key
    /// fails.
    pub fn new(constraints: impl IntoIterator<Item = Constraint>) -> Result<Self, ConstraintError> {
        let mut keyed: Vec<(String, Constraint)> = constraints
            .into_iter()
            .map(|constraint| Ok((constraint.ordering_key()?, constraint)))
            .collect::<Result<_, ConstraintError>>()?;
        keyed.sort_by(|(left, _), (right, _)| left.cmp(right));
        let mut ordered = Vec::with_capacity(keyed.len());
        let mut tie_runs = Vec::new();
        let mut members = keyed.into_iter().peekable();
        while let Some((key, first)) = members.next() {
            let mut run = vec![first];
            while let Some((_, member)) = members.next_if(|(next, _)| *next == key) {
                run.push(member);
            }
            tie_runs.push(run.len());
            ordered.extend(group_equivalent(run));
        }
        Ok(Self {
            constraints: ordered.into(),
            tie_runs: tie_runs.into(),
        })
    }

    /// Return the members of each tie run, in order.
    fn runs(&self) -> impl Iterator<Item = &[Constraint]> {
        let mut rest: &[Constraint] = &self.constraints;
        self.tie_runs.iter().map(move |&length| {
            let (run, after) = rest.split_at(length);
            rest = after;
            run
        })
    }

    /// Return the members, in canonical order.
    #[must_use]
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Return whether the system holds no member.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Decide the conjunction under `bindings`.
    ///
    /// Members are evaluated in order: the first [`Violated`](Outcome::Violated)
    /// one answers, and otherwise any [`Undecided`](Outcome::Undecided)
    /// one makes the system undecided, reported as
    /// [`ConstraintEvent::UndecidedMember`]. A member's own events are reported as
    /// [`ConstraintEvent::InMember`].
    ///
    /// # Errors
    ///
    /// Returns the first member's error, in order.
    pub fn evaluate(
        &self,
        bindings: &Bindings,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        let mut is_undecided = false;
        for (index, constraint) in self.constraints.iter().enumerate() {
            let observer = MemberObserver { context, index };
            let member_context = context.with_observer(&observer);
            match constraint.evaluate(bindings, &member_context)? {
                Outcome::Violated => return Ok(Outcome::Violated),
                Outcome::Undecided => {
                    context.notify(&ConstraintEvent::UndecidedMember { index });
                    is_undecided = true;
                }
                Outcome::Satisfied => {}
            }
        }
        Ok(if is_undecided {
            Outcome::Undecided
        } else {
            Outcome::Satisfied
        })
    }

    /// Return the conjunction as an expression: `true` for no member, a
    /// member's expression for one, and their conjunction otherwise.
    ///
    /// # Errors
    ///
    /// Returns the first member's conversion error, in order.
    pub fn to_expression(&self) -> Result<Expression, ConstraintError> {
        Ok(conjoin(convert_members(&self.constraints)?))
    }

    /// Decide whether some assignment satisfies every member.
    ///
    /// An empty system is satisfied. Otherwise, in order: every member is
    /// converted; `symbol_types` must cover the conjunction's identifiers,
    /// native constants aside; each member's expression must be a
    /// predicate; and the context's solver is asked. A refused or given-up
    /// question is undecided, reported as [`ConstraintEvent::Refused`] or
    /// [`ConstraintEvent::GaveUp`].
    ///
    /// A set member's expression is screened with
    /// [`Hazard::find_for_membership`] before the solver's own screen, so a
    /// member of another numeric kind than its variable's is refused: the
    /// solver would read `x == 2.0` by value, where membership keeps `2` and
    /// `2.0` apart. An equation's equality of an integer with a real is
    /// answered by value, as the evaluator decides it.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::UnliftableMember`] or
    /// [`ConstraintError::Custom`] for a conversion,
    /// [`ConstraintError::MissingSymbolTypes`], [`ConstraintError::IllTyped`],
    /// and [`ConstraintError::Solve`].
    pub fn check_satisfiability(
        &self,
        symbol_types: &dyn SymbolTypes,
        limits: CheckLimits,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        if self.constraints.is_empty() {
            return Ok(Outcome::Satisfied);
        }
        let members = convert_members(&self.constraints)?;
        let conjunction = conjoin(members.clone());
        check_symbol_types(conjunction.free_identifiers(), symbol_types, context)?;
        screen_members(&members, None, symbol_types, context)?;
        let memberships = memberships_of(self.constraints.iter(), &members);
        ask(
            &Question::Satisfiability(&conjunction),
            &memberships,
            symbol_types,
            limits,
            context,
        )
    }

    /// Decide whether the system is satisfiable given `bindings`.
    ///
    /// An empty system is satisfied. Otherwise, in order:
    ///
    /// 1. every binding becomes an expression, a literal value its literal;
    /// 2. a set member whose variable is bound to a literal is decided by
    ///    itself, and the other members are the residual;
    /// 3. `symbol_types` must cover the identifiers the substituted
    ///    residual leaves free, native constants aside;
    /// 4. each residual member's expression must be a predicate, with the
    ///    bindings;
    /// 5. a bound native constant the system refers to makes it undecided,
    ///    reported as [`ConstraintEvent::BoundNativeConstants`];
    /// 6. the decided members fold as [`evaluate`](Self::evaluate) folds
    ///    them, a violation answering at once, and no residual answering
    ///    their fold;
    /// 7. the substituted residual is asked about, and the two outcomes
    ///    fold: a violation wins, then undecidedness.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::UnusableBinding`] for a binding that is
    /// not an expression or a literal, and the errors of
    /// [`check_satisfiability`](Self::check_satisfiability).
    pub fn check_satisfiability_with_bindings(
        &self,
        bindings: &Bindings,
        symbol_types: &dyn SymbolTypes,
        limits: CheckLimits,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        if self.constraints.is_empty() {
            return Ok(Outcome::Satisfied);
        }
        let environment = build_environment(bindings)?;
        let (leaves, rest): (Vec<&Constraint>, Vec<&Constraint>) =
            self.constraints.iter().partition(|constraint| {
                matches!(constraint, Constraint::Set(set)
                if environment.get(set.variable()).is_some_and(|bound| {
                    matches!(bound.kind(), ExpressionKind::Literal(_))
                }))
            });
        let rest_members = rest
            .iter()
            .map(|constraint| constraint.to_expression())
            .collect::<Result<Vec<_>, _>>()?;
        let residual = conjoin(rest_members.clone());
        let mut left_free: HashSet<Identifier> = HashSet::new();
        for identifier in residual.free_identifiers() {
            match environment.get(&identifier) {
                Some(bound) => left_free.extend(bound.free_identifiers()),
                None => {
                    left_free.insert(identifier);
                }
            }
        }
        check_symbol_types(left_free, symbol_types, context)?;
        screen_members(&rest_members, Some(&environment), symbol_types, context)?;
        let mut scope = residual.free_identifiers();
        for leaf in &leaves {
            if let Constraint::Set(set) = leaf {
                scope.insert(set.variable().clone());
            }
        }
        let mut constants: Vec<Identifier> = environment
            .keys()
            .filter(|identifier| {
                scope.contains(*identifier) && context.is_native_constant(identifier)
            })
            .cloned()
            .collect();
        if !constants.is_empty() {
            constants.sort_by_key(Identifier::id);
            context.notify(&ConstraintEvent::BoundNativeConstants {
                identifiers: &constants,
            });
            return Ok(Outcome::Undecided);
        }
        let mut leaves_outcome = Outcome::Satisfied;
        for leaf in &leaves {
            match leaf.evaluate(bindings, context)? {
                Outcome::Violated => return Ok(Outcome::Violated),
                Outcome::Undecided => leaves_outcome = Outcome::Undecided,
                Outcome::Satisfied => {}
            }
        }
        if rest.is_empty() {
            return Ok(leaves_outcome);
        }
        let substituted = residual
            .substitute(&environment)
            .map_err(ConstraintError::Substitution)?;
        check_symbol_types(substituted.free_identifiers(), symbol_types, context)?;
        let memberships = memberships_of(rest.iter().copied(), &rest_members)
            .into_iter()
            .map(|membership| {
                membership
                    .substitute(&environment)
                    .map_err(ConstraintError::Substitution)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let residual_outcome = ask(
            &Question::Satisfiability(&substituted),
            &memberships,
            symbol_types,
            limits,
            context,
        )?;
        Ok(match (leaves_outcome, residual_outcome) {
            (Outcome::Violated, _) | (_, Outcome::Violated) => Outcome::Violated,
            (Outcome::Undecided, _) | (_, Outcome::Undecided) => Outcome::Undecided,
            (Outcome::Satisfied, Outcome::Satisfied) => Outcome::Satisfied,
        })
    }

    /// Decide whether every assignment satisfying this system satisfies
    /// `other`.
    ///
    /// In order: both systems are converted; `symbol_types` must cover both
    /// sides' identifiers, native constants aside; each member of this
    /// system, then of `other`, must be a predicate; and the solver is
    /// asked, both sides' set members screened first as
    /// [`check_satisfiability`](Self::check_satisfiability) screens them.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`check_satisfiability`](Self::check_satisfiability).
    pub fn check_implication(
        &self,
        other: &Self,
        symbol_types: &dyn SymbolTypes,
        limits: CheckLimits,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        let antecedent_members = convert_members(&self.constraints)?;
        let consequent_members = convert_members(&other.constraints)?;
        let antecedent = conjoin(antecedent_members.clone());
        let consequent = conjoin(consequent_members.clone());
        let mut mentioned = antecedent.free_identifiers();
        mentioned.extend(consequent.free_identifiers());
        check_symbol_types(mentioned, symbol_types, context)?;
        screen_members(&antecedent_members, None, symbol_types, context)?;
        screen_members(&consequent_members, None, symbol_types, context)?;
        let mut memberships = memberships_of(self.constraints.iter(), &antecedent_members);
        memberships.extend(memberships_of(
            other.constraints.iter(),
            &consequent_members,
        ));
        ask(
            &Question::Implication {
                antecedent: &antecedent,
                consequent: &consequent,
            },
            &memberships,
            symbol_types,
            limits,
            context,
        )
    }

    /// Return whether `other` holds structurally equivalent members: its
    /// tie runs have this system's lengths, and each run holds the members
    /// of this system's run, in any order.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        let matched = self.match_runs(other, |left, right| {
            Ok::<_, std::convert::Infallible>(left.is_structurally_equivalent(right))
        });
        matches!(matched, Ok(true))
    }

    /// Return whether the tie runs of `self` and `other` have one length
    /// each and hold the same members as multisets under `is_equivalent`,
    /// or the first error it returns.
    fn match_runs<E>(
        &self,
        other: &Self,
        mut is_equivalent: impl FnMut(&Constraint, &Constraint) -> Result<bool, E>,
    ) -> Result<bool, E> {
        if self.tie_runs != other.tie_runs {
            return Ok(false);
        }
        for (left, right) in self.runs().zip(other.runs()) {
            if let ([left], [right]) = (left, right) {
                if !is_equivalent(left, right)? {
                    return Ok(false);
                }
                continue;
            }
            let mut unmatched: Vec<&Constraint> = right.iter().collect();
            for member in left {
                let mut found = None;
                for (position, candidate) in unmatched.iter().enumerate() {
                    if is_equivalent(member, candidate)? {
                        found = Some(position);
                        break;
                    }
                }
                let Some(position) = found else {
                    return Ok(false);
                };
                unmatched.swap_remove(position);
            }
        }
        Ok(true)
    }
}

/// Return the members of one tie run with each equivalent group next to one
/// another, the groups in the order of their first member.
fn group_equivalent(run: Vec<Constraint>) -> Vec<Constraint> {
    if run.len() < 2 {
        return run;
    }
    let mut groups: Vec<Vec<Constraint>> = Vec::new();
    for member in run {
        match groups
            .iter_mut()
            .find(|group| group[0].is_structurally_equivalent(&member))
        {
            Some(group) => group.push(member),
            None => groups.push(vec![member]),
        }
    }
    groups.into_iter().flatten().collect()
}

impl PartialEq for ConstraintSystem {
    /// Compare as [`is_structurally_equivalent`](Self::is_structurally_equivalent)
    /// does: each tie run as a multiset, in canonical order.
    fn eq(&self, other: &Self) -> bool {
        self.is_structurally_equivalent(other)
    }
}

impl Eq for ConstraintSystem {}

impl Hash for ConstraintSystem {
    /// Feed each tie run's length, then its members' hashes in sorted
    /// order, so the order within a run, which equivalence ignores, does not
    /// change the hash.
    fn hash<H: Hasher>(&self, state: &mut H) {
        for run in self.runs() {
            state.write_usize(run.len());
            if let [member] = run {
                member.hash(state);
                continue;
            }
            let mut hashes: Vec<u64> = run
                .iter()
                .map(|member| {
                    let mut hasher = DefaultHasher::new();
                    member.hash(&mut hasher);
                    hasher.finish()
                })
                .collect();
            hashes.sort_unstable();
            hashes.hash(state);
        }
    }
}

impl fmt::Display for ConstraintSystem {
    /// Write the members in canonical order, joined by `and`, or `true` for
    /// the empty system.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.constraints.is_empty() {
            return f.write_str("true");
        }
        for (position, constraint) in self.constraints.iter().enumerate() {
            if position > 0 {
                f.write_str(" and ")?;
            }
            write!(f, "{constraint}")?;
        }
        Ok(())
    }
}

impl AlphaEquivalence for ConstraintSystem {
    type Error = ConstraintError;

    /// Compare the members under `renaming`, each tie run as a multiset:
    /// the runs must have one length each, in order.
    ///
    /// # Errors
    ///
    /// Returns the first custom member's failure, in order.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, ConstraintError> {
        self.match_runs(other, |left, right| {
            left.is_alpha_equivalent_under(right, renaming)
        })
    }
}

/// The observer a member reports to while a system evaluates it: it wraps
/// each event in [`ConstraintEvent::InMember`].
struct MemberObserver<'a> {
    context: &'a ConstraintContext<'a>,
    index: usize,
}

impl ConstraintObserver for MemberObserver<'_> {
    fn notify(&self, event: &ConstraintEvent<'_>) {
        self.context.notify(&ConstraintEvent::InMember {
            index: self.index,
            event,
        });
    }
}

/// Return each member's expression, in order.
fn convert_members(constraints: &[Constraint]) -> Result<Vec<Expression>, ConstraintError> {
    constraints.iter().map(Constraint::to_expression).collect()
}

/// Return the conjunction of `members`: `true` for none, the member itself
/// for one.
fn conjoin(members: Vec<Expression>) -> Expression {
    Expression::all(members)
}

/// Return the environment of `bindings`: each an expression, a literal
/// value its literal.
fn build_environment(
    bindings: &Bindings,
) -> Result<HashMap<Identifier, Expression>, ConstraintError> {
    bindings
        .iter()
        .map(|(identifier, binding)| {
            let expression = match binding {
                Binding::Expression(expression) => expression.clone(),
                Binding::Value(_) => super::equation::lift_binding(identifier, binding)?,
            };
            Ok((identifier.clone(), expression))
        })
        .collect()
}

/// Check that `symbol_types` covers `mentioned`, native constants aside.
fn check_symbol_types(
    mentioned: HashSet<Identifier>,
    symbol_types: &dyn SymbolTypes,
    context: &ConstraintContext<'_>,
) -> Result<(), ConstraintError> {
    let mut missing: Vec<Identifier> = mentioned
        .into_iter()
        .filter(|identifier| {
            symbol_types.symbol_type(identifier).is_none()
                && !context.is_native_constant(identifier)
        })
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    missing.sort_by_key(Identifier::id);
    Err(ConstraintError::MissingSymbolTypes(missing))
}

/// Check that each of `members` can be a predicate, in order, with
/// `environment` if given.
fn screen_members(
    members: &[Expression],
    environment: Option<&HashMap<Identifier, Expression>>,
    symbol_types: &dyn SymbolTypes,
    context: &ConstraintContext<'_>,
) -> Result<(), ConstraintError> {
    let mut screen = BooleanScreen::new()
        .with_sorts(context.sorts())
        .with_symbol_types(symbol_types);
    if let Some(environment) = environment {
        screen = screen.with_environment(environment);
    }
    for member in members {
        screen
            .check_predicate(member)
            .map_err(ConstraintError::IllTyped)?;
    }
    Ok(())
}

/// Return the expressions of the set members of `constraints`, whose
/// converted expressions are `members`, in order.
fn memberships_of<'c>(
    constraints: impl IntoIterator<Item = &'c Constraint>,
    members: &[Expression],
) -> Vec<Expression> {
    constraints
        .into_iter()
        .zip(members)
        .filter(|(constraint, _)| matches!(constraint, Constraint::Set(_)))
        .map(|(_, member)| member.clone())
        .collect()
}

/// Return the first hazard [`Hazard::find_for_membership`] finds in
/// `memberships`, the expressions of a question's set members.
fn find_membership_hazard(
    memberships: &[Expression],
    symbol_types: &dyn SymbolTypes,
    context: &ConstraintContext<'_>,
) -> Option<Hazard> {
    memberships.iter().find_map(|membership| {
        Hazard::find_for_membership(membership, symbol_types, context.sorts())
    })
}

/// Ask `question` of the context's solver, and read its answer: undecided
/// for a refusal or an `unknown`, each reported.
///
/// `memberships`, the expressions of the question's set members, are
/// screened for type-strict membership first, once the solver is known to
/// answer the question's kind, and a hazard there refuses the question as
/// the solver's own screen does.
fn ask(
    question: &Question<'_>,
    memberships: &[Expression],
    symbol_types: &dyn SymbolTypes,
    limits: CheckLimits,
    context: &ConstraintContext<'_>,
) -> Result<Outcome, ConstraintError> {
    let kind: QueryKind = question.kind();
    let refusal = context
        .solver()
        .can_answer(kind)
        .then(|| find_membership_hazard(memberships, symbol_types, context))
        .flatten();
    let answer = if let Some(hazard) = refusal {
        Answer::Unknown(UnknownReason::Refused(hazard))
    } else {
        let query = QueryContext::new(symbol_types)
            .with_sorts(context.sorts())
            .with_limits(limits);
        context
            .solver()
            .ask(question, &query)
            .map_err(ConstraintError::Solve)?
    };
    Ok(match answer {
        Answer::Yes => Outcome::Satisfied,
        Answer::No => Outcome::Violated,
        Answer::Unknown(UnknownReason::Refused(hazard)) => {
            context.notify(&ConstraintEvent::Refused {
                kind,
                hazard: &hazard,
            });
            Outcome::Undecided
        }
        Answer::Unknown(UnknownReason::GaveUp { reason }) => {
            context.notify(&ConstraintEvent::GaveUp {
                kind,
                reason: &reason,
            });
            Outcome::Undecided
        }
    })
}
