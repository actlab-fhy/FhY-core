//! Counting a space's complete configurations, and drawing one uniformly
//! from them, through the space's relaxation: the space without its
//! conditions, forbidden clauses and param constraints, whose counts have a
//! closed form.

use std::collections::HashMap;
use std::num::NonZeroU32;

use num_bigint::BigUint;

use crate::identifier::Identifier;
use crate::param::{ParamContext, ParamDomain};

use super::configuration::{Activity, Configuration};
use super::domain::{Coordinate, StepDomain};
use super::error::TraceError;
use super::exploration::{Cardinality, walk};
use super::oracle::ExhaustiveOracle;
use super::recorder::Recorded;
use super::rng::Rng;
use super::space::{Decision, Space};
use super::step::{
    ParamStepDomain, are_bounds, decision_domain, param_step_domain, try_extend, unrank_permutation,
};

/// The decisions of a space as a tree: per decision its children, by
/// alternative for a choice, and the top-level decisions.
struct Tree {
    /// The top-level decisions' canonical positions.
    tops: Vec<usize>,
    /// Per decision, per alternative, the canonical positions of the
    /// decisions under it; empty for a variable.
    children: Vec<Vec<Vec<usize>>>,
}

impl Tree {
    /// Return the tree of `space`.
    fn of(space: &Space) -> Self {
        let count = space.decision_count();
        let mut children: Vec<Vec<Vec<usize>>> = (0..count)
            .map(|position| match space.decision_at(position) {
                Decision::Choice(choice) => vec![Vec::new(); choice.alternatives().len()],
                Decision::Variable(_) => Vec::new(),
            })
            .collect();
        let mut tops = Vec::new();
        for position in 0..count {
            match space.parent_at(position) {
                Some((choice, alternative)) => children[choice][alternative].push(position),
                None => tops.push(position),
            }
        }
        Self { tops, children }
    }
}

/// A component's count, before the components combine.
enum Count {
    Finite(BigUint),
    Unbounded(Identifier),
    Unknown(Identifier),
}

impl Count {
    /// Return the product of `counts`, each at least one when finite: an
    /// unknown factor makes it unknown, else an unbounded one unbounded.
    fn product(counts: impl IntoIterator<Item = Self>) -> Self {
        let mut product = BigUint::from(1_u8);
        let mut unbounded = None;
        for count in counts {
            match count {
                Self::Finite(count) => product *= count,
                Self::Unknown(decision) => return Self::Unknown(decision),
                Self::Unbounded(decision) => {
                    unbounded.get_or_insert(decision);
                }
            }
        }
        unbounded.map_or(Self::Finite(product), Self::Unbounded)
    }

    /// Return the sum of `counts`, as [`product`](Self::product) combines.
    fn sum(counts: impl IntoIterator<Item = Self>) -> Self {
        let mut sum = BigUint::ZERO;
        let mut unbounded = None;
        for count in counts {
            match count {
                Self::Finite(count) => sum += count,
                Self::Unknown(decision) => return Self::Unknown(decision),
                Self::Unbounded(decision) => {
                    unbounded.get_or_insert(decision);
                }
            }
        }
        unbounded.map_or(Self::Finite(sum), Self::Unbounded)
    }
}

/// Return the relaxed count of the decision at `position`: a variable's
/// domain's, or a choice's sum over its alternatives of the product of
/// their decisions'.
fn count_relaxed(space: &Space, tree: &Tree, position: usize) -> Result<Count, TraceError> {
    match space.decision_at(position) {
        Decision::Choice(_) => {
            let alternatives = tree.children[position]
                .iter()
                .map(|children| {
                    children
                        .iter()
                        .map(|&child| count_relaxed(space, tree, child))
                        .collect::<Result<Vec<_>, _>>()
                        .map(Count::product)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Count::sum(alternatives))
        }
        decision @ Decision::Variable(variable) => match decision_domain(decision) {
            Ok(domain) => Ok(Count::Finite(domain.cardinality())),
            Err(TraceError::NotEnumerable { decision }) => {
                Ok(match param_step_domain(variable.get().param()) {
                    ParamStepDomain::Unknown => Count::Unknown(decision),
                    ParamStepDomain::Finite(_) | ParamStepDomain::Unbounded => {
                        Count::Unbounded(decision)
                    }
                })
            }
            Err(error) => Err(error),
        },
    }
}

/// Return the relaxed counts of every decision, by canonical position.
///
/// # Errors
///
/// Returns [`TraceError::NotEnumerable`] for a variable with no finite
/// domain, and [`TraceError::Hook`].
fn count_every_relaxed(space: &Space, tree: &Tree) -> Result<Vec<BigUint>, TraceError> {
    (0..space.decision_count())
        .map(|position| match count_relaxed(space, tree, position)? {
            Count::Finite(count) => Ok(count),
            Count::Unbounded(decision) | Count::Unknown(decision) => {
                Err(TraceError::NotEnumerable { decision })
            }
        })
        .collect()
}

/// Draw a complete configuration of `space` uniformly, as
/// [`Space::sample_uniform`] documents.
pub(super) fn sample_uniformly(
    space: &Space,
    rng: &mut Rng,
    context: &ParamContext<'_>,
    attempts: NonZeroU32,
) -> Result<Recorded, TraceError> {
    let tree = Tree::of(space);
    let counts = count_every_relaxed(space, &tree)?;
    let total: BigUint = tree.tops.iter().map(|&top| &counts[top]).product();
    for _ in 0..attempts.get() {
        let mut point = HashMap::new();
        let mut index = rng.below_big(&total);
        for &top in &tree.tops {
            let digit = &index % &counts[top];
            index /= &counts[top];
            decode(space, &tree, &counts, top, digit, &mut point);
        }
        let Some((configuration, multiplicity)) = apply(space, &counts, &point, context)? else {
            continue;
        };
        if rng.below_big(&multiplicity) == BigUint::ZERO {
            let trace = configuration.trace(context)?;
            return Ok(Recorded::new(trace, Some(configuration)));
        }
    }
    Err(TraceError::AttemptsExhausted {
        attempts: attempts.get(),
    })
}

/// Decode `digit`, below the relaxed count of the decision at `position`,
/// into coordinates of it and the decisions under its chosen alternative.
fn decode(
    space: &Space,
    tree: &Tree,
    counts: &[BigUint],
    position: usize,
    digit: BigUint,
    point: &mut HashMap<usize, Coordinate>,
) {
    match space.decision_at(position) {
        Decision::Choice(_) => {
            let mut digit = digit;
            for (alternative, children) in tree.children[position].iter().enumerate() {
                let size: BigUint = children.iter().map(|&child| &counts[child]).product();
                if digit < size {
                    let index = u64::try_from(alternative).unwrap_or(u64::MAX);
                    point.insert(position, Coordinate::Index(index));
                    for &child in children {
                        let child_digit = &digit % &counts[child];
                        digit /= &counts[child];
                        decode(space, tree, counts, child, child_digit, point);
                    }
                    return;
                }
                digit -= size;
            }
        }
        decision @ Decision::Variable(_) => {
            let coordinate = match decision_domain(decision) {
                Ok(StepDomain::Order(order)) => {
                    Coordinate::Order(unrank_permutation(order.elements().len(), &digit))
                }
                _ => Coordinate::Index(u64::try_from(digit).unwrap_or(0)),
            };
            point.insert(position, coordinate);
        }
    }
}

/// Return the configuration of `space` the relaxed `point` gives, and the
/// number of relaxed points that give it, or `None` when a value or a
/// forbidden clause refuses it.
fn apply(
    space: &Space,
    counts: &[BigUint],
    point: &HashMap<usize, Coordinate>,
    context: &ParamContext<'_>,
) -> Result<Option<(Configuration, BigUint)>, TraceError> {
    let mut configuration = Configuration::empty(space);
    let mut multiplicity = BigUint::from(1_u8);
    for &position in space.order_positions() {
        let Some(coordinate) = point.get(&position) else {
            continue;
        };
        let node = space.decision_at(position);
        let name = node.name();
        if configuration.activity(name) == Some(Activity::Active) {
            let domain = decision_domain(node)?;
            let Some(value) = domain.value_at(coordinate) else {
                return Ok(None);
            };
            match try_extend(&configuration, name, value, context)? {
                Some(extended) => configuration = extended,
                None => return Ok(None),
            }
        } else {
            let is_parent_active = space.parent_at(position).is_none_or(|(choice, _)| {
                configuration.activity(space.decision_at(choice).name()) == Some(Activity::Active)
            });
            if is_parent_active {
                multiplicity *= &counts[position];
            }
        }
    }
    Ok(Some((configuration, multiplicity)))
}

/// Count the complete configurations of `space`, as [`Space::cardinality`]
/// documents.
pub(super) fn count_space(
    space: &Space,
    context: &ParamContext<'_>,
    budget: u64,
) -> Result<Cardinality, TraceError> {
    let tree = Tree::of(space);
    let components = find_components(space, &tree);
    let mut answers = Vec::new();
    for component in components {
        let answer = if is_closed_form(space, &component) {
            let counts = component
                .tops
                .iter()
                .map(|&top| count_relaxed(space, &tree, top))
                .collect::<Result<Vec<_>, _>>()?;
            match Count::product(counts) {
                Count::Finite(count) => Cardinality::Exact(count),
                Count::Unbounded(decision) => Cardinality::Unbounded { decision },
                Count::Unknown(decision) => Cardinality::Unknown { decision },
            }
        } else {
            count_by_enumeration(space, &component, context, budget)?
        };
        answers.push(answer);
    }
    Ok(combine(answers))
}

/// A set of top-level decisions no condition or forbidden clause links to
/// another set, and every decision at or under them.
struct Component {
    tops: Vec<usize>,
    members: Vec<bool>,
}

/// Return the top-level decision the decision at `position` is at or
/// under.
fn find_top(space: &Space, position: usize) -> usize {
    let mut current = position;
    while let Some((choice, _)) = space.parent_at(current) {
        current = choice;
    }
    current
}

/// Return the representative of `position`'s set in the union-find forest
/// `leader`, halving the path on the way.
fn root(leader: &mut [usize], position: usize) -> usize {
    let mut current = position;
    while leader[current] != current {
        leader[current] = leader[leader[current]];
        current = leader[current];
    }
    current
}

/// Return the components of `space`, in canonical order of their first
/// top-level decision.
fn find_components(space: &Space, tree: &Tree) -> Vec<Component> {
    let count = space.decision_count();
    let mut leader: Vec<usize> = (0..count).collect();
    let link = |left: usize, right: usize, leader: &mut Vec<usize>| {
        let (left, right) = (root(leader, left), root(leader, right));
        if left != right {
            leader[left.max(right)] = left.min(right);
        }
    };
    for position in 0..count {
        if let Some((_, references)) = space.condition_with_references_at(position) {
            let target = find_top(space, position);
            for &reference in references {
                link(target, find_top(space, reference), &mut leader);
            }
        }
    }
    for index in 0..space.forbidden().len() {
        let references = space.forbidden_references(index);
        if let Some((&first, rest)) = references.split_first() {
            for &reference in rest {
                link(
                    find_top(space, first),
                    find_top(space, reference),
                    &mut leader,
                );
            }
        }
    }
    let mut components: Vec<Component> = Vec::new();
    let mut by_root: HashMap<usize, usize> = HashMap::new();
    for &top in &tree.tops {
        let group = root(&mut leader, top);
        let index = *by_root.entry(group).or_insert_with(|| {
            components.push(Component {
                tops: Vec::new(),
                members: vec![false; count],
            });
            components.len() - 1
        });
        components[index].tops.push(top);
    }
    for position in 0..count {
        let group = root(&mut leader, find_top(space, position));
        if let Some(&index) = by_root.get(&group) {
            components[index].members[position] = true;
        }
    }
    components
}

/// Return whether `component`'s count has a closed form: no condition or
/// forbidden clause names its decisions, and no variable's param
/// constrains its values beyond the bounds of an integer domain.
fn is_closed_form(space: &Space, component: &Component) -> bool {
    let is_linked = (0..space.decision_count())
        .any(|position| component.members[position] && space.condition_at(position).is_some())
        || (0..space.forbidden().len()).any(|index| {
            space
                .forbidden_references(index)
                .iter()
                .any(|&reference| component.members[reference])
        });
    if is_linked {
        return false;
    }
    (0..space.decision_count())
        .filter(|&position| component.members[position])
        .all(|position| match space.decision_at(position) {
            Decision::Choice(_) => true,
            Decision::Variable(variable) => {
                let part = variable.get();
                let param = part.param();
                param.constraints().is_empty()
                    || (matches!(
                        param.domain(),
                        ParamDomain::Integer(_) | ParamDomain::IntervalInteger(_)
                    ) && part.search_domain().is_ok_and(|domain| domain.is_none())
                        && are_bounds(param))
            }
        })
}

/// Count `component`'s complete configurations by enumerating them,
/// checking at most `budget` of its paths.
fn count_by_enumeration(
    space: &Space,
    component: &Component,
    context: &ParamContext<'_>,
    budget: u64,
) -> Result<Cardinality, TraceError> {
    let mut oracle = ExhaustiveOracle::new();
    let mut found = BigUint::ZERO;
    let mut checked: u64 = 0;
    loop {
        if checked >= budget {
            return Ok(Cardinality::AtLeast(found));
        }
        checked += 1;
        let result = walk(
            space,
            |position| component.members[position],
            &mut oracle,
            context,
        );
        match result {
            Ok(_) => found += 1_u8,
            Err(error) if ExhaustiveOracle::is_backtrack(&error) => {}
            Err(TraceError::NotEnumerable { decision }) => {
                return Ok(classify_unenumerable(space, decision));
            }
            Err(error) => return Err(error),
        }
        if !oracle.advance() {
            return Ok(Cardinality::Exact(found));
        }
    }
}

/// Return the count a variable with no finite domain, `decision`, makes.
fn classify_unenumerable(space: &Space, decision: Identifier) -> Cardinality {
    let is_custom = space.decision(&decision).is_some_and(|node| match node {
        Decision::Variable(variable) => {
            matches!(
                param_step_domain(variable.get().param()),
                ParamStepDomain::Unknown
            )
        }
        Decision::Choice(_) => false,
    });
    if is_custom {
        Cardinality::Unknown { decision }
    } else {
        Cardinality::Unbounded { decision }
    }
}

/// Return the count the components' `answers` make together, as
/// [`Space::cardinality`] documents.
fn combine(answers: Vec<Cardinality>) -> Cardinality {
    if answers.contains(&Cardinality::Exact(BigUint::ZERO)) {
        return Cardinality::Exact(BigUint::ZERO);
    }
    if let Some(unknown) = answers
        .iter()
        .find(|answer| matches!(answer, Cardinality::Unknown { .. }))
    {
        return unknown.clone();
    }
    if let Some(unbounded) = answers
        .iter()
        .find(|answer| matches!(answer, Cardinality::Unbounded { .. }))
    {
        let others_have_one = answers.iter().all(|answer| match answer {
            Cardinality::Exact(count) | Cardinality::AtLeast(count) => {
                *count >= BigUint::from(1_u8)
            }
            Cardinality::Unbounded { .. } | Cardinality::Unknown { .. } => true,
        });
        return if others_have_one {
            unbounded.clone()
        } else {
            match unbounded {
                Cardinality::Unbounded { decision } => Cardinality::Unknown {
                    decision: decision.clone(),
                },
                other => other.clone(),
            }
        };
    }
    let mut product = BigUint::from(1_u8);
    let mut is_exact = true;
    for answer in answers {
        match answer {
            Cardinality::Exact(count) => product *= count,
            Cardinality::AtLeast(count) => {
                product *= count;
                is_exact = false;
            }
            Cardinality::Unbounded { .. } | Cardinality::Unknown { .. } => {}
        }
    }
    if is_exact {
        Cardinality::Exact(product)
    } else {
        Cardinality::AtLeast(product)
    }
}
