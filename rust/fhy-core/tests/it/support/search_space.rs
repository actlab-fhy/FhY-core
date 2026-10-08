//! Helpers of the search-space tests: builders of params, variables,
//! alternatives, choices, spaces and configurations; a solver whose
//! simplifier decides ground equations; and two test implementations of
//! the traits, shaped like MOGA-VM's `ArrayTileKnob` and
//! `RealizationOption`, with the resolver that reads them back.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};

use fhy_core::constraint::{
    Constraint, ConstraintSystem, CustomConstraint, EquationConstraint, OpaqueValue, Polarity,
    SetConstraint, Value,
};
use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BoxError, Foreign, ForeignError, ForeignPart, NoForeign, Part, Resolve};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, CustomDomain, IntegerDomain, Param, ParamContext, ParamDomain, Sign,
    ZeroInclusion,
};
use fhy_core::search_space::wire::{ChoiceData, VariableData};
use fhy_core::search_space::{
    Alternative, Choice, Condition, Configuration, ConfigurationErrors, Forbidden,
    PlainAlternative, PlainVariable, Space, SpaceError, Variable,
};
use fhy_core::solver::{GroundSimplifier, Solver};
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use serde::{Deserialize, Serialize};

use super::constraint::{int, member_set};

/// Return a solver whose simplifier decides ground equations, so the
/// conditions and forbidden clauses over numeric variables evaluate.
pub(crate) fn ground_solver() -> Solver {
    Solver::new().with_simplifier(GroundSimplifier::new())
}

/// Return the value a configuration gives a choice that chooses the
/// alternative named `alternative`.
pub(crate) fn chosen(alternative: &Identifier) -> Value {
    Value::Identifier(alternative.clone())
}

/// Return whether `left` and `right` are alpha-equivalent with no binder in
/// scope, in each direction.
///
/// # Panics
///
/// Panics if a comparison fails.
pub(crate) fn compare_alpha_both_ways<T: AlphaEquivalence>(left: &T, right: &T) -> [bool; 2]
where
    T::Error: std::fmt::Debug,
{
    [
        left.is_alpha_equivalent(right)
            .expect("the comparison succeeds"),
        right
            .is_alpha_equivalent(left)
            .expect("the comparison succeeds"),
    ]
}

/// Return the integer values of `values`.
pub(crate) fn int_values(values: &[i64]) -> Vec<Value> {
    values.iter().copied().map(int).collect()
}

/// Return the param over the categories `values`, its variable fresh,
/// narrowed by the constraints `constrain` builds over that variable.
///
/// # Panics
///
/// Panics if the domain or a constraint is refused.
pub(crate) fn categorical_where(
    values: Vec<Value>,
    constrain: impl FnOnce(&Identifier) -> Vec<Constraint>,
) -> Param {
    let variable = Identifier::new("p");
    let constraints = constrain(&variable);
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(CategoricalDomain::new(values).expect("the categories are valid")),
        variable,
        constraints,
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return the param over the categories `values` whose variable is
/// `variable`, so that params built apart can be structurally equivalent.
///
/// # Panics
///
/// Panics if the domain is refused.
pub(crate) fn categorical_of(variable: &Identifier, values: Vec<Value>) -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(CategoricalDomain::new(values).expect("the categories are valid")),
        variable.clone(),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return the param over the categories `values`, its variable fresh.
pub(crate) fn categorical(values: Vec<Value>) -> Param {
    categorical_where(values, |_| Vec::new())
}

/// Return the param over the integer categories `values`.
pub(crate) fn int_param(values: &[i64]) -> Param {
    categorical(int_values(values))
}

/// Return the param over the non-negative integers, its variable fresh.
pub(crate) fn natural_param() -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
        Identifier::new("n"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return the equation `variable % 2 == 1`, a constraint no integer bound
/// decides, so a param holding it asks the solver to check a value.
pub(crate) fn odd(variable: &Identifier) -> Constraint {
    Constraint::from(EquationConstraint::new(
        super::param::reference(variable)
            .floor_mod(super::param::literal(2))
            .equals(super::param::literal(1)),
    ))
}

/// Return the param over the odd non-negative integers, its variable
/// fresh: a natural number constrained by [`odd`], which only a solver
/// that decides ground equations checks a value against.
///
/// # Panics
///
/// Panics if the param is refused.
pub(crate) fn odd_natural_param() -> Param {
    let param = natural_param();
    let solver = Solver::new();
    param
        .with_constraint(odd(param.variable()), &ParamContext::new(&solver))
        .expect("the param is valid")
}

/// Return the plain variable `name` over `param`.
pub(crate) fn plain_variable(name: &Identifier, param: Param) -> Part<dyn Variable> {
    Part::new(PlainVariable::new(name.clone(), param))
}

/// Return the plain variable `name` over the integer categories `values`.
pub(crate) fn int_variable(name: &Identifier, values: &[i64]) -> Part<dyn Variable> {
    plain_variable(name, int_param(values))
}

/// Return the plain alternative `name` holding `variables` and `choices`.
///
/// # Panics
///
/// Panics if the alternative's names repeat.
pub(crate) fn plain_alternative(
    name: &Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
) -> Part<dyn Alternative> {
    Part::new(
        PlainAlternative::new(name.clone(), variables, choices)
            .expect("the alternative's names are distinct"),
    )
}

/// Return the plain alternative `name` holding nothing.
pub(crate) fn bare_alternative(name: &Identifier) -> Part<dyn Alternative> {
    plain_alternative(name, Vec::new(), Vec::new())
}

/// Return the choice `name` among `alternatives`.
///
/// # Panics
///
/// Panics if the choice is refused.
pub(crate) fn choice_of(name: &Identifier, alternatives: Vec<Part<dyn Alternative>>) -> Choice {
    Choice::new(name.clone(), alternatives).expect("the choice is valid")
}

/// Return the choice nesting `depth` levels of choices, one alternative
/// each, the innermost alternative holding `leaf_variables`; or the
/// refusal of the outermost level, which alone may be refused.
///
/// # Panics
///
/// Panics if a level below the outermost is refused, or `depth` is zero.
pub(crate) fn build_choice_chain(
    depth: usize,
    leaf_variables: Vec<Part<dyn Variable>>,
) -> Result<Choice, SpaceError> {
    assert!(depth > 0, "a chain has at least one choice");
    let leaf = plain_alternative(&Identifier::new("leaf"), leaf_variables, Vec::new());
    let mut choice = Choice::new(Identifier::new("level"), vec![leaf]);
    for _ in 1..depth {
        let inner = choice.expect("an inner level is within the cap");
        let holder = plain_alternative(&Identifier::new("holder"), Vec::new(), vec![inner]);
        choice = Choice::new(Identifier::new("level"), vec![holder]);
    }
    choice
}

/// Return the space `name` of the top-level `variables` and `choices`,
/// with no condition or forbidden clause.
///
/// # Panics
///
/// Panics if the space is refused.
pub(crate) fn space_of(
    name: &Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
) -> Space {
    Space::new(name.clone(), variables, choices, Vec::new(), Vec::new())
        .expect("the space is valid")
}

/// Return the system of `constraints`.
///
/// # Panics
///
/// Panics if a key fails.
pub(crate) fn system(constraints: impl IntoIterator<Item = Constraint>) -> ConstraintSystem {
    ConstraintSystem::new(constraints).expect("the system's keys do not fail")
}

/// Return the set constraint that the choice `choice` chose one of
/// `alternatives`.
pub(crate) fn chooses(choice: &Identifier, alternatives: &[&Identifier]) -> Constraint {
    Constraint::from(SetConstraint::new(
        choice.clone(),
        member_set(alternatives.iter().map(|alternative| chosen(alternative))),
        Polarity::In,
    ))
}

/// Return the condition that `target` is active while `constraints` hold.
pub(crate) fn condition(
    target: &Identifier,
    constraints: impl IntoIterator<Item = Constraint>,
) -> Condition {
    Condition::new(target.clone(), system(constraints))
}

/// Return the clause forbidding every configuration satisfying
/// `constraints`.
pub(crate) fn forbidden(constraints: impl IntoIterator<Item = Constraint>) -> Forbidden {
    Forbidden::new(system(constraints))
}

/// Return the configuration of `space` with `entries`, checked with the
/// ground solver.
pub(crate) fn try_configure(
    space: &Space,
    entries: impl IntoIterator<Item = (Identifier, Value)>,
) -> Result<Configuration, ConfigurationErrors> {
    let solver = ground_solver();
    Configuration::new(space, entries, &ParamContext::new(&solver))
}

/// Return the configuration of `space` with `entries`.
///
/// # Panics
///
/// Panics if the configuration is refused.
pub(crate) fn configure(
    space: &Space,
    entries: impl IntoIterator<Item = (Identifier, Value)>,
) -> Configuration {
    try_configure(space, entries).unwrap_or_else(|errors| panic!("refused: {errors}"))
}

/// The type id of [`TileKnob`].
pub(crate) const TILE_KNOB: &str = "test.tile_knob";
/// The type id of [`Realization`].
pub(crate) const REALIZATION: &str = "test.realization";

/// A variable shaped like MOGA-VM's `ArrayTileKnob`: a tile size over the
/// index symbols it tiles, which are references.
#[derive(Debug, Clone)]
pub(crate) struct TileKnob {
    pub(crate) name: Identifier,
    pub(crate) param: Param,
    pub(crate) notes: Vec<Note>,
    pub(crate) index_symbols: Vec<Identifier>,
}

/// The payload of a [`TileKnob`]'s foreign part.
#[derive(Serialize, Deserialize)]
struct TileKnobPayload {
    name: Identifier,
    param: Param,
    notes: Vec<Note>,
    index_symbols: Vec<Identifier>,
}

impl TileKnob {
    /// Return the knob `name` over `param` tiling `index_symbols`.
    pub(crate) fn part(
        name: &Identifier,
        param: Param,
        index_symbols: &[&Identifier],
    ) -> Part<dyn Variable> {
        Part::new(Self {
            name: name.clone(),
            param,
            notes: Vec::new(),
            index_symbols: index_symbols.iter().map(|&symbol| symbol.clone()).collect(),
        })
    }
}

impl ForeignPart for TileKnob {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("TileKnob")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        let payload = TileKnobPayload {
            name: self.name.clone(),
            param: self.param.clone(),
            notes: self.notes.clone(),
            index_symbols: self.index_symbols.clone(),
        };
        let data = serde_json::to_string(&payload).map_err(|error| ForeignError::Failed {
            type_id: TILE_KNOB.to_owned(),
            source: Box::new(error),
        })?;
        Ok(Foreign::new(TILE_KNOB, data))
    }
}

impl Variable for TileKnob {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(TILE_KNOB)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn notes(&self) -> &[Note] {
        &self.notes
    }

    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        Ok(self.index_symbols == other.index_symbols)
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Variable,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        Ok(self.index_symbols.len() == other.index_symbols.len()
            && self
                .index_symbols
                .iter()
                .zip(&other.index_symbols)
                .all(|(left, right)| renaming.is_corresponding(left, right)))
    }

    fn eq_part(&self, other: &dyn Variable) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|other| {
            self.name == other.name
                && self.param == other.param
                && self.notes == other.notes
                && self.index_symbols == other.index_symbols
        })
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        let mut state = state;
        self.name.hash(&mut state);
        self.param.hash(&mut state);
        self.notes.hash(&mut state);
        self.index_symbols.hash(&mut state);
    }
}

/// An alternative shaped like MOGA-VM's `RealizationOption`: it binds the
/// axes of its walk, and its own data is a walk order over identifiers
/// (references, the axes among them) and a tag compared by value.
#[derive(Debug, Clone)]
pub(crate) struct Realization {
    pub(crate) name: Identifier,
    pub(crate) variables: Vec<Part<dyn Variable>>,
    pub(crate) choices: Vec<Choice>,
    pub(crate) notes: Vec<Note>,
    pub(crate) axes: Vec<Identifier>,
    pub(crate) order: Vec<Identifier>,
    pub(crate) tag: i64,
}

/// The payload of a [`Realization`]'s foreign part.
#[derive(Serialize, Deserialize)]
struct RealizationPayload {
    name: Identifier,
    variables: Vec<VariableData>,
    choices: Vec<ChoiceData>,
    notes: Vec<Note>,
    axes: Vec<Identifier>,
    order: Vec<Identifier>,
    tag: i64,
}

impl Realization {
    /// Return the realization `name` holding `variables`, binding `axes`,
    /// whose walk order is `order` and whose tag is `tag`.
    pub(crate) fn new(
        name: &Identifier,
        variables: Vec<Part<dyn Variable>>,
        axes: &[&Identifier],
        order: &[&Identifier],
        tag: i64,
    ) -> Self {
        Self {
            name: name.clone(),
            variables,
            choices: Vec::new(),
            notes: Vec::new(),
            axes: axes.iter().map(|&axis| axis.clone()).collect(),
            order: order.iter().map(|&axis| axis.clone()).collect(),
            tag,
        }
    }

    /// Return this realization holding the sub-choices `choices`.
    pub(crate) fn with_choices(self, choices: Vec<Choice>) -> Self {
        Self { choices, ..self }
    }

    /// Return this realization as a part.
    pub(crate) fn into_part(self) -> Part<dyn Alternative> {
        Part::new(self)
    }
}

impl ForeignPart for Realization {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Realization")
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        let payload = RealizationPayload {
            name: self.name.clone(),
            variables: self
                .variables
                .iter()
                .map(VariableData::of)
                .collect::<Result<_, _>>()?,
            choices: self
                .choices
                .iter()
                .map(ChoiceData::of)
                .collect::<Result<_, _>>()?,
            notes: self.notes.clone(),
            axes: self.axes.clone(),
            order: self.order.clone(),
            tag: self.tag,
        };
        let data = serde_json::to_string(&payload).map_err(|error| ForeignError::Failed {
            type_id: REALIZATION.to_owned(),
            source: Box::new(error),
        })?;
        Ok(Foreign::new(REALIZATION, data))
    }
}

impl Alternative for Realization {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(REALIZATION)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        &self.variables
    }

    fn choices(&self) -> &[Choice] {
        &self.choices
    }

    fn notes(&self) -> &[Note] {
        &self.notes
    }

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        Ok(self.axes.clone())
    }

    fn is_extension_structurally_equivalent(
        &self,
        other: &dyn Alternative,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        Ok(self.order == other.order && self.tag == other.tag)
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Alternative,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        Ok(self.tag == other.tag
            && self.order.len() == other.order.len()
            && self
                .order
                .iter()
                .zip(&other.order)
                .all(|(left, right)| renaming.is_corresponding(left, right)))
    }

    fn eq_part(&self, other: &dyn Alternative) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|other| {
            self.name == other.name
                && self.variables == other.variables
                && self.choices == other.choices
                && self.notes == other.notes
                && self.axes == other.axes
                && self.order == other.order
                && self.tag == other.tag
        })
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        let mut state = state;
        self.name.hash(&mut state);
        self.variables.hash(&mut state);
        self.choices.hash(&mut state);
        self.notes.hash(&mut state);
        self.axes.hash(&mut state);
        self.order.hash(&mut state);
        self.tag.hash(&mut state);
    }
}

/// The resolver of the test implementations: [`TileKnob`] and
/// [`Realization`] by their type ids, and no other part.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ImplementorResolver;

/// Return the failure of reading a payload of `type_id`.
fn unreadable(
    type_id: &str,
    error: impl std::error::Error + Send + Sync + 'static,
) -> ForeignError {
    ForeignError::Failed {
        type_id: type_id.to_owned(),
        source: Box::new(error),
    }
}

impl Resolve<Part<dyn Variable>> for ImplementorResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
        if foreign.type_id() != TILE_KNOB {
            return NoForeign.resolve(foreign);
        }
        let payload: TileKnobPayload =
            serde_json::from_str(foreign.data()).map_err(|error| unreadable(TILE_KNOB, error))?;
        Ok(Part::new(TileKnob {
            name: payload.name,
            param: payload.param,
            notes: payload.notes,
            index_symbols: payload.index_symbols,
        }))
    }
}

impl Resolve<Part<dyn Alternative>> for ImplementorResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
        if foreign.type_id() != REALIZATION {
            return NoForeign.resolve(foreign);
        }
        let payload: RealizationPayload =
            serde_json::from_str(foreign.data()).map_err(|error| unreadable(REALIZATION, error))?;
        let solver = Solver::new();
        let context = ParamContext::new(&solver);
        let variables = payload
            .variables
            .into_iter()
            .map(|variable| variable.build(self, &context))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| unreadable(REALIZATION, error))?;
        let choices = payload
            .choices
            .into_iter()
            .map(|choice| choice.build(self, &context))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| unreadable(REALIZATION, error))?;
        Ok(Part::new(Realization {
            name: payload.name,
            variables,
            choices,
            notes: payload.notes,
            axes: payload.axes,
            order: payload.order,
            tag: payload.tag,
        }))
    }
}

impl Resolve<Part<dyn OpaqueValue>> for ImplementorResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
        NoForeign.resolve(foreign)
    }
}

impl Resolve<Part<dyn CustomConstraint>> for ImplementorResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
        NoForeign.resolve(foreign)
    }
}

impl Resolve<Part<dyn CustomDomain>> for ImplementorResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
        NoForeign.resolve(foreign)
    }
}
