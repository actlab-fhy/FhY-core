//! Builders of objectives, keys and measurements for the measurement
//! stories and properties.

use fhy_core::search_space::{ConfigurationKey, Direction, Measurement, Objective};

use super::search::{build_tiling_space, tiling_configurations, tiling_entries};
use super::search_space::configure;

/// Return the objective `name`, better in `direction`.
///
/// # Panics
///
/// Panics if `name` is empty.
pub(crate) fn objective(name: &str, direction: Direction) -> Objective {
    Objective::new(name, direction).expect("the objective has a name")
}

/// Return the key of the tiling space's complete configuration at `index`
/// (0 to 4) of its lexicographic order.
pub(crate) fn tiling_key(index: usize) -> ConfigurationKey {
    let tiling = build_tiling_space();
    let configuration = &tiling_configurations(&tiling)[index];
    configure(&tiling.space, tiling_entries(&tiling, configuration)).key()
}

/// Return the successful measurement of the tiling configuration at
/// `index` holding `values`.
///
/// # Panics
///
/// Panics if the measurement is refused.
pub(crate) fn measured(index: usize, values: &[(&Objective, f64)]) -> Measurement {
    Measurement::ok(
        tiling_key(index),
        values
            .iter()
            .map(|(objective, value)| ((*objective).clone(), *value))
            .collect(),
    )
    .unwrap_or_else(|error| panic!("the measurement is refused: {error}"))
}
