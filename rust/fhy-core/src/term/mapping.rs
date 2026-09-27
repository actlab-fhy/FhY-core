//! Alpha equivalence of two maps keyed by identifiers.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::identifier::Identifier;

use super::binder::AlphaEquivalence;
use super::renaming::AlphaRenaming;

/// Return whether the identifier-keyed maps `left` and `right` are
/// alpha-equivalent under `renaming`: their keys are references that
/// correspond by `renaming`, and the values of corresponding keys are
/// alpha-equivalent under it.
///
/// `left` is given as its pairs, in the order the values are compared. The
/// maps must have as many entries; each key of `left` resolves
/// ([`AlphaRenaming::resolve`]) to a distinct key of `right`, with no two
/// left keys resolving to one; each pair of keys must correspond
/// ([`AlphaRenaming::is_corresponding`], which refuses a capture); and then
/// each pair of values is compared, stopping at the first that differs. The
/// keys are not binders: a node whose keys bind enters a frame for them
/// first, and compares the values under it.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::expression::Expression;
/// use fhy_core::identifier::Identifier;
/// use fhy_core::term::{AlphaRenaming, is_mapping_alpha_equivalent_under};
///
/// let (a, b, x, y) = (Identifier::new("a"), Identifier::new("b"), Identifier::new("x"), Identifier::new("y"));
/// let left = HashMap::from([(a.clone(), Expression::from(x.clone()) + 1)]);
/// let right = HashMap::from([(b.clone(), Expression::from(y.clone()) + 1)]);
/// let renaming = AlphaRenaming::new(HashMap::from([(a, b), (x, y)]))
///     .expect("the renaming is injective");
///
/// let Ok(is_equivalent) = is_mapping_alpha_equivalent_under(&left, &right, &renaming);
/// assert!(is_equivalent);
/// let Ok(is_equivalent) =
///     is_mapping_alpha_equivalent_under(&left, &right, &AlphaRenaming::default());
/// assert!(!is_equivalent);
/// ```
///
/// # Errors
///
/// Returns the first value comparison's error.
pub fn is_mapping_alpha_equivalent_under<'a, V, L, S>(
    left: L,
    right: &HashMap<Identifier, V, S>,
    renaming: &AlphaRenaming,
) -> Result<bool, V::Error>
where
    V: AlphaEquivalence + 'a,
    L: IntoIterator<Item = (&'a Identifier, &'a V)>,
    L::IntoIter: ExactSizeIterator,
    S: BuildHasher,
{
    let left = left.into_iter();
    if left.len() != right.len() {
        return Ok(false);
    }
    let mut resolved_pairs = Vec::with_capacity(left.len());
    let mut resolved_keys = HashSet::with_capacity(left.len());
    for (key, value) in left {
        let resolved = renaming.resolve(key);
        if !resolved_keys.insert(resolved) {
            return Ok(false);
        }
        resolved_pairs.push((key, resolved, value));
    }
    let Some(right_values) = resolved_pairs
        .iter()
        .map(|(_, resolved, _)| right.get(*resolved))
        .collect::<Option<Vec<&V>>>()
    else {
        return Ok(false);
    };
    for ((key, resolved, value), right_value) in resolved_pairs.iter().zip(right_values) {
        if !renaming.is_corresponding(key, resolved)
            || !value.is_alpha_equivalent_under(right_value, renaming)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}
