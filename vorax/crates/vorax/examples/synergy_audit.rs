//! Reproduce information lost by the legacy value features, without changing advice.
use serde_json::{json, Value};
use vorax::{data::data, game::{dispatch, ultimate_activity, THRESHOLDS}, types::*, value};

fn monster(archetype: Archetype, rarity: Rarity) -> Option<Monster> {
    Some(Monster { archetype, rarity, activity: 100, quantity: 100 })
}

fn board(tanks: Tanks, instruments: &[(u8, i32)]) -> GameState {
    let mut s = GameState { tanks, round: 6, phase: Phase::RoundEnd, t8000: true,
                           t28000: true, rng_state: 12345, ..GameState::default() };
    for &(id, acquired_round) in instruments { s.instruments.push(Equipped { id, acquired_round }); }
    s
}

fn compare(name: &str, a: &GameState, b: &GameState) -> Value {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    value::features(data(), a, &mut x);
    value::features(data(), b, &mut y);
    let next_a = dispatch(a, Action::Confirm);
    let next_b = dispatch(b, Action::Confirm);
    json!({"case": name, "legacy_features_identical": x == y,
           "before": [ultimate_activity(a), ultimate_activity(b)],
           "legacy_prediction": [value::board_value(a), value::board_value(b)],
           "after_one_round_same_seed": [ultimate_activity(&next_a), ultimate_activity(&next_b)],
           "instrument_ids": [a.instruments.iter().map(|e| (e.id, e.acquired_round)).collect::<Vec<_>>(),
                              b.instruments.iter().map(|e| (e.id, e.acquired_round)).collect::<Vec<_>>()],
           "archetype_rarity": [a.monsters().map(|m| (m.archetype, m.rarity)).collect::<Vec<_>>(),
                                 b.monsters().map(|m| (m.archetype, m.rarity)).collect::<Vec<_>>() ]})
}

fn main() {
    use Archetype::*;
    use Rarity::*;
    let rare_agito = board([monster(Agito, Rare), monster(Pest, Normal), None, None, None, None], &[(4, 1)]);
    let rare_pest = board([monster(Agito, Normal), monster(Pest, Rare), None, None, None, None], &[(4, 1)]);
    let tanks = [monster(Pest, Magic), monster(Pest, Magic), monster(Pest, Magic), None, None, None];
    let double_restraint = board(tanks, &[(13, 1), (13, 3), (1, 1)]);
    let double_specimen = board(tanks, &[(13, 1), (1, 3), (1, 1)]);
    let periodic = data().instrument(25).unwrap();
    let vorax::data::Timing::Periodic(every) = periodic.timing else { panic!("periodic instrument required") };
    let due = board(tanks, &[(periodic.id, 6 - every)]);
    let later = board(tanks, &[(periodic.id, 7 - every)]);
    let result = json!({"thresholds_in_engine": THRESHOLDS,
                       "balanced_same_sum": {"three_by_three": 3 * 3, "two_by_four": 2 * 4},
                       "equal_addition_at_q200_a50": {"add50_quantity": 50 * 50, "add50_activity": 200 * 50},
                       "cases": [compare("archetype_and_rarity_joint_condition", &rare_agito, &rare_pest),
                                 compare("duplicate_instrument_counts", &double_restraint, &double_specimen),
                                 compare("periodic_acquisition_round", &due, &later)]});
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
