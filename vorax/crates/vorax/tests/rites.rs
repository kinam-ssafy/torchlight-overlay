//! Known rites are evaluated as a chain, not unconditionally selected by their name.

use serde_json::json;
use vorax::advisor::{self, ScreenState};

/// The recorded traces were made with tli-vorax-simulator's linear board value (7b4c9ba), not the trees the program
/// ships: every test plays with that one.
fn linear_board_value() {
    let _ = vorax::data::use_model(Some(include_str!("../../../fixtures/boardValue-linear.json")));
    assert!(matches!(vorax::data().model, Some(vorax::data::ValueModel::Rounds { .. })), "the tests need the linear board value");
}

fn board(hand: &[&str]) -> ScreenState {
    let tank = json!({ "archetype": "pest", "rarity": "magic", "activity": 81, "quantity": 340 });
    serde_json::from_value(json!({ "tanks": [tank, tank, tank, tank, tank, tank], "hand": hand, "instruments": [], "phase": "potion", "round": 9 })).unwrap()
}

#[test]
fn a_late_rite_is_evaluated_alongside_immediate_effects() {
    linear_board_value();
    let a = advisor::advise(&board(&["Mutagenic Powder", "Bewitching Tincture", "First Rite Holy Water: Confession"]), 2, false, false);
    assert!(!a.ranking.is_empty());
    assert!(!a.note.as_deref().is_some_and(|n| n.contains("always take it")), "{:?}", a.note);
    assert!(a.unknown.is_empty());
}

#[test]
fn purest_holy_water_is_simulated() {
    linear_board_value();
    assert!(!advisor::is_always_take("Purest Holy Water"));
    assert!(!advisor::is_always_take("Ritual Holy Water: Flagellation"));
    assert!(!advisor::is_always_take("Xenopanacea: Brood"));
}
