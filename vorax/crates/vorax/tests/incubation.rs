use vorax::{advisor::{self, Best, ScreenState}, game::{dispatch, ultimate_activity}, types::*, special};
use serde_json::json;

fn state(round: i32, hand: &[&str]) -> GameState {
    let screen: ScreenState = serde_json::from_value(json!({
        "tanks":[{"archetype":"aemberon","rarity":"normal","activity":1,"quantity":36},
          {"archetype":"agito","rarity":"rare","activity":162,"quantity":36},
          {"archetype":"boneSentry","rarity":"normal","activity":1,"quantity":36},null,null,null],
        "instruments":[],"hand":hand,"round":round,"phase":"potion","potionRefreshes":1
    })).unwrap();
    let mut unknown = vec![];
    let s = advisor::to_state(&screen, &mut unknown);
    assert!(unknown.is_empty());
    s
}

fn play(s: &GameState, index: i64, target: Option<i64>) -> GameState {
    let mut s = dispatch(s, Action::SelectCard { index });
    if let Some(tank) = target { s = dispatch(&s, Action::ToggleTarget { tank }); }
    dispatch(&s, Action::Confirm)
}

#[test]
fn observed_cocoon_then_gen1_advance_round_and_add_two_choices() {
    let s = state(2,&[special::NAMES[0]]);
    let next = play(&s,0,None);
    assert_eq!(next.round,3);
    assert_eq!(next.tanks[3],Some(Monster { archetype:Archetype::Pest,rarity:Rarity::Boss,activity:1,quantity:1 }));
    assert_eq!(ultimate_activity(&next), ultimate_activity(&s)+1);
    assert_eq!(next.hand.len(),5);
    assert_eq!(next.hand[0].id,43);
    assert_eq!(next.hand[1].id,44);
    let gen2 = play(&next,0,Some(0));
    assert_eq!(gen2.round,4);
    let first = gen2.tanks[0].unwrap();
    assert_eq!((first.archetype,first.activity,first.quantity),(Archetype::Pest,11,36));
    assert_eq!(gen2.hand.len(),5);
    assert_eq!(gen2.hand[0].id,45);
    assert_eq!(gen2.hand[1].id,46);
}

#[test]
fn gen_replication_preserves_stats_and_bonuses_both_copies() {
    for (id, bonus) in [(44,10),(46,20),(48,30)] {
        let mut s = state(3,&[special::NAMES[(id-42) as usize]]);
        s.tanks = [Some(Monster { archetype:Archetype::Pest,rarity:Rarity::Normal,activity:11,quantity:36 }),None,None,None,None,None];
        let next = play(&s,0,None);
        let expected = Monster { archetype:Archetype::Pest,rarity:Rarity::Normal,activity:11,quantity:36+bonus };
        assert_eq!(next.tanks[0],Some(expected));
        assert_eq!(next.tanks[1],Some(expected));
        assert_eq!(next.hand.len(),if id==48 {4} else {5});
    }
}

#[test]
fn reroll_discards_gen_offer_and_selecting_an_ordinary_card_ends_the_chain() {
    let s = state(3,&[special::NAMES[1],special::NAMES[2],"Probiotic Enzyme Solution"]);
    assert_eq!(s.special_offer,1);
    let rolled = dispatch(&s,Action::Refresh);
    assert_eq!(rolled.special_offer,0);
    assert_eq!(rolled.hand.len(),3);
    assert!(rolled.hand.iter().all(|c| !special::contains(c.id)));
    assert_eq!(rolled.potion_refreshes,0);
    let ordinary = play(&s,2,None);
    assert_eq!(ordinary.special_offer,0);
    assert_eq!(ordinary.hand.len(),3);
}

#[test]
fn reroll_clears_every_gen_stage_and_cannot_restore_it_next_round() {
    for stage in 1..=3 {
        let names: Vec<_> = special::offer(stage).iter()
            .map(|&id| special::NAMES[(id - special::COCOON) as usize]).collect();
        let mut s = state(3, &names);
        assert_eq!(s.special_offer, stage);
        s = dispatch(&s, Action::SelectCard { index: 0 });
        s = dispatch(&s, Action::ToggleTarget { tank: 0 });
        let rolled = dispatch(&s, Action::Refresh);
        assert_eq!(rolled.special_offer, 0);
        assert!(rolled.selected.is_none());
        assert!(rolled.targets.is_empty());
        assert_eq!(rolled.tanks, s.tanks);
        assert_eq!(rolled.round, s.round);
        assert_eq!(rolled.potion_refreshes, 0);
        assert!(rolled.hand.iter().all(|c| !special::contains(c.id)));
        // Use a deterministic ordinary offer to isolate subsequent round state
        // from the random reroll's possible targeting or potion-kit choices.
        let mut ordinary = rolled;
        ordinary.hand = state(rolled.round, &["Probiotic Enzyme Solution"]).hand;
        let next = play(&ordinary, 0, None);
        assert_eq!(next.round, rolled.round + 1);
        assert_eq!(next.special_offer, 0);
        assert!(next.hand.iter().all(|c| !special::contains(c.id)));
        // An unavailable reroll must leave the visible offer and RNG intact.
        let mut exhausted = s;
        exhausted.potion_refreshes = 0;
        assert_eq!(dispatch(&exhausted, Action::Refresh), exhausted);
    }
}

#[test]
fn an_unknown_effect_is_not_dropped_from_the_comparison() {
    let screen: ScreenState = serde_json::from_value(json!({
        "tanks":[null,null,null,null,null,null],"instruments":[],
        "hand":["Viviparous Hormone","New unimplemented potion"],"round":2,"phase":"potion"
    })).unwrap();
    for advice in [advisor::advise(&screen,1,false,false),advisor::advise_plan(&screen,Some(1),false)] {
        assert!(matches!(advice.best,Best::Nothing));
        assert!(advice.ranking.is_empty());
        assert_eq!(advice.unknown,["New unimplemented potion"]);
    }
}

#[test]
fn intervening_tool_rewards_and_their_reroll_preserve_the_next_gen_offer() {
    let mut s = state(2, &[special::NAMES[0]]);
    s.tanks[1].as_mut().unwrap().activity = 1000;
    let mut next = play(&s, 0, None);
    assert_eq!(next.phase, Phase::InstrumentSelect);
    assert_eq!(next.pending_picks, 2);
    assert_eq!(next.special_offer, 1);
    next.instrument_refreshes = 1;
    next = dispatch(&next, Action::Refresh);
    assert_eq!(next.instrument_refreshes, 0);
    assert_eq!(next.special_offer, 1);
    next = play(&next, 0, None);
    assert_eq!(next.pending_picks, 1);
    assert_eq!(next.special_offer, 1);
    next = play(&next, 0, None);
    assert_eq!(next.phase, Phase::PotionSelect);
    assert_eq!(next.round, 3);
    assert_eq!(next.special_offer, 1);
    assert_eq!(next.hand.len(), 5);
    assert_eq!((next.hand[0].id, next.hand[1].id), (43, 44));
}

#[test]
fn holy_chain_defers_its_effect_until_the_fourth_rite() {
    let mut s = state(3, &[special::NAMES[8]]);
    s.t8000 = true;
    s.t28000 = true;
    let original = s.tanks;
    for (next_id, round) in [(51,4),(52,5),(53,6)] {
        s = play(&s, 0, None);
        assert_eq!(s.round, round);
        assert_eq!(s.tanks, original);
        assert_eq!(s.hand.len(), 4);
        assert_eq!(s.hand[0].id, next_id);
    }
    let a: i64 = original.iter().flatten().map(|m| m.activity).sum();
    let q: i64 = original.iter().flatten().map(|m| m.quantity).sum();
    s = play(&s, 0, None);
    assert_eq!(s.monsters().count(), 6);
    assert!(s.monsters().all(|m| m.archetype == Archetype::Agito && m.activity == a && m.quantity == q));
    assert_eq!(s.special_offer, 0);
}

#[test]
fn dissection_requires_a_bone_and_distinguishes_extra_from_replacement() {
    let mut s = state(3, &[special::NAMES[13]]);
    s.t8000 = true;
    s.t28000 = true;
    s.tanks[3] = Some(Monster {archetype:Archetype::Pest,rarity:Rarity::Normal,activity:8,quantity:5});
    let next = play(&s, 0, None);
    assert!(next.monsters().all(|m| m.archetype != Archetype::Pest));
    assert_eq!(next.tanks[0].unwrap().quantity, 235);
    assert_eq!(next.hand[0].id, 56);
    assert_eq!(next.hand.len(), 4);
    s.tanks[2] = None;
    let absent = play(&s, 0, None);
    assert_eq!(absent.tanks, s.tanks);
    assert_eq!(absent.special_offer, 0);
    let mut s = state(6, &[special::NAMES[15]]);
    s.t8000 = true;
    s.t28000 = true;
    let next = play(&s, 0, None);
    assert_eq!(next.hand.len(), 1);
    assert_eq!(next.hand[0].id, 58);
    let ending = play(&next, 0, None);
    assert_eq!(ending.monsters().count(), 6);
    assert!(ending.monsters().all(|m| m.archetype == Archetype::BoneSentry && m.activity >= 1195 && m.quantity >= 1195));
}

#[test]
fn xeno_replacement_stops_when_no_other_race_remains() {
    let mut s = state(3, &[special::NAMES[17]]);
    s.t8000 = true;
    s.t28000 = true;
    // With three different races, one mutation cannot exhaust the other races.
    let next = play(&s, 0, None);
    assert_eq!(next.special_offer, 12);
    assert_eq!(next.hand.len(), 2);
    assert_eq!((next.hand[0].id,next.hand[1].id),(59,60));
    s.tanks = [s.tanks[0],None,None,None,None,None];
    let stopped = play(&s, 0, None);
    assert_eq!(stopped.special_offer, 0);
    assert_eq!(stopped.tanks[0].unwrap().activity,334);
    assert_eq!(stopped.hand.len(), 3);
}

#[test]
fn unknown_family_name_does_not_bypass_the_unknown_effect_hold() {
    let screen: ScreenState = serde_json::from_value(json!({
        "tanks":[null,null,null,null,null,null],"instruments":[],
        "hand":["Xenopanacea: unverified future variant"],"round":2,"phase":"potion"
    })).unwrap();
    assert!(matches!(advisor::advise(&screen,1,false,false).best,Best::Nothing));
}
