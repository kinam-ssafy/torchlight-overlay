use vorax::{advisor::{self,ScreenState},bot::*};

#[test]
fn value_candidate_recovers_the_omitted_high_activity_target_without_dropping_old_choices() {
    let screen:ScreenState=serde_json::from_str(include_str!("../../../fixtures/pruned-value-target.json")).unwrap();
    let s=advisor::to_state(&screen,&mut vec![]);
    let seed=mix(1_300_000_005,41_001);
    let old=shortlist(&s,seed,&advisor::LOOK_AHEAD);
    let extra=shortlist(&s,seed,&LookAhead{value_candidates:true,..advisor::LOOK_AHEAD});
    let target=Opt::card(0,&[1]);
    assert!(!old.iter().any(|(o,_)|*o==target));
    assert!(extra.iter().any(|(o,_)|*o==target));
    for option in old {assert!(extra.contains(&option));}
    let all=options(&s);
    assert!(extra.iter().all(|(o,_)|all.contains(o)));
    // A fresh continuation sample confirms why preserving this target matters
    // on this reproducible board; it is not a claim about whole-game win rates.
    let opts=[Opt::card(2,&[2]),target];
    let scores=potion_futures(&s,mix(1_300_000_005,91_003),&advisor::LOOK_AHEAD,&opts,0,512,false);
    assert!(scores[1].goal>scores[0].goal);
}
