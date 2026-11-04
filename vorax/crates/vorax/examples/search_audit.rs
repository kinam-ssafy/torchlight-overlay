//! Diagnose root pruning with independent continuation seeds; no game RNG oracle.
use rayon::prelude::*;
use serde_json::{json, Value};
use std::time::Instant;
use vorax::{advisor::LOOK_AHEAD, bot::*, data::data, game::*, types::*};

fn opt(o: Opt) -> Value {
    match o { Opt::Refresh => json!({"refresh":true}),
        Opt::Card {index,targets} => json!({"index":index,"targets":targets.as_slice()}) }
}

fn ranked_shortlist(s: &GameState, seed: u32, policy: Policy) -> Vec<Opt> {
    let mut ranked: Vec<_> = options(s).iter().map(|o| (*o,expected(s,o,&policy,seed,0))).collect();
    ranked.sort_by(|a,b| b.1.total_cmp(&a.1));
    let mut result=Vec::new();
    for &(o,_) in &ranked {
        if !result.iter().any(|r: &Opt| r.index()==o.index()) {result.push(o);}
    }
    for &(o,_) in &ranked {
        if result.len()>=LOOK_AHEAD.top {break;}
        if !result.contains(&o) {result.push(o);}
    }
    if can_refresh(s) {result.push(Opt::Refresh);}
    result
}

fn main() {
    let args:Vec<String>=std::env::args().collect();
    let games:u32=args.get(1).map(|s|s.parse().unwrap()).unwrap_or(32);
    let out=args.get(2).expect("search_audit [games] output.json");
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();
    vorax::data::configure_experiment(None,Some(Settings{potion_refreshes:3,instrument_refreshes:1}),840000,false,0).unwrap();
    let policy=Policy{value:Some(true),..Policy::new(2,0)};
    let chooser=Chooser::Ev(policy);
    let mut boards=Vec::new();
    for seed in 1_300_000_000..1_300_000_000+games {
        let mut s=create_game(data().default_settings,None,Some(seed));
        let mut seen=[false;14];
        for step in 0..300 {
            match s.phase {
                Phase::GameOver=>break,
                Phase::RoundEnd=>s=dispatch(&s,Action::Confirm),
                Phase::InstrumentSelect=>s=match Picker::Value.pick(&s,mix(seed,5000+step)) {
                    Pick::Refresh=>dispatch(&s,Action::Refresh),Pick::Card(i)=>pick_and_confirm(&s,i)},
                Phase::PotionSelect=>{
                    if [3,6,9].contains(&s.round) && !seen[s.round as usize] && s.open_box.is_none() {
                        seen[s.round as usize]=true;
                        boards.push((seed,s));
                    }
                    let c=chooser.choose(&s,mix(seed,step));
                    s=if c.value<=c.stay.unwrap_or(ultimate_activity(&s) as f64) && can_refresh(&s) {
                        dispatch(&s,Action::Refresh)
                    } else {play_opt(&s,&c.best.expect("valid hand"))};
                }
            }
        }
        assert_eq!(s.phase,Phase::GameOver);
    }
    let started=Instant::now();
    let results:Vec<_>=boards.par_iter().enumerate().map(|(i,(seed,s))|{
        let probe_seed=mix(*seed,41_001);
        let mut all:Vec<_>=options(s).iter().copied().collect();
        if can_refresh(s) {all.push(Opt::Refresh);}
        let baseline:Vec<_>=shortlist(s,probe_seed,&LOOK_AHEAD).iter().map(|r|r.0).collect();
        let value=ranked_shortlist(s,probe_seed,Policy{value:Some(true),..Policy::new(8,0)});
        let horizon=ranked_shortlist(s,probe_seed,Policy{value:Some(true),..Policy::new(8,2)});
        let la=LookAhead{playouts:256,..LOOK_AHEAD};
        let probe=potion_futures(s,probe_seed,&la,&all,0,256,false);
        let best_at=|indices:&[usize], scores:&[Tally]| {
            let mut best=indices[0];
            for &i in &indices[1..] {if scores[i].compare(&scores[best],Goal::Target).is_gt() {best=i;}}
            best
        };
        let best=|opts:&[Opt]| all[best_at(&opts.iter().map(|o|all.iter().position(|v|v==o).unwrap()).collect::<Vec<_>>(),&probe)];
        let budget=baseline.len() as u32*256;
        let indices:Vec<_>=(0..all.len()).collect();
        let uniform=potion_futures(s,probe_seed,&la,&all,0,budget/all.len() as u32,false);
        let pilot_n=16.min(budget/all.len() as u32/2).max(1);
        let pilot=potion_futures(s,probe_seed,&la,&all,0,pilot_n,false);
        let mut survivors=indices.clone();
        survivors.sort_by(|&a,&b|pilot[b].compare(&pilot[a],Goal::Target));
        survivors.truncate(baseline.len());
        let extra=(budget-pilot_n*all.len() as u32)/survivors.len() as u32;
        let selected_opts:Vec<_>=survivors.iter().map(|&i|all[i]).collect();
        let extra_scores=potion_futures(s,probe_seed,&la,&selected_opts,pilot_n,pilot_n+extra,false);
        let mut combined=pilot;
        for (&i,t) in survivors.iter().zip(extra_scores) {
            combined[i].goal+=t.goal; combined[i].n+=t.n; combined[i].reached+=t.reached;
            combined[i].monsters+=t.monsters; combined[i].groups+=t.groups;
        }
        let winners=[best(&baseline),best(&value),best(&horizon),best(&all),
                     all[best_at(&indices,&uniform)],all[best_at(&survivors,&combined)]];
        let mut selected=Vec::new();
        for o in winners {if !selected.contains(&o){selected.push(o);}}
        let check=potion_futures(s,mix(*seed,91_003),&la,&selected,0,512,false);
        let estimate=|o:Opt|check[selected.iter().position(|v|*v==o).unwrap()];
        eprintln!("board {}/{} round {} options {}",i+1,boards.len(),s.round,all.len());
        json!({"seed":seed,"round":s.round,"options":all.len(),"budget_options":baseline.len(),
            "all_best_omitted":!baseline.contains(&winners[3]),
            "winners":winners.map(opt),"confirmation":winners.map(estimate),
            "screen":{"tanks":s.tanks.iter().map(|m|m.map(|m|json!({"archetype":m.archetype,"rarity":m.rarity,"activity":m.activity,"quantity":m.quantity}))).collect::<Vec<_>>(),
                "instruments":s.instruments.iter().map(|e|data().instrument(e.id).unwrap().name.clone()).collect::<Vec<_>>(),
                "instrumentRounds":s.instruments.iter().map(|e|e.acquired_round).collect::<Vec<_>>(),
                "hand":s.hand.iter().map(|c|data().potion(c.id).name.clone()).collect::<Vec<_>>(),
                "phase":"potion","round":s.round,"potionRefreshes":s.potion_refreshes,"instrumentRefreshes":s.instrument_refreshes},
            "model": "embedded incumbent", "target":840000})
    }).collect();
    std::fs::write(out,serde_json::to_string_pretty(&json!({"games":games,"probe_futures":256,"confirmation_futures":512,
        "seed0":1300000000u32,"candidate_methods":["immediate","value","two_decisions","all_actions_extra_budget","uniform_fixed_budget","pilot_fixed_budget"],
        "seconds":started.elapsed().as_secs_f64(),"boards":results})).unwrap()).unwrap();
}
