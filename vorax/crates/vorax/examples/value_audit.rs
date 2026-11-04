//! Deterministic CPU export/latency audit, including every special-offer stage.
use std::{hint::black_box, io::Write, time::Instant};
use vorax::{data, types::*, value};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(args.len() >= 3, "value_audit model.json target-score output.jsonl [success]");
    let model = std::fs::read_to_string(&args[0]).unwrap();
    let target: i64 = args[1].parse().unwrap();
    data::configure_experiment(Some(&model), None, target, args.get(3).is_some_and(|a| a == "success"), 0).unwrap();
    let d = data::data();
    let mut states = Vec::new();
    for round in 1..=13 {
        for stage in 0..=12 {
            for variation in 0..8usize {
                let mut s = GameState { round, special_offer: stage, phase: Phase::PotionSelect, ..GameState::default() };
                for i in 0..(variation % 6 + 1) {
                    s.tanks[i] = Some(Monster {
                        archetype: [Archetype::Pest,Archetype::Agito,Archetype::BoneSentry,Archetype::Aemberon][(i+variation)%4],
                        rarity: [Rarity::Normal,Rarity::Magic,Rarity::Rare,Rarity::Boss][(i+variation/2)%4],
                        activity: 1 + (i*31+variation*100) as i64,
                        quantity: 1 + (i*17+variation*73) as i64,
                    });
                }
                for i in 0..variation%4 {
                    s.instruments.push(Equipped {id:d.instruments[(variation+i)%d.instruments.len()].id,acquired_round:1});
                }
                states.push(s);
            }
        }
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
    for s in &states {
        let mut sparse = Vec::new();
        value::features(d,s,&mut sparse);
        let mut x = vec![0.;value::feature_count(d)];
        for (i,v) in sparse { x[i as usize]=v; }
        x.extend(value::chain_features(d,s));
        writeln!(out,"{}",serde_json::json!({"round":s.round,"stage":s.special_offer,"features":x,"prediction":value::board_value(s)})).unwrap();
    }
    out.flush().unwrap();
    for s in &states { black_box(value::board_value(black_box(s))); }
    let mut times = Vec::new();
    for _ in 0..5 {
        let start=Instant::now();
        for _ in 0..256 {
            for s in &states { black_box(value::board_value(black_box(s))); }
        }
        times.push(start.elapsed().as_secs_f64()*1e6/(256*states.len()) as f64);
    }
    times.sort_by(f64::total_cmp);
    println!("{}",serde_json::json!({"boards":states.len(),"stages":13,"rounds":13,
        "single_thread_median_us_per_value":times[2],"all_repeats_us":times,
        "scope":"value function only, excludes OCR, full solver and screen rendering"}));
}
