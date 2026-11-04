//! Special chains from Korean screenshots and TLIDB definitions (2026-11-04).
//! No invented random entry weights. Experimental: Matriarch tier bonuses,
//! unspecified rarity distributions and reactive ordering need game fixtures.
//! Controlled research scenarios must not be presented as live occurrence rates.
use crate::{context::{base_stats, Ctx}, data::{PotionDef, PotionRarity, Targeting}, types::*};

pub const COCOON: u8 = 42;
pub const MATRIARCH: u8 = 49;
pub const LAST: u8 = 60;
pub const NAMES: [&str; 19] = [
    "Incubation Elixir: Hollow Cocoon",
    "F1 Gen: Parasite Egg", "F1 Gen: Vermin Egg Cluster",
    "F2 Gen: Parasite Pupae", "F2 Gen: Vermin Pupal Cluster",
    "F3 Gen: Parasitic Butterfly", "F3 Gen: Emergent Butterfly Swarm",
    "Incubation Elixir: Hollow Matriarch",
    "First Rite Holy Water: Confession", "Ritual Holy Water: Flagellation",
    "Third Rite Holy Water: Illumination", "Fourth Rite Holy Water: Eucharist",
    "Dissection Infusion: Morbid Corpse", "Dissection I: Vermectomy",
    "Dissection II: Daemonectomy", "Dissection III: Craniotomy",
    "Dissection Infusion: Intact Skeleton", "Xenopanacea: Osteotrophy", "Xenopanacea: Brood",
];

pub fn definitions() -> impl Iterator<Item = PotionDef> {
    NAMES.into_iter().enumerate().map(|(i, name)| PotionDef {
        id: COCOON + i as u8, name: name.into(), rarity: PotionRarity::Supreme,
        targeting: if matches!(i, 1 | 3 | 5) { Targeting::Select1 } else { Targeting::Auto },
    })
}

pub fn contains(id: u8) -> bool { (COCOON..=LAST).contains(&id) }

/// The next round's extra offer, not an extra action in the current round.
pub fn next(id: u8, tanks: &Tanks) -> u8 {
    match id {
        42 => 1, 43 | 44 => 2, 45 | 46 => 3, 47 | 48 => 4,
        50 => 5, 51 => 6, 52 => 7, 54 => 8,
        55..=57 if tanks.iter().flatten().any(|m| m.archetype == Archetype::BoneSentry) => id - 46,
        59 | 60 if tanks.iter().flatten().any(|m| m.archetype != Archetype::Aemberon) => 12,
        _ => 0,
    }
}

pub fn offer(stage: u8) -> &'static [u8] {
    match stage { 1 => &[43,44], 2 => &[45,46], 3 => &[47,48], 4 => &[49],
        5 => &[51], 6 => &[52], 7 => &[53], 8 => &[55], 9 => &[56], 10 => &[57],
        11 => &[58], 12 => &[59,60], _ => &[] }
}

/// Only these two tooltips explicitly replace the ordinary next-round offer.
pub fn replaces(stage: u8) -> bool { matches!(stage, 11 | 12) }

pub fn stage(hand: &Hand) -> u8 {
    (1..=12).find(|&stage| offer(stage).iter().all(|id| hand.iter().any(|c| c.kind == CardKind::Potion && c.id == *id))).unwrap_or(0)
}

pub fn apply(c: &mut Ctx, id: u8, targets: &[u8]) {
    use Archetype::*;
    use Rarity::*;
    match id {
        // The observed cocoon has a red name, Pest icon, Activity 1, Quantity 1.
        COCOON => { c.add(Pest, Boss, 1 - base_stats(Boss).0, 0); }
        43 | 45 | 47 => {
            if let Some(&tank) = targets.first() {
                c.mutate(tank as usize, Some(Pest), None);
                c.boost(tank as i32, ((id - 43) / 2 + 1) as i64 * 10, 0);
            }
        }
        44 | 46 | 48 => {
            let pests = c.by_archetype(Pest);
            if let Some(tank) = c.pick(&pests) {
                let monster = c.mon(tank).unwrap();
                let bonus = ((id - 44) / 2 + 1) as i64 * 10;
                let (a,q) = base_stats(monster.rarity);
                c.boost(tank as i32, 0, bonus);
                // Replication copies the existing stats, rather than spawning base stats.
                c.add(Pest, monster.rarity, monster.activity - a, monster.quantity + bonus - q);
            }
        }
        MATRIARCH => {
            let sources = c.non_empty();
            let bonus: i64 = sources.iter().filter_map(|&i| c.mon(i as usize))
                .filter(|m| m.archetype == Pest)
                .map(|m| match m.rarity { Normal => 15, Magic => 30, Rare | Boss => 75 }).sum();
            let tank = c.fuse(sources.as_slice(), Pest, Boss);
            if tank >= 0 {
                let m = c.mon(tank as usize).unwrap();
                c.boost(tank, m.activity * bonus / 100, m.quantity * bonus / 100);
            }
        }
        // TLIDB effect definitions, retrieved 2026-11-04. The first three rites
        // only schedule the next card; they do not multiply the current board.
        50..=52 => {}
        53 => {
            let sources = c.non_empty();
            let (mut a, mut q) = (0, 0);
            for &i in &sources { let m = c.mon(i as usize).unwrap(); a += m.activity; q += m.quantity; }
            for &i in &sources { c.remove(i as usize); }
            for _ in 0..6 {
                let r = c.roll_rarity();
                let base = base_stats(r);
                c.add(Agito, r, a - base.0, q - base.1);
            }
        }
        54 => {
            if c.non_empty().len() == TANK_COUNT {
                let i = c.highest();
                c.mutate(i as usize, Some(BoneSentry), None);
            } else { c.add(BoneSentry, Boss, 0, 0); }
        }
        55..=57 => {
            if c.by_archetype(BoneSentry).is_empty() { return; }
            let race = match id { 55 => Pest, 56 => Aemberon, _ => Agito };
            for &i in &c.by_archetype(race) { c.remove(i as usize); }
            c.all(if id == 56 {199} else {0}, if id == 56 {0} else {199});
        }
        58 => {
            for &i in &c.filter(|_, m| m.archetype != BoneSentry) { c.remove(i as usize); }
            c.add(BoneSentry, Boss, 0, 0);
            // Bounded by available slots even when reactive additions fill them.
            for _ in c.non_empty().len()..TANK_COUNT {
                let r = c.roll_rarity(); c.add(BoneSentry, r, 0, 0);
            }
            let bones = c.by_archetype(BoneSentry);
            let bonus = 199 * bones.len() as i64;
            for &i in &bones { c.boost(i as i32, bonus, bonus); }
        }
        59 | 60 => {
            if let Some(i) = c.pick(&c.non_empty()) {
                c.mutate(i, Some(Aemberon), None);
                c.boost(i as i32, if id == 59 {333} else {0}, if id == 60 {333} else {0});
            }
        }
        _ => {}
    }
}
