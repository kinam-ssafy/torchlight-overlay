//! The equipped instruments, from the icons in the top-right panel (src/solver/scan/instruments.ts, iconMatch.ts).

use crate::data::ScanData;
use crate::layout::Layout;
use crate::pixels::{signature, Img, Rect};
use crate::reading::{FieldRead, Fields, Ranks};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

const BRIGHT: u8 = 60;
const FILLED_MIN: f64 = 0.008;
const ICON_MIN: f64 = 0.85;
const ICON_GAP: f64 = 0.08;
pub const STARTING_ITEM: &str = "(starting item)";

/// Near black: the square an icon sits on.
const DARK: u8 = 25;
/// An icon's square is mostly near black (49–95% measured); an empty slot over the lit scenery has none, yet enough
/// bright pixels to pass FILLED_MIN (1–7% at 1080p), so it read as an icon: 4 instruments where 3 were held.
const BACKDROP_MIN: f64 = 0.3;

pub fn holds_an_icon(img: &Img, r: &Rect) -> bool {
    let x0 = r.x.floor();
    let y0 = r.y.floor();
    let (mut n, mut dark) = (0usize, 0usize);
    let mut y = y0;
    while y < (img.height as f64).min(y0 + r.h) {
        let mut x = x0;
        while x < (img.width as f64).min(x0 + r.w) {
            let (rr, gg, bb) = img.px(x as usize, y as usize);
            let m = rr.max(gg).max(bb);
            if m >= BRIGHT {
                n += 1;
            }
            if m < DARK {
                dark += 1;
            }
            x += 1.0;
        }
        y += 1.0;
    }
    n as f64 / (r.w * r.h) >= FILLED_MIN && dark as f64 / (r.w * r.h) >= BACKDROP_MIN
}

const N: usize = 24;
const REACH: i32 = 2;

fn centre(p: &[f64]) -> (f64, f64) {
    let lo = p.iter().fold(f64::INFINITY, |m, &v| if v < m { v } else { m });
    let (mut s, mut x, mut y) = (0.0, 0.0, 0.0);
    for (i, &v) in p.iter().enumerate() {
        let w = v - lo;
        s += w;
        x += w * (i % N) as f64;
        y += w * (i / N) as f64;
    }
    if s != 0.0 {
        (x / s, y / s)
    } else {
        ((N - 1) as f64 / 2.0, (N - 1) as f64 / 2.0)
    }
}

fn moved(p: &[f64], dx: i32, dy: i32) -> Vec<f64> {
    let bg = p.iter().fold(f64::INFINITY, |m, &v| if v < m { v } else { m });
    let mut q = vec![0.0; p.len()];
    let mut sum = 0.0;
    for i in 0..p.len() {
        let x = (i % N) as i32 - dx;
        let y = (i / N) as i32 - dy;
        q[i] = if x >= 0 && (x as usize) < N && y >= 0 && (y as usize) < N { p[y as usize * N + x as usize] } else { bg };
        sum += q[i];
    }
    let mean = sum / q.len() as f64;
    let mut norm = 0.0;
    for v in q.iter_mut() {
        *v -= mean;
        norm += *v * *v;
    }
    let norm = if norm.sqrt() != 0.0 { norm.sqrt() } else { 1.0 };
    for v in q.iter_mut() {
        *v /= norm;
    }
    q
}

fn js_round(x: f64) -> f64 {
    crate::pixels::js_round(x)
}

/// How alike a reference icon and a panel pattern are, allowing for the crop being a few pixels off.
pub fn icon_similarity(rf: &[f64], p: &[f64], cache: &mut HashMap<(i32, i32), Vec<f64>>) -> f64 {
    if p.len() != N * N || rf.len() != N * N {
        return rf.iter().enumerate().fold(0.0, |s, (i, x)| s + x * p.get(i).copied().unwrap_or(0.0));
    }
    let (rx, ry) = centre(rf);
    let (px, py) = centre(p);
    let (bx, by) = (js_round(rx - px) as i32, js_round(ry - py) as i32);
    let mut best = -1.0f64;
    for dy in by - REACH..=by + REACH {
        for dx in bx - REACH..=bx + REACH {
            let q = cache.entry((dx, dy)).or_insert_with(|| if dx == 0 && dy == 0 { p.to_vec() } else { moved(p, dx, dy) });
            let mut s = 0.0;
            for i in 0..q.len() {
                s += rf[i] * q[i];
            }
            if s > best {
                best = s;
            }
        }
    }
    best
}

pub struct InstrumentsRead {
    pub names: Vec<String>,
    pub fields: Fields,
    pub ranks: Ranks,
    pub known: Vec<String>,
    pub locations: Vec<Rect>,
    slots: usize,
}

/// Icons observed before the first tool was selected. Their identity comes from the
/// start event, not a guess at the nearest tool. Persisted locally by the overlay.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StartingItems(Vec<Vec<f64>>);

impl StartingItems {
    pub fn learn(&mut self, img: &Img, l: &Layout) {
        let Some(rects) = icon_rects(img, l) else { return; };
        for r in rects {
            if !holds_an_icon(img, &r) { continue; }
            self.learn_at(img, r);
        }
    }

    pub fn valid(&self) -> bool {
        self.0.len() <= 64 && self.0.iter().all(|v| v.len() == N * N && v.iter().all(|x| x.is_finite()))
    }

    pub fn learn_at(&mut self, img: &Img, r: Rect) -> bool {
        let p = signature(img, r, N, N);
        let mut cache = HashMap::new();
        if self.0.iter().any(|v| icon_similarity(v, &p, &mut cache) >= 0.95) { return false; }
        if self.0.len() >= 64 { return false; }
        self.0.push(p);
        true
    }
}

pub fn read_instruments(img: &Img, l: &Layout, data: &ScanData) -> InstrumentsRead {
    read_icons_at(img, &l.instrument_icon, data, false, &StartingItems::default())
}

/// Localized panels space their icons differently. Locate separated foreground clusters in the
/// existing icon strip. Keep each slot's confidence separately: an unknown starting item
/// must not discard a confidently located tool beside it. The tracker validates ownership.
pub fn refine_instruments(img: &Img, l: &Layout, data: &ScanData) -> Option<InstrumentsRead> {
    refine_with_starting(img, l, data, &StartingItems::default())
}

pub fn refine_with_starting(img: &Img, l: &Layout, data: &ScanData, starting: &StartingItems) -> Option<InstrumentsRead> {
    let rects = icon_rects(img, l)?;
    let read = read_icons_at(img, &rects, data, true, starting);
    // A recognized starting item occupies a slot even though it is not a tool.
    (read.slots == rects.len()).then_some(read)
}

pub fn icon_rects(img: &Img, l: &Layout) -> Option<Vec<Rect>> {
    let first = l.instrument_icon.first()?;
    let last = l.instrument_icon.last()?;
    let x0 = (last.x - 60.0 * l.k).max(0.0) as usize;
    let x1 = (first.x + first.w + 40.0 * l.k).min(img.width as f64) as usize;
    let y0 = first.y.max(0.0) as usize;
    let y1 = (first.y + first.h + 8.0 * l.k).min(img.height as f64) as usize;
    let peak = (y0..y1).flat_map(|y| (x0..x1).map(move |x| img.px(x,y)))
        .map(|(r,g,b)| r.max(g).max(b)).max().unwrap_or(0);
    let foreground = (peak as f64 * 0.20).max(DARK as f64) as u8;
    let mut groups: Vec<(usize, usize)> = vec![];
    for x in x0..x1 {
        let count = (y0..y1).filter(|&y| { let (r,g,b) = img.px(x,y); r.max(g).max(b) >= foreground }).count();
        if count < (3.0 * l.k).ceil() as usize { continue; }
        if let Some((_, end)) = groups.last_mut().filter(|(_, end)| (x - *end) as f64 <= 16.0 * l.k) { *end = x; }
        else { groups.push((x,x)); }
    }
    // The strip also crosses animated scenery. Validate each component's size and
    // dark icon backdrop before counting slots; scenery must not reject the real panel.
    groups.retain(|(a,b)| {
        let width = (*b-*a+1) as f64;
        width >= 12.0*l.k && width <= 100.0*l.k
            && holds_an_icon(img,&Rect::new((*a+*b) as f64/2.0-first.w/2.0,first.y,first.w,first.h))
    });
    if groups.is_empty() || groups.len() > l.instrument_icon.len() { return None; }
    Some(groups.iter().rev().map(|(a,b)| Rect::new((*a+*b) as f64/2.0-first.w/2.0, first.y, first.w, first.h)).collect())
}

/// One-pixel wires alias when reduced to the 24x24 reference grid. Compare their low-frequency
/// shape as a second check; never use this to change the normal matcher's winning identity.
fn softened(v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; N * N];
    for y in 0..N {
        for x in 0..N {
            let mut count = 0.0;
            for yy in y.saturating_sub(1)..=(y + 1).min(N - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(N - 1) {
                    out[y * N + x] += v[yy * N + xx];
                    count += 1.0;
                }
            }
            out[y * N + x] /= count;
        }
    }
    let mean = out.iter().sum::<f64>() / out.len() as f64;
    let norm = out.iter().map(|x| (x - mean).powi(2)).sum::<f64>().sqrt().max(f64::EPSILON);
    for x in &mut out { *x = (*x - mean) / norm; }
    out
}

fn read_icons_at(img: &Img, rects: &[Rect], data: &ScanData, refine: bool, starting: &StartingItems) -> InstrumentsRead {
    let mut from_right: Vec<String> = Vec::new();
    let mut reads: Vec<FieldRead> = Vec::new();
    let mut ranks_from_right: Ranks = Vec::new();
    let mut locations = Vec::new();
    let mut slots = 0;
    for r in rects {
        if !holds_an_icon(img, r) {
            break;
        }
        slots += 1;
        let p = signature(img, *r, 24, 24);
        let mut lined = HashMap::new();
        let mut ranked: Vec<(String, f64)> = data
            .icons
            .iter()
            .filter(|(n, _)| n.as_str() != "none")
            .map(|(n, vs)| (n.clone(), vs.iter().map(|v| icon_similarity(v, &p, &mut lined)).fold(f64::NEG_INFINITY, f64::max)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let start_score = starting.0.iter().map(|v| icon_similarity(v, &p, &mut lined)).fold(-1.0, f64::max);
        let tool_score = ranked.iter().find(|(n, _)| n != STARTING_ITEM).map_or(-1.0, |(_, s)| *s);
        if start_score >= 0.90 && start_score - tool_score >= ICON_GAP {
            continue;
        }
        if refine && (ranked[0].1 < ICON_MIN || ranked[0].1 - ranked.get(1).map_or(0.0, |x| x.1) < ICON_GAP) {
            let soft = softened(&p);
            let mut cache = HashMap::new();
            let mut stable: Vec<_> = data.icons.iter().filter(|(n, _)| n.as_str() != "none")
                .map(|(n, vs)| (n.clone(), vs.iter().filter(|v| v.len() == N * N)
                    .map(|v| icon_similarity(&softened(v), &soft, &mut cache)).fold(-1.0, f64::max))).collect();
            stable.sort_by(|a, b| b.1.total_cmp(&a.1));
            if stable[0].0 == ranked[0].0 && stable[0].1 >= 0.90
                && stable[0].1 - stable.get(1).map_or(0.0, |x| x.1) >= 0.12 {
                ranked = stable;
            }
        }
        if ranked[0].0 == STARTING_ITEM {
            continue;
        }
        let second = ranked.get(1).map(|x| x.1);
        reads.push(FieldRead { sure: ranked[0].1 >= ICON_MIN && ranked[0].1 - second.unwrap_or(0.0) >= ICON_GAP, guess: Some(ranked[0].0.clone()), score: Some(ranked[0].1), second });
        from_right.push(ranked[0].0.clone());
        locations.push(*r);
        ranks_from_right.push(ranked.into_iter().filter(|(n, _)| n != STARTING_ITEM).collect());
    }
    from_right.reverse();
    reads.reverse();
    ranks_from_right.reverse();
    locations.reverse();
    let mut fields = Fields::new();
    for (i, f) in reads.into_iter().enumerate() {
        fields.insert(format!("instrument{i}"), f);
    }
    let known = data.icons.keys().filter(|n| n.as_str() != "none" && n.as_str() != STARTING_ITEM).cloned().collect();
    InstrumentsRead { names: from_right, fields, ranks: ranks_from_right, known, locations, slots }
}
