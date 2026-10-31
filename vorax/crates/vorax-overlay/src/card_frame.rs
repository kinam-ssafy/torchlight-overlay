//! Fit the visible card edges, including neutral borders. Localized cards have different widths and spacing, so an OCR
//! crop is not a reliable rectangle to draw. Searches are bounded to the already detected hand.
use vorax_scan::{layout::{CardPos, Layout}, pixels::Img};

pub type CardFrame = (CardPos, f64, f64);

fn gold_pixel(img: &Img, x: f64, y: f64) -> bool {
    let (x, y) = (x.round() as isize, y.round() as isize);
    if x < 0 || y < 0 || x >= img.width as isize || y >= img.height as isize { return false; }
    let (r, g, b) = img.px(x as usize, y as usize);
    (r >= 100 && g >= 60 && g <= 240 && b < 210 && r as i32 - g as i32 >= 15 && g as i32 - b as i32 >= 25)
        || (b >= 80 && g >= 50 && b >= g && b as i32 - r as i32 >= 35)
}

fn gold(img: &Img, x: f64, y: f64) -> bool {
    [(0.0, 0.0), (-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0)].iter().any(|(dx, dy)| gold_pixel(img, x + dx, y + dy))
}

/// A long boundary is evidence regardless of rarity colour. Sample across it, not along it:
/// scattered white title glyphs must not stand in for an entire card edge.
fn contrast(img: &Img, x: f64, y: f64, dx: f64, dy: f64) -> bool {
    let brightness = |x: f64, y: f64| {
        let (x, y) = (x.round() as isize, y.round() as isize);
        if x < 0 || y < 0 || x >= img.width as isize || y >= img.height as isize { return None; }
        let (r, g, b) = img.px(x as usize, y as usize);
        Some((r as f64 + g as f64 + b as f64) / 3.0)
    };
    match (brightness(x - dx, y - dy), brightness(x + dx, y + dy)) {
        (Some(a), Some(b)) => (a - b).abs() > 22.0,
        _ => false,
    }
}

fn line_peak(lo: f64, hi: f64, minimum: f64, score: impl Fn(f64) -> f64) -> Option<f64> {
    let values: Vec<_> = (lo.floor() as i32..=hi.ceil() as i32).map(|n| (n as f64, score(n as f64))).collect();
    let &(at, best) = values.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    if best < minimum { return None; }
    // The glowing edge is several pixels wide. Use the middle of the strongest local band.
    let near: Vec<_> = values.iter().filter(|(x, s)| (*x - at).abs() <= 5.0 && *s >= best - 0.04).collect();
    Some(near.iter().map(|(x, _)| *x).sum::<f64>() / near.len() as f64)
}

fn edge_peak(lo: f64, hi: f64, colour_min: f64, colour: impl Fn(f64) -> f64, neutral: impl Fn(f64) -> f64) -> Option<f64> {
    line_peak(lo, hi, colour_min.min(0.85), |at| {
        let (c, n) = (colour(at), neutral(at));
        let c = if c >= colour_min { c } else { 0.0 };
        let n = if n >= 0.85 { n } else { 0.0 };
        c.max(n)
    })
}

fn vertical(img: &Img, c: &CardPos, height: f64, lo: f64, hi: f64) -> Option<f64> {
    let score = |x: f64, neutral: bool| {
        (0..40).filter(|i| {
            let y = c.cy + ((*i as f64 / 39.0) - 0.5) * height * 0.6;
            let px = x - c.angle.tan() * (y - c.cy);
            if neutral { contrast(img, px, y, 3.0 * height / 255.0, 0.0) } else { gold(img, px, y) }
        }).count() as f64 / 40.0
    };
    edge_peak(lo, hi, 0.75, |x| score(x, false), |x| score(x, true))
}

pub fn detect(img: &Img, l: &Layout, size: usize) -> Option<Vec<CardFrame>> {
    let cards = l.hand_at(size);
    let (first, last) = (cards.first()?, cards.last()?);
    let (base_w, base_h) = (230.0 * l.k, 340.0 * l.k);
    let edges: Vec<_> = cards.iter().map(|c| (
        // Hover spreads the fan horizontally as well as lifting the selected card.
        vertical(img, c, base_h, c.cx - base_w, c.cx - base_w * 0.1),
        vertical(img, c, base_h, c.cx + base_w * 0.1, c.cx + base_w * 1.2),
    )).collect();
    // A raised card can cover its neighbour's left edge. At least one card is fully exposed;
    // the widest valid pair supplies the shared width, while the fan's outer edges stay visible.
    let width = cards.iter().zip(&edges).filter_map(|(c, (left, right))| {
        let w = (right.as_ref()? - left.as_ref()?) * c.angle.cos();
        (base_w * 0.75..=base_w * 1.65).contains(&w).then_some(w)
    }).max_by(f64::total_cmp)?;
    let left = edges.first()?.0?;
    let right = edges.last()?.1?;
    let first_mid = left + width / (2.0 * first.angle.cos());
    let last_mid = right - width / (2.0 * last.angle.cos());
    if last_mid - first_mid < width * (size - 1) as f64 * 0.4 { return None; }
    let mut out = vec![];
    let mut known_height = None;
    let mut normal_top = None;
    for (j, c) in cards.iter().enumerate().rev() {
        let fraction = (c.cx - first.cx) / (last.cx - first.cx);
        let mid = first_mid + fraction * (last_mid - first_mid);
        let lx = mid - width / (2.0 * c.angle.cos());
        let visible = if j + 1 == cards.len() { 0.88 } else { 0.65 };
        let horizontal = |y: f64| (0..40).filter(|i| {
            let x = lx + width * (0.1 + *i as f64 / 39.0 * (visible - 0.1));
            gold(img, x, y + c.angle.tan() * (x - mid))
        }).count() as f64 / 40.0;
        let neutral_edge = |y: f64| (0..40).filter(|i| {
            let x = lx + width * (0.1 + *i as f64 / 39.0 * (visible - 0.1));
            let y = y + c.angle.tan() * (x - mid);
            contrast(img, x, y, 0.0, 4.0 * l.k)
        }).count() as f64 / 40.0;
        // Prefer the row established by the exposed card. Gold header text above the fan can
        // otherwise outscore a partly occluded edge. A genuinely raised card uses the wider search.
        let top = normal_top.and_then(|y| edge_peak(y - base_h * 0.06, y + base_h * 0.06, 0.75, horizontal, neutral_edge))
            .or_else(|| edge_peak(c.cy - base_h * 0.9, c.cy - base_h * 0.25, 0.75, horizontal, neutral_edge))?;
        normal_top.get_or_insert(top);
        let predicted = known_height.map(|h| top + h / c.angle.cos());
        // Limit the lower-edge search by the exposed card's height, excluding the confirm button.
        let bottom = match predicted {
            Some(y) => edge_peak(y - base_h * 0.035, y + base_h * 0.035, 0.45, horizontal, neutral_edge).unwrap_or(y),
            // Compare both sources of evidence before choosing a row. Taking the first gold
            // match lets the reroll button below a grey card replace its real lower edge.
            None => edge_peak(c.cy + base_h * 0.25, c.cy + base_h * 0.7, 0.45, horizontal, neutral_edge).or_else(|| {
                // A portrait can cover the last card's bottom. Recover the shared height from
                // another complete card, rather than replacing the whole fan with OCR rectangles.
                cards[..j].iter().find_map(|other| {
                    let f = (other.cx - first.cx) / (last.cx - first.cx);
                    let om = first_mid + f * (last_mid - first_mid);
                    let ol = om - width / (2.0 * other.angle.cos());
                    let score = |y: f64| (0..40).filter(|i| {
                        let x = ol + width * (0.1 + *i as f64 / 39.0 * 0.55);
                        gold(img, x, y + other.angle.tan() * (x - om))
                    }).count() as f64 / 40.0;
                    let ot = line_peak(top - base_h * 0.06, top + base_h * 0.06, 0.75, score)?;
                    let ob = line_peak(other.cy + base_h * 0.25, other.cy + base_h * 0.7, 0.45, score)?;
                    let h = (ob - ot) * other.angle.cos();
                    (base_h * 0.75..=base_h * 1.65).contains(&h).then_some(top + h / c.angle.cos())
                })
            })?,
        };
        let height = (bottom - top) * c.angle.cos();
        if !(base_h * 0.75..=base_h * 1.65).contains(&height) { return None; }
        known_height.get_or_insert(height);
        let cy = (top + bottom) / 2.0;
        let cx = mid - c.angle.tan() * (cy - c.cy);
        out.push((CardPos { cx, cy, angle: c.angle }, width, height));
    }
    out.reverse();
    let low = out.iter().map(|(_, _, h)| *h).fold(f64::INFINITY, f64::min);
    let high = out.iter().map(|(_, _, h)| *h).fold(0.0, f64::max);
    if high - low > base_h * 0.08 { return None; }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_outline_is_invented_on_a_blank_screen() {
        let img = Img::new(1920, 1080);
        assert!(detect(&img, &vorax_scan::layout::layout(1920, 1080), 3).is_none());
    }

    #[test]
    fn korean_potion_fan_with_a_blue_card_has_all_three_outlines() {
        let frame = crate::pipeline::decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/potion-ko-1080p.png")).unwrap();
        let fitted = detect(&frame, &vorax_scan::layout::layout(1920, 1080), 3).expect("blue and gold card edges");
        for ((c, w, h), (x, y)) in fitted.iter().zip([(889.0, 825.0), (1077.0, 819.0), (1265.0, 825.0)]) {
            assert!((c.cx - x).abs() < 7.0 && (c.cy - y).abs() < 7.0, "card center: {c:?}");
            assert!((w - 224.0).abs() < 10.0 && (h - 300.0).abs() < 10.0);
        }
    }

    #[test]
    fn a_dim_lower_edge_does_not_drop_the_whole_korean_instrument_fan() {
        let frame = crate::pipeline::decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/instrument-ko-1080p.png")).unwrap();
        let fitted = detect(&frame, &vorax_scan::layout::layout(1920, 1080), 3).expect("shared card dimensions");
        for ((c, w, h), (x, y)) in fitted.iter().zip([(889.0, 825.0), (1077.0, 819.0), (1265.0, 825.0)]) {
            assert!((c.cx - x).abs() < 7.0 && (c.cy - y).abs() < 7.0, "card center: {c:?}");
            assert!((w - 224.0).abs() < 10.0 && (h - 300.0).abs() < 10.0);
        }
    }

    #[test]
    fn a_raised_card_does_not_move_the_occluded_neighbour_or_match_the_confirm_button() {
        let frame = crate::pipeline::decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/instrument-selected-ko-1080p.png")).unwrap();
        let fitted = detect(&frame, &vorax_scan::layout::layout(1920,1080), 3).expect("raised fan geometry");
        for ((c,w,h),(x,y)) in fitted.iter().zip([(887.0,794.0),(1076.0,817.0),(1263.0,822.0)]) {
            assert!((c.cx-x).abs()<5.0 && (c.cy-y).abs()<5.0, "{c:?}");
            assert!((w-226.0).abs()<6.0 && (h-300.0).abs()<6.0);
        }
    }

    #[test]
    fn recorded_fans_keep_geometry_under_brightness_and_resolution_changes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames");
        for file in ["initial-ko-1080p.png", "potion-ko-1080p.png", "instrument-ko-1080p.png", "instrument-selected-ko-1080p.png", "potion-round2-ko-1080p.png", "potion-round3-ko-1080p.png", "potion-round4-ko-1080p.png", "potion-round5-selected-ko-1080p.png"] {
            let source = crate::pipeline::decode_png(&root.join(file)).unwrap();
            for scale in [1.0, 2.0/3.0, 5.0/6.0] { for gain in [0.85, 1.0, 1.15] {
                let mut img = vorax_scan::pixels::resize(&source, (1920.0*scale) as usize, (1080.0*scale) as usize);
                for p in img.data.chunks_exact_mut(4) { for v in &mut p[..3] { *v = (*v as f64*gain).min(255.0) as u8; } }
                let frames = detect(&img, &vorax_scan::layout::layout(img.width,img.height),3).unwrap_or_else(|| panic!("{file} scale={scale} gain={gain}"));
                for (i,(c,w,h)) in frames.iter().enumerate() {
                    let cy = if (i==0 && file=="instrument-selected-ko-1080p.png") || (i==2 && file=="potion-round5-selected-ko-1080p.png") {794.0} else if i==1 {819.0} else {825.0};
                    // The round-five capture also hovers the middle card: its neighbours fan out.
                    // These are measured fixture coordinates, not assumptions in the detector.
                    let xs = if file=="potion-round5-selected-ko-1080p.png" {[851.0,1077.0,1305.0]} else {[889.0,1077.0,1265.0]};
                    assert!((c.cx-xs[i]*scale).abs()<8.0*scale && (c.cy-cy*scale).abs()<8.0*scale, "{file} {scale} {gain} {i}: {frames:?}");
                    assert!((w-226.0*scale).abs()<9.0*scale && (h-300.0*scale).abs()<9.0*scale, "{file} {scale} {gain}: {w}x{h}");
                }
            } }
        }
    }

    #[test]
    fn every_rarity_order_and_raised_slot_keeps_full_card_geometry() {
        use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke, Transform, FillRule};
        let palette = [[145, 145, 145], [255, 190, 65], [40, 155, 230]];
        for order in 0..27 { for raised in 0..4 { for scale in [1.0, 0.75] {
            let (w, h) = ((1920.0 * scale) as u32, (1080.0 * scale) as u32);
            let mut pm = Pixmap::new(w, h).unwrap();
            pm.fill(Color::from_rgba8(65, 45, 40, 255));
            // Gold controls below the fan must not outrank neutral card borders.
            let mut button = Paint::default();
            button.set_color_rgba8(190, 135, 50, 255);
            pm.fill_rect(tiny_skia::Rect::from_xywh(1190.0, 1004.0, 180.0, 35.0).unwrap(),
                &button, Transform::from_scale(scale as f32, scale as f32), None);
            let positions = [(886.0, 825.0, -0.0576), (1076.0, 820.0, 0.0), (1265.0, 825.0, 0.0506)];
            let mut draw_card = |j: usize| {
                let (x, y, angle) = positions[j];
                let y = y - if raised == j { 30.0 } else { 0.0 };
                let mut pb = PathBuilder::new();
                pb.push_rect(tiny_skia::Rect::from_xywh(-112.5, -151.0, 225.0, 302.0).unwrap());
                let path = pb.finish().unwrap();
                let transform = Transform::from_rotate((angle as f64).to_degrees() as f32).post_translate(x as f32, y as f32).post_scale(scale as f32, scale as f32);
                let mut p = Paint::default();
                p.set_color_rgba8(20, 20, 20, 255);
                pm.fill_path(&path, &p, FillRule::Winding, transform, None);
                let [r, g, b] = palette[(order / 3usize.pow(j as u32)) % 3];
                p.set_color_rgba8(r, g, b, 255);
                pm.stroke_path(&path, &p, &Stroke { width: 4.0, ..Default::default() }, transform, None);
            };
            // Selected cards are in front of both neighbours.
            for j in 0..3 { if j != raised { draw_card(j); } }
            if raised < 3 { draw_card(raised); }
            let img = Img { width: w as usize, height: h as usize, data: pm.data().to_vec() };
            let fitted = detect(&img, &vorax_scan::layout::layout(w as usize, h as usize), 3)
                .unwrap_or_else(|| panic!("rarities={order} raised={raised} scale={scale}"));
            for (j, (c, cw, ch)) in fitted.iter().enumerate() {
                let (x, y, _) = positions[j];
                let y = y - if raised == j { 30.0 } else { 0.0 };
                assert!((c.cx - x * scale).abs() < 7.0 * scale && (c.cy - y * scale).abs() < 7.0 * scale,
                    "rarities={order} raised={raised} scale={scale} slot={j}: {c:?}");
                assert!((*cw - 225.0 * scale).abs() < 7.0 * scale && (*ch - 302.0 * scale).abs() < 7.0 * scale,
                    "rarities={order} raised={raised} scale={scale}: {cw}x{ch}");
            }
        } } }
    }

    #[test]
    fn localized_card_edges_override_the_smaller_ocr_layout_at_two_resolutions() {
        use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke, Transform, FillRule};
        for scale in [1.0, 0.75] {
            let (w, h) = ((1920.0 * scale) as u32, (1080.0 * scale) as u32);
            let mut pm = Pixmap::new(w, h).unwrap();
            pm.fill(Color::from_rgba8(40, 20, 15, 255));
            let positions = [(886.0, 825.0, -0.0576), (1076.0, 820.0, 0.0), (1265.0, 825.0, 0.0506)];
            for (x, y, angle) in positions {
                let mut pb = PathBuilder::new();
                pb.push_rect(tiny_skia::Rect::from_xywh(-112.5, -151.0, 225.0, 302.0).unwrap());
                let path = pb.finish().unwrap();
                let transform = Transform::from_rotate((angle as f64).to_degrees() as f32).post_translate(x as f32, y as f32).post_scale(scale as f32, scale as f32);
                let mut p = Paint::default();
                p.set_color_rgba8(45, 25, 10, 255);
                pm.fill_path(&path, &p, FillRule::Winding, transform, None);
                p.set_color_rgba8(255, 190, 65, 255);
                pm.stroke_path(&path, &p, &Stroke { width: 4.0, ..Default::default() }, transform, None);
            }
            let img = Img { width: w as usize, height: h as usize, data: pm.data().to_vec() };
            let fitted = detect(&img, &vorax_scan::layout::layout(w as usize, h as usize), 3).expect("localized gold borders");
            for ((c, cw, ch), (x, y, _)) in fitted.iter().zip(positions) {
                assert!((c.cx - x * scale).abs() < 4.0, "x: {} vs {}", c.cx, x * scale);
                assert!((c.cy - y * scale).abs() < 4.0, "y: {} vs {}", c.cy, y * scale);
                assert!((*cw - 225.0 * scale).abs() < 4.0);
                assert!((*ch - 302.0 * scale).abs() < 4.0);
            }
        }
    }
}
