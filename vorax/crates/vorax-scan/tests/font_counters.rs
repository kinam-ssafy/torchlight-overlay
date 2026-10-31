//! Independently rendered whole numbers exercise segmentation, spacing, scale and contrast.
use vorax_scan::{pixels::{gray, resize, Img, Rect}, scan, scan_data, text::{glyphs, read_counter, TextKind}};

fn decode(file: &str) -> Img {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames").join(file);
    let decoder = png::Decoder::new(std::fs::File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut bytes = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut bytes).unwrap();
    let mut source = Img::new(info.width as usize, info.height as usize);
    for (src, dst) in bytes.chunks_exact(info.color_type.samples()).zip(source.data.chunks_exact_mut(4)) {
        dst[..3].copy_from_slice(&src[..3]); dst[3] = 255;
    }
    source
}

#[test]
fn every_round_reads_at_multiple_resolutions_and_brightness_levels() {
    let source = decode("round-font-counters.png");
    for scale in [2.0/3.0, 5.0/6.0, 1.0, 4.0/3.0, 1.5] {
        for gain in [0.7, 0.85, 1.0, 1.15, 1.3] {
            let mut img = resize(&source, (source.width as f64*scale).round() as usize, (source.height as f64*scale).round() as usize);
            for pixel in img.data.chunks_exact_mut(4) {
                for channel in &mut pixel[..3] { *channel = (*channel as f64*gain).min(255.0) as u8; }
            }
            let g = gray(&img);
            for number in 1..=13 {
                let r = Rect::new(0.0, (number - 1) as f64*80.0*scale, 96.0*scale, 80.0*scale);
                let result = read_counter(&img, &g, &r, TextKind::Round, &scan_data().digits["round"]).unwrap();
                assert_eq!(result.text, number.to_string(), "scale={scale} gain={gain}: {result:?}");
                assert!(result.score >= 0.85 && result.score - result.second >= 0.10,
                    "number={number} scale={scale} gain={gain}: {result:?}");
            }
        }
    }
}

#[test]
fn whole_counter_band_wins_over_clipped_digits_for_every_round() {
    let source = decode("potion-round4-ko-1080p.png");
    let sheet = decode("round-font-counters.png");
    for number in 1..=13 {
        let boxes = glyphs(&sheet, &Rect::new(0.0, (number-1) as f64*80.0, 96.0, 80.0), TextKind::Round);
        let x0 = boxes.first().unwrap().x as usize;
        let x1 = boxes.last().map(|b| (b.x+b.w) as usize).unwrap();
        let y0 = boxes.iter().map(|b| b.y as usize).min().unwrap();
        let y1 = boxes.iter().map(|b| (b.y+b.h) as usize).max().unwrap();
        let mut frame = source.clone();
        for y in 355..411 { for x in 80..155 {
            frame.data[(y*frame.width+x)*4..(y*frame.width+x)*4+3].copy_from_slice(&[24,20,16]);
        } }
        // Match the visible baseline and right alignment; leave /13 and other board data intact.
        for y in y0..y1 { for x in x0..x1 {
            let dst = ((y-y0+368)*frame.width+154-(x1-x0)+x-x0)*4;
            let src = (y*sheet.width+x)*4;
            frame.data[dst..dst+4].copy_from_slice(&sheet.data[src..src+4]);
        } }
        for scale in [2.0/3.0, 5.0/6.0, 1.0] { for gain in [0.85, 1.0, 1.15] {
            let mut img = resize(&frame, (1920.0*scale) as usize, (1080.0*scale) as usize);
            for p in img.data.chunks_exact_mut(4) { for c in &mut p[..3] { *c = (*c as f64*gain).min(255.0) as u8; } }
            let result = scan(&img, scan_data(), false);
            assert!(result.fields["round"].sure, "number={number} scale={scale} gain={gain}: {:?}", result.fields["round"]);
            assert_eq!(result.state.round, Some(number as i32), "scale={scale} gain={gain}: {:?}", result.fields["round"]);
        } }
    }
}
