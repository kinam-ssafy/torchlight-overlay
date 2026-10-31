//! Build counter references with the production glyph extractor: cargo run --release -p vorax-scan
//! --example round_font_refs -- atlas.png round-font.json. Atlas: ten 64x80 cells, digits 0..9.
use vorax_scan::{pixels::{Img, Rect, gray, glyph_vector}, text::{glyphs, TextKind}};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut decoder = png::Decoder::new(std::fs::File::open(&args[1]).unwrap());
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().unwrap();
    let mut bytes = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut bytes).unwrap();
    assert_eq!((info.width, info.height), (640, 80));
    let mut img = Img::new(640, 80);
    for (src, dst) in bytes.chunks_exact(info.color_type.samples()).zip(img.data.chunks_exact_mut(4)) {
        dst[..3].copy_from_slice(&src[..3]); dst[3] = 255;
    }
    let g = gray(&img);
    let mut refs = indexmap::IndexMap::new();
    for i in 0..10 {
        let gs = glyphs(&img, &Rect::new(i as f64 * 64.0, 0.0, 64.0, 80.0), TextKind::Round);
        assert_eq!(gs.len(), 1, "digit {i}");
        refs.insert(i.to_string(), vec![glyph_vector(&g, img.width, img.height, &gs[0], gs[0].h)]);
    }
    std::fs::write(&args[2], serde_json::to_vec(&refs).unwrap()).unwrap();
}
