//! A frame of the game into advice and marks: read the board, follow the instruments through the game, ask the
//! advisor, lay out the marks for the screen the overlay covers. Shared by the live app and --preview.

use crate::draw::{plan_for, Plan};
use vorax::advisor::{advise_plan, Advice};
use vorax_scan::layout::layout_at;
use vorax_scan::live::carried_round;
use vorax_scan::pixels::Img;
use vorax_scan::reading::{Problem, Reading};
use vorax_scan::titles::pictures;
#[cfg(test)]
use vorax_scan::titles::read_titles;
use vorax_scan::track::{track_instruments, InstrumentTracker};
use vorax_scan::window::{crop_to, windowed_game};
use vorax_scan::{scan, scan_data};

pub type TextReader<'a> = Option<&'a mut dyn FnMut(&Img) -> String>;

/// Futures per option the Solver plays for the overlay's advice. More futures make close calls less a matter of
/// luck in the Solver's own playouts. On the same 3,000 seeds with the full Solver (Oct 24, 2026): the linear board
/// value at 1024 futures reached 840k in 81.5% of games, the trained trees at 256 in 79.2% and at 512 in 81.0%;
/// the linear one is about five times faster to evaluate, so 1024 of its futures take less time than 256 with the
/// trees (the hardest recorded board: about 0.2 s on 12 threads).
pub const FUTURES: u32 = 1024;
pub fn futures() -> u32 {
    std::env::var("VORAX_FUTURES").ok().and_then(|s| s.parse().ok()).filter(|n| [256, 512, 1024].contains(n)).unwrap_or(FUTURES)
}

/// Unread fields can change both the legal targets and the ranking. Suppress advice until another read.
pub fn trustworthy(reading: &Reading) -> bool {
    !reading.has(|p| matches!(p, Problem::Unreadable { .. } | Problem::UnknownArt { .. } | Problem::UnknownName { .. }))
        && !reading.state.hand.is_empty()
        && (0..reading.state.hand.len()).all(|i| reading.fields.get(&format!("hand{i}")).is_some_and(|f| f.sure))
        && (0..reading.state.instruments.len()).all(|i| reading.fields.get(&format!("instrument{i}")).is_some_and(|f| f.sure))
        && (reading.state.phase == vorax::advisor::ScreenPhase::Protocol || reading.fields.get("refreshes").is_some_and(|f| f.sure))
}

pub enum Outcome {
    /// not the Surgical Preparation screen
    NotGame,
    Uncertain { reading: Box<Reading> },
    /// a card flying off as it's played: the last advice still stands
    Animating,
    /// the game closing after round 13's last pick (its end animation, the rewards): not a board
    Closing,
    /// advice for the board, with its marks
    Advice { reading: Box<Reading>, advice: Box<Advice>, plan: Plan, read_ms: f64, cached: bool },
}

pub struct Pipeline {
    last_advice: Option<(String, u32, Advice)>,
    card_frames: Option<Vec<crate::card_frame::CardFrame>>,
    pub tracker: InstrumentTracker,
    pub shelley: i32,
    /// instrument rerolls left, as last read on an instrument offer this game (a potion board doesn't show them)
    instrument_refreshes: Option<i32>,
    /// the last board advised on, with the size of its picture, for advising again when a setting changes (and
    /// for its round, when the next board's can't be read)
    last: Option<(Reading, usize, usize)>,
    /// this game reached the Surgical Protocols (rounds 11-13): a board after them with no cards in hand is the game
    /// closing (its end animation, the rewards). Read as a board it gave six 1횞1 monsters in "round 1"; round 13's
    /// own board may not be read first (caught mid-animation, or its number read as 12)
    protocols: bool,
    /// the last read was the Vorax screen opened again on the game under way (Escape and back, to skip long
    /// animations), not a new game
    pub resumed: bool,
}

/// Whether the screen opening is the game under way going on rather than a new one, from the round of the board
/// before (`before`) and the round now read for sure (`now`). A new game starts in round 1, and none goes on past
/// round 13. Icons in the instrument panel prove nothing: a new game can start with an item from the map in it
/// (a boss drop), read as the nearest instrument; taken for the game under way, it stayed in as an instrument.
fn resumes(before: Option<i32>, now: Option<i32>) -> bool {
    match (before, now) {
        (Some(b), Some(n)) => n > 1 && b < 13 && n >= b,
        _ => false,
    }
}

/// Instrument rerolls a game starts with, by Intern Shelley level (src/data/shelley.ts).
pub fn shelley_rerolls(level: i32) -> i32 {
    [0, 1, 1, 2, 2, 2].get((level - 1).clamp(0, 5) as usize).copied().unwrap_or(0)
}

impl Pipeline {
    pub fn new(shelley: i32) -> Pipeline {
        Pipeline { last_advice: None, card_frames: None, tracker: InstrumentTracker::default(), shelley, instrument_refreshes: None, last: None, protocols: false, resumed: false }
    }

    /// The board's instrument rerolls: as read on an instrument offer, else carried from the last one this game,
    /// else what the Shelley level starts a game with. (Upstream counts them as 0 on a potion board.)
    fn fill_rerolls(&mut self, reading: &mut Reading) {
        match reading.state.phase {
            vorax::advisor::ScreenPhase::Instrument => {
                if reading.fields.get("refreshes").is_some_and(|f| f.sure) {
                    self.instrument_refreshes = reading.state.instrument_refreshes;
                }
            }
            _ => {
                if reading.state.instrument_refreshes.is_none() {
                    reading.state.instrument_refreshes = Some(self.instrument_refreshes.unwrap_or(shelley_rerolls(self.shelley)));
                }
            }
        }
    }

    /// The last board advised again (the Shelley level changed): its marks, for a picture of this size.
    pub fn readvise(&mut self) -> Option<(Plan, usize, usize)> {
        let (mut reading, w, h) = self.last.clone()?;
        reading.state.shelley = Some(self.shelley);
        if reading.state.phase != vorax::advisor::ScreenPhase::Instrument {
            reading.state.instrument_refreshes = None;
        }
        self.fill_rerolls(&mut reading);
        if !trustworthy(&reading) { return None; }
        let (advice, _) = self.advise(&reading.state);
        let mut plan = plan_for(&advice, &reading.state, &reading, &layout_at(w, h, reading.ui()).shifted_hand(reading.hand_dy.unwrap_or(0.0)));
        self.align_card(&mut plan, &advice);
        Some((plan, w, h))
    }

    /// `frame`: the captured game window; `trigger`: "screen" when the log says a new game opened.
    pub fn run(&mut self, frame: &Img, trigger: Option<&str>, ocr: TextReader) -> Outcome {
        let t = std::time::Instant::now();
        self.resumed = false;
        let data = scan_data();
        let img = match windowed_game(frame) {
            Some(a) => crop_to(frame, &a),
            None => frame.clone(),
        };
        // with a text reader: the cards by their printed names, by their pictures only where a name doesn't settle
        let mut fitted = None;
        let raw = match ocr {
            Some(read) => {
                let mut names = scan(&img, data, false);
                let (dy, ui) = (names.hand_dy.unwrap_or(0.0), names.ui());
                let korean = std::env::var("VORAX_LANGUAGE").map_or(true, |v| v == "ko");
                let layout = layout_at(img.width, img.height, ui).shifted_hand(dy);
                if korean {
                    if let Some(panel) = vorax_scan::instruments::refine_instruments(&img, &layout, data) {
                        names.state.instruments = panel.names;
                        names.instrument_ranks = Some(panel.ranks);
                        names.fields.retain(|key, _| !key.starts_with("instrument"));
                        names.fields.extend(panel.fields);
                    }
                }
                fitted = crate::card_frame::detect(&img, &layout, names.hand_size.unwrap_or(3));
                // OCR and drawing share the measured card geometry, including a selected card raised
                // above the fan. Convert the physical centre to the title reader's reference centre.
                let positions = fitted.as_ref().filter(|_| korean).map(|frames| frames.iter().map(|(c, _, h)|
                    vorax_scan::layout::CardPos { cy: c.cy - h / 12.0 + 62.5 * layout.k, ..*c }
                ).collect::<Vec<_>>());
                let pics = |phase: &str, size: usize, slots: &[usize]| pictures(&img, data, phase, size, slots, dy, ui);
                vorax_scan::titles::read_titles_styled(&img, names, data, read, Some(&pics), korean, positions.as_deref())
            }
            None => scan(&img, data, true),
        };
        if raw.has(|p| matches!(p, Problem::Animating)) {
            return Outcome::Animating;
        }
        if raw.has(|p| matches!(p, Problem::NotFound)) {
            return Outcome::NotGame;
        }
        // the screen opened again on the game under way (Escape and back skips long animations): what's known of it
        // (instruments and their rounds, rerolls) is kept
        let round_sure = raw.fields.get("round").is_some_and(|f| f.sure);
        let before = self.last.as_ref().and_then(|(r, ..)| r.state.round);
        self.resumed = trigger == Some("screen") && resumes(before, round_sure.then_some(raw.state.round.unwrap_or(1)));
        let trigger = if self.resumed { None } else { trigger };
        if trigger == Some("screen") {
            self.instrument_refreshes = None;
            self.protocols = false;
        }
        // a new game opens with "screen" and an instrument offer: cards in hand
        let protocol = raw.state.phase == vorax::advisor::ScreenPhase::Protocol;
        if self.protocols && !protocol && raw.state.hand.is_empty() {
            return Outcome::Closing;
        }
        self.protocols |= protocol;
        // A manual first capture can be a reward instrument offer midway through round 1.
        // Seed only confidently read owned icons; an explicit new-game event still resets them.
        if self.last.is_none() && self.tracker.owned.is_empty() && trigger.is_none()
            && raw.state.phase == vorax::advisor::ScreenPhase::Instrument && !raw.state.instruments.is_empty() {
            if !(0..raw.state.instruments.len()).all(|i| raw.fields.get(&format!("instrument{i}")).is_some_and(|f| f.sure)) {
                // An unread owned panel is not evidence that this is a fresh game with zero tools.
                return Outcome::Uncertain { reading: Box::new(raw) };
            }
            self.tracker.owned = raw.state.instruments.clone();
            self.tracker.rounds = (0..self.tracker.owned.len()).map(|i| if i == 0 && raw.state.round == Some(1) { 1 } else { 0 }).collect();
        }
        let tr = track_instruments(&self.tracker, &raw, trigger);
        self.tracker = tr.tracker.clone();
        let mut reading = raw;
        reading.state.instruments = tr.instruments.clone();
        reading.state.instrument_rounds = Some(tr.rounds.clone());
        reading.fields.retain(|key, _| !key.starts_with("instrument"));
        reading.fields.extend(tr.fields);
        // a round not read for sure: the one before stands in, not round 1 (before any board, round 1 as upstream's
        // Solver starts). So does one that can't be: a protocol board below round 11, or a round going back when the
        // screen opened again on the game under way (still fading in, round 12 read as 2)
        let before_round = self.last.as_ref().and_then(|(r, ..)| r.state.round);
        let read_round = reading.state.round.unwrap_or(1);
        let impossible = (reading.state.phase == vorax::advisor::ScreenPhase::Protocol && read_round < 11) || (self.resumed && before_round.is_some_and(|b| read_round < b));
        if let Some(f) = reading.fields.get("round").filter(|f| !f.sure || impossible) {
            let before = self.last.as_ref().map_or(Some(1), |(r, ..)| r.state.round);
            reading.state.round = Some(carried_round(before, reading.state.phase, f.guess.as_deref()));
        }
        reading.state.shelley = Some(self.shelley);
        self.fill_rerolls(&mut reading);
        let read_ms = t.elapsed().as_secs_f64() * 1000.0;
        if !trustworthy(&reading) { return Outcome::Uncertain { reading: Box::new(reading) }; }
        let (advice, cached) = self.advise(&reading.state);
        let l = layout_at(img.width, img.height, reading.ui()).shifted_hand(reading.hand_dy.unwrap_or(0.0));
        self.card_frames = fitted.or_else(|| crate::card_frame::detect(&img, &l, reading.state.hand.len()));
        let mut plan = plan_for(&advice, &reading.state, &reading, &l);
        self.align_card(&mut plan, &advice);
        self.last = Some((reading.clone(), img.width, img.height));
        Outcome::Advice { reading: Box::new(reading), advice: Box::new(advice), plan, read_ms, cached }
    }

    /// A pick the tracker isn't sure of yet, first seen on this board: worth another look once the icon has landed.
    pub fn wants_relook(&self) -> bool {
        self.tracker.unsure.iter().any(|q| q.looks == 0)
    }

    fn align_card(&self, plan: &mut Plan, advice: &Advice) {
        if plan.card.is_some() {
            if let (Some(frames), vorax::advisor::Best::Card(card)) = (&self.card_frames, &advice.best) {
                plan.card = frames.get(card.index).copied();
            }
        }
    }

    fn advise(&mut self, state: &vorax::advisor::ScreenState) -> (Advice, bool) {
        let key = serde_json::to_string(state).expect("finite screen state");
        let count = futures();
        if let Some((previous, n, advice)) = &self.last_advice {
            if previous == &key && *n == count {
                let mut cached = advice.clone();
                cached.ms = 0;
                return (cached, true);
            }
        }
        let advice = advise_plan(state, Some(count), true);
        self.last_advice = Some((key, count, advice.clone()));
        (advice, false)
    }
}

pub fn decode_png(path: &std::path::Path) -> Result<Img, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut dec = png::Decoder::new(file);
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = info.color_type.samples();
    let mut img = Img::new(w, h);
    for i in 0..w * h {
        let p = &buf[i * ch..i * ch + ch];
        let px = match ch {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        };
        img.data[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
    Ok(img)
}

pub fn encode_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgba).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::resumes;

    #[test]
    fn uncertain_input_never_produces_actionable_advice() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/board-1080p.png")).unwrap();
        let mut reading = scan(&frame, scan_data(), true);
        reading.problems.clear();
        assert!(trustworthy(&reading));
        reading.problems.push(Problem::Unreadable { field: "tank0.quantity".into() });
        assert!(!trustworthy(&reading));
        reading.problems.clear();
        reading.problems.push(Problem::UnknownName { name: "new card".into() });
        assert!(!trustworthy(&reading));
    }

    #[test]
    fn visible_rerolls_override_pet_fallback() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/board-1080p.png")).unwrap();
        let mut reading = scan(&frame, scan_data(), true);
        let mut pipeline = Pipeline::new(6);
        reading.state.phase = vorax::advisor::ScreenPhase::Instrument;
        reading.state.instrument_refreshes = Some(0);
        reading.fields.insert("refreshes".into(), vorax_scan::reading::FieldRead::sure(true));
        pipeline.fill_rerolls(&mut reading);
        reading.state.phase = vorax::advisor::ScreenPhase::Potion;
        reading.state.instrument_refreshes = None;
        pipeline.fill_rerolls(&mut reading);
        assert_eq!(reading.state.instrument_refreshes, Some(0));
        assert_eq!(shelley_rerolls(1), 0);
        assert_eq!(shelley_rerolls(2), 1);
        assert_eq!(shelley_rerolls(6), 2);
        assert_eq!(shelley_rerolls(0), 0);
    }

    #[test]
    fn korean_titles_recover_gold_text_and_keep_all_three_positions() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/board-1080p.png")).unwrap();
        let data = scan_data();
        let names = scan(&frame, data, false);
        // Live Korean OCR: the middle gold heading needs the alternate contrast pass. No picture
        // fallback is supplied, so a missing title must never compact the hand into two positions.
        let mut text = ["비대한 늬하수체", "역! 가 최소2그룹 변이", "얼룩진 간", "오그라는 손가락"].into_iter();
        let read = read_titles(&frame, names, data, &mut |_| text.next().expect("bounded OCR retry").to_owned(), None);
        assert_eq!(read.state.hand, ["Hypertrophic Pituitary", "Prismatic Liver", "Contracted Claws"]);
        assert!(!read.has(|p| matches!(p, Problem::UnknownArt { .. })));
        assert!(text.next().is_none());
    }

    #[test]
    fn first_instrument_then_blue_potion_keeps_recommending() {
        use super::*;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames");
        let instrument = decode_png(&root.join("board-1080p.png")).unwrap();
        let potion = decode_png(&root.join("potion-ko-1080p.png")).unwrap();
        let mut pipe = Pipeline::new(5);
        assert!(matches!(pipe.run(&instrument, Some("screen"), None), Outcome::Advice { .. }));
        // Recover within the bounded contrast passes when the first crops are unreadable.
        let mut titles = ["1 그를 몬스터를 선택 그 우측의 몬스터를", "끄入4하 다즈 요에", "끈적한 담즙 용액", "허물 용액", "종기 치료 연고"].into_iter();
        let mut read = |_: &Img| titles.next().expect("bounded title retries").to_owned();
        let Outcome::Advice { reading, .. } = pipe.run(&potion, Some("board"), Some(&mut read)) else { panic!("potion advice after instrument pick") };
        assert_eq!(reading.state.hand, ["Viscous Bile Solution", "Slough Solution", "Cleansing Ointment"]);
        assert_eq!(reading.state.instruments, ["Contracted Claws"]);
        assert_eq!(reading.state.round, Some(1));
        assert_eq!(reading.state.potion_refreshes, Some(3));
        assert!(titles.next().is_none());
    }

    #[test]
    fn successive_korean_hands_keep_positions_and_updated_tanks() {
        use super::*;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames");
        let mut pipe = Pipeline::new(3);
        for (file, titles, expected) in [
            ("board-1080p.png", ["비대한 뇌하수체", "얼룩진 간", "오그라든 손가락"], ["Hypertrophic Pituitary", "Prismatic Liver", "Contracted Claws"]),
            ("potion-ko-1080p.png", ["끈적한 담즙 용액", "허물 용액", "종기 치료 연고"], ["Viscous Bile Solution", "Slough Solution", "Cleansing Ointment"]),
            // Actual heading-only Windows OCR: one residual syllable error has a clear margin.
            ("instrument-ko-1080p.png", ["견치 줄칼", "꿈틀대는 적수", "트와이스 본의 부리"], ["Canine Teeth File", "Vermicular Spinal Cord", "Beak of the Twice-Born"]),
        ] {
            let frame = decode_png(&root.join(file)).unwrap();
            let mut titles = titles.into_iter();
            let mut read = |_: &Img| titles.next().expect("one heading per card").to_owned();
            let Outcome::Advice { reading, .. } = pipe.run(&frame, Some("board"), Some(&mut read)) else { panic!("no advice for {file}") };
            assert_eq!(reading.state.hand, expected);
            assert!(trustworthy(&reading));
            assert!(titles.next().is_none());
            if file == "instrument-ko-1080p.png" {
                assert_eq!(reading.state.instrument_refreshes, Some(1));
                assert_eq!(reading.state.instruments, ["Contracted Claws"]);
                assert!(reading.state.tanks[1].is_none());
                let tank = reading.state.tanks[0].as_ref().unwrap();
                assert_eq!((tank.activity, tank.quantity), (2, 72));
            }
        }
    }

    #[test]
    fn ambiguous_card_guess_is_not_actionable() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/board-1080p.png")).unwrap();
        let mut reading = scan(&frame, scan_data(), true);
        reading.problems.clear();
        reading.fields.get_mut("hand0").unwrap().sure = false;
        assert!(!trustworthy(&reading));
    }

    #[test]
    fn restart_on_a_reward_offer_preserves_confident_owned_icons_but_new_game_resets() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/instrument-ko-1080p.png")).unwrap();
        for (trigger, expected) in [(None, vec!["Contracted Claws"]), (Some("screen"), vec![])] {
            let mut pipe = Pipeline::new(3);
            let mut titles = ["견치 줄칼", "꿈틀대는 적수", "트와이스 본의 부리"].into_iter();
            let mut read = |_: &Img| titles.next().unwrap().to_owned();
            let Outcome::Advice { reading, .. } = pipe.run(&frame, trigger, Some(&mut read)) else { panic!("reward offer must be readable") };
            assert_eq!(reading.state.instruments, expected);
        }
    }

    #[test]
    fn selected_korean_card_uses_its_measured_title_and_restores_two_owned_icons() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/instrument-selected-ko-1080p.png")).unwrap();
        let mut pipe = Pipeline::new(3);
        let mut titles = ["무두질 가죽 구속대", "흑염소 거트", "꿈틀대는 적수"].into_iter();
        let mut read = |_: &Img| titles.next().expect("one bounded heading per card").to_owned();
        let Outcome::Advice { reading, plan, .. } = pipe.run(&frame, None, Some(&mut read)) else { panic!("selected card must remain readable") };
        assert_eq!(reading.state.hand, ["Tanned Leather Restraint", "Black Goat Gut Suture", "Vermicular Spinal Cord"]);
        assert_eq!(reading.state.instruments, ["Contracted Claws", "Beak of the Twice-Born"]);
        assert!(trustworthy(&reading));
        let (_, w, h) = plan.card.expect("recommended card frame");
        assert!((w-226.0).abs()<6.0 && (h-300.0).abs()<6.0);
    }

    #[test]
    fn an_unread_reroll_counter_cannot_suggest_spending_it() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/board-1080p.png")).unwrap();
        let mut reading = scan(&frame, scan_data(), true);
        reading.problems.clear();
        reading.fields.insert("refreshes".into(), vorax_scan::reading::FieldRead::sure(false));
        assert!(!trustworthy(&reading));
    }

    #[test]
    fn round_two_restores_all_three_icons_and_full_counter() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/potion-round2-ko-1080p.png")).unwrap();
        let mut pipe = Pipeline::new(3);
        let mut titles = ["Vivigenic Powder", "Twin Hormone - Bone Sentry", "Digestive Enzyme Solution"].into_iter();
        let mut read = |_: &Img| vorax_scan::korean::name(titles.next().unwrap()).to_owned();
        let Outcome::Advice { mut reading, plan, .. } = pipe.run(&frame, None, Some(&mut read)) else { panic!("round two must be readable") };
        assert_eq!(reading.state.round, Some(2));
        assert_eq!(reading.state.potion_refreshes, Some(3));
        assert_eq!(reading.state.instruments, ["Contracted Claws", "Beak of the Twice-Born", "Black Goat Gut Suture"]);
        let tank = reading.state.tanks[2].as_ref().unwrap();
        assert_eq!((tank.activity, tank.quantity), (300,151));
        assert!(trustworthy(&reading));
        assert!(plan.card.is_some() || plan.pill.is_some());
        reading.fields.get_mut("instrument2").unwrap().sure = false;
        assert!(!trustworthy(&reading), "an uncertain equipped tool changes the simulation");
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Requires the installed Windows Korean OCR model; run explicitly on the release machine"]
    fn recorded_korean_run_with_real_windows_ocr() {
        use super::*;
        unsafe { windows::Win32::System::WinRT::RoInitialize(windows::Win32::System::WinRT::RO_INIT_MULTITHREADED).unwrap(); }
        let ocr = crate::app::ocr::Ocr::new().unwrap();
        assert_eq!(ocr.language.split('-').next(), Some("ko"));
        let mut pipe = Pipeline::new(3);
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames");
        for (file, hand, owned, round) in [
            ("initial-ko-1080p.png", ["Hypertrophic Pituitary", "Prismatic Liver", "Contracted Claws"], 0, 1),
            ("potion-ko-1080p.png", ["Viscous Bile Solution", "Slough Solution", "Cleansing Ointment"], 1, 1),
            ("instrument-ko-1080p.png", ["Canine Teeth File", "Vermicular Spinal Cord", "Beak of the Twice-Born"], 1, 1),
            ("instrument-selected-ko-1080p.png", ["Tanned Leather Restraint", "Black Goat Gut Suture", "Vermicular Spinal Cord"], 2, 1),
            ("potion-round2-ko-1080p.png", ["Vivigenic Powder", "Twin Hormone - Bone Sentry", "Digestive Enzyme Solution"], 3, 2),
        ] {
            let frame = decode_png(&root.join(file)).unwrap();
            let mut read = |crop: &Img| ocr.read(crop).unwrap();
            let outcome = pipe.run(&frame, Some("board"), Some(&mut read));
            let Outcome::Advice { reading, plan, read_ms, advice, .. } = outcome else {
                if let Outcome::Uncertain { reading } = outcome { panic!("{file}: {:?} {:?}", reading.problems, reading.fields); }
                panic!("{file}: no advice");
            };
            assert_eq!(reading.state.hand, hand, "{file}");
            assert_eq!(reading.state.instruments, ["Contracted Claws", "Beak of the Twice-Born", "Black Goat Gut Suture"][..owned], "{file}");
            assert_eq!(reading.state.round, Some(round), "{file}");
            if let Some((_, width, height)) = plan.card {
                assert!((215.0..235.0).contains(&width) && (288.0..315.0).contains(&height), "{file}: {width}x{height}");
            }
            eprintln!("{file}: read={read_ms:.0}ms solve={}ms tools={owned} round={round}", advice.ms);
        }
    }

    #[test]
    fn localized_counters_keep_values_under_brightness_changes() {
        use super::*;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames");
        for (file, round, refresh) in [("initial-ko-1080p.png",1,1),("potion-ko-1080p.png",1,3),("potion-round2-ko-1080p.png",2,3)] {
            let source = decode_png(&root.join(file)).unwrap();
            let expected_tanks = serde_json::to_value(scan(&source,scan_data(),false).state.tanks).unwrap();
            for gain in [0.85, 1.0, 1.15] {
                let mut img = source.clone();
                for p in img.data.chunks_exact_mut(4) { for v in &mut p[..3] { *v = (*v as f64*gain).min(255.0) as u8; } }
                let reading = scan(&img,scan_data(),false);
                assert!(reading.fields["round"].sure && reading.fields["refreshes"].sure, "{file} {gain}: {:?}",reading.fields);
                assert_eq!(reading.state.round,Some(round));
                assert_eq!(reading.fields["refreshes"].guess,Some(refresh.to_string()));
                assert_eq!(serde_json::to_value(&reading.state.tanks).unwrap(),expected_tanks,"{file} {gain}: tank presence and values");
                if round == 2 {
                    let l = layout_at(img.width,img.height,reading.ui());
                    let panel = vorax_scan::instruments::refine_instruments(&img,&l,scan_data()).expect("three equipped icons");
                    assert_eq!(panel.names,["Contracted Claws","Beak of the Twice-Born","Black Goat Gut Suture"]);
                }
            }
        }
    }

    #[test]
    fn unchanged_state_reuses_exact_advice_but_pet_change_recalculates() {
        use super::*;
        let frame = decode_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/frames/board-1080p.png")).unwrap();
        let mut pipe = Pipeline::new(0);
        let Outcome::Advice { advice: first, cached: false, .. } = pipe.run(&frame, None, None) else { panic!("first calculation") };
        let Outcome::Advice { advice: second, cached: true, .. } = pipe.run(&frame, None, None) else { panic!("cached calculation") };
        assert_eq!(first.best, second.best);
        assert_eq!(first.ranking, second.ranking);
        pipe.shelley = 6;
        assert!(matches!(pipe.run(&frame, None, None), Outcome::Advice { cached: false, .. }));
    }

    #[test]
    fn the_screen_opened_again_mid_game_goes_on() {
        assert!(resumes(Some(6), Some(6)));
        assert!(resumes(Some(6), Some(7)));
        assert!(resumes(Some(12), Some(12)));
    }

    #[test]
    fn a_new_game_is_a_new_game() {
        // a new game starts in round 1, whatever is in the instrument panel (a boss drop from the map)
        assert!(!resumes(Some(13), Some(1)));
        assert!(!resumes(Some(6), Some(1)));
        // nothing goes on after round 13, nor back in rounds, nor when the round isn't read for sure
        assert!(!resumes(Some(13), Some(13)));
        assert!(!resumes(Some(8), Some(5)));
        assert!(!resumes(Some(6), None));
        // the overlay's first board
        assert!(!resumes(None, Some(6)));
    }
}
