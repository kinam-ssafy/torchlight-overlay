//! Korean display names and OCR aliases. Game terms, independent of the solver's stable English identifiers.
use std::{collections::HashMap, sync::OnceLock};

pub fn names() -> &'static HashMap<String, String> {
    static NAMES: OnceLock<HashMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| serde_json::from_str(include_str!("../../../data/names-ko.json")).expect("Korean game names"))
}

pub fn name(english: &str) -> &str {
    names().get(english).map(String::as_str).unwrap_or(english)
}

pub fn hangul(c: char) -> bool { ('\u{ac00}'..='\u{d7a3}').contains(&c) }

pub(crate) fn one_jamo_apart(a: char, b: char) -> bool {
    if !hangul(a) || !hangul(b) { return false; }
    let (a, b) = (a as u32 - 0xac00, b as u32 - 0xac00);
    [a / 588 == b / 588, a / 28 % 21 == b / 28 % 21, a % 28 == b % 28].into_iter().filter(|same| *same).count() == 2
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_are_unique_and_known() {
        let data = crate::scan_data();
        let all: Vec<_> = data.names.potion.iter().chain(&data.names.instrument).chain(&data.names.protocol).collect();
        let mut values = std::collections::HashSet::new();
        for (en, ko) in names() {
            assert!(all.contains(&en), "unknown card: {en}");
            assert!(ko.chars().any(hangul));
            assert!(values.insert(ko), "duplicate translation: {ko}");
        }
        assert_eq!(name("Incubation Sac"), "부화낭");
        assert_eq!(name("unseen card"), "unseen card");
    }

    #[test]
    fn every_korean_title_ranks_its_english_identifier_first() {
        use crate::titles::name_fit;
        let d = crate::scan_data();
        for phase in [&d.names.potion, &d.names.instrument, &d.names.protocol] {
            for english in phase {
                let korean = name(english);
                if korean == english { continue; }
                let ocr = korean.replace(' ', "").replace('-', " ");
                let actual = name_fit(english, &ocr);
                let other = phase.iter().filter(|n| *n != english).map(|n| name_fit(n, &ocr)).fold(f64::NEG_INFINITY, f64::max);
                assert!(actual >= 6.0 && actual - other >= 3.0, "ambiguous {english} ({korean}): {actual} vs {other}");
            }
        }
        assert!(name_fit("Incubation Sac", "잘못된 화면") < 6.0);
        assert!(name_fit("Incubation Sac", "") < 6.0);
    }

    #[test]
    fn syllable_components_recover_ocr_noise_without_matching_unrelated_text() {
        use crate::titles::name_fit;
        let d = crate::scan_data();
        let all: Vec<_> = d.names.potion.iter().chain(&d.names.instrument).chain(&d.names.protocol).collect();
        for (english, text) in [("Canine Teeth File", "건치 줄칼"), ("Canine Teeth File", "견지 줄칼"),
            ("Prismatic Liver", "얼록진 간"), ("Vermicular Spinal Cord", "꿈를대는 적수")]
        {
            let score = name_fit(english, text);
            let other = all.iter().filter(|n| n.as_str() != english).map(|n| name_fit(n, text)).fold(f64::NEG_INFINITY, f64::max);
            assert!(score >= 6.0 && score - other >= 3.0, "{text}: {score} vs {other}");
        }
        for text in ["", "잘못된 화면", "라운드 종료 시 모든 몬스터 활성", "부화", "견치", "줄칼"] {
            assert!(all.iter().all(|n| name_fit(n, text) < 6.0), "unrelated or incomplete: {text}");
        }
        assert!(one_jamo_apart('견', '건'));
        assert!(!one_jamo_apart('견', '견'));
        assert!(!one_jamo_apart('견', '칼'));
    }
}
