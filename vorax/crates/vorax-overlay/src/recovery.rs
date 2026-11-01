//! A visible explanation, never a guessed replacement for an unread tool.
use vorax_scan::{layout::Layout, pixels::{crop, resize, Img, Rect}, reading::Reading};
use crate::draw::{Pill, Plan};

pub enum Detail {
    Tool(String),
    StartingItem,
}

/// The upright detail card is left of the offer. Require both tall borders and
/// horizontal edges so ordinary labels or a partial animation cannot become text evidence.
pub fn detail_rect(img: &Img) -> Option<Rect> {
    let gold = |x: usize, y: usize| {
        let (r,g,b) = img.px(x,y);
        r >= 100 && g >= 65 && r as i32 - b as i32 >= 55 && g as i32 - b as i32 >= 25
    };
    let (y0,y1) = (img.height / 3, img.height * 94 / 100);
    let columns: Vec<_> = (img.width / 60..img.width / 3).filter(|&x|
        (y0..y1).filter(|&y| gold(x,y)).count() > img.height * 28 / 100).collect();
    let (&left,&right) = (columns.first()?,columns.last()?);
    if !(img.width / 10..img.width / 4).contains(&(right-left)) { return None; }
    let rows: Vec<_> = (y0..y1).filter(|&y|
        (left..=right).filter(|&x| gold(x,y)).count() > (right-left) * 75 / 100).collect();
    let (&top,&bottom) = (rows.first()?,rows.last()?);
    let ratio = (bottom-top) as f64 / (right-left) as f64;
    if !(1.25..1.75).contains(&ratio) { return None; }
    Some(Rect::new(left as f64,top as f64,(right-left) as f64,(bottom-top) as f64))
}

fn text_crop(img: &Img, r: Rect, top: f64, height: f64) -> Img {
    let part = crop(img, Rect::new(r.x+r.w*0.07,r.y+r.h*top,r.w*0.86,r.h*height));
    resize(&part, part.width*2, part.height*2)
}

pub fn read_detail(img: &Img, read: &mut dyn FnMut(&Img)->String) -> Option<Detail> {
    let r = detail_rect(img)?;
    let title = read(&text_crop(img,r,0.40,0.085));
    let footer = read(&text_crop(img,r,0.87,0.095));
    if !compact(&footer).contains("수술도구") { return None; }
    if let Some(name) = vorax_scan::titles::confident_instrument(&title) { return Some(Detail::Tool(name)); }
    let body = read(&text_crop(img,r,0.49,0.32));
    initial_only(&title,&body).then_some(Detail::StartingItem)
}

fn compact(s: &str) -> String { s.chars().filter(|c| c.is_alphanumeric()).collect() }

fn initial_only(title: &str, body: &str) -> bool {
    let title = compact(title);
    let body = compact(body);
    title.chars().filter(|c| vorax_scan::korean::hangul(*c)).count() >= 3
        && body.starts_with("초기몬스터") && body.ends_with("변경된다")
        && !["라운드","매번","추가로","종료","이후"].iter().any(|v| body.contains(v))
}

pub fn guidance(img: &Img, r: &Reading, l: &Layout, icon: Option<Rect>) -> Plan {
    let mut plan = Plan { help:true, pill: Some(Pill { main:"인식 확인 필요".into(), sub:String::new(), dim:false }), ..Plan::default() };
    let bad = |prefix: &str| r.fields.iter().find(|(k,f)| k.starts_with(prefix) && !f.sure).map(|(k,_)|k.as_str());
    let text = if bad("instrument").is_some() {
        plan.button = icon;
        if icon.is_some() { "오른쪽 위 표시한 도구를 클릭해 설명을 열어 주세요" }
        else { "오른쪽 위 도구를 하나씩 클릭해 설명을 열어 주세요" }
    } else if let Some(key) = bad("hand") {
        if let Some(i) = key.strip_prefix("hand").and_then(|s|s.parse::<usize>().ok()) {
            plan.card = crate::card_frame::detect(img,l,r.hand_size.unwrap_or(3)).and_then(|frames|frames.get(i).copied());
        }
        "표시한 카드를 클릭해 펼쳐 주세요. 확인 버튼은 누르지 마세요"
    } else if bad("tank").is_some() || bad("round").is_some() || bad("refreshes").is_some() {
        "설명창을 닫고 탱크 수치와 라운드가 보이게 해 주세요"
    } else {
        "카드 선택 화면을 열고 전환 효과가 끝날 때까지 기다려 주세요"
    };
    plan.pill.as_mut().unwrap().sub = text.into();
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_unknown_title_alone_does_not_prove_a_starting_item() {
        assert!(initial_only("거머리 통","초기 몬스터가 4 그룹 노멀 독충으로 변경된다."));
        for body in ["", "라운드 종료 시 초기 몬스터가 변경된다", "초기 몬스터가 변경된다. 이후 활성 +10", "1 그룹 몬스터 활성 +30"] {
            assert!(!initial_only("거머리 통",body));
        }
        assert!(!initial_only("", "초기 몬스터가 4 그룹 노멀 독충으로 변경된다."));
    }
}
