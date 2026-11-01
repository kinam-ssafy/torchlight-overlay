use vorax_scan::live::{classify_current, OpeningOffer, Settler, Trigger};

const OPEN: &str = "[Game] TipMsgShowMgr@DispatchPageRunChange PageName = S13GamePlayMain , PageRunState = Run";
const ESCAPE: &str = "[2026.11.01-12.11.17:624][665]TLLua: Display: [Game] TipMsgShowMgr@DispatchPageRunChange PageName = S13GamePlayMain , PageRunState = Destory ";

#[test]
fn escape_cancels_pending_work_and_late_board_replies_until_reopened() {
    for close in [ESCAPE,
        "[Game] TipMsgShowMgr@DispatchPageRunChange PageName = S13GamePlayMain , PageRunState = Destroy",
        "[Game] TipMsgShowMgr@DispatchPageRunChange PageName = S13GamePlayMain , PageRunState = Hide",
        "[Game] Page@ CloseView 5217_S13GamePlayMainCtrl",
        "[Game] S13GamePlayMain Hide"] {
        let mut state = Settler::default();
        let mut opening = OpeningOffer::default();
        opening.feed(&["S13GamePlay_C_StartGameBtn", close]);
        assert!(!opening.is_initial());
        state.feed_current(&[OPEN, "ProtoId = 5619"], 0.0);
        state.feed_current(&[close, "ProtoId = 5619"], 100.0);
        assert!(state.over, "{close}");
        assert_eq!(state.due(10_000.0), None, "closed pages must never retry");
        state.feed_current(&[OPEN], 11_000.0);
        assert!(!state.over);
        assert_eq!(state.due(11_400.0), Some(Trigger::Screen));
        assert_eq!(state.due(12_000.0), None);
    }
}

#[test]
fn page_noise_and_other_pages_do_not_close_or_reopen_the_board() {
    for noise in [
        "[Game] PageStack@ PageName = S13GamePlayMain, PageAction = HideLastFuncPage, LifeState = Run",
        "[Game] TipMsgShowMgr@DispatchPageRunChange PageName = S13GamePlayMainOther , PageRunState = Destory",
        "[Game] S13GamePlayMain RunOther",
        "[Game] Page@ CloseView 5217_S13GamePlayMainOtherCtrl",
        "[Game] UILevel@ CheckLoadSubLevel S13GamePlayMain S13GamePlay",
        "AudioMgr requested row 'S13GamePlayMain_C_CloseTipBtn'",
    ] { assert_eq!(classify_current(noise), None, "{noise}"); }
    assert_eq!(classify_current("[Game] PageBase@ OpenFlow5! OpenView S13GamePlayMain"), Some(Trigger::Screen));
    assert_eq!(classify_current("[Game] PageBase@ OpenView 5217_S13GamePlayMainCtrl nil 123"), Some(Trigger::Screen));
    assert_eq!(classify_current("S13GamePlayMain_C_RefreshBtn2"), Some(Trigger::Refresh));
}

#[test]
fn restarting_after_escape_restores_closed_state_without_replaying_work() {
    let path = std::env::temp_dir().join(format!("vorax-visibility-{}.log", std::process::id()));
    std::fs::write(&path, format!("{OPEN}\n{ESCAPE}\nProtoId = 5619\n")).unwrap();
    let mut state = Settler::from_recent_log(&path);
    assert!(state.over);
    assert_eq!(state.due(10_000.0), None);
    std::fs::write(&path, format!("{ESCAPE}\n{OPEN}\n")).unwrap();
    let mut state = Settler::from_recent_log(&path);
    assert!(!state.over);
    assert_eq!(state.due(10_000.0), None);
    std::fs::remove_file(path).unwrap();
}
