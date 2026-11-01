use vorax_scan::live::OpeningOffer;

#[test]
fn initial_offer_requires_start_evidence_and_rerolls_do_not_equip_tools() {
    let mut context = OpeningOffer::default();
    context.feed(&["S13GamePlayMain Run", "S13GamePlayMain_C_RefreshBtn2", "ProtoId = 5619"]);
    assert!(!context.is_initial(), "reopening and refreshing cannot prove a new game");
    context.feed(&["S13GamePlay_C_StartGameBtn", "S13GamePlayMain Run"]);
    assert!(context.is_initial());
    for _ in 0..3 {
        context.feed(&["S13GamePlayMain_C_RefreshBtn2", "ProtoId = 5619", "Play_UI_S13_KYZ_Card_Selected_Shuffle"]);
        assert!(context.is_initial());
    }
    context.feed(&["UIButton@ normal-click ComfirmBtn", "ProtoId = 5619"]);
    assert!(!context.is_initial());
    context.feed(&["S13GamePlayMain Run"]);
    assert!(!context.is_initial(), "a reward offer may also be in round 1");
    context.feed(&["S13GamePlay_C_StartGameBtn", "KYZ_Card_Selected_Confirm"]);
    assert!(!context.is_initial());
    context.feed(&["S13GamePlay_C_StartGameBtn", "S13GamePlayMain Hide"]);
    assert!(!context.is_initial(), "closed history must not seed a later board");
}

#[test]
fn bounded_log_bootstrap_does_not_treat_a_truncated_start_as_proof() {
    let path = std::env::temp_dir().join(format!("vorax-opening-{}.log", std::process::id()));
    std::fs::write(&path, b"S13GamePlay_C_StartGameBtn\nS13GamePlayMain_C_RefreshBtn2\nProtoId = 5619\n").unwrap();
    assert!(OpeningOffer::from_recent_log(&path).is_initial());
    let mut truncated = b"S13GamePlay_C_StartGameBtn\n".to_vec();
    truncated.extend(vec![b' '; (8 << 20) + 1]);
    truncated.extend(b"\nS13GamePlayMain Run\n");
    std::fs::write(&path, truncated).unwrap();
    assert!(!OpeningOffer::from_recent_log(&path).is_initial());
    std::fs::remove_file(path).unwrap();
}
