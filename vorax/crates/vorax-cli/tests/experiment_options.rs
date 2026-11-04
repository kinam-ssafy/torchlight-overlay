use std::process::Command;

#[test]
fn malformed_research_settings_cannot_silently_run_the_default_experiment() {
    for args in [
        vec!["info","--target-score","not-a-number"],
        vec!["info","--scenario","unknown"],
        vec!["info","--scenario","5"],
        vec!["info","--goal"],
        vec!["train","--goal","mean"],
        vec!["gate","--target-score","1120000"],
        vec!["sim","--against","unused.txt","--target-score","1120000"],
    ] {
        let result=Command::new(env!("CARGO_BIN_EXE_vorax")).args(&args).output().unwrap();
        assert!(!result.status.success(),"accepted invalid configuration: {args:?}");
        assert!(!result.stderr.is_empty(),"missing explanation: {args:?}");
    }
    let valid=Command::new(env!("CARGO_BIN_EXE_vorax"))
        .args(["info","--target-score","1120000","--scenario","4","--goal","success"])
        .output().unwrap();
    assert!(valid.status.success(),"{}",String::from_utf8_lossy(&valid.stderr));
}
