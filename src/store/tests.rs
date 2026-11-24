use super::runs::{prune_terminal_runs, MAX_LOCAL_RUNS_PER_VM, MAX_LOCAL_RUNS_TOTAL};
use super::*;

#[test]
fn local_run_history_is_bounded_without_pruning_active_runs() {
    let mut records = HashMap::new();
    for index in 0..=MAX_LOCAL_RUNS_PER_VM {
        let mut run = RunRecord::started("vm".into(), None, "true".into());
        run.state = RunState::Completed;
        run.completed_at =
            Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(index as u64));
        records.insert(format!("run_{index}"), run);
    }
    prune_terminal_runs(&mut records, "vm").unwrap();
    assert_eq!(records.len(), MAX_LOCAL_RUNS_PER_VM);
    assert!(!records.contains_key("run_0"));

    for run in records.values_mut() {
        run.state = RunState::Running;
    }
    records.insert(
        "one_too_many".into(),
        RunRecord::started("vm".into(), None, "sleep 10".into()),
    );
    assert_eq!(
        prune_terminal_runs(&mut records, "vm").unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn global_history_evicts_only_terminal_cache_entries() {
    let mut records = HashMap::new();
    for index in 0..MAX_LOCAL_RUNS_TOTAL {
        let mut run = RunRecord::started(format!("vm_{index}"), None, "true".into());
        run.state = RunState::Completed;
        run.completed_at =
            Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(index as u64));
        records.insert(format!("run_{index}"), run);
    }
    records.insert(
        "new".into(),
        RunRecord::started("new_vm".into(), None, "sleep 1".into()),
    );

    prune_terminal_runs(&mut records, "new_vm").unwrap();

    assert_eq!(records.len(), MAX_LOCAL_RUNS_TOTAL);
    assert!(records.contains_key("new"));
    assert!(!records.contains_key("run_0"));
}

#[tokio::test]
async fn sessions_keep_their_ids_and_become_idle_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::durable(dir.path(), 0).unwrap();
    store
        .insert(VmRecord::new(
            "vm_a".into(),
            VmState::Running,
            Resources {
                vcpu: 1,
                memory_mib: 1,
                disk_mib: 1,
            },
        ))
        .await;
    let opened = store
        .open_session("vm_a", None, Some("/work".into()), HashMap::new())
        .await
        .unwrap()
        .unwrap();

    let restarted = Store::durable(dir.path(), 0).unwrap();
    restarted
        .insert(VmRecord::new(
            "vm_a".into(),
            VmState::Running,
            Resources {
                vcpu: 1,
                memory_mib: 1,
                disk_mib: 1,
            },
        ))
        .await;
    let vm = restarted.get("vm_a").await.unwrap();
    let session = vm.sessions.get(&opened.session_id).unwrap();
    assert_eq!(session.session_idx, 0);
    assert_eq!(session.cwd.as_deref(), Some("/work"));
}

#[tokio::test]
async fn an_interrupted_run_is_reconciled_as_failed() {
    let dir = tempfile::tempdir().unwrap();
    let runs = RunStore::durable(dir.path()).unwrap();
    runs.insert(
        "run_a".into(),
        RunRecord::started("vm_a".into(), Some("sess_a".into()), "sleep 10".into()),
    )
    .await
    .unwrap();

    let restarted = RunStore::durable(dir.path()).unwrap();
    let run = restarted.get("run_a").await.unwrap();
    assert_eq!(run.state, RunState::Failed);
    assert!(run.completed_at.is_some());
    assert!(run.fail_reason.unwrap().contains("restarted"));
}

#[tokio::test]
async fn portable_state_moves_sessions_and_completed_runs_between_workers() {
    let source_dir = tempfile::tempdir().unwrap();
    let source = Store::durable(source_dir.path(), 0).unwrap();
    let source_runs = RunStore::durable(source_dir.path()).unwrap();
    source
        .insert(VmRecord::new(
            "vm_a".into(),
            VmState::Suspended,
            Resources {
                vcpu: 1,
                memory_mib: 1,
                disk_mib: 1,
            },
        ))
        .await;
    let opened = source
        .open_session(
            "vm_a",
            None,
            Some("/workspace".into()),
            HashMap::from([("COLOR".into(), "blue".into())]),
        )
        .await
        .unwrap()
        .unwrap();
    let mut run = RunRecord::started("vm_a".into(), Some(opened.session_id.clone()), "pwd".into());
    run.state = RunState::Completed;
    run.completed_at = Some(SystemTime::now());
    run.stdout = Some("/workspace\n".into());
    run.exit_code = Some(0);
    source_runs.insert("run_a".into(), run).await.unwrap();

    let state = source_dir.path().join("portable.json");
    write_portable_vm_state(&state, "vm_a", &source, &source_runs)
        .await
        .unwrap();

    let target_dir = tempfile::tempdir().unwrap();
    let target = Store::durable(target_dir.path(), 0).unwrap();
    let target_runs = RunStore::durable(target_dir.path()).unwrap();
    target
        .insert(VmRecord::new(
            "vm_a".into(),
            VmState::Suspended,
            Resources {
                vcpu: 1,
                memory_mib: 1,
                disk_mib: 1,
            },
        ))
        .await;
    restore_portable_vm_state(&state, "vm_a", &target, &target_runs)
        .await
        .unwrap();

    let vm = target.get("vm_a").await.unwrap();
    let session = vm.sessions.get(&opened.session_id).unwrap();
    assert_eq!(session.cwd.as_deref(), Some("/workspace"));
    assert_eq!(session.env.get("COLOR").map(String::as_str), Some("blue"));
    let run = target_runs.get("run_a").await.unwrap();
    assert_eq!(run.state, RunState::Completed);
    assert_eq!(run.stdout.as_deref(), Some("/workspace\n"));
}

#[tokio::test]
async fn admission_charges_each_live_vm_its_backend_memory_overhead() {
    let store = Store {
        vm_memory_overhead_mib: 768,
        ..Store::default()
    };
    let total = Resources {
        vcpu: 4,
        memory_mib: 4096,
        disk_mib: 4096,
    };
    let shape = Resources {
        vcpu: 1,
        memory_mib: 2048,
        disk_mib: 1,
    };
    let first = VmRecord::new("vm_a".into(), VmState::Running, shape.clone());
    let second = VmRecord::new("vm_b".into(), VmState::Running, shape);

    store.try_reserve(&first, &total).await.unwrap();
    assert_eq!(store.resources_in_use().await.memory_mib, 2048 + 768);
    assert!(store.try_reserve(&second, &total).await.is_err());
}
