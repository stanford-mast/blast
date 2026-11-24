use super::*;
use crate::{store::VmState, worker::heartbeat::WorkerResources};

#[test]
fn newer_state_replaces_the_same_durable_delivery_slot() {
    let mut deliveries = VecDeque::new();
    let vm = |state: &str| Delivery::PutVm {
        vm_id: "vm".into(),
        body: WorkerVmRecord {
            state: state.into(),
            resources: WorkerResources {
                vcpu: 1,
                memory_mib: 1,
                disk_mib: 1,
            },
            image: None,
            parent_vm_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            last_active_at: None,
        },
    };
    enqueue(&mut deliveries, vm("running"));
    enqueue(&mut deliveries, vm("suspended"));
    enqueue(
        &mut deliveries,
        Delivery::Activity(Activity::operation("sync", "completed", None)),
    );

    assert_eq!(deliveries.len(), 2);
    assert!(matches!(
        &deliveries[0],
        Delivery::PutVm { body, .. } if body.state == "suspended"
    ));
    assert!(matches!(&deliveries[1], Delivery::Activity(_)));
}

#[test]
fn connected_restart_discards_ownership_deliveries_only() {
    let mut pending = Pending {
        deliveries: VecDeque::from([
            Delivery::DeleteVm {
                vm_id: "old".into(),
            },
            Delivery::Activity(Activity::operation("run", "completed", None)),
            Delivery::PutRun {
                vm_id: "vm".into(),
                run_id: "run".into(),
                body: WorkerRunRecord {
                    state: "completed".into(),
                    command: "true".into(),
                    exit_code: Some(0),
                    started_at: "2026-01-01T00:00:00Z".into(),
                    finished_at: Some("2026-01-01T00:00:01Z".into()),
                    stdout: None,
                    stderr: None,
                    session_id: None,
                    stdout_encoding: None,
                    stderr_encoding: None,
                    stdout_truncated: false,
                    stderr_truncated: false,
                    fail_reason: None,
                },
            },
        ]),
        acknowledged: 0,
    };

    pending.discard_ownership();

    assert_eq!(pending.deliveries.len(), 2);
    assert!(matches!(pending.deliveries[0], Delivery::Activity(_)));
    assert!(matches!(pending.deliveries[1], Delivery::PutRun { .. }));
}

#[test]
fn abandoning_a_vm_discards_only_its_pending_ownership_record() {
    let data_dir = tempfile::tempdir().unwrap();
    let records = Records::new(
        data_dir.path(),
        ControlPlane::local(data_dir.path()).unwrap(),
    )
    .unwrap();
    let vm = |id: &str| {
        VmRecord::new(
            id.into(),
            VmState::Suspended,
            crate::backend::Resources {
                vcpu: 1,
                memory_mib: 1,
                disk_mib: 1,
            },
        )
    };
    records.vm(&vm("rejected")).unwrap();
    records.vm(&vm("owned")).unwrap();

    records.abandon_vm("rejected").unwrap();

    let pending = records.queue.pending.lock().unwrap();
    assert_eq!(pending.deliveries.len(), 1);
    assert!(matches!(
        &pending.deliveries[0],
        Delivery::PutVm { vm_id, .. } if vm_id == "owned"
    ));
    drop(pending);
}

#[tokio::test]
async fn delete_is_committed_before_it_returns_and_never_journaled() {
    let data_dir = std::env::temp_dir().join(format!("blast-records-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let control_plane = ControlPlane::local(&data_dir).unwrap();
    control_plane
        .put("vms/vm", &serde_json::json!({"state": "running"}))
        .await
        .unwrap();
    let records = Records::new(&data_dir, control_plane).unwrap();

    records.vm_deleted("vm").await.unwrap();

    assert!(!data_dir.join("control-plane/vms__vm.json").exists());
    assert!(records.queue.pending.lock().unwrap().deliveries.is_empty());
    std::fs::remove_dir_all(data_dir).unwrap();
}

#[tokio::test]
async fn failed_delete_leaves_the_control_plane_record_untouched() {
    let data_dir = std::env::temp_dir().join(format!("blast-records-{}", ulid::Ulid::new()));
    let record = data_dir.join("control-plane/vms__vm.json");
    std::fs::create_dir_all(&record).unwrap();
    let records = Records::new(&data_dir, ControlPlane::local(&data_dir).unwrap()).unwrap();

    assert!(records.vm_deleted("vm").await.is_err());

    assert!(record.is_dir());
    assert!(records.queue.pending.lock().unwrap().deliveries.is_empty());
    std::fs::remove_dir_all(data_dir).unwrap();
}
