use super::*;

fn manifest(vm_id: &str) -> VmManifest {
    VmManifest {
        vm_id: vm_id.into(),
        name: Some("n".into()),
        description: None,
        image: Some("alpine:3.20".into()),
        parent_vm_id: None,
        workdir: None,
        created_at: 1_700_000_000,
        resources: Resources {
            vcpu: 1,
            memory_mib: 512,
            disk_mib: 1024,
        },
        backend_handle: None,
    }
}

#[tokio::test]
async fn dir_storage_round_trips_a_snapshot() {
    let data = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let store = SnapshotStore::with_shared_storage(
        data.path(),
        Storage::Dir(storage.path().to_owned()),
        true,
    );
    let dir = store.snap_dir("vm_a");
    fs::create_dir_all(dir.join("sub")).await.unwrap();
    fs::write(dir.join("memory.bin"), b"memory").await.unwrap();
    fs::write(dir.join("sub/disk.img"), b"disk").await.unwrap();
    store.write_backend_tag("vm_a", "docker").await.unwrap();
    store.write_manifest(&manifest("vm_a")).await.unwrap();
    store.persist("vm_a").await.unwrap();

    // A second persist replaces the stored copy, dropping files that went away.
    fs::remove_file(dir.join("sub/disk.img")).await.unwrap();
    store.persist("vm_a").await.unwrap();
    assert!(!storage.path().join("vms/vm_a/sub/disk.img").exists());

    fs::remove_dir_all(&dir).await.unwrap();
    assert_eq!(
        store.find_manifest("vm_a").await.unwrap(),
        Some(manifest("vm_a"))
    );
    assert_eq!(store.ensure_local("vm_a").await.unwrap(), dir);
    assert_eq!(fs::read(dir.join("memory.bin")).await.unwrap(), b"memory");
    store.check_backend_tag("vm_a", "docker").await.unwrap();
    assert!(store.check_backend_tag("vm_a", "smolvm").await.is_err());

    store.forget("vm_a").await.unwrap();
    assert!(!dir.exists());
    store.remove_stored("vm_a").await;
    assert!(storage.path().join("vms/vm_a").exists());
    assert!(store.ensure_local("vm_a").await.is_err());
}

#[tokio::test]
async fn deletion_respects_directory_storage_ownership() {
    for (shared, durable_remains) in [(false, false), (true, true)] {
        let data = tempfile::tempdir().unwrap();
        let storage = tempfile::tempdir().unwrap();
        let store = SnapshotStore::with_shared_storage(
            data.path(),
            Storage::Dir(storage.path().to_owned()),
            shared,
        );
        let vm_id = if shared { "vm_shared" } else { "vm_local" };
        fs::create_dir_all(store.snap_dir(vm_id)).await.unwrap();
        fs::write(store.snap_dir(vm_id).join("disk"), b"state")
            .await
            .unwrap();
        store.write_backend_tag(vm_id, "docker").await.unwrap();
        store.write_manifest(&manifest(vm_id)).await.unwrap();
        store.persist(vm_id).await.unwrap();

        store.forget(vm_id).await.unwrap();
        store.remove_stored(vm_id).await;

        assert_eq!(
            storage.path().join("vms").join(vm_id).exists(),
            durable_remains
        );
    }
}

#[tokio::test]
async fn pending_generation_records_survive_restart_and_can_be_abandoned() {
    let data = tempfile::tempdir().unwrap();
    let pending = PendingGeneration {
        vm_generation: 7,
        snapshot_id: "01ARZ3NDEKTSV4RRFFQ69G5FAX".into(),
        uploaded: true,
        kind: PendingKind::DirtyCheckpoint,
    };
    SnapshotStore::new(data.path(), Storage::None)
        .set_pending_generation("vm_pending", &pending)
        .await
        .unwrap();
    let restarted = SnapshotStore::new(data.path(), Storage::None);
    let restored = restarted
        .pending_generation("vm_pending")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(restored.kind, PendingKind::DirtyCheckpoint);
    assert!(restored.uploaded);
    restarted.abandon_generation("vm_pending").await.unwrap();
    assert!(restarted
        .pending_generation("vm_pending")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn prepare_claim_requires_a_tagged_complete_shared_snapshot() {
    let source = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let writer = SnapshotStore::new(source.path(), Storage::Dir(storage.path().to_owned()));
    fs::create_dir_all(writer.snap_dir("vm_claim"))
        .await
        .unwrap();
    fs::write(writer.snap_dir("vm_claim").join("disk"), b"state")
        .await
        .unwrap();
    writer.write_manifest(&manifest("vm_claim")).await.unwrap();
    assert!(writer.persist("vm_claim").await.is_err());

    let reader = SnapshotStore::new(target.path(), Storage::Dir(storage.path().to_owned()));
    assert!(reader
        .prepare_claim("vm_claim", 1, "01ARZ3NDEKTSV4RRFFQ69G5FAV", "docker")
        .await
        .is_err());

    writer
        .write_backend_tag("vm_claim", "docker")
        .await
        .unwrap();
    let generation = "01ARZ3NDEKTSV4RRFFQ69G5FAW";
    writer
        .persist_generation("vm_claim", generation)
        .await
        .unwrap();
    reader
        .prepare_claim("vm_claim", 1, generation, "docker")
        .await
        .unwrap();
    assert!(reader
        .prepare_claim("vm_claim", 1, generation, "smolvm")
        .await
        .is_err());
}

#[tokio::test]
async fn abandoning_a_claim_removes_only_local_state() {
    let source = tempfile::tempdir().unwrap();
    let target = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let writer = SnapshotStore::with_shared_storage(
        source.path(),
        Storage::Dir(storage.path().to_owned()),
        true,
    );
    fs::create_dir_all(writer.snap_dir("vm_claim"))
        .await
        .unwrap();
    fs::write(writer.snap_dir("vm_claim").join("disk"), b"state")
        .await
        .unwrap();
    writer
        .write_backend_tag("vm_claim", "docker")
        .await
        .unwrap();
    writer.write_manifest(&manifest("vm_claim")).await.unwrap();
    let generation = "01ARZ3NDEKTSV4RRFFQ69G5FAY";
    writer
        .persist_generation("vm_claim", generation)
        .await
        .unwrap();

    let reader = SnapshotStore::with_shared_storage(
        target.path(),
        Storage::Dir(storage.path().to_owned()),
        true,
    );
    reader
        .prepare_claim("vm_claim", 9, generation, "docker")
        .await
        .unwrap();
    reader.claim("vm_claim", 9).await.unwrap();
    assert!(!reader.abandon_claim("vm_claim", 8).await.unwrap());
    assert!(reader.snap_dir("vm_claim").exists());
    assert!(reader.abandon_claim("vm_claim", 9).await.unwrap());
    assert!(!reader.snap_dir("vm_claim").exists());
    assert!(storage.path().join("vms/vm_claim").exists());
    assert!(!reader.abandon_claim("vm_claim", 9).await.unwrap());
}

#[tokio::test]
async fn restore_ignores_uncommitted_generation_and_falls_back_from_corruption() {
    let data = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let store = SnapshotStore::new(data.path(), Storage::Dir(storage.path().to_owned()));
    let local = store.snap_dir("vm_generation");
    fs::create_dir_all(&local).await.unwrap();
    fs::write(local.join("disk"), b"committed").await.unwrap();
    store
        .write_backend_tag("vm_generation", "docker")
        .await
        .unwrap();
    store
        .write_manifest(&manifest("vm_generation"))
        .await
        .unwrap();
    store.persist("vm_generation").await.unwrap();

    let root = storage.path().join("vms/vm_generation");
    let first = store
        .selected_generation("vm_generation")
        .await
        .unwrap()
        .unwrap();
    let orphan = root.join("generations/orphan");
    fs::create_dir_all(&orphan).await.unwrap();
    fs::write(orphan.join("disk"), b"partial-new-state")
        .await
        .unwrap();
    fs::remove_dir_all(&local).await.unwrap();
    store.ensure_local("vm_generation").await.unwrap();
    assert_eq!(fs::read(local.join("disk")).await.unwrap(), b"committed");

    fs::write(local.join("disk"), b"newer").await.unwrap();
    store.persist("vm_generation").await.unwrap();
    let current = store
        .selected_generation("vm_generation")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(current, first);
    fs::write(
        root.join("generations").join(&current).join("disk"),
        b"corrupt",
    )
    .await
    .unwrap();
    let _ = fs::remove_dir_all(&local).await;
    assert!(store.ensure_local("vm_generation").await.is_err());

    fs::remove_file(root.join("generations").join(&first).join("disk"))
        .await
        .unwrap();
    let _ = fs::remove_dir_all(&local).await;
    assert!(store.ensure_local("vm_generation").await.is_err());
}

#[cfg(unix)]
#[tokio::test]
async fn persist_rejects_symlink_snapshot_entries() {
    use std::os::unix::fs::symlink;

    let data = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    let store = SnapshotStore::new(data.path(), Storage::Dir(storage.path().to_owned()));
    let local = store.snap_dir("vm_symlink");
    fs::create_dir_all(&local).await.unwrap();
    store
        .write_backend_tag("vm_symlink", "docker")
        .await
        .unwrap();
    store.write_manifest(&manifest("vm_symlink")).await.unwrap();
    symlink(outside.path(), local.join("disk-link")).unwrap();

    let error = store.persist("vm_symlink").await.unwrap_err();
    assert!(format!("{error:#}").contains("non-regular snapshot entry"));
}

#[tokio::test]
async fn a_stored_snapshot_without_its_manifest_is_incomplete() {
    let data = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let partial = storage.path().join("vms/vm_b");
    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join("memory.bin"), b"half").unwrap();
    let store = SnapshotStore::new(data.path(), Storage::Dir(storage.path().to_owned()));
    assert_eq!(store.find_manifest("vm_b").await.unwrap(), None);
    assert!(store.ensure_local("vm_b").await.is_err());
    assert!(!store.snap_dir("vm_b").exists());
}

#[tokio::test]
async fn object_storage_waits_for_the_heartbeat() {
    let data = tempfile::tempdir().unwrap();
    let store = SnapshotStore::new(data.path(), Storage::object(None));
    fs::create_dir_all(store.snap_dir("vm_c")).await.unwrap();
    store.write_backend_tag("vm_c", "docker").await.unwrap();
    store.write_manifest(&manifest("vm_c")).await.unwrap();
    assert!(!store.can_store().await);
    let err = store.persist("vm_c").await.unwrap_err();
    assert!(
        format!("{err:#}").contains("has not answered a heartbeat"),
        "{err:#}"
    );
    // The local manifest is found; whether storage holds another VM is
    // unknown until the control plane answers.
    assert_eq!(
        store.find_manifest("vm_c").await.unwrap(),
        Some(manifest("vm_c"))
    );
    assert!(store.find_manifest("vm_other").await.is_err());
}

#[tokio::test]
async fn unsafe_ids_never_touch_the_disk() {
    let data = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let store = SnapshotStore::new(data.path(), Storage::Dir(storage.path().to_owned()));
    assert!(store.ensure_local("..").await.is_err());
    assert!(store.persist("..").await.is_err());
    assert!(store.write_manifest(&manifest("../x")).await.is_err());
    assert!(store.forget("..").await.is_err());
    assert!(data.path().exists());
}

#[test]
fn unsafe_ids_and_paths_are_refused() {
    assert!(is_safe_vm_id("vm_01J0ABC-x"));
    for bad in ["", "../x", "a/b", "a.b", "vm id"] {
        assert!(!is_safe_vm_id(bad), "{bad}");
    }
    assert!(is_safe_relative("a/b.bin"));
    for bad in ["", "/etc/passwd", "../x", "a/../../x", "a//b", "./a"] {
        assert!(!is_safe_relative(bad), "{bad}");
    }
}

#[tokio::test]
async fn ensure_local_reuses_the_selected_generation_it_restored() {
    let data = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    let store = SnapshotStore::with_shared_storage(
        data.path(),
        Storage::Dir(storage.path().to_owned()),
        true,
    );
    let dir = store.snap_dir("vm_a");
    fs::create_dir_all(&dir).await.unwrap();
    fs::write(dir.join("memory.bin"), b"memory").await.unwrap();
    store.write_backend_tag("vm_a", "docker").await.unwrap();
    store.write_manifest(&manifest("vm_a")).await.unwrap();
    store.persist("vm_a").await.unwrap();
    store.release_local_snapshot("vm_a").await.unwrap();

    assert_eq!(store.ensure_local("vm_a").await.unwrap(), dir);
    // The restored copy is used as is: storage is not read again.
    fs::remove_dir_all(storage.path().join("vms/vm_a/generations"))
        .await
        .unwrap();
    assert_eq!(store.ensure_local("vm_a").await.unwrap(), dir);
    assert_eq!(fs::read(dir.join("memory.bin")).await.unwrap(), b"memory");

    // A rewritten local snapshot is no longer taken for that generation.
    store.invalidate_local_generation("vm_a").await.unwrap();
    assert!(store.ensure_local("vm_a").await.is_err());
}

#[tokio::test]
async fn a_pinned_snapshot_outlives_its_replacement_and_removal() {
    let data = tempfile::tempdir().unwrap();
    let store = SnapshotStore::new(data.path(), Storage::None);
    let dir = store.snap_dir("vm_a");
    fs::create_dir_all(dir.join("sub")).await.unwrap();
    fs::write(dir.join("sub/memory.bin"), b"old").await.unwrap();

    let pinned = store.pin_local("vm_a").await.unwrap();
    fs::write(dir.join("next"), b"new").await.unwrap();
    fs::rename(dir.join("next"), dir.join("sub/memory.bin"))
        .await
        .unwrap();
    store.forget("vm_a").await.unwrap();

    assert_eq!(
        fs::read(pinned.path().join("sub/memory.bin"))
            .await
            .unwrap(),
        b"old"
    );
    let path = pinned.path().to_owned();
    drop(pinned);
    assert!(!path.exists());
}
