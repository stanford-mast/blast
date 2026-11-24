#!/bin/bash
set -euo pipefail

binary=${1:-target/debug/blast}
repo=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
pid=
cleanup() {
  if [[ -n ${pid:-} ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

chmod +x "$repo/tests/fake-smolvm.sh"

start_worker() {
  local memory=$1 disk=$2
  local data="$tmp/data-$memory-$disk"
  local storage="$tmp/storage-$memory-$disk"
  local config="$tmp/blast-$memory-$disk.toml"
  mkdir -p "$data/transient-snapshots/stale" "$storage"
  printf 'incomplete' > "$data/transient-snapshots/stale/checkpoint"
  cat > "$config" <<EOF
port = 17240
data_dir = "$data"
[backend]
kind = "smolvm"
binary = "$repo/tests/fake-smolvm.sh"
[worker]
storage_dir = "$storage"
storage_dir_shared = true
[resources]
vcpu = 4
memory_mib = $memory
disk_mib = $disk
admission_queue_secs = 10
[lifecycle]
pause_ttl_secs = 3600
suspend_ttl_secs = 3600
memory_pressure_thresh = 0.0
vcpu_pressure_thresh = 0.0
EOF
  "$repo/$binary" --config "$config" > "$tmp/worker.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 100); do
    curl -sf http://127.0.0.1:17240/v1/live >/dev/null && break
    sleep .05
  done
  curl -sf http://127.0.0.1:17240/v1/live >/dev/null
  [[ ! -e "$data/transient-snapshots" ]]
  DATA_DIR=$data
  STORAGE_DIR=$storage
}

stop_worker() {
  kill "$pid"
  wait "$pid"
  pid=
}

start_worker_without_storage() {
  local data="$tmp/data-no-storage"
  local config="$tmp/blast-no-storage.toml"
  mkdir -p "$data"
  cat > "$config" <<EOF
port = 17240
data_dir = "$data"
[backend]
kind = "smolvm"
binary = "$repo/tests/fake-smolvm.sh"
[resources]
vcpu = 4
memory_mib = 256
disk_mib = 1000
admission_queue_secs = 10
[lifecycle]
pause_ttl_secs = 3600
suspend_ttl_secs = 3600
memory_pressure_thresh = 0.0
vcpu_pressure_thresh = 0.0
EOF
  "$repo/$binary" --config "$config" > "$tmp/no-storage-worker.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 100); do
    curl -sf http://127.0.0.1:17240/v1/live >/dev/null && break
    sleep .05
  done
  curl -sf http://127.0.0.1:17240/v1/live >/dev/null
  DATA_DIR=$data
}

fork_vm() {
  local memory=$1 disk=$2
  curl -s --max-time 15 -X POST http://127.0.0.1:17240/v1/fork \
    -H 'content-type: application/json' \
    -d "{\"image\":\"test\",\"resources\":{\"vcpu\":1,\"memory_mib\":$memory,\"disk_mib\":$disk}}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "vm_id" in d, d; print(d["vm_id"])'
}

vm_state() {
  curl -sf "http://127.0.0.1:17240/v1/vms/$1" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])'
}

start_worker_without_storage
local_only=$(fork_vm 192 100)
fork_vm 128 100 >/dev/null
[[ $(vm_state "$local_only") == idle ]]
[[ -d "$DATA_DIR/snapshots/$local_only" ]]
curl -sf --max-time 10 -X POST "http://127.0.0.1:17240/v1/vms/$local_only/runs" \
  -H 'content-type: application/json' -d '{"command":"true","timeout":8}' >/dev/null
[[ $(vm_state "$local_only") == running ]]
stop_worker

start_worker 256 1000
first=$(fork_vm 192 100)
fork_vm 128 100 >/dev/null
[[ $(vm_state "$first") == idle ]]
[[ $(find "$DATA_DIR/machines" -type f | wc -l) == 1 ]]
stop_worker

start_worker 1024 150
first=$(fork_vm 128 100)
fork_vm 128 100 >/dev/null
[[ $(vm_state "$first") == idle ]]
[[ $(find "$DATA_DIR/machines" -type f | wc -l) == 1 ]]
stop_worker

start_worker 320 1000
busy=$(fork_vm 128 100)
idle=$(fork_vm 128 100)
curl -sf --max-time 10 -X POST "http://127.0.0.1:17240/v1/vms/$busy/runs" \
  -H 'content-type: application/json' -d '{"command":"sleep","timeout":8}' >/dev/null &
run_pid=$!
sleep .2
status=$(curl -s -o "$tmp/busy-delete" -w '%{http_code}' --max-time 10 -X DELETE \
  "http://127.0.0.1:17240/v1/vms/$busy")
[[ $status == 409 ]]
python3 - "$tmp/busy-delete" <<'PY'
import json
import sys
error = json.load(open(sys.argv[1]))["error"]
assert error["code"] == "conflict", error
PY
fork_vm 128 100 >/dev/null
wait "$run_pid"
[[ $(vm_state "$busy") == running ]]
[[ $(vm_state "$idle") == idle ]]

for _ in $(seq 1 100); do
  commit=$(find "$STORAGE_DIR/vms/$idle/.blast-commits" -type f -print -quit 2>/dev/null || true)
  [[ -n $commit ]] && break
  sleep .05
done
[[ -n ${commit:-} ]]
curl -sf --max-time 10 -X DELETE "http://127.0.0.1:17240/v1/vms/$idle" >/dev/null
[[ -f $commit ]]

child=$(curl -sf --max-time 15 -X POST http://127.0.0.1:17240/v1/fork \
  -H 'content-type: application/json' \
  -d "{\"source_vm_id\":\"$busy\"}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["vm_id"])')
[[ -n $child ]]
[[ ! -e "$DATA_DIR/transient-snapshots" || -z $(find "$DATA_DIR/transient-snapshots" -mindepth 1 -print -quit) ]]
run_id=$(curl -sf --max-time 10 -X POST "http://127.0.0.1:17240/v1/vms/$child/runs" \
  -H 'content-type: application/json' -d '{"command":"true","timeout":8}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
curl -sf --max-time 10 -X DELETE "http://127.0.0.1:17240/v1/vms/$child" >/dev/null
status=$(curl -s -o /dev/null -w '%{http_code}' \
  "http://127.0.0.1:17240/v1/vms/$child/runs/$run_id")
[[ $status == 404 ]]

echo "process lifecycle pressure tests passed"
