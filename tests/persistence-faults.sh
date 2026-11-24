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

cat > "$tmp/connected.toml" <<EOF
data_dir = "$tmp/connected"
[backend]
kind = "smolvm"
binary = "$repo/tests/fake-smolvm.sh"
[worker]
control_plane_endpoint = "http://127.0.0.1:1"
api_key = "test"
pool_id = "test"
EOF
if "$repo/$binary" --config "$tmp/connected.toml" >"$tmp/connected.log" 2>&1; then
  echo "connected worker started without explicit resource limits" >&2
  exit 1
fi
grep -Fq 'connected workers require explicit [resources]' "$tmp/connected.log"

data="$tmp/data"
cat > "$tmp/blast.toml" <<EOF
port = 17241
data_dir = "$data"
[backend]
kind = "smolvm"
binary = "$repo/tests/fake-smolvm.sh"
[resources]
vcpu = 2
memory_mib = 512
disk_mib = 1024
admission_queue_secs = 1
[lifecycle]
pause_ttl_secs = 3600
suspend_ttl_secs = 3600
memory_pressure_thresh = 0.0
vcpu_pressure_thresh = 0.0
EOF
mkdir -p "$data"
chmod +x "$repo/tests/fake-smolvm.sh"
XDG_DATA_HOME="$data" "$repo/$binary" --config "$tmp/blast.toml" >"$tmp/worker.log" 2>&1 &
pid=$!
for _ in $(seq 1 100); do
  curl -sf http://127.0.0.1:17241/v1/live >/dev/null && break
  sleep .05
done
curl -sf http://127.0.0.1:17241/v1/live >/dev/null

vm=$(curl -sf -X POST http://127.0.0.1:17241/v1/fork \
  -H 'content-type: application/json' \
  -d '{"image":"test","resources":{"vcpu":1,"memory_mib":128,"disk_mib":100}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["vm_id"])')

curl -sf -X POST "http://127.0.0.1:17241/v1/vms/$vm/sync" \
  -H 'content-type: application/json' \
  -d '{"op":"write","writes":[{"path":"/tmp/too-large","upload_id":"large","size":1073741825,"content":""}]}' \
  > "$tmp/sync-quota"
python3 - "$tmp/sync-quota" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))["results"][0]
assert result["error"]["code"] == "resource_exhausted", result
PY

curl -sf -X POST "http://127.0.0.1:17241/v1/vms/$vm/sync" \
  -H 'content-type: application/json' \
  -d '{"op":"write","writes":[{"path":"/tmp/changing","upload_id":"changing","size":100,"content":""}]}' \
  >/dev/null
curl -sf -X POST "http://127.0.0.1:17241/v1/vms/$vm/sync" \
  -H 'content-type: application/json' \
  -d '{"op":"write","writes":[{"path":"/tmp/changing","upload_id":"changing","size":99,"content":""}]}' \
  > "$tmp/sync-immutable"
python3 - "$tmp/sync-immutable" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))["results"][0]
assert result["error"]["code"] == "bad_request", result
PY

machines_before=$(find "$data/machines" -type f | wc -l)
rm -rf "$data/owned-vms"
printf 'fault\n' > "$data/owned-vms"
status=$(curl -s -o "$tmp/snapshot-response" -w '%{http_code}' -X POST \
  http://127.0.0.1:17241/v1/fork -H 'content-type: application/json' \
  -d '{"image":"test","resources":{"vcpu":1,"memory_mib":128,"disk_mib":100}}')
[[ $status == 500 ]]
[[ $(find "$data/machines" -type f | wc -l) == "$machines_before" ]]
curl -sf http://127.0.0.1:17241/v1/live >/dev/null

rm -f "$data/runs.json"
mkdir "$data/runs.json"
status=$(curl -s -o "$tmp/run-response" -w '%{http_code}' -X POST \
  "http://127.0.0.1:17241/v1/vms/$vm/runs" \
  -H 'content-type: application/json' -d '{"command":"true","timeout":1}')
[[ $status == 503 ]]
curl -sf http://127.0.0.1:17241/v1/live >/dev/null

rm -rf "$data/runs.json"
printf '{}\n' > "$data/runs.json"
for _ in $(seq 1 100); do
  [[ -f $data/pending_records.jsonl ]] && break
  sleep .02
done
rm -f "$data/pending_records.jsonl"
mkdir "$data/pending_records.jsonl"
status=$(curl -s -o "$tmp/outbox-response" -w '%{http_code}' -X POST \
  "http://127.0.0.1:17241/v1/vms/$vm/runs" \
  -H 'content-type: application/json' -d '{"command":"true","timeout":1}')
[[ $status == 503 ]]
curl -sf http://127.0.0.1:17241/v1/live >/dev/null

rm -rf "$data/pending_records.jsonl"
run_id=$(curl -sf -X POST "http://127.0.0.1:17241/v1/vms/$vm/runs" \
  -H 'content-type: application/json' \
  -d '{"command":"true","background":true,"timeout":10}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
rm -f "$data/pending_records.jsonl"
mkdir "$data/pending_records.jsonl"
sleep 4
status=$(curl -s -o "$tmp/delete-during-terminal-retry" -w '%{http_code}' -X DELETE \
  "http://127.0.0.1:17241/v1/vms/$vm")
[[ $status == 409 ]]
rm -rf "$data/pending_records.jsonl"
for _ in $(seq 1 100); do
  state=$(curl -sf "http://127.0.0.1:17241/v1/vms/$vm/runs/$run_id" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  [[ $state != running ]] && break
  sleep .05
done
[[ $state == completed ]]

kill "$pid"
wait "$pid"
pid=
if grep -qE "panicked at|thread .* panicked" "$tmp/worker.log"; then
  cat "$tmp/worker.log" >&2
  exit 1
fi

echo "process resource and persistence fault tests passed"
