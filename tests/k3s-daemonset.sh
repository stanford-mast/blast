#!/bin/bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
binary=${1:-$repo/target/release/blast}
namespace=blast-k3s-e2e
host_root=/var/lib/blast-k3s-e2e
kubeconfig=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
selector='app.kubernetes.io/name=blast'
base_image='docker.io/library/python@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90'

for command in k3s kubectl curl python3; do
  command -v "$command" >/dev/null || { echo "$command is required" >&2; exit 1; }
done
[[ -x $binary ]] || { echo "build BLAST first: cargo build --release" >&2; exit 1; }
[[ -c /dev/kvm ]] || { echo "/dev/kvm is required" >&2; exit 1; }

original_kvm=$(stat -c %a /dev/kvm)
cleanup() {
  kubectl --kubeconfig "$kubeconfig" delete namespace "$namespace" --wait=true --timeout=60s >/dev/null 2>&1 || true
  chmod "$original_kvm" /dev/kvm
  rm -rf "$host_root"
}
diagnostics() {
  kubectl --kubeconfig "$kubeconfig" -n "$namespace" get pods -o wide >&2 || true
  kubectl --kubeconfig "$kubeconfig" -n "$namespace" logs -l "$selector" --all-containers --tail=100 >&2 || true
}
trap cleanup EXIT
trap diagnostics ERR

rm -rf "$host_root"
kubectl --kubeconfig "$kubeconfig" delete namespace "$namespace" --wait=true --timeout=60s \
  >/dev/null 2>&1 || true
install -d -o 65532 -g 65532 -m 0700 "$host_root"
chmod a+rw /dev/kvm

image_name=docker.io/library/blast-k3s-e2e
image_tag="$image_name:$(sha256sum "$binary" "$repo/tests/fake-smolvm-k3s.sh" | sha256sum | cut -c1-16)"
k3s ctr images pull "$base_image" >/dev/null
k3s ctr images export --platform "linux/$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')" "$host_root/base.tar" "$base_image"
image_digest=$(python3 "$repo/tests/build-k3s-image.py" "$host_root/base.tar" "$host_root/worker.tar" \
  "$binary" "$repo/tests/fake-smolvm-k3s.sh" "$image_tag")
image_ref="$image_name@$image_digest"
k3s ctr images import "$host_root/worker.tar" >/dev/null
k3s ctr images tag "$image_tag" "$image_ref" >/dev/null

kubectl --kubeconfig "$kubeconfig" create namespace "$namespace"
kubectl --kubeconfig "$kubeconfig" -n "$namespace" create configmap blast-tools \
  --from-file=control-plane.py="$repo/tests/fake-control-plane.py"
kubectl --kubeconfig "$kubeconfig" -n "$namespace" create secret generic blast-pool \
  --from-literal=join-key=e2e --from-literal=local-api-token=e2e-api

kubectl --kubeconfig "$kubeconfig" -n "$namespace" apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: blast-control-plane
  labels: {app: blast-control-plane}
spec:
  containers:
    - name: server
      image: docker.io/library/python@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
      command: [python3, /tools/control-plane.py]
      volumeMounts: [{name: tools, mountPath: /tools}]
  volumes: [{name: tools, configMap: {name: blast-tools}}]
---
apiVersion: v1
kind: Service
metadata: {name: blast-control-plane}
spec:
  selector: {app: blast-control-plane}
  ports: [{port: 8080, targetPort: 8080}]
EOF

kubectl kustomize --load-restrictor=LoadRestrictionsNone "$repo/tests/kubernetes" \
  | sed "s|blast-k3s-e2e:IMAGE|$image_ref|" \
  | kubectl --kubeconfig "$kubeconfig" apply -f -

kubectl --kubeconfig "$kubeconfig" -n "$namespace" wait --for=condition=Ready pod/blast-control-plane --timeout=120s
kubectl --kubeconfig "$kubeconfig" -n "$namespace" rollout status daemonset/blast --timeout=120s
pod=$(kubectl --kubeconfig "$kubeconfig" -n "$namespace" get pod -l "$selector" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl --kubeconfig "$kubeconfig" -n "$namespace" get pod "$pod" -o json | python3 -c '
import json, sys
pod = json.load(sys.stdin)
container = pod["spec"]["containers"][0]
security = pod["spec"]["securityContext"]
assert container["image"] == sys.argv[1]
assert container["imagePullPolicy"] == "Never"
assert security["runAsNonRoot"] and security["runAsUser"] == 65532 and security["runAsGroup"] == 65532
assert pod["spec"]["automountServiceAccountToken"] is False
assert pod["spec"]["serviceAccountName"] == "blast"
assert not any(volume.get("projected") for volume in pod["spec"]["volumes"])
assert container["resources"]["requests"] and container["resources"]["limits"]
machine_id = next(mount for mount in container["volumeMounts"] if mount["name"] == "machine-id")
assert machine_id["readOnly"] is True
' "$image_ref"
worker_id=$(cat "$host_root/worker_id")
incarnation=$(cat "$host_root/worker_incarnation")
kubectl --kubeconfig "$kubeconfig" -n "$namespace" delete pod "$pod" --wait=true
kubectl --kubeconfig "$kubeconfig" -n "$namespace" rollout status daemonset/blast --timeout=120s
connected_pod=$(kubectl --kubeconfig "$kubeconfig" -n "$namespace" get pod -l "$selector" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
[[ $connected_pod != "$pod" ]]
[[ $(cat "$host_root/worker_id") == "$worker_id" ]]
connected_incarnation=$(cat "$host_root/worker_incarnation")
[[ $connected_incarnation != "$incarnation" ]]

# The connected worker intentionally exposes only loopback diagnostics; API
# traffic arrives over its outbound tunnel. Switch the same DaemonSet to the
# embedded control plane to exercise its full API through a port-forward.
kubectl --kubeconfig "$kubeconfig" -n "$namespace" create configmap blast \
  --from-literal=blast.toml="$(cat <<'EOF'
port = 7240
data_dir = "/var/lib/blast"
[backend]
kind = "smolvm"
binary = "/usr/local/bin/fake-smolvm"
[resources]
vcpu = 2
memory_mib = 512
disk_mib = 2048
admission_queue_secs = 5
[lifecycle]
pause_ttl_secs = 1
suspend_ttl_secs = 2
vcpu_pressure_thresh = 0.0
memory_pressure_thresh = 0.0
EOF
)" --dry-run=client -o yaml | kubectl --kubeconfig "$kubeconfig" apply -f -
kubectl --kubeconfig "$kubeconfig" -n "$namespace" patch daemonset blast --type=json \
  -p="[{\"op\":\"remove\",\"path\":\"/spec/template/spec/containers/0/readinessProbe\"},{\"op\":\"add\",\"path\":\"/spec/template/metadata/annotations\",\"value\":{\"e2e/restarted\":\"$(date +%s%N)\"}}]"
kubectl --kubeconfig "$kubeconfig" -n "$namespace" rollout status daemonset/blast --timeout=120s
pod=$(kubectl --kubeconfig "$kubeconfig" -n "$namespace" get pod -l "$selector" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl --kubeconfig "$kubeconfig" -n "$namespace" port-forward "pod/$pod" 17240:7240 >"$host_root/port-forward.log" 2>&1 &
forward=$!
trap 'kill "$forward" 2>/dev/null || true; cleanup' EXIT
for _ in $(seq 1 100); do curl -sf http://127.0.0.1:17240/v1/live >/dev/null && break; sleep .1; done
curl -sf http://127.0.0.1:17240/v1/live >/dev/null

vm=$(curl -sf -X POST http://127.0.0.1:17240/v1/fork -H 'authorization: Bearer e2e-api' -H 'content-type: application/json' \
  -d '{"image":"e2e","resources":{"vcpu":1,"memory_mib":128,"disk_mib":256}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["vm_id"])')
output=$(curl -sf -X POST "http://127.0.0.1:17240/v1/vms/$vm/runs" -H 'authorization: Bearer e2e-api' -H 'content-type: application/json' \
  -d '{"command":"printf run-ok","timeout":5}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["stdout"])')
[[ $output == run-ok ]]
content=$(printf persistent | base64 -w0)
curl -sf -X POST "http://127.0.0.1:17240/v1/vms/$vm/sync" -H 'authorization: Bearer e2e-api' -H 'content-type: application/json' \
  -d "{\"op\":\"write\",\"writes\":[{\"path\":\"/tmp/blast-k3s-state\",\"size\":10,\"content\":\"$content\"}]}" >/dev/null
for _ in $(seq 1 100); do
  state=$(curl -sf -H 'authorization: Bearer e2e-api' "http://127.0.0.1:17240/v1/vms/$vm" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  [[ $state == idle && -d $host_root/snapshots/$vm ]] && break
  sleep .1
done
[[ $state == idle && -d $host_root/snapshots/$vm ]]

kill "$forward"
wait "$forward" 2>/dev/null || true
kubectl --kubeconfig "$kubeconfig" -n "$namespace" delete pod "$pod" --wait=true
kubectl --kubeconfig "$kubeconfig" -n "$namespace" rollout status daemonset/blast --timeout=120s
new_pod=$(kubectl --kubeconfig "$kubeconfig" -n "$namespace" get pod -l "$selector" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
[[ $new_pod != "$pod" ]]
[[ $(cat "$host_root/worker_id") == "$worker_id" ]]

kubectl --kubeconfig "$kubeconfig" -n "$namespace" port-forward "pod/$new_pod" 17240:7240 >"$host_root/port-forward-2.log" 2>&1 &
forward=$!
for _ in $(seq 1 100); do curl -sf http://127.0.0.1:17240/v1/live >/dev/null && break; sleep .1; done
curl -sf http://127.0.0.1:17240/v1/live >/dev/null
restored=$(curl -sf -X POST "http://127.0.0.1:17240/v1/vms/$vm/sync" -H 'authorization: Bearer e2e-api' -H 'content-type: application/json' \
  -d '{"op":"read","path":"/tmp/blast-k3s-state"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])')
[[ $restored == persistent ]]
curl -sf -X DELETE -H 'authorization: Bearer e2e-api' "http://127.0.0.1:17240/v1/vms/$vm" >/dev/null
status=$(curl -s -H 'authorization: Bearer e2e-api' -o /dev/null -w '%{http_code}' "http://127.0.0.1:17240/v1/vms/$vm")
[[ $status == 404 ]]

echo "k3s DaemonSet process E2E passed"
