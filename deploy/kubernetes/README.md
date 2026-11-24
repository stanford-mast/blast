# Kubernetes worker

The example runs one host-bound BLAST worker on every labeled node. It is a
DaemonSet, not a scalable Deployment: the worker identity and warm VM state in
`/var/lib/blast` belong to exactly one machine.

Prepare each node before applying the manifest:

1. Install a compatible `smolvm` and BLAST image. Replace the example image
   with an immutable digest.
2. Create `/var/lib/blast`, owned by uid/gid `65532`, on node-local durable
   storage. Do not use a shared volume and do not copy this directory between
   nodes.
3. Install the KubeVirt KVM device plugin, which advertises
   `devices.kubevirt.io/kvm` and injects `/dev/kvm` with its device-cgroup
   permission. Keep the pod non-root. If the plugin preserves restrictive
   host ownership, configure its device permissions for uid/gid `65532`.
4. Label and taint the node:

   ```sh
   kubectl label node NODE blastproject.org/worker=true
   kubectl taint node NODE blastproject.org/worker=true:NoSchedule
   ```

5. Replace the control-plane endpoint, pool id, resources, and lifecycle
   values. The BLAST limits must leave capacity for Kubernetes and the host.
6. Create the pool join secret and apply the manifest:

   ```sh
   kubectl -n blast-system create secret generic blast-pool \
     --from-literal=join-key='...' \
     --from-literal=local-api-token='...'
   kubectl apply -f deploy/kubernetes/daemonset.yaml
   ```

The host machine ID is mounted read-only so a restarted pod rejoins as the same
worker. A fresh incarnation fences the stopped process. The control plane
issues object-storage credentials over heartbeats; no cloud credentials or
database credentials belong in the pod. Both control-plane and object-storage
traffic are outbound. The local API has no ingress policy and is not exposed by
a Service.

SIGTERM drains work and persists VMs. The 900-second Kubernetes grace exceeds
BLAST's 600-second drain timeout. The disruption budget keeps at least one
compatible worker available, and the autoscaler annotation prevents an
uncoordinated eviction. A single-node pool therefore requires an explicit PDB
override before maintenance. Before draining a node, confirm its VMs have
durable snapshots or explicitly run `blast leave`; deleting a node also
deletes its local warm state.

`tests/k3s-daemonset.sh` exercises the pod boundary on a real single-node k3s
cluster. It renders this manifest through `tests/kubernetes`, builds and imports
a content-addressed local worker image, and checks the non-root identity,
disabled service-account token, read-only machine identity, and resource
settings on the resulting pod. Its overlay makes the worker privileged only to
grant `/dev/kvm` through a raw hostPath without installing a device plugin. It
uses a deterministic fake hypervisor so it can test orchestration without
downloading a guest image; it does not qualify SmolVM, the KVM device plugin, or
a production worker image.
