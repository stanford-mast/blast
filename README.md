<div align="center">
  <img src="website/public/blast_icon_only.ico" width="200" height="200" alt="BLAST Logo">
</div>

<p align="center" style="font-size: 24px">Open-source VMs-as-a-service</p>

<div align="center">

[![Website](https://img.shields.io/badge/blastproject.org-FFE067)](https://blastproject.org)
[![Documentation](https://img.shields.io/badge/Docs-FFE067)](https://blastproject.org/docs)
[![Discord](https://img.shields.io/badge/Discord-FFE067)](https://discord.gg/NqrkJwYYh4)

</div>

## Quick Start

BLAST runs on Linux with KVM and the SmolVM 1.37.0 release, which it no longer
installs itself:

```bash
sudo mkdir -p /opt/smolvm
curl -fsSL "https://github.com/smol-machines/smolvm/releases/download/v1.37.0/smolvm-1.37.0-linux-$(uname -m | sed s/aarch64/arm64/).tar.gz" \
  | sudo tar -xz -C /opt/smolvm --strip-components=1
sudo ln -sf /opt/smolvm/smolvm /usr/local/bin/smolvm
sudo usermod -aG kvm "$USER"   # takes effect at your next login
```

SmolVM runs unprivileged, so BLAST's user must be able to open `/dev/kvm`; on a
host without a `kvm` group, run `sudo groupadd -r kvm && sudo chgrp kvm /dev/kvm && sudo chmod 660 /dev/kvm` before `usermod`.

```bash
cargo install --locked --git https://github.com/stanford-mast/blast
blast
```

```bash
VM1=$(curl -s -X POST localhost:7240/v1/fork \
  -H "Content-Type: application/json" \
  -d '{"image":"ubuntu:24.04"}' | jq -r .vm_id)

VM2=$(curl -s -X POST localhost:7240/v1/fork \
  -H "Content-Type: application/json" \
  -d "{\"source_vm_id\":\"$VM1\",\"name\":\"feature-xyz\"}" | jq -r .vm_id)

curl -X POST localhost:7240/v1/vms/$VM2/runs \
  -H "Content-Type: application/json" \
  -d '{"command":"echo hello from fork"}'
```

Set `[api] token`, or keep the worker on localhost: without a token, a standalone worker's API is open to anyone who can reach it.

## Purpose

BLAST is a single binary for local sandbox orchestration given a pool of CPU, memory, disk. More precisely, BLAST abstracts over local sandboxes such as SmolVM, Hypeman, Docker to provides a simple API to fork and run sandboxed commands, sync data, and monitor VMs, sessions, runs while automatically scheduling and placing forks and runs, snapshots, syncing snapshots to durable storage, migration, and managing resource pressure.

## Features

Compared to existing solutions for local sandboxing or BYOC sandbox compute, BLAST features:

* **Single 10 MB binary.** No Terraform. No Packer. No extra dependencies beyond SmolVM.
* **Most permissible license.** Code is MIT-licensed and about 15,000 lines, built to keep enterprise security reviews as simple as possible.
* **Full orchestration.** Unlike other solutions that simply provide utilities for creating sandboxes, BLAST takes a given pool of CPU, memory, disk and optimally serves forking and running sandboxed commands.
* **Compatible with VM providers.** For a unified control plane across bursty cloud compute and user-provided compute, BLAST is built to integrate with a control plane.
* Built and actively maintained by a growing team of open-source sandboxing and orchestration enthusiasts

## Documentation

Visit [documentation](https://blastproject.org/docs) to learn more.
