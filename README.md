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

```bash
cargo install blast_core
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

## Purpose

BLAST is a single binary for local sandbox orchestration given a pool of CPU, memory, disk. More precisely, BLAST abstracts over local sandboxes such as SmolVM, Hypeman, Docker to provides a simple API to fork and run sandboxed commands, sync data, and monitor VMs, sessions, runs while automatically scheduling and placing forks and runs, snapshots, syncing snapshots to durable storage, migration, and managing resource pressure.

## Features

Compared to existing solutions for local sandboxing or BYOC sandbox compute, BLAST features:

* **Single 7 MB binary.** No Terraform. No Packer. No extra dependencies.
* **Most permissible license.** Code is MIT-licensed and just 3,586 lines, built to keep enterprise security reviews as simple as possible.
* **Full orchestration.** Unlike other solutions that simply provide utilities for creating sandboxes, BLAST takes a given pool of CPU, memory, disk and optimally serves forking and running sandboxed commands.
* **Compatible with VM providers.** For a unified control plane across bursty cloud compute and user-provided compute, BLAST is built to integrate with a control plane.
* Built and actively maintained by a growing team of open-source sandboxing and orchestration enthusiasts

## Documentation

Visit [documentation](https://blastproject.org/docs) to learn more.
