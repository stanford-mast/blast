#!/usr/bin/env python3
import gzip
import hashlib
import io
import json
import pathlib
import platform
import sys
import tarfile


source_path, output_path, binary_path, fake_path, tag = sys.argv[1:]


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


with tarfile.open(source_path) as source:
    members = {member.name: source.extractfile(member).read() for member in source if member.isfile()}

index = json.loads(members["index.json"])
descriptor = index["manifests"][0]
document = json.loads(members[f"blobs/sha256/{descriptor['digest'].split(':')[1]}"])
if "manifests" in document:
    architecture = {"x86_64": "amd64", "aarch64": "arm64"}[platform.machine()]
    descriptor = next(item for item in document["manifests"] if item["platform"]["architecture"] == architecture and item["platform"]["os"] == "linux")
    document = json.loads(members[f"blobs/sha256/{descriptor['digest'].split(':')[1]}"])

config_descriptor = document["config"]
manifest_media_type = descriptor["mediaType"]
config = json.loads(members[f"blobs/sha256/{config_descriptor['digest'].split(':')[1]}"])
layer_buffer = io.BytesIO()
with tarfile.open(fileobj=layer_buffer, mode="w") as layer:
    for source, destination in ((binary_path, "usr/local/bin/blast"), (fake_path, "usr/local/bin/fake-smolvm")):
        data = pathlib.Path(source).read_bytes()
        info = tarfile.TarInfo(destination)
        info.mode = 0o755
        info.size = len(data)
        layer.addfile(info, io.BytesIO(data))
layer_data = layer_buffer.getvalue()
compressed = gzip.compress(layer_data, mtime=0)
layer_digest = digest(compressed)
members[f"blobs/sha256/{layer_digest.split(':')[1]}"] = compressed

config.setdefault("rootfs", {}).setdefault("diff_ids", []).append(digest(layer_data))
config.setdefault("history", []).append({"created_by": "BLAST k3s process E2E"})
config.setdefault("config", {})["Entrypoint"] = ["/usr/local/bin/blast"]
config["config"]["Cmd"] = []
config_data = json.dumps(config, separators=(",", ":")).encode()
config_digest = digest(config_data)
members[f"blobs/sha256/{config_digest.split(':')[1]}"] = config_data

document["config"] = {**config_descriptor, "digest": config_digest, "size": len(config_data)}
document["layers"].append({
    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
    "digest": layer_digest,
    "size": len(compressed),
})
manifest_data = json.dumps(document, separators=(",", ":")).encode()
manifest_digest = digest(manifest_data)
members[f"blobs/sha256/{manifest_digest.split(':')[1]}"] = manifest_data
members["index.json"] = json.dumps({
    "schemaVersion": 2,
    "manifests": [{
        "mediaType": manifest_media_type,
        "digest": manifest_digest,
        "size": len(manifest_data),
        "annotations": {"org.opencontainers.image.ref.name": tag},
    }],
}, separators=(",", ":")).encode()

with tarfile.open(output_path, "w") as output:
    for name, data in members.items():
        info = tarfile.TarInfo(name)
        info.mode = 0o644
        info.size = len(data)
        output.addfile(info, io.BytesIO(data))
print(manifest_digest)
