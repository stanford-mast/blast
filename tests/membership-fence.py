#!/usr/bin/env python3
import base64
import hashlib
import http.server
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse


class ControlPlane(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    reject_heartbeats = False
    first_incarnation = None
    incarnations = []
    worker_ids = []
    tunnel_opened = threading.Event()
    accepting = threading.Event()

    def log_message(self, *_args):
        pass

    def do_PUT(self):
        size = int(self.headers.get("content-length", "0"))
        self.rfile.read(size)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        incarnation = query["incarnation"][0]
        type(self).incarnations.append(incarnation)
        if type(self).first_incarnation is None:
            type(self).first_incarnation = incarnation
        if type(self).reject_heartbeats:
            # Left unanswered past its headers, as an unreachable control
            # plane leaves it, until the test lets heartbeats through again.
            self.send_response(503)
            self.end_headers()
            type(self).accepting.wait(30)
            self.close_connection = True
            return
        worker_id = urllib.parse.urlparse(self.path).path.rsplit("/", 1)[-1]
        type(self).worker_ids.append(worker_id)
        storage = {
            "bucket": "unused",
            "region": "us-east-1",
            "prefix": worker_id,
            "credentials": {
                "access_key_id": "unused",
                "secret_access_key": "unused",
                "session_token": "",
                "expires_at": "2099-01-01T00:00:00Z",
            },
        }
        body = json.dumps({
            "worker_id": worker_id,
            "heartbeat_interval_ms": 1000,
            "membership_lease_ms": 4000,
            "desired_state": "active",
            "tunnel_url": f"ws://127.0.0.1:{self.server.server_port}/tunnel",
            "storage": storage,
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        key = self.headers["Sec-WebSocket-Key"]
        accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
        ).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.send_header("Sec-WebSocket-Protocol", "worker-tunnel.v1")
        self.end_headers()
        type(self).tunnel_opened.set()
        self.connection.settimeout(10)
        try:
            while self.connection.recv(4096):
                pass
        except (OSError, TimeoutError):
            pass

    def do_POST(self):
        if not urllib.parse.urlparse(self.path).path.endswith("/vms/reconcile"):
            self.send_response(404)
            self.end_headers()
            return
        size = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(size) or b"{}")
        body = json.dumps({"retained_vm_ids": request.get("vm_ids", [])}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start(binary, config):
    return subprocess.Popen(
        [binary, "--config", str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def toml_string(value):
    return json.dumps(str(value).replace("\\", "/"))


def write_config(path, *, port, data_dir, control_plane, resources=True):
    limits = """
[resources]
vcpu = 1
memory_mib = 256
disk_mib = 1024
""" if resources else ""
    path.write_text(f'''port = {port}
data_dir = {toml_string(data_dir)}
[backend]
kind = "smolvm"
binary = {toml_string(sys.executable)}
{limits}
[worker]
control_plane_endpoint = "http://127.0.0.1:{control_plane}"
api_key = "test"
pool_id = "00000000-0000-0000-0000-000000000001"
''')


def main():
    binary = str(pathlib.Path(sys.argv[1]).resolve())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ControlPlane)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp) / "native-路径" / ("durable-" + "x" * 80)
        root.mkdir(parents=True)
        config = root / "blast.toml"
        data_dir = root / "data-状态"
        write_config(
            config,
            port=free_port(),
            data_dir=data_dir,
            control_plane=server.server_port,
        )

        invalid = root / "missing-resources.toml"
        write_config(
            invalid,
            port=free_port(),
            data_dir=root / "invalid",
            control_plane=server.server_port,
            resources=False,
        )
        rejected = start(binary, invalid)
        output, _ = rejected.communicate(timeout=5)
        if rejected.returncode == 0 or "require explicit [resources]" not in output:
            raise AssertionError(output)

        first = start(binary, config)
        if not ControlPlane.tunnel_opened.wait(5):
            raise AssertionError("worker did not establish its tunnel")

        host_identity = (data_dir / "worker_host").read_text().strip()
        if sys.platform == "win32" and not host_identity.startswith("windows:"):
            raise AssertionError(f"unexpected Windows host identity: {host_identity}")

        contender = start(binary, config)
        contender_output, _ = contender.communicate(timeout=5)
        if contender.returncode == 0 or "another BLAST process" not in contender_output:
            raise AssertionError(contender_output)

        # A worker that cannot renew its lease fences itself before the
        # lease runs out, then heartbeats again under a new incarnation.
        ControlPlane.reject_heartbeats = True
        failed_at = time.monotonic()
        while len(set(ControlPlane.incarnations)) < 2:
            if first.poll() is not None:
                raise AssertionError(first.communicate()[0])
            if time.monotonic() - failed_at >= 4:
                raise AssertionError("worker did not fence before membership expiry")
            time.sleep(.05)

        ControlPlane.reject_heartbeats = False
        ControlPlane.accepting.set()
        ControlPlane.tunnel_opened.clear()
        second = first
        try:
            if not ControlPlane.tunnel_opened.wait(10):
                raise AssertionError("worker did not rejoin")
            deadline = time.time() + 3
            while time.time() < deadline and len(set(ControlPlane.incarnations)) < 2:
                time.sleep(.05)
            if len(set(ControlPlane.incarnations)) != 2:
                raise AssertionError(f"incarnations: {ControlPlane.incarnations}")
            if ControlPlane.incarnations[-1] == ControlPlane.first_incarnation:
                raise AssertionError("the rejoin reused a fenced incarnation")
            if len(set(ControlPlane.worker_ids)) != 1:
                raise AssertionError(f"worker ids: {ControlPlane.worker_ids}")
            if (data_dir / "worker_host").read_text().strip() != host_identity:
                raise AssertionError("the rejoin changed the durable host identity")
            worker_id = (data_dir / "worker_id").read_text().strip()
            if worker_id != ControlPlane.worker_ids[-1]:
                raise AssertionError("durable worker id differs from the admitted worker")
            incarnation = (data_dir / "worker_incarnation").read_text().strip()
            if incarnation != ControlPlane.incarnations[-1]:
                raise AssertionError("durable incarnation differs from the admitted worker")
            leftovers = list(data_dir.glob(".*.tmp"))
            if leftovers:
                raise AssertionError(f"atomic-write temporary files remain: {leftovers}")
        finally:
            second.terminate()
            second.wait(timeout=5)
    server.shutdown()
    print("membership self-fence process test passed")


if __name__ == "__main__":
    main()
