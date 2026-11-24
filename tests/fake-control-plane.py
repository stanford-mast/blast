#!/usr/bin/env python3
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    generation = 0

    def log_message(self, format, *args):
        print(format % args, flush=True)

    def reply(self, status=200, body=None):
        encoded = json.dumps(body or {}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_PUT(self):
        size = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(size) or b"{}")
        match = re.fullmatch(r"/api/v1/pools/[^/]+/workers/([^/?]+)", self.path.split("?", 1)[0])
        if match:
            self.reply(body={
                "worker_id": match.group(1),
                "heartbeat_interval_ms": 1000,
                "membership_lease_ms": 10000,
                "desired_state": "active",
                "tunnel_url": "ws://blast-control-plane:8080/unavailable",
                "storage": {
                    "bucket": "unused",
                    "region": "us-east-1",
                    "prefix": "unused",
                    "credentials": {
                        "access_key_id": "unused",
                        "secret_access_key": "unused",
                        "session_token": "",
                        "expires_at": "2099-01-01T00:00:00Z"
                    }
                }
            })
        else:
            self.reply()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path.endswith("/vms/reconcile"):
            size = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
            self.reply(body={"retained_vm_ids": body.get("vm_ids", [])})
        elif path.endswith("/snapshots"):
            Handler.generation += 1
            self.reply(body={
                "vm_generation": Handler.generation,
                "snapshot_id": f"01K3S{Handler.generation:021d}",
                "lease_renew_after_secs": 60
            })
        elif self.path.endswith("/api/v1/internal/store/telemetry/events"):
            self.reply(body={"accepted": 1, "refused": 0})
        else:
            self.reply()

    do_DELETE = do_PUT
    do_PATCH = do_PUT


ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
