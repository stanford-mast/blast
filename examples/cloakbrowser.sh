#!/usr/bin/env bash
# Fast browser sessions with BLAST: fork a *running* SmolVM that already has
# a browser warm, instead of booting a fresh VM and cold-starting the
# browser inside it every time. The fork inherits the running process's
# live memory state, so the forked VM's browser is instantly usable, no
# boot, no browser cold-start.
#
# Uses real CloakBrowser (github.com/CloakHQ/CloakBrowser) via its official
# Docker image. `cloakserve` runs it as a CDP server; Playwright, bundled in
# the image, drives it over CDP. Nothing to install.
#
# This is SmolVM-specific: only SmolVM's fork does a real memory
# snapshot/restore of a running process. Point BLAST_URL at a `blast`
# instance configured with kind = "smolvm".
set -euo pipefail

BLAST_URL="${BLAST_URL:-http://localhost:7240}"
IMAGE="${CLOAKBROWSER_IMAGE:-cloakhq/cloakbrowser}"

fork() {
  curl -sf -X POST "$BLAST_URL/v1/fork" -H "Content-Type: application/json" -d "$1"
}
run() {
  curl -sf -X POST "$BLAST_URL/v1/vms/$1/runs" -H "Content-Type: application/json" -d "$2"
}
sync_read() {
  curl -sf -X POST "$BLAST_URL/v1/vms/$1/sync" -H "Content-Type: application/json" \
    -d "{\"op\":\"read\",\"path\":\"$2\"}"
}
vm_id() { python3 -c 'import json,sys; print(json.load(sys.stdin)["vm_id"])'; }

echo "Booting the base VM (cold: image pull + browser launch)..."
BASE=$(fork "{\"image\":\"$IMAGE\",\"name\":\"cloakbrowser-base\",\"resources\":{\"vcpu\":2,\"memory_mib\":2048,\"disk_mib\":10240}}" | vm_id)
# A fork boots the image's filesystem, not its entrypoint, so the image's own
# entrypoint.sh (which normally starts Xvfb before cloakserve) never runs.
# Starting it explicitly here does the same thing.
run "$BASE" '{"command":"Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp & sleep 1 && DISPLAY=:99 nohup cloakserve >/tmp/cloakserve.log 2>&1 & sleep 3 && curl -sf http://localhost:9222/json/version","timeout":60}' > /dev/null
echo "Base VM $BASE ready, browser warm."

echo ""
echo "Forking a session from the warm base VM..."
T0=$(date +%s%N)
SESSION=$(fork "{\"source_vm_id\":\"$BASE\",\"name\":\"cloakbrowser-session\"}" | vm_id)
T1=$(date +%s%N)
echo "Session VM $SESSION ready in $(( (T1 - T0) / 1000000 ))ms, browser already running."

echo ""
echo "Navigating and taking a screenshot in the forked session..."
NAV='python3 -c "from playwright.sync_api import sync_playwright as sp
with sp() as p:
    b = p.chromium.connect_over_cdp(\"http://localhost:9222\")
    ctx = b.contexts[0] if b.contexts else b.new_context()
    page = ctx.new_page()
    page.goto(\"https://example.com\")
    page.screenshot(path=\"/tmp/screenshot.png\")
    print(page.title())"'
run "$SESSION" "{\"command\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$NAV"),\"timeout\":30}"
SYNC=$(sync_read "$SESSION" /tmp/screenshot.png)
echo "$SYNC" | python3 -c '
import json, sys, base64
r = json.load(sys.stdin)
data = base64.b64decode(r["content"]) if r.get("encoding") == "base64" else r["content"].encode()
open("screenshot.png", "wb").write(data)
print(f"screenshot saved to ./screenshot.png ({len(data)} bytes)")
'

echo ""
echo "Cleaning up..."
curl -sf -X DELETE "$BLAST_URL/v1/vms/$SESSION" > /dev/null
curl -sf -X DELETE "$BLAST_URL/v1/vms/$BASE" > /dev/null
echo "done."
