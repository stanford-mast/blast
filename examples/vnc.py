import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
import re
from multiprocessing import Process

def setup_requirements():
    print("Installing dependencies...")
    subprocess.run("sudo apt-get update", shell=True, check=True)
    subprocess.run(
        "sudo apt-get install -y openbox tigervnc-standalone-server novnc websockify dbus-x11 x11-utils xdotool",
        shell=True,
        check=True,
    )
    print("Fixing /tmp/.X11-unix permissions...")
    subprocess.run("sudo mkdir -p /tmp/.X11-unix", shell=True, check=True)
    subprocess.run("sudo chmod 1777 /tmp/.X11-unix", shell=True, check=True)

def configure_xstartup():
    vnc_dir = Path.home() / ".vnc"
    vnc_dir.mkdir(exist_ok=True)
    xstartup = vnc_dir / "xstartup"
    xstartup.write_text("""#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
[ -r "$HOME/.Xresources" ] && xrdb "$HOME/.Xresources"
vncconfig -iconic &
exec dbus-launch --exit-with-session openbox-session
""")
    xstartup.chmod(0o755)
    print("Configured ~/.vnc/xstartup to use Openbox")

def get_chromium_class():
    print("Detecting Chromium WM_CLASS...")
    proc = subprocess.run(
        "xprop -name Chromium | grep WM_CLASS",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    match = re.search(r'"(.*?)",\s*"(.*?)"', proc.stdout)
    if match:
        return match.group(2)
    return "Chromium-browser"

def configure_openbox(wm_class):
    ob_dir = Path.home() / ".config" / "openbox"
    ob_dir.mkdir(parents=True, exist_ok=True)
    rc = ob_dir / "rc.xml"
    rc.write_text(f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<openbox_config>
  <applications>
    <application class=\"{wm_class}\">
      <decor>no</decor>
      <fullscreen>yes</fullscreen>
      <maximize>yes</maximize>
    </application>
  </applications>
  <theme>
    <titleLayout>NL</titleLayout>
    <keepBorder>no</keepBorder>
  </theme>
</openbox_config>
""")
    subprocess.run("openbox --reconfigure || true", shell=True)

def patch_novnc_ui():
    html_path = "/usr/share/novnc/vnc.html"
    patch_version = "v0.1.20"

    if not os.path.exists(html_path):
        print("\u26a0\ufe0f noVNC HTML not found!")
        return

    with open(html_path, "r") as f:
        html = f.read()

    while True:
        new_html = re.sub(
            r"<!-- custom patch v0\.\d+\.\d+ -->\s*(<style>.*?</style>\s*)?(<script>.*?</script>\s*)?",
            "",
            html,
            flags=re.DOTALL
        )
        if new_html == html:
            break
        html = new_html

    if f"<!-- custom patch {patch_version} -->" in html:
        print(f"\u2705 noVNC UI already patched with {patch_version}.")
        return

    patch = f"""<!-- custom patch {patch_version} -->
<style>
    #noVNC_control_bar_anchor,
    #noVNC_control_bar,
    #noVNC_status,
    #noVNC_connect_dlg,
    #noVNC_control_bar_hint,
    #noVNC_transition {{
        display: none !important;
    }}
</style>
<script>
window.addEventListener('load', function () {{
    const style = document.createElement('style');
    style.textContent = `
        #noVNC_control_bar_anchor,
        #noVNC_control_bar,
        #noVNC_status,
        #noVNC_connect_dlg,
        #noVNC_control_bar_hint,
        #noVNC_transition {{
            display: none !important;
        }}
    `;
    document.head.appendChild(style);
    const button = document.querySelector("#noVNC_connect_button");
    if (button) button.click();
}});
</script>"""

    patched = html.replace("</head>", patch + "\n</head>")

    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name

    subprocess.run(f"sudo cp {shlex.quote(tmp_path)} {html_path}", shell=True)
    os.unlink(tmp_path)

    print(f"\u2705 Patched {html_path} with {patch_version}")

def cleanup_display(display_no, vnc_port, http_port):
    subprocess.run(f"vncserver -kill :{display_no}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for port in (vnc_port, http_port):
        subprocess.run(f"fuser -k {port}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def launch_session(display_no=1, geometry="1280x720", target_url="https://practicepanther.com"):
    vnc_port = 5900 + display_no
    http_port = 6080 + display_no

    env = os.environ.copy()
    env["DISPLAY"] = f":{display_no}"
    os.environ["DISPLAY"] = f":{display_no}"

    cleanup_display(display_no, vnc_port, http_port)
    configure_xstartup()
    patch_novnc_ui()

    wm_class = get_chromium_class()
    configure_openbox(wm_class)

    print(f"Starting TigerVNC on display :{display_no}")
    subprocess.run(
        f"vncserver :{display_no} -geometry {geometry} -localhost no -SecurityTypes None --I-KNOW-THIS-IS-INSECURE",
        shell=True,
        check=True,
        env=env
    )

    print(f"Starting noVNC proxy on port {http_port}")
    web_proc = subprocess.Popen(
        f"websockify --web /usr/share/novnc {http_port} localhost:{vnc_port}",
        shell=True,
        env=env
    )
    time.sleep(2)

    iframe_url = f"http://localhost:{http_port}/vnc.html"

    print("Launching Chromium via Playwright")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=f'/tmp/playwright_{display_no}',
            headless=False,
            args=[
                f'--app={target_url}',
                '--disable-infobars',
                '--disable-blink-features=AutomationControlled',
                '--disable-features=TranslateUI,OverlayScrollbar,ExperimentalFullscreenExitUI',
                '--kiosk',
                '--start-fullscreen',
                '--start-maximized',
                '--disable-translate',
                '--disable-dev-shm-usage'
            ],
            ignore_default_args=['--enable-automation'],
            env=env
        )
        time.sleep(2)
        context.pages[0].goto(target_url)
        subprocess.run("xdotool search --class chromium windowactivate --sync key F11", shell=True, env=env)
        print("\U0001f4cc Embed iframe using:", iframe_url)
        while True:
            time.sleep(1)
        context.close()

    web_proc.terminate()
    subprocess.run(f"vncserver -kill :{display_no}", shell=True)
    print(f"\U0001f9fc Session on display :{display_no} cleaned up.")

def run_parallel_sessions():
    setup_requirements()

    sessions = [
        (1, "https://practicepanther.com"),
        (2, "https://example.com")
    ]

    procs = []
    for display_no, url in sessions:
        p = Process(target=launch_session, args=(display_no, "1280x720", url))
        p.start()
        procs.append(p)

    input("\nPress ENTER to terminate all sessions...\n")

    for display_no, _ in sessions:
        subprocess.run(f"vncserver -kill :{display_no}", shell=True)
        subprocess.run(f"fuser -k {5900 + display_no}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"fuser -k {6080 + display_no}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for p in procs:
        p.terminate()
        p.join()

    print("\n✅ All sessions cleaned up.")

if __name__ == "__main__":
    run_parallel_sessions()
