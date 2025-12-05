import subprocess
import time
import paramiko
import re
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
NUC_IP = "172.16.0.5"
# NUC_USERNAME = "your_username"
NUC_PASSWORD = " " 
TMUX_SESSION_NAME = "robot_server"

RESTART_LIMIT = 5
RESTART_WINDOW_SEC = 600   # 10 minutes
RESTART_HISTORY = []

# Error patterns (customize)
CAMERA_ERROR = r"camera"
ROBOT_ERROR = r"polymetis"
CRITICAL_ERROR = r"Too many critical failures"


# -----------------------------
# TMUX HELPERS (on the NUC)
# -----------------------------
def send_ssh_cmd(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode(), stderr.read().decode()


def tmux_running(ssh):
    out, _ = send_ssh_cmd(
        ssh, f"tmux has-session -t {TMUX_SESSION_NAME} 2>/dev/null && echo OK || echo NO"
    )
    return "OK" in out


def tmux_start_server(ssh):
    print("[Supervisor] Starting robot server in tmux...")
    send_ssh_cmd(
        ssh,
        f'tmux new -d -s {TMUX_SESSION_NAME} "bash -c \'conda activate polymetis-local && cd droid && python scripts/server/run_server.py\'"'
    )


def tmux_stop_server(ssh):
    print("[Supervisor] Stopping robot server...")
    send_ssh_cmd(ssh, f"tmux send-keys -t {TMUX_SESSION_NAME} C-c")


# -----------------------------
# START / STOP main.py LOCALLY
# -----------------------------
def start_main():
    print("[Supervisor] Starting main.py...")
    cmd = [
        "bash", "-c",
        "conda activate robot && python -u examples/droid/main.py --remote_host=0.0.0.0 --remote_port=8000 --external_camera=right"
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def stop_main(proc):
    if proc:
        print("[Supervisor] Stopping main.py...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


# -----------------------------
# LOGIC HELPERS
# -----------------------------
def track_restart():
    now = datetime.now()
    RESTART_HISTORY.append(now)
    # remove old ones
    RESTART_HISTORY[:] = [
        t for t in RESTART_HISTORY if (now - t).total_seconds() <= RESTART_WINDOW_SEC
    ]
    return len(RESTART_HISTORY) >= RESTART_LIMIT


# -----------------------------
# MAIN SUPERVISOR LOOP
# -----------------------------
def supervisor():
    print("[Supervisor] Connecting to NUC via SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(NUC_IP, password=NUC_PASSWORD)
    # ssh.connect(NUC_IP, username=NUC_USERNAME, password=NUC_PASSWORD)

    # Ensure robot server is running
    if not tmux_running(ssh):
        tmux_start_server(ssh)

    proc = start_main()

    while True:
        line = proc.stdout.readline()
        if not line:
            # process died
            print("[Supervisor] main.py exited unexpectedly")
            if track_restart():
                print("Too many restarts → stopping everything.")
                break
            time.sleep(20)
            proc = start_main()
            continue

        print("[main.py]", line.strip())

        # ---- ERROR DETECTION ----
        lower = line.lower()
        if re.search(CAMERA_ERROR, lower):
            print("[Supervisor] Camera error → restart main.py only.")
            stop_main(proc)
            if track_restart():
                print("Too many restarts → shutting down.")
                break
            time.sleep(20)
            proc = start_main()

        elif re.search(ROBOT_ERROR, lower):
            print("[Supervisor] Robot connection error → restarting robot server & main.py.")

            # restart main first
            stop_main(proc)

            # restart robot server
            tmux_stop_server(ssh)
            time.sleep(5)
            tmux_start_server(ssh)
            time.sleep(2)

            if track_restart():
                print("Too many restarts → shutting down.")
                break

            proc = start_main()

        elif re.search(CRITICAL_ERROR, lower):
            print("[Supervisor] CRITICAL error → stopping everything.")
            break

    # Final cleanup
    stop_main(proc)
    tmux_stop_server(ssh)
    ssh.close()
    print("[Supervisor] Shutdown complete.")


if __name__ == "__main__":
    supervisor()
