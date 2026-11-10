"""Linux user units contain even double-forked/setsid experiment children."""
import array
import hashlib
from pathlib import Path
import socket
import subprocess


def service_name(root):
    return "ignis-experiments-" + hashlib.sha256(str(root).encode()).hexdigest()[:12] + ".service"


def unit_name(root, job_id):
    suffix = hashlib.sha256((str(root) + "/" + job_id).encode()).hexdigest()[:24]
    return "ignis-experiment-" + suffix + ".service"


def systemctl(*args, check=True):
    result = subprocess.run(["systemctl", "--user", *args], text=True, capture_output=True, timeout=30)
    if check and result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip())
    return result


def inspect(unit):
    result = systemctl("show", unit, "--property=LoadState,ActiveState,SubState,Result,ControlGroup,ExecMainStatus", check=False)
    if result.returncode:
        raise ValueError("cannot query user systemd: " + result.stderr.strip())
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def members(group):
    if not group:
        return []
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    if not root.exists():
        return []
    paths = [root / "cgroup.procs", *root.glob("**/cgroup.procs")]
    found = set()
    for path in paths:
        try:
            found.update(int(pid) for pid in path.read_text().split())
        except FileNotFoundError:
            pass
    return sorted(found)


def stop(unit):
    before = inspect(unit)
    if before.get("LoadState") == "not-found":
        return
    systemctl("stop", unit)
    after = inspect(unit)
    if after.get("ActiveState") not in {"inactive", "failed"} or members(before.get("ControlGroup")):
        raise ValueError("unit cleanup incomplete: " + unit)
    systemctl("reset-failed", unit, check=False)


def address(root, job_id):
    return "\0ignis-lease-" + hashlib.sha256((str(root) + job_id).encode()).hexdigest()


def listener(root, job_id):
    channel = socket.socket(socket.AF_UNIX)
    channel.bind(address(root, job_id))
    channel.listen(1)
    channel.settimeout(15)
    return channel


def send_lease(channel, fd):
    connection, _ = channel.accept()
    with connection:
        connection.sendmsg([b"L"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])


def receive_lease(root, job_id):
    with socket.socket(socket.AF_UNIX) as channel:
        channel.settimeout(15)
        channel.connect(address(root, job_id))
        message, controls, _, _ = channel.recvmsg(1, socket.CMSG_SPACE(array.array("i").itemsize))
    if message != b"L" or not controls:
        raise ValueError("GPU lease was not delivered")
    fds = array.array("i")
    fds.frombytes(controls[0][2][:fds.itemsize])
    return fds[0]


def launch(root, runtime, record):
    directory = root / ".scratch/experiments/jobs" / record["id"]
    # Resolve private toolchain paths from the configured checkout, then let the
    # runner set its candidate cwd and private build/cache locations.
    script = 'set -e; cd "$1"; if [ -f "$2" ]; then source "$2"; fi; exec "$3" "$4" --root "$1" run "$5"'
    command = ["systemd-run", "--user", "--quiet", "--unit=" + record["unit"],
               "--property=Type=exec", "--property=RemainAfterExit=yes",
               "--property=KillMode=control-group", "--property=TimeoutStopSec=5s",
               "--property=RuntimeMaxSec=" + str(record["payload"]["request"]["timeout_s"]),
               "--property=StandardOutput=append:" + str(directory / "unit.log"),
               "--property=StandardError=append:" + str(directory / "unit.log"),
               "/bin/bash", "--noprofile", "--norc", "-c", script, "ignis-experiment",
               str(root), str(directory / "config/.remote.env"), "/usr/bin/python3", str(runtime), record["id"]]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("cannot start experiment unit: " + result.stderr.strip())
