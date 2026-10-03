"""
Stop the running remote Brumley-Boneh attack -- safely and automatically.

Kills the attacker recorded in `full_run_current.txt` (and its victim child, via
the Windows process tree) plus any STRAY `attack_remote.py` / `bb_victim_server.py`
python processes.  It does NOT delete the checkpoint, so the run can be resumed
with the same command (`attack_remote.py full --config ...`).

Why a helper: launches are detached background processes; this finds and ends
exactly this attack's processes (not every python) in one command.

Usage (from anywhere; Windows):
    python stop_attack.py            # kill the current run (pointer + strays)
    python stop_attack.py --all      # kill ANY attack_remote/victim processes
    python stop_attack.py --dry-run  # just show what WOULD be killed
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABB = os.path.dirname(_HERE)
_RESULTS = os.path.join(_ABB, "bb_results", "remote_unpinned_victim")
POINTER = os.path.join(_RESULTS, "full_run_current.txt")
CHECKPOINT = os.path.join(_RESULTS, "full_run_checkpoint.json")

# python processes whose command line contains any of these are "this attack".
_MARKERS = ("attack_remote.py", "bb_victim_server.py")


def _python_procs():
    """[(pid, cmdline)] for every running python.exe (via PowerShell CIM)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    if isinstance(data, dict):
        data = [data]
    return [(int(d["ProcessId"]), d.get("CommandLine") or "") for d in data]


def _pointer_pid():
    try:
        with open(POINTER, encoding="utf-8-sig") as fh:
            return int(fh.read().split("|")[0].strip())
    except Exception:
        return None


def _taskkill(pid, dry):
    if dry:
        print("  [dry-run] would kill PID %d (and its tree)" % pid)
        return
    r = subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                       capture_output=True, text=True)
    print("  killed PID %d%s" % (pid, "" if r.returncode == 0
                                  else " (%s)" % r.stdout.strip()))


def main():
    argv = sys.argv[1:]
    dry = "--dry-run" in argv
    kill_all = "--all" in argv

    targets = set()
    if not kill_all:
        pid = _pointer_pid()
        if pid:
            targets.add(pid)
            print("current run attacker PID (from pointer): %d" % pid)
        else:
            print("no readable pointer (%s)" % POINTER)

    # Always also match by command line (catches the victim child if the
    # attacker already died, strays, or --all).
    for pid, cmd in _python_procs():
        if any(m in cmd for m in _MARKERS):
            targets.add(pid)

    if not targets:
        print("no attack_remote / victim processes found -- nothing to kill.")
    else:
        print("targets: %s" % ", ".join(map(str, sorted(targets))))
        for pid in sorted(targets):
            _taskkill(pid, dry)

    # Report checkpoint status (resume point) -- never delete it here.
    if os.path.exists(CHECKPOINT):
        try:
            ck = json.load(open(CHECKPOINT))
            total = ck.get("qbits", 0) - 1 - ck.get("stop_bit", 0)
            print("checkpoint intact: next_idx=%s/%s  queries=%s  (resumable)"
                  % (ck.get("next_idx"), total, ck.get("queries_so_far")))
        except Exception as exc:
            print("checkpoint present but unreadable: %r" % exc)
    else:
        print("no checkpoint present (a finished/never-started run).")

    # Confirm none remain (skip in dry-run).
    if not dry:
        remaining = [pid for pid, cmd in _python_procs()
                     if any(m in cmd for m in _MARKERS)]
        print("attack processes remaining: %d" % len(remaining))


if __name__ == "__main__":
    main()
