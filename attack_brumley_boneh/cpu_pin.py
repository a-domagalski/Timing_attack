"""
Measurement-environment pinning for the Brumley-Boneh `real` timing oracle.

Why
---
The work log's root cause for the failed `real`-mode recovery is MULTIPLICATIVE
timing noise (~20-28% rel. std) from an unpinned CPU: turbo / SpeedStep /
thermal frequency scaling on a general-purpose Windows host, plus scheduler
migration jitter.  This module reduces the *controllable* part of that noise
in-process, cheaply and reversibly:

  * single-core CPU affinity  -> stops cross-core migration (each core has its
    own frequency/thermal state, so migrating aliases as timing noise);
  * high process/thread priority -> fewer pre-emptions / scheduler gaps;
  * optional warm-up busy-loop -> reach a steady turbo/thermal frequency before
    measuring, so the run doesn't straddle the boost-then-settle transition.

IMPORTANT — what this does NOT do
---------------------------------
Affinity + priority do NOT disable Turbo Boost / frequency scaling itself; they
only remove migration and pre-emption jitter.  Killing the *multiplicative*
frequency drift needs a system-wide power/BIOS change (admin), which this module
deliberately does NOT apply silently.  Use :func:`power_plan_guide` to print the
exact `powercfg` commands, or :func:`apply_high_performance_power_plan` (opt-in,
``confirm=True``) to run them.  On server-class fixed-clock hardware (the regime
Brumley-Boneh measured in) this pinning is the software analogue of the fixed
clock their hardware provided for free.
"""

import os
import sys
import time


# --------------------------------------------------------------------------- #
# In-process pinning (safe, reversible, no admin required).
# --------------------------------------------------------------------------- #
def pin_process(core: int = 0, high_priority: bool = True,
                verbose: bool = True) -> dict:
    """Pin this process/thread to a single core and raise its priority.

    Returns a dict describing what actually took effect (best-effort: failures
    are reported, never raised, so a measurement run is never aborted by it).
    """
    result = {"platform": sys.platform, "core": core, "affinity_set": False,
              "priority_set": False, "notes": []}
    try:
        if sys.platform.startswith("win"):
            _pin_windows(core, high_priority, result)
        else:
            _pin_posix(core, high_priority, result)
    except Exception as exc:  # never let pinning abort a run
        result["notes"].append("pin_process error: %r" % exc)
    if verbose:
        print("[cpu_pin] core=%d affinity=%s priority=%s%s"
              % (core, result["affinity_set"], result["priority_set"],
                 ("  (" + "; ".join(result["notes"]) + ")")
                 if result["notes"] else ""))
    return result


def _pin_windows(core: int, high_priority: bool, result: dict) -> None:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    hproc = k32.GetCurrentProcess()
    hthread = k32.GetCurrentThread()

    # SetProcessAffinityMask(HANDLE, DWORD_PTR)
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    mask = ctypes.c_size_t(1 << core)
    if k32.SetProcessAffinityMask(hproc, mask):
        result["affinity_set"] = True
    else:
        result["notes"].append(
            "SetProcessAffinityMask failed (err %d)" % ctypes.get_last_error())

    # also pin the current thread (SetThreadAffinityMask returns prev mask)
    k32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetThreadAffinityMask.restype = ctypes.c_size_t
    k32.SetThreadAffinityMask(hthread, ctypes.c_size_t(1 << core))

    if high_priority:
        HIGH_PRIORITY_CLASS = 0x00000080
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        if k32.SetPriorityClass(hproc, HIGH_PRIORITY_CLASS):
            result["priority_set"] = True
        else:
            result["notes"].append(
                "SetPriorityClass failed (err %d)" % ctypes.get_last_error())


def _pin_posix(core: int, high_priority: bool, result: dict) -> None:
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})
            result["affinity_set"] = True
        except OSError as exc:
            result["notes"].append("sched_setaffinity: %r" % exc)
    else:
        result["notes"].append("no os.sched_setaffinity on this platform")
    if high_priority:
        try:
            os.nice(-10)              # needs privilege; best-effort
            result["priority_set"] = True
        except OSError as exc:
            result["notes"].append("nice(-10) needs privilege: %r" % exc)


def warmup(seconds: float = 1.0) -> None:
    """Busy-loop to drive the CPU to a steady turbo/thermal frequency before
    measuring, so the measured region doesn't straddle the boost-then-settle
    ramp.  Does nothing if seconds <= 0."""
    if seconds and seconds > 0:
        end = time.perf_counter() + seconds
        x = 0
        while time.perf_counter() < end:
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        return x


# --------------------------------------------------------------------------- #
# System-wide power plan (admin change) -- opt-in, never applied automatically.
# --------------------------------------------------------------------------- #
def power_plan_guide() -> str:
    """Return the exact commands to fix the CPU frequency on Windows (the lever
    that actually removes the multiplicative drift).  Run these yourself in an
    elevated shell; they are reversible (restore with the balanced GUID)."""
    return (
        "Fix the clock (elevated PowerShell/cmd) -- removes turbo drift:\n"
        "  powercfg /setactive SCHEME_MIN            "
        "# High performance scheme\n"
        "  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR "
        "PROCTHROTTLEMIN 100   # min CPU state 100%\n"
        "  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR "
        "PROCTHROTTLEMAX 100   # max CPU state 100%\n"
        "  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR "
        "PERFBOOSTMODE 0       # disable turbo boost\n"
        "  powercfg /setactive SCHEME_CURRENT\n"
        "Restore:  powercfg /setactive SCHEME_BALANCED\n"
        "(BIOS: additionally disable Turbo/SpeedStep and SMT for best results.)"
    )


def apply_high_performance_power_plan(confirm: bool = False,
                                      verbose: bool = True) -> bool:
    """Apply the fixed-clock power plan via powercfg.  SYSTEM-WIDE, needs admin.

    Refuses unless ``confirm=True``.  Reversible via
    ``powercfg /setactive SCHEME_BALANCED``.  Returns True if all commands ran.
    """
    if not confirm:
        if verbose:
            print("[cpu_pin] refusing to change the system power plan without "
                  "confirm=True. Commands:\n" + power_plan_guide())
        return False
    if not sys.platform.startswith("win"):
        if verbose:
            print("[cpu_pin] power plan helper is Windows-only.")
        return False
    import subprocess
    cmds = [
        ["powercfg", "/setactive", "SCHEME_MIN"],
        ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
         "PROCTHROTTLEMIN", "100"],
        ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
         "PROCTHROTTLEMAX", "100"],
        ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
         "PERFBOOSTMODE", "0"],
        ["powercfg", "/setactive", "SCHEME_CURRENT"],
    ]
    ok = True
    for c in cmds:
        try:
            subprocess.run(c, check=True, capture_output=True)
        except Exception as exc:
            ok = False
            if verbose:
                print("[cpu_pin] '%s' failed: %r (admin shell needed?)"
                      % (" ".join(c), exc))
    return ok
