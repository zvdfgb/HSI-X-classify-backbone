"""Temporary, verified sleep protection for the already-running fusion queue.

No power-plan or training changes. The OS releases this thread's request on
exit, and this helper exits after both the queue and its training child exit.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "report/fusion_papers/fusion_papers_seed6_20261031"
STATUS = BATCH / "运行状态.json"
EVIDENCE = BATCH / "power_protection.json"
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
kernel.SetThreadExecutionState.restype = wintypes.DWORD
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.CloseHandle.argtypes = [wintypes.HANDLE]


class PowerStatus(ctypes.Structure):
    _fields_ = [("ac", wintypes.BYTE), ("battery_flags", wintypes.BYTE),
                ("battery_percent", wintypes.BYTE), ("system_flags", wintypes.BYTE),
                ("battery_lifetime", wintypes.DWORD), ("battery_full_lifetime", wintypes.DWORD)]


kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(PowerStatus)]
kernel.GetSystemPowerStatus.restype = wintypes.BOOL


def write(info):
    temp = EVIDENCE.with_suffix(".tmp")
    temp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, EVIDENCE)


def is_running(pid):
    if not pid:
        return False
    handle = kernel.OpenProcess(0x00100000, False, int(pid))
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    info = {"helper_pid": os.getpid(), "scope": "temporary idle sleep/hibernate prevention during this training batch",
            "power_plan_changed": False, "display_required": False,
            "started_at": datetime.now().astimezone().isoformat()}
    try:
        while True:
            state = json.loads(STATUS.read_text(encoding="utf-8"))
            if not (is_running(state.get("queue_pid")) or is_running(state.get("training_pid"))):
                break
            previous = kernel.SetThreadExecutionState(0x80000001)
            if previous == 0:
                raise ctypes.WinError(ctypes.get_last_error())
            supply = PowerStatus()
            valid = kernel.GetSystemPowerStatus(ctypes.byref(supply))
            info.update(state="active", native_request_succeeded=True,
                        ac_connected=supply.ac == 1 if valid else None,
                        battery_percent=int(supply.battery_percent) if valid else None,
                        checked_at=datetime.now().astimezone().isoformat())
            write(info)
            time.sleep(30)
    except BaseException as exc:
        info.update(state="failed", error=repr(exc))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        if info.get("state") != "failed":
            info["state"] = "released"
        info["updated_at"] = datetime.now().astimezone().isoformat()
        write(info)
