"""Temporary native power request while this approved queue/child is alive."""
import sys
import time
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.keep_awake_fusion import kernel, ctypes, PowerStatus, is_running
from scripts.prepare_sr_native import read, write


if __name__ == "__main__":
    destination = ROOT / "report/fusion_papers/sr_lg_prior_seed678_20261103"
    info = {"helper_pid": os.getpid(), "power_plan_changed": False, "display_required": False,
            "scope": "temporary idle-sleep protection while queue or training child is alive"}
    try:
        while True:
            state = read(destination / "运行状态.json")
            if not any(is_running(state.get(k)) for k in ("queue_pid", "training_pid")):
                break
            if not kernel.SetThreadExecutionState(0x80000001):
                raise ctypes.WinError(ctypes.get_last_error())
            supply = PowerStatus(); valid = kernel.GetSystemPowerStatus(ctypes.byref(supply))
            info.update(state="active", native_request_succeeded=True, queue_pid=state.get("queue_pid"),
                training_pid=state.get("training_pid"), ac_connected=supply.ac==1 if valid else None,
                battery_percent=int(supply.battery_percent) if valid else None,
                checked_at=datetime.now().astimezone().isoformat())
            write(destination / "power_protection.tmp", info)
            os.replace(destination / "power_protection.tmp", destination / "power_protection.json")
            time.sleep(20)
    except BaseException as exc:
        info.update(state="failed", error=repr(exc)); raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        if info.get("state") != "failed": info["state"] = "released"
        info["updated_at"] = datetime.now().astimezone().isoformat()
        write(destination / "power_protection.json", info)
