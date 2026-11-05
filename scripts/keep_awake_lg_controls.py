"""Temporary sleep guard for the authorized six-run operator-control queue."""
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.keep_awake_fusion import kernel, is_running, ctypes, PowerStatus
from scripts.prepare_sr_native import read, write

if __name__=='__main__':
    dest=ROOT/'report/sr_lg_operator_controls_20261104'
    info={'helper_pid':os.getpid(),'power_plan_changed':False,'display_required':False}
    try:
        while True:
            state=read(dest/'state.json')
            if not any(is_running(state.get(k)) for k in ('queue_pid','training_pid')):break
            if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError()
            info.update(state='active',checked_at=time.time())
            tmp=dest/'power.tmp';write(tmp,info);os.replace(tmp,dest/'power.json')
            time.sleep(15)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        info.update(state='released',released_at=time.time())
        tmp=dest/'power.tmp';write(tmp,info);os.replace(tmp,dest/'power.json')
