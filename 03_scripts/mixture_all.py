"""혼합물설계 Phase A 단일프로세스 래퍼 — pythonw(콘솔없음) 실행용.

★ __main__ 가드 필수(spawn 워커 재실행 방지). chdir(CWD 고정, yolo11n.pt 로컬). stdout→파일(pythonw None).
순서: 풀인덱스 없으면 구축 → 러너(21회) → finalize.
"""
import os
import sys
from pathlib import Path


def run():
    os.chdir(r"C:\YangHyunHo\DFire")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    f = open(r"C:\YangHyunHo\DFire\runs\mixture_detached.log", "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = f
    sys.stderr = f
    try:
        os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)
    except Exception:
        pass
    sys.path.insert(0, r"C:\YangHyunHo\DFire\03_scripts")
    print("===== mixture_all START (pythonw, guarded) =====", flush=True)
    try:
        if not Path(r"C:\YangHyunHo\DFire\runs\mixture_pool_index.json").exists():
            import build_mixture_pools
            build_mixture_pools.main()
        import run_mixture_pilot
        run_mixture_pilot.main()
        import finalize_mixture
        finalize_mixture.main()
        print("===== mixture_all END =====", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("===== mixture_all ERROR =====", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run()
