"""C1~C3 멀티시드 단일프로세스 래퍼 — pythonw(콘솔없음) → 콘솔 종료 이벤트 면역.

★ Windows 멀티프로세싱(DataLoader spawn) 필수: 모든 실행은 `if __name__ == '__main__'` 가드 안에서만.
  (가드 없으면 워커가 모듈 재실행 → main() 재호출 → RuntimeError 크래시 루프 → 창 깜빡임)
pythonw는 sys.stdout/stderr=None → 파일로 재지정(파이썬+fd 레벨) 필수.
"""
import os
import sys
from pathlib import Path


def run():
    os.chdir(r"C:\YangHyunHo\DFire")   # bat 없이 실행되므로 CWD 고정(캐시된 yolo11n.pt 로컬 사용, 재다운로드 방지)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    log = Path(r"C:\YangHyunHo\DFire\runs\c123_detached.log")
    f = open(log, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = f            # pythonw: None → 파일
    sys.stderr = f
    try:
        os.dup2(f.fileno(), 1)   # C 레벨(ultralytics/torch)도 파일
        os.dup2(f.fileno(), 2)
    except Exception:
        pass
    sys.path.insert(0, r"C:\YangHyunHo\DFire\03_scripts")
    print("===== c123_all START (pythonw, guarded) =====", flush=True)
    try:
        import run_c123_multiseed
        run_c123_multiseed.main()
        import finalize_c123
        finalize_c123.main()
        print("===== c123_all END =====", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("===== c123_all ERROR =====", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # spawn 워커 안전
    run()
