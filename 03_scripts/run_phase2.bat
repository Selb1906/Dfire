@echo off
REM Phase 2 detached — Design A 종료 후 자동 체이닝: X(크로스평가) -> T1(전이학습) -> 기록+push.
cd /d C:\YangHyunHo\DFire
set PY=C:\Python313\python.exe
echo ===== Phase2 START %DATE% %TIME% ===== >> runs\phase2_detached.log
"%PY%" 03_scripts\eval_dfull_on_aihub.py >> runs\phase2_detached.log 2>&1
echo ===== X done, transfer start %DATE% %TIME% ===== >> runs\phase2_detached.log
"%PY%" 03_scripts\run_transfer.py         >> runs\phase2_detached.log 2>&1
echo ===== T1 done, finalizing %DATE% %TIME% ===== >> runs\phase2_detached.log
"%PY%" 03_scripts\finalize_phase2.py      >> runs\phase2_detached.log 2>&1
echo ===== Phase2 END %DATE% %TIME% ===== >> runs\phase2_detached.log
