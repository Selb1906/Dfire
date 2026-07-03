@echo off
REM AIHub C3 detached — 멀티시드 종료 후 체이닝. C3 학습+평가 → 4셀 기록+push+작업 자삭제.
cd /d C:\YangHyunHo\DFire
set PY=C:\Python313\python.exe
echo ===== C3 START %DATE% %TIME% ===== >> runs\c3_detached.log
"%PY%" 03_scripts\run_aihub_c3.py >> runs\c3_detached.log 2>&1
echo ===== training done, finalizing %DATE% %TIME% ===== >> runs\c3_detached.log
"%PY%" 03_scripts\finalize_c3.py   >> runs\c3_detached.log 2>&1
echo ===== C3 END %DATE% %TIME% ===== >> runs\c3_detached.log
