@echo off
REM AIHub 멀티시드 detached — seed 1·2 재학습 + 5셋 평가 → mean±std 기록+push+작업 자삭제.
cd /d C:\YangHyunHo\DFire
set PY=C:\Python313\python.exe
echo ===== MSeed START %DATE% %TIME% ===== >> runs\multiseed_detached.log
"%PY%" 03_scripts\run_aihub_multiseed.py >> runs\multiseed_detached.log 2>&1
echo ===== training done, finalizing %DATE% %TIME% ===== >> runs\multiseed_detached.log
"%PY%" 03_scripts\finalize_multiseed.py  >> runs\multiseed_detached.log 2>&1
echo ===== MSeed END %DATE% %TIME% ===== >> runs\multiseed_detached.log
