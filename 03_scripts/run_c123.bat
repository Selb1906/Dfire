@echo off
REM C1~C3 멀티시드 detached — seed1·2 재학습(seed0 재사용) + val/held-out 평가 → 4셀통일 기록+push+자삭제.
cd /d C:\YangHyunHo\DFire
set PY=C:\Python313\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
echo ===== C123 START %DATE% %TIME% ===== >> runs\c123_detached.log
"%PY%" 03_scripts\run_c123_multiseed.py >> runs\c123_detached.log 2>&1
echo ===== training done, finalizing %DATE% %TIME% ===== >> runs\c123_detached.log
"%PY%" 03_scripts\finalize_c123.py       >> runs\c123_detached.log 2>&1
echo ===== C123 END %DATE% %TIME% ===== >> runs\c123_detached.log
