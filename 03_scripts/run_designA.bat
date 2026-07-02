@echo off
REM Design A detached runner — 세션과 무관하게 in→naive 학습 후 TRAINING_LOG 기록+push.
REM 실행: powershell Start-Process 로 detached 기동. 로그는 runs\designA_detached.log.
cd /d C:\YangHyunHo\DFire
set PY=C:\Python313\python.exe
echo ===== Design A detached START %DATE% %TIME% ===== >> runs\designA_detached.log
"%PY%" 03_scripts\run_inout.py       >> runs\designA_detached.log 2>&1
echo ===== training done, finalizing %DATE% %TIME% ===== >> runs\designA_detached.log
"%PY%" 03_scripts\finalize_inout.py  >> runs\designA_detached.log 2>&1
echo ===== Design A detached END %DATE% %TIME% ===== >> runs\designA_detached.log
