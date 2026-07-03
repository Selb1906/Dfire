@echo off
REM IFireSmoke detached — C3 종료 후 체이닝. 학습+평가 → §5.9 기록+push+작업 자삭제.
cd /d C:\YangHyunHo\DFire
set PY=C:\Python313\python.exe
echo ===== IFS START %DATE% %TIME% ===== >> runs\ifs_detached.log
"%PY%" 03_scripts\run_ifiresmoke.py     >> runs\ifs_detached.log 2>&1
echo ===== training done, finalizing %DATE% %TIME% ===== >> runs\ifs_detached.log
"%PY%" 03_scripts\finalize_ifiresmoke.py >> runs\ifs_detached.log 2>&1
echo ===== IFS END %DATE% %TIME% ===== >> runs\ifs_detached.log
