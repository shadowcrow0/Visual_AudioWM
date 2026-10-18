@echo off
REM 把 PyMC（lnrm 擬合要用）裝進 PsychoPy 獨立版的 user-site，不需要管理員。
REM 第一個參數可指定 adaptiveSFT fork 的路徑；預設是 Desktop 上那份。
set "PY=C:\Program Files\PsychoPy\python.exe"
set "FORK=%~1"
if "%FORK%"=="" set "FORK=%USERPROFILE%\OneDrive - University of Texas at San Antonio\Desktop\adaptiveSFT"
if not exist "%PY%" (echo 找不到 %PY% & exit /b 1)
if not exist "%FORK%\adaptivesft\psi.py" (echo 找不到 fork：%FORK% & exit /b 1)
"%PY%" -m pip install --user pymc numba arviz nutpie || exit /b 1
"%PY%" -m pip install --user --no-deps -e "%FORK%" || exit /b 1
"%PY%" -c "import pymc, pytensor, numba, arviz, nutpie, adaptivesft; print('OK pymc', pymc.__version__, 'nutpie', nutpie.__version__)"
echo.
echo 裝好了。Coder 開 VAWM_calibrate.py 直接跑（METHOD = "lnrm"）。
pause
