@echo off
title VSL Recognition - Nhan Dien Ngon Ngu Ky Hieu Viet Nam
color 0A

echo ==========================================================
echo    VSL RECOGNITION - NHAN DIEN NGON NGU KY HIEU VIET NAM
echo ==========================================================
echo.
echo [*] Dang khoi dong server...
echo [*] Trang web se mo tai http://127.0.0.1:5000
echo.

:: UTF-8
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: Mo trinh duyet
start "" http://127.0.0.1:5000

:: Chay server
if exist ".venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe app.py
) else (
    python app.py
)

pause
