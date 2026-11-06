@echo off
title Vietnamese to Hmong Daw S2S Translation - Ngrok Server
echo ========================================================
echo   He Thong Dich Tieng Viet sang Tieng H'Mong (S2S)
echo   Khoi chay Server FastAPI + Ngrok Public Tunnel
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Dang khoi dong may chu S2S API tai cong 8000...
start /b .\.venv\Scripts\python.exe -m uvicorn src.api.app_stream:app --host 0.0.0.0 --port 8000

echo Cho 5 giay de mo hinh nap san sang...
timeout /t 5 /nobreak >nul

echo [2/2] Dang mo duong truyen ngrok public...
ngrok http 8000
