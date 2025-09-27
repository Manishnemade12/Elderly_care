@echo off
echo Starting Fall Detection System...
echo.

echo Starting Python Backend...
start "Fall Detection Backend" cmd /k "cd udaya && fall-detection-env\Scripts\python.exe app.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting React Frontend...
start "Fall Detection Frontend" cmd /k "npm run dev"

echo.
echo System is starting up...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause > nul