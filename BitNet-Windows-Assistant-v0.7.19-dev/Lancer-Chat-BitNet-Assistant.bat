@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

rem Add common Visual Studio Build Tools paths to this process only, when present.
set "VSROOT=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
if exist "%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
if exist "%VSROOT%\VC\Tools\Llvm\x64\bin\clang++.exe" set "PATH=%VSROOT%\VC\Tools\Llvm\x64\bin;%PATH%"
if exist "%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

echo Starting BitNet Windows Assistant - Chat + RAG UI...
echo Chat local via llama-completion (recommended) or compatible llama-cli. Local PDF/text RAG, document summaries and opt-in Web search are available; OCR is not implemented yet.
echo.
where python >nul 2>nul
if not errorlevel 1 goto run_python
where py >nul 2>nul
if not errorlevel 1 goto run_py_launcher
echo Python was not found. Install Python 3.10+ with Tkinter or use a configured launcher.
pause
exit /b 1

:run_python
python -m bitnet_windows.chat_gui
if errorlevel 1 goto error
endlocal
exit /b 0

:run_py_launcher
py -3 -m bitnet_windows.chat_gui
if errorlevel 1 goto error
endlocal
exit /b 0

:error
echo.
echo The chat UI stopped with an error. Check the output above.
pause
endlocal
exit /b 1
