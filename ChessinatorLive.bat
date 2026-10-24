@echo off
cd /d "%~dp0"
if not exist engine\stockfish.exe python setup_stockfish.py
python analyze.py %*
pause
