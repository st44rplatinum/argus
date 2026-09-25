@echo off
rem Phase labels for the 100-case subset, then the comparison. Resumable.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -u scripts\build_phase_labels.py > data\derived\phase_run.log 2>&1
".venv\Scripts\python.exe" scripts\compare_encoders.py > data\derived\phase_compare.txt 2>&1
