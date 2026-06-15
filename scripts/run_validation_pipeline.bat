@echo off
REM ---------------------------------------------------------------------------
REM AllowanceMap Phase 4 (E): autonomous fraud-lead validation loop.
REM Refreshes OIG LEIE + CMS Revoked ground truth, rebuilds labels + lead queue,
REM re-runs the temporal backtest, and records the self-eval dashboard.
REM
REM Intended to be fired on a cadence by Windows Task Scheduler (monthly).
REM Self-contained: needs Python (+ pandas/pyarrow) and network access to
REM oig.hhs.gov and data.cms.gov. Does NOT call the Claude API (no key needed).
REM
REM Register (monthly, day 1, 03:00 -- run from repo root):
REM   schtasks /create /tn "AllowanceMap Fraud Validation" ^
REM     /tr "%CD%\scripts\run_validation_pipeline.bat" /sc MONTHLY /d 1 /st 03:00 /f
REM Remove:  schtasks /delete /tn "AllowanceMap Fraud Validation" /f
REM Run now: schtasks /run /tn "AllowanceMap Fraud Validation"
REM ---------------------------------------------------------------------------
setlocal

REM cd to the repo root (this script lives in <repo>\scripts\).
cd /d "%~dp0.."

REM Prefer the python on PATH; override by setting ALLOWANCEMAP_PYTHON.
set "PY=%ALLOWANCEMAP_PYTHON%"
if "%PY%"=="" set "PY=python"

set "LOGDIR=local_pipeline\anomaly\validation"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo [%date% %time%] starting validation pipeline >> "%LOGDIR%\pipeline_cron.log"
"%PY%" -m anomaly.run_pipeline >> "%LOGDIR%\pipeline_cron.log" 2>&1
echo [%date% %time%] pipeline exit code %ERRORLEVEL% >> "%LOGDIR%\pipeline_cron.log"

endlocal
