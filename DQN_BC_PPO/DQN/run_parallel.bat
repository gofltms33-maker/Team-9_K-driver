@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem ─────────────────────────────────────────────────────────────
rem  팀원별 PPO 실험 3개를 창 3개로 동시에 띄운다.
rem    run_parallel.bat A   →  A1 A2 A3  (학습률 0.001)
rem    run_parallel.bat B   →  B1 B2 B3  (학습률 0.0003)
rem    run_parallel.bat C   →  C1 C2 C3  (학습률 0.0001)
rem  감마는 1 = 0.95, 2 = 0.99, 3 = 0.995
rem  결과는 results\<실험이름>_날짜_시각\ 에 저장된다.
rem
rem  Ctrl+C 로 멈춘 실험을 이어서 돌리기:
rem    run_parallel.bat C resume   →  results\C1_*, C2_*, C3_* 중 가장 최근 폴더를 각각 이어서 학습
rem ─────────────────────────────────────────────────────────────

set LR=
rem 실험 하나당 총 학습 스텝 = 저장되는 데이터 행 수 (2048 단위로 올림). 9개 x 1,120,000 ≈ 1.008 x 10^7
set STEPS=1120000

if /I "%1"=="A" set LR=1e-3
if /I "%1"=="B" set LR=3e-4
if /I "%1"=="C" set LR=1e-4
if "%LR%"=="" (
    echo 사용법: run_parallel.bat A ^| B ^| C
    exit /b 1
)
set T=%1

if /I "%2"=="resume" goto RESUME

start "%T%1  lr=%LR% gamma=0.95"  cmd /k python train.py --name %T%1 --lr %LR% --gamma 0.95  --timesteps %STEPS% --threads 1 --record
timeout /t 20 /nobreak >nul
start "%T%2  lr=%LR% gamma=0.99"  cmd /k python train.py --name %T%2 --lr %LR% --gamma 0.99  --timesteps %STEPS% --threads 1 --record
timeout /t 20 /nobreak >nul
start "%T%3  lr=%LR% gamma=0.995" cmd /k python train.py --name %T%3 --lr %LR% --gamma 0.995 --timesteps %STEPS% --threads 1 --record
exit /b 0

:RESUME
for %%n in (1 2 3) do call :RESUME_ONE %%n
exit /b 0

:RESUME_ONE
set D=
for /f "delims=" %%d in ('dir /b /ad /o:n "results\%T%%1_*" 2^>nul') do set D=%%d
if "%D%"=="" (
    echo results\%T%%1_* 폴더가 없습니다.
    exit /b 0
)
start "%T%%1 resume" cmd /k python train.py --resume "results\%D%" --threads 1
timeout /t 20 /nobreak >nul
exit /b 0
