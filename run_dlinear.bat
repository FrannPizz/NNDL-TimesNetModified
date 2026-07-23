@echo off
REM Lancia i 4 esperimenti DLinear su ECL (pred_len 96/192/336/720) in sequenza.
REM Riproduce le righe ECL_*_DLinear di result_long_term_forecast.txt.
REM (Spostato qui da scripts\DLinear_ECL_all.bat quando la cartella scripts,
REM  che conteneva solo lanciatori ridondanti o .sh Linux di TSLib, e' stata rimossa.)
REM Uso: run_dlinear.bat (dalla cartella del progetto, usa il venv se esiste)

setlocal
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

call :run 96   || exit /b 1
call :run 192  || exit /b 1
call :run 336  || exit /b 1
call :run 720  || exit /b 1

echo.
echo Tutti e 4 gli esperimenti completati.
echo Metriche in result_long_term_forecast.txt - curve di errore in result\DLinear\
exit /b 0

:run
echo.
echo ======================================================
echo   DLinear ECL_96_%1  -  pred_len %1
echo ======================================================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id ECL_96_%1 ^
    --model DLinear ^
    --data custom ^
    --features M ^
    --seq_len 96 ^
    --label_len 48 ^
    --pred_len %1 ^
    --e_layers 2 ^
    --d_layers 1 ^
    --factor 3 ^
    --enc_in 321 ^
    --dec_in 321 ^
    --c_out 321 ^
    --des Exp ^
    --itr 1 ^
    --num_workers 0 ^
    --track_batch_error ^
    --eval_every 100
if errorlevel 1 (
    echo.
    echo [ERRORE] ECL_96_%1 fallito, mi fermo qui.
    exit /b 1
)
exit /b 0
