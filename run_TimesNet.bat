@echo off
REM 4 run: TimesNet x pred_len 96/192/336/720, config leggera.
REM d_model 64 / d_ff 64 + AMP per stare nei tempi su RTX 4060 (dichiararlo nel report).
REM Curve train/val per batch solo su pred_len 96; gli altri solo MSE/MAE.
REM Uso: run_TimesNet.bat (doppio click o da cmd, dalla cartella Time-Series-Library)

setlocal
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

call :run 96  TimesNet --track_batch_error    || exit /b 1
call :run 192 TimesNet --no_track_batch_error || exit /b 1
call :run 336 TimesNet --no_track_batch_error || exit /b 1
call :run 720 TimesNet --no_track_batch_error || exit /b 1

echo.
echo FATTO. Metriche in result_long_term_forecast.txt - curve in result\
exit /b 0

:run
echo.
echo ==================== %2  pred_len=%1 ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id ECL_96_%1 ^
    --model %2 ^
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
    --d_model 64 ^
    --d_ff 64 ^
    --top_k 5 ^
    --des Exp ^
    --itr 1 ^
    --train_epochs 10 ^
    --num_workers 0 ^
    --use_amp ^
    --eval_every 100 ^
    %3
if errorlevel 1 (
    echo.
    echo [ERRORE] %2 pred_len=%1 fallito, mi fermo qui.
    exit /b 1
)
exit /b 0
