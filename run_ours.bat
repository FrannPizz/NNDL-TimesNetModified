@echo off
REM Tabella principale: OurTimesNet "completo" (period_mode=fft, block inception,
REM reshape 2D) sui 4 orizzonti 96/192/336/720. E' la NOSTRA reimplementazione
REM di TimesNet (models/OurTimesNet.py): da confrontare con le righe ECL128_*
REM (TimesNet di riferimento TSLib, stessa config d_model/d_ff 128 + AMP).
REM Curve train/val per batch solo su pred_len 96; gli altri solo MSE/MAE.
REM model_id OURS128_* per non sovrascrivere i run precedenti.
REM Uso: run_ours.bat (dalla cartella del progetto)

setlocal
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

call :run 96  --track_batch_error    || exit /b 1
call :run 192 --no_track_batch_error || exit /b 1
call :run 336 --no_track_batch_error || exit /b 1
call :run 720 --no_track_batch_error || exit /b 1

echo.
echo FATTO. Metriche in result_long_term_forecast.txt (righe OURS128_*) - curve in result\OurTimesNet\
exit /b 0

:run
echo.
echo ==================== OurTimesNet (fft)  pred_len=%1 ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id OURS128_96_%1 ^
    --model OurTimesNet ^
    --period_mode fft ^
    --block_type inception ^
    --use_2d 1 ^
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
    --d_model 128 ^
    --d_ff 128 ^
    --top_k 5 ^
    --des Exp ^
    --itr 1 ^
    --train_epochs 10 ^
    --num_workers 0 ^
    --use_amp ^
    --eval_every 100 ^
    %2
if errorlevel 1 (
    echo.
    echo [ERRORE] OurTimesNet pred_len=%1 fallito, mi fermo qui.
    exit /b 1
)
exit /b 0
