@echo off
REM Tabella principale: TimesNetModified "completo" (periodo fisso 24 +
REM blocco Inception + reshape 2D) sui 4 orizzonti 96/192/336/720.
REM E' la NOSTRA variante di TimesNet (models/TimesNetModified.py).
REM Da confrontare con:
REM   - le righe ECL128_* (TimesNet TSLib di riferimento, FFT top-k): il
REM     confronto isola l'effetto del periodo fisso 24 vs scoperta FFT;
REM   - le righe ABL_* di run_ablation.bat: isolano Inception e reshape 2D.
REM Config d_model/d_ff 128 + AMP come i run ECL128_*, cosi' il confronto
REM e' ad armi pari. Curve train/val per batch solo su pred_len 96.
REM model_id MOD128_* per non sovrascrivere gli altri run in result\.
REM Uso: run_modified.bat (dalla cartella del progetto)

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
echo FATTO. Metriche in result_long_term_forecast.txt (righe MOD128_*) - curve in result\TimesNetModified\
exit /b 0

:run
echo.
echo ==================== TimesNetModified (fixed24 + inception + 2D)  pred_len=%1 ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id MOD128_96_%1 ^
    --model TimesNetModified ^
    --use_inception 1 ^
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
    echo [ERRORE] TimesNetModified pred_len=%1 fallito, mi fermo qui.
    exit /b 1
)
exit /b 0
