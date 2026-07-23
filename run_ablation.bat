@echo off
REM Studio di ablation su OurTimesNet: 4 varianti, SOLO pred_len=96
REM (prassi standard: l'ablation si fa a un orizzonte fisso, non su tutta
REM la griglia). Tutte con d_model/d_ff 128 + AMP come i run principali.
REM
REM Varianti (la versione "completa" fft+inception+2D e' gia' in run_ours.bat):
REM   1. ABL_fixed24     : periodo FISSO 24 (giornaliero) al posto della scoperta FFT
REM   2. ABL_fixed24_168 : periodi fissi 24+168 (giornaliero + settimanale)
REM   3. ABL_simpleblock : conv 3x3 singola al posto del blocco Inception
REM   4. ABL_no2d        : conv 1D, senza reshape 2D (l'idea centrale del paper)
REM
REM NB: i periodi fissi sono giustificati SOLO dal training set
REM (analysis/train_periodogram.py) + conoscenza di dominio - niente leakage.
REM Ogni variante ha un model_id diverso: i flag non entrano nel nome del
REM checkpoint, senza model_id distinti i run si sovrascriverebbero.
REM Uso: run_ablation.bat (dalla cartella del progetto)

setlocal
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

REM       model_id          period_mode  fixed_periods  block_type  use_2d
call :run ABL_fixed24       fixed        24             inception   1 || exit /b 1
call :run ABL_fixed24_168   fixed        24,168         inception   1 || exit /b 1
call :run ABL_simpleblock   fft          24             simple      1 || exit /b 1
call :run ABL_no2d          fft          24             inception   0 || exit /b 1

echo.
echo FATTO. Metriche in result_long_term_forecast.txt (righe ABL_*) - curve in result\OurTimesNet\
exit /b 0

:run
echo.
echo ==================== OurTimesNet %1  pred_len=96 ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id %1_96_96 ^
    --model OurTimesNet ^
    --period_mode %2 ^
    --fixed_periods %3 ^
    --block_type %4 ^
    --use_2d %5 ^
    --data custom ^
    --features M ^
    --seq_len 96 ^
    --label_len 48 ^
    --pred_len 96 ^
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
    --track_batch_error
if errorlevel 1 (
    echo.
    echo [ERRORE] variante %1 fallita, mi fermo qui.
    exit /b 1
)
exit /b 0
