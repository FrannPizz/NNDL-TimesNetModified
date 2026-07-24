@echo off
REM Studio di ablation su TimesNetModified: 3 varianti, SOLO pred_len=96
REM (prassi standard: l'ablation si fa a un orizzonte fisso, non su tutta
REM la griglia). Tutte con d_model/d_ff 128 + AMP come i run principali.
REM
REM Le 3 varianti + il modello completo (in run_modified.bat) formano la
REM matrice 2x2 use_inception x use_2d, per isolare ogni ingrediente:
REM
REM                     use_2d=1            use_2d=0
REM   use_inception=1   COMPLETO (MOD128)  ABL_no2d
REM   use_inception=0   ABL_simpleblock    ABL_simple1d
REM
REM   ABL_simpleblock : conv 3x3 singola al posto del blocco Inception
REM                     (quanto conta il multi-scala?)
REM   ABL_no2d        : conv 1D, senza reshape 2D
REM                     (l'idea centrale del paper: il 2D serve davvero?)
REM   ABL_simple1d    : entrambe tolte (TimesNet ridotto all'osso)
REM
REM Il periodo resta fisso 24 (hardcoded nel modello, giustificato dal
REM training set in analysis/period_study.py - niente leakage).
REM Ogni variante ha un model_id diverso: i flag non entrano nel nome del
REM checkpoint, senza model_id distinti i run si sovrascriverebbero.
REM Uso: run_ablation.bat (dalla cartella del progetto)

setlocal
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

REM       model_id          use_inception  use_2d
call :run ABL_simpleblock   0              1 || exit /b 1
call :run ABL_no2d          1              0 || exit /b 1
call :run ABL_simple1d      0              0 || exit /b 1

echo.
echo FATTO. Metriche in result_long_term_forecast.txt (righe ABL_*) - curve in result\TimesNetModified\
exit /b 0

:run
echo.
echo ==================== TimesNetModified %1  pred_len=96 ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id %1_96_96 ^
    --model TimesNetModified ^
    --use_inception %2 ^
    --use_2d %3 ^
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
