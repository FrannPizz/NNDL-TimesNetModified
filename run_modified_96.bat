@echo off
REM TimesNetModified: TUTTE e 4 le opzioni (matrice use_inception x use_2d),
REM su UN SOLO orizzonte pred_len=96 -> il minimo per confrontarle.
REM Confrontare su 1 orizzonte e' prassi standard per l'ablation (vedi anche
REM run_ablation.bat) e sufficiente per il report.
REM
REM                     use_2d=1            use_2d=0
REM   use_inception=1   COMPLETO (MOD128)  ABL_no2d
REM   use_inception=0   ABL_simpleblock    ABL_simple1d
REM
REM   COMPLETO        : periodo fisso 24 + blocco Inception + reshape 2D (giorni x ore)
REM   ABL_simpleblock : conv 3x3 singola al posto del blocco Inception
REM   ABL_no2d        : conv 1D, senza reshape 2D
REM   ABL_simple1d    : entrambe tolte (ridotto all'osso)
REM
REM Config d_model/d_ff 128 + AMP come i run ECL128_* (TimesNet TSLib di
REM riferimento) -> confronto ad armi pari. Periodo 24 hardcoded nel modello
REM (giustificato da analysis/period_study.py sul train set, niente leakage).
REM Curve train/val per batch attive (siamo su un solo orizzonte).
REM model_id distinti: i flag non entrano nel nome del checkpoint, senza
REM model_id diversi i run si sovrascriverebbero.
REM Uso: run_modified_96.bat (dalla cartella del progetto)

setlocal
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
REM venv: prima quello locale del clone; se manca (non versionato su git),
REM riusa quello gia' pronto della cartella Progetto NNDL\Time-Series-Library.
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=C:\Users\alber\Desktop\Progetto NNDL\Time-Series-Library\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

REM       model_id          use_inception  use_2d
call :run MOD128            1              1 || exit /b 1
call :run ABL_simpleblock   0              1 || exit /b 1
call :run ABL_no2d          1              0 || exit /b 1
call :run ABL_simple1d      0              0 || exit /b 1

echo.
echo FATTO. Metriche in result_long_term_forecast.txt (righe MOD128_/ABL_*) - curve in result\TimesNetModified\
exit /b 0

:run
echo.
echo ==================== TimesNetModified %1 (use_inception=%2 use_2d=%3)  pred_len=96 ====================
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
