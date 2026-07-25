@echo off
REM Completa le 2 celle mancanti dell'ablation TimesNetModified:
REM   - ABL_simpleblock (use_inception=0, use_2d=1) @ pred_len 720
REM   - ABL_no2d        (use_inception=1, use_2d=0) @ pred_len 192
REM Stessa config dei run principali: d_model/d_ff 128 + AMP, 10 epoche.
REM Curve/CSV spostate nelle cartelle dedicate result\<VARIANTE>\.
REM Uso: run_modified_missing.bat (dalla cartella del progetto)

setlocal enabledelayedexpansion
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=C:\Users\alber\Desktop\Progetto NNDL\Time-Series-Library\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

REM        variante          use_inception  use_2d  pred_len
call :run ABL_simpleblock    0              1       720 || exit /b 1
call :run ABL_no2d           1              0       192 || exit /b 1

echo.
echo FATTO. Celle mancanti completate - curve in result\ABL_simpleblock\ e result\ABL_no2d\
exit /b 0

REM ---------------------------------------------------------------
:run
set "VAR=%~1"
set "INC=%2"
set "D2=%3"
set "PL=%4"
if not exist "result\%VAR%" mkdir "result\%VAR%"
echo.
echo ==================== TimesNetModified %VAR% (use_inception=%INC% use_2d=%D2%)  pred_len=%PL% ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id %VAR%_96_%PL% ^
    --model TimesNetModified ^
    --use_inception %INC% ^
    --use_2d %D2% ^
    --data custom ^
    --features M ^
    --seq_len 96 ^
    --label_len 48 ^
    --pred_len %PL% ^
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
    echo [ERRORE] %VAR% pred_len=%PL% fallito, mi fermo qui.
    exit /b 1
)
if exist "result\TimesNetModified\%VAR%_96_%PL%_*" move /Y "result\TimesNetModified\%VAR%_96_%PL%_*" "result\%VAR%\" >nul
exit /b 0
