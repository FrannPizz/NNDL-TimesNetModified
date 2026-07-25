@echo off
REM TimesNetModified: le 4 varianti (matrice use_inception x use_2d) su TUTTI
REM e 4 gli orizzonti 96/192/336/720. Rifa' il training da zero.
REM
REM                     use_2d=1            use_2d=0
REM   use_inception=1   MOD128 (completo)  ABL_no2d
REM   use_inception=0   ABL_simpleblock    ABL_simple1d
REM
REM Config d_model/d_ff 128 + AMP come i run ECL128_* (confronto ad armi pari).
REM Curve train/val per batch attive su tutti gli orizzonti.
REM I risultati (curve PNG + CSV) di ogni variante vengono spostati in una
REM cartella dedicata:  result\<VARIANTE>\  (MOD128, ABL_simpleblock, ...).
REM Le metriche finali restano (come sempre) in result_long_term_forecast.txt.
REM Uso: run_modified_all.bat (dalla cartella del progetto)

setlocal enabledelayedexpansion
set "PYTHONUTF8=1"
set "ROOT=%~dp0"
REM venv: prima quello locale del clone; se manca, riusa quello di Progetto NNDL
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=C:\Users\alber\Desktop\Progetto NNDL\Time-Series-Library\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

cd /d "%ROOT%"

REM          variante          use_inception  use_2d
call :variant MOD128           1              1 || exit /b 1
call :variant ABL_simpleblock  0              1 || exit /b 1
call :variant ABL_no2d         1              0 || exit /b 1
call :variant ABL_simple1d     0              0 || exit /b 1

echo.
echo FATTO. Metriche in result_long_term_forecast.txt - curve in result\<VARIANTE>\
exit /b 0

REM ---------------------------------------------------------------
:variant
set "VAR=%1"
set "INC=%2"
set "D2=%3"
if not exist "result\%VAR%" mkdir "result\%VAR%"
for %%P in (96 192 336 720) do (
    call :run "%VAR%" !INC! !D2! %%P || exit /b 1
)
REM sposta le curve/CSV di questa variante nella sua cartella dedicata
if exist "result\TimesNetModified\%VAR%_*" move /Y "result\TimesNetModified\%VAR%_*" "result\%VAR%\" >nul
exit /b 0

REM ---------------------------------------------------------------
:run
echo.
echo ==================== TimesNetModified %~1 (use_inception=%2 use_2d=%3)  pred_len=%4 ====================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id %~1_96_%4 ^
    --model TimesNetModified ^
    --use_inception %2 ^
    --use_2d %3 ^
    --data custom ^
    --features M ^
    --seq_len 96 ^
    --label_len 48 ^
    --pred_len %4 ^
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
    echo [ERRORE] %~1 pred_len=%4 fallito, mi fermo qui.
    exit /b 1
)
exit /b 0
