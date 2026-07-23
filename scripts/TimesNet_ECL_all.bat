@echo off
REM Lancia i 4 esperimenti TimesNet su ECL (pred_len 96/192/336/720) in sequenza.
REM Equivalente Windows di scripts\long_term_forecast\ECL_script\TimesNet.sh
REM Uso: scripts\TimesNet_ECL_all.bat (da qualunque cartella, usa il venv del progetto)
REM NB: chiudi eventuali PDF aperti da test_results\ prima di lanciare, o il run
REM     fallira' con PermissionError quando prova a sovrascriverli.

setlocal
set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"

cd /d "%ROOT%"

call :run 96   || exit /b 1
call :run 192  || exit /b 1
call :run 336  || exit /b 1
call :run 720  || exit /b 1

echo.
echo Tutti e 4 gli esperimenti completati.
echo Metriche in result_long_term_forecast.txt - curve di errore in result\TimesNet\
exit /b 0

:run
echo.
echo ======================================================
echo   TimesNet ECL_96_%1  -  pred_len %1
echo ======================================================
"%PYTHON%" -u run.py ^
    --task_name long_term_forecast ^
    --is_training 1 ^
    --root_path ./dataset/electricity/ ^
    --data_path electricity.csv ^
    --model_id ECL_96_%1 ^
    --model TimesNet ^
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
    --d_model 256 ^
    --d_ff 512 ^
    --top_k 5 ^
    --des Exp ^
    --itr 1 ^
    --num_workers 0
if errorlevel 1 (
    echo.
    echo [ERRORE] ECL_96_%1 fallito, mi fermo qui.
    exit /b 1
)
exit /b 0
