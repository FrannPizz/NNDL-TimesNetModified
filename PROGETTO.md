# Progetto NNDL — Forecasting su Electricity con TimesNet

Struttura del progetto e istruzioni per riprodurre gli esperimenti.
Base di codice: [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library),
usata **solo** per l'infrastruttura (data loader, ciclo di training) e per il
modello di riferimento. Il contributo del progetto è `models/OurTimesNet.py`
e gli script in `analysis/`.

## Ruolo di ciascun modello

| File | Ruolo |
|---|---|
| `models/TimesNet.py` | Riferimento TSLib, **non modificato**: riproduce i numeri del paper |
| `models/DLinear.py` | Baseline lineare TSLib, non modificata |
| `models/OurTimesNet.py` | **Nostra reimplementazione** di TimesNet, parametrizzata per le ablation |
| `analysis/seasonal_naive.py` | Baseline seasonal-naive (nessun training) |
| `models/TimesNetFixedPeriod.py` | Prototipo iniziale (periodo 24 hardcoded), superato da `OurTimesNet --period_mode fixed`; tenuto solo come storico |

## Flag di OurTimesNet (aggiunti a `run.py`)

- `--period_mode fft|fixed` — scoperta dei periodi via FFT (paper) vs periodi imposti
- `--fixed_periods 24` / `24,168` — periodi imposti (in ore)
- `--block_type inception|simple` — blocco multi-kernel (paper) vs singola conv 3×3
- `--use_2d 1|0` — reshape 2D per periodo (idea centrale del paper) vs conv 1D

**Nota**: i flag non entrano nel nome del checkpoint → usare sempre un
`--model_id` diverso per variante (gli script `.bat` lo fanno già).

## Scelta dei periodi fissi senza leakage

I periodi 24 e 168 sono giustificati **prima** di ogni esperimento da:

1. conoscenza di dominio (dati orari → ciclo giornaliero e settimanale);
2. `analysis/train_periodogram.py`: periodogramma e ACF calcolati **solo sul
   training split** (primo 70% delle righe, identico a `Dataset_Custom`).
   Risultato: picco dominante a 24h (con armoniche 12/8/6h) e massimo locale
   netto a 168h (ampiezza ~830 vs mediana ~120 nella banda 100–300h).
   Figura: `result/analysis/train_periodogram_acf.png`.

I periodi non sono mai stati selezionati o raffinati guardando metriche di
validation/test.

## Esperimenti

Tutti con `seq_len=96`, split 70/10/20, metriche MSE/MAE su dati
standardizzati (stesso protocollo per tutti i modelli).

1. **Già eseguiti**: DLinear ×4 orizzonti; TimesNet riferimento d64 ×4 e
   d128 ×4 (righe `ECL_*` / `ECL128_*` in `result_long_term_forecast.txt`).
2. **Seasonal-naive** (`python analysis/seasonal_naive.py`) — già eseguita,
   righe `SeasonalNaive24`: MSE 0.321 / 0.304 / 0.327 / 0.367.
3. **`run_ours.bat`** — OurTimesNet completo (fft+inception+2D, d128) sui 4
   orizzonti → tabella principale, righe `OURS128_*`.
4. **`run_ablation.bat`** — 4 varianti a `pred_len=96` (prassi: ablation a
   orizzonte fisso), righe `ABL_*`:
   - `ABL_fixed24` — periodo fisso giornaliero
   - `ABL_fixed24_168` — giornaliero + settimanale
   - `ABL_simpleblock` — conv 3×3 al posto dell'Inception block
   - `ABL_no2d` — senza reshape 2D (testa l'idea centrale del paper)

Ordine consigliato: 3 poi 4 (run notturni; config d128+AMP come i run già
fatti, così il confronto con `ECL128_*` è ad armi pari).

## Mappa script → righe dei risultati

Ogni riga di `result_long_term_forecast.txt` è riproducibile da esattamente
uno script alla radice del progetto:

| Script | Righe prodotte | Stato |
|---|---|---|
| `run_dlinear.bat` | `ECL_*_DLinear` | ✅ eseguito |
| `run_TimesNet.bat` | `ECL_*_TimesNet` (d_model 64, primo tentativo) | ✅ eseguito |
| `run_128_night.bat` | `ECL128_*` (TimesNet riferimento, d_model 128) | ✅ eseguito |
| `python analysis/seasonal_naive.py` | `SeasonalNaive24` | ✅ eseguito |
| `run_ours.bat` | `OURS128_*` | ⬜ da eseguire (GPU) |
| `run_ablation.bat` | `ABL_*` | ⬜ da eseguire (GPU) |

## Pulizia del repo (23 lug 2026)

Rimossi i file TSLib mai usati dal nostro flusso: `scripts/` (lanciatori
ridondanti e `.sh` Linux; `DLinear_ECL_all.bat` è stato spostato alla radice
come `run_dlinear.bat`), `utils/ADFtest.py`, `utils/dtw.py`,
`utils/losses.py`, `utils/m4_summary.py`, `utils/masking.py` (servivano per
benchmark M4, modelli ad attention e metrica DTW, tutti fuori dal nostro
scope). Nota: `utils/augmentation.py` importa `utils.dtw` dentro alcune
funzioni di augmentation DTW-based — innocuo con `--augmentation_ratio 0`
(il nostro caso), da ripristinare solo se si attivassero quelle augmentation.

## Requisiti risposta del docente (recap)

- ✅ contributo centrale = nostra implementazione (`OurTimesNet.py`), TSLib solo riferimento
- ✅ periodi fissi motivati senza leakage (solo train + dominio)
- ✅ un solo task (long-term forecasting), anomaly detection esclusa
