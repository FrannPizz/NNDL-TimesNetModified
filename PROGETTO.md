# Progetto NNDL — Forecasting su Electricity con TimesNet

Struttura del progetto e istruzioni per riprodurre gli esperimenti.
Base di codice: [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library),
usata **solo** per l'infrastruttura (data loader, ciclo di training) e per il
modello di riferimento. Il contributo del progetto è `models/TimesNetModified.py`
e gli script in `analysis/`.

## Ruolo di ciascun modello

| File | Ruolo |
|---|---|
| `models/TimesNet.py` | Riferimento TSLib, **non modificato**: riproduce i numeri del paper |
| `models/DLinear.py` | Baseline lineare TSLib, non modificata |
| `models/TimesNetModified.py` | **Nostra variante** di TimesNet: periodo fisso 24 + flag di ablation `--use_inception` / `--use_2d` |
| `analysis/seasonal_naive.py` | Baseline seasonal-naive (nessun training) |

## Flag di ablation di TimesNetModified (aggiunti a `run.py`)

- `--use_inception 1|0` — blocco Inception multi-kernel (paper) vs singola conv 3×3
- `--use_2d 1|0` — reshape 1D→2D per periodo (idea centrale del paper) vs conv 1D lungo il tempo

Il periodo è fisso a 24 (hardcoded in `FIXED_PERIOD`), quindi lo studio del
periodo fisso vs scoperta FFT si fa confrontando `TimesNetModified` (completo)
con `TimesNet` di riferimento (righe `ECL128_*`).

**Nota**: i flag non entrano nel nome del checkpoint → usare sempre un
`--model_id` diverso per variante (gli script `.bat` lo fanno già).

## Scelta del periodo fisso senza leakage

Il periodo 24 è giustificato **prima** di ogni esperimento da:

1. conoscenza di dominio (dati orari → ciclo giornaliero);
2. `analysis/period_study.py`: periodogramma e ACF calcolati **solo sul
   training split** (primo 70% delle righe, identico a `Dataset_Custom`).
   Risultato: picco dominante a 24h (con armoniche 12/8/6h) e massimo locale
   netto a 168h (ampiezza ~830 vs mediana ~120 nella banda 100–300h).
   Figura: `result/analysis/period_study.png`.

Il periodo non è mai stato selezionato o raffinato guardando metriche di
validation/test.

## Esperimenti

Tutti con `seq_len=96`, split 70/10/20, metriche MSE/MAE su dati
standardizzati (stesso protocollo per tutti i modelli).

1. **Già eseguiti**: DLinear ×4 orizzonti; TimesNet riferimento d64 ×4 e
   d128 ×4 (righe `ECL_*` / `ECL128_*` in `result_long_term_forecast.txt`).
2. **Seasonal-naive** (`python analysis/seasonal_naive.py`) — già eseguita,
   righe `SeasonalNaive24`: MSE 0.321 / 0.304 / 0.327 / 0.367.
3. **`run_modified.bat`** — TimesNetModified completo (fixed24 + inception +
   2D, d128) sui 4 orizzonti → tabella principale, righe `MOD128_*`.
4. **`run_ablation.bat`** — 3 varianti a `pred_len=96` (prassi: ablation a
   orizzonte fisso), righe `ABL_*`. Con il modello completo formano la
   matrice 2×2 `use_inception` × `use_2d`:
   - `ABL_simpleblock` (inception=0, 2d=1) — conv 3×3 al posto dell'Inception
   - `ABL_no2d` (inception=1, 2d=0) — senza reshape 2D (idea centrale del paper)
   - `ABL_simple1d` (inception=0, 2d=0) — entrambe tolte (TimesNet all'osso)

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
| `run_modified.bat` | `MOD128_*` | ⬜ da eseguire (GPU) |
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

- ✅ contributo centrale = nostra implementazione (`TimesNetModified.py`), TSLib solo riferimento
- ✅ periodo fisso motivato senza leakage (solo train + dominio)
- ✅ un solo task (long-term forecasting), anomaly detection esclusa
