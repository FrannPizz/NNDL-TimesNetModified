# Long-Term Forecasting on Electricity: TimesNet with a Fixed Period

**Final project — Neural Networks and Deep Learning**
Authors: Filippo Facco, Francesco Pizzato

This project studies [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) on
multivariate **long-term forecasting** over the **Electricity (ECL)** dataset
(321 hourly series). Our contribution is `models/TimesNetModified.py`: a rewrite of
the TimesBlock in which the period is **not** discovered by an FFT at every forward
pass, but **fixed to 24 hours**, justified a priori on the training split alone. On
top of this variant we build a 2×2 ablation that isolates the two architectural
ingredients of the paper: the multi-kernel Inception block and the 1D→2D reshape.

The surrounding infrastructure (data loader, training loop, evaluation) comes from
[Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML,
used as the code base and as the reference implementation we compare against.

Experimental protocol and results are reported and discussed in the project report;
this file documents the code and how to reproduce every run.

---

## 1. What is ours and what comes from TSLib

| File | Origin | Role |
|---|---|---|
| `models/TimesNetModified.py` | **ours** | The variant under study: fixed period 24 + `--use_inception` / `--use_2d` flags |
| `analysis/period_study.py` | **ours** | Selects the period (periodogram + ACF on the training split only) |
| `analysis/seasonal_naive.py` | **ours** | Seasonal-naive baseline, no training involved |
| `run.py` | TSLib + our patches | Added `--use_inception`, `--use_2d`, `--track_batch_error`, `--eval_every` |
| `exp/exp_long_term_forecasting.py` | TSLib + our patches | Added val/test evaluation every *N* training batches |
| `models/TimesNet.py`, `models/DLinear.py` | TSLib, **unmodified** | Reference model and linear baseline |
| `layers/`, `data_provider/`, `utils/` | TSLib | Embedding, 2D Inception block, data loader, metrics |

A single task is addressed: `long_term_forecast`. Imputation, anomaly detection and
classification were left untouched.

## 2. The model: `TimesNetModified`

Three changes to the original TimesBlock, all marked `# MODIFY` in the source.

**MODIFY 1 — fixed period.** TimesNet runs an FFT on every batch, keeps the `top_k`
dominant periods and averages the resulting branches weighted by spectral amplitude.
Here the period is a constant, `FIXED_PERIOD = 24` (`models/TimesNetModified.py:38`).
Consequences: a single branch instead of `top_k`, no FFT in the forward pass, and no
padding at all (every sequence length we use — 192, 288, 432, 816 — is a multiple
of 24).

**MODIFY 2 — `--use_inception {0,1}`.** With `1` the model uses the paper's
multi-kernel Inception block (6 kernels, from 1×1 to 11×11); with `0`, a single 3×3
convolution. This isolates the contribution of multi-scale kernels.

**MODIFY 3 — `--use_2d {0,1}`.** With `1` the sequence is folded into a
`(days × hours)` "image" and convolved in 2D — the core idea of TimesNet. With `0`
the model stays in 1D along time, using `InceptionBlock1D` (same multi-kernel scheme,
`Conv1d` instead of `Conv2d`). This isolates the contribution of the reshape.

The folding is:

```python
out.reshape(B, T // period, period, N).permute(0, 3, 1, 2)   # T = seq_len + pred_len
```

so **rows = days** (*inter*-period variation) and **columns = hours**
(*intra*-period variation). A daily pattern therefore shows up as vertical stripes.

The four combinations of the two flags form the ablation matrix:

|  | `use_2d=1` | `use_2d=0` |
|---|---|---|
| **`use_inception=1`** | `MOD128` — full model | `ABL_no2d` |
| **`use_inception=0`** | `ABL_simpleblock` | `ABL_simple1d` |

> ⚠️ The two flags **do not appear in the checkpoint name**: runs sharing the same
> `--model_id` overwrite each other. Always pass a distinct `--model_id` per variant
> (the commands in section 5 already do).

## 3. Choosing the period without leakage

The value 24 is fixed **before** any experiment, based on:

1. **domain knowledge** — hourly sampling, electricity consumption follows a daily
   cycle;
2. **`analysis/period_study.py`** — periodogram and ACF computed **on the training
   split only** (first 70% of the rows, the same partition used by `Dataset_Custom`),
   with the scaler fitted on training data alone. Result: a dominant peak at **24 h**
   with harmonics at 12/8/6 h, and a clear local maximum at 168 h (amplitude ≈830
   against a median ≈120 in the 100–300 h band). Figure:
   `result/analysis/period_study.png`.

The period was never selected or refined by looking at validation or test metrics.

## 4. Setup

Requires **Python 3.11** and an NVIDIA GPU (the runs work on CPU, but are impractically
slow).

```bash
py -3.11 -m venv .venv
```

```bash
.venv\Scripts\activate
```

PyTorch must be installed **separately** from the CUDA channel — it is deliberately
not listed in `requirements.txt`, to avoid pinning the CPU-only build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements.txt
```

The dataset is not included in the repository. Download `electricity.csv` from the
[official TSLib link](https://github.com/thuml/Time-Series-Library#usage) (Google
Drive / Tsinghua Cloud) and place it at:

```
dataset/electricity/electricity.csv
```

> On Windows, set the environment variable `PYTHONUTF8=1` before running: without it,
> logging some characters can raise an encoding error mid-run.

## 5. Reproducing the experiments

Every command below is a complete, self-contained invocation: copy it, run it from
the project root with the virtual environment active, and it reproduces exactly one
line of `result_long_term_forecast.txt`. Nothing is hidden in wrapper scripts.

`--top_k` is **ignored** by `TimesNetModified` (its period is fixed). It is kept in
those commands only so the comparison against `TimesNet` stays readable side by side.

### 5.1 Seasonal-naive baseline — no training, ~10 s

The scripts under `analysis/` use relative paths (`../dataset/...`), so they must be
run from inside that folder:

```bash
cd analysis
```

```bash
python seasonal_naive.py
```

Produces the `SeasonalNaive24` lines in
`result/analysis/result_long_term_forecast.txt`. Return to the project root with
`cd ..` before running anything else.

### 5.2 Period study — no training, ~1 min

```bash
python period_study.py
```

Run from inside `analysis/` as above. Prints the 5 dominant periods of the training
split and saves `result/analysis/period_study.png`.

### 5.3 DLinear — linear baseline

TSLib default capacity (`d_model 512`, `d_ff 2048`), no mixed precision.

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_96_96 --model DLinear --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des Exp --itr 1 --num_workers 0 --track_batch_error --eval_every 100
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_96_192 --model DLinear --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des Exp --itr 1 --num_workers 0 --track_batch_error --eval_every 100
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_96_336 --model DLinear --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des Exp --itr 1 --num_workers 0 --track_batch_error --eval_every 100
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_96_720 --model DLinear --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des Exp --itr 1 --num_workers 0 --track_batch_error --eval_every 100
```

### 5.5 TimesNet, reference — `d_model 128`

The `ECL128_*` lines: the actual comparison target for our variant, same capacity and
same mixed-precision setting. About 4 hours in total on an RTX 4060 Laptop.

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL128_96_96 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL128_96_192 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --no_track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL128_96_336 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --no_track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL128_96_720 --model TimesNet --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --no_track_batch_error
```

### 5.6 TimesNetModified — full model (`use_inception=1`, `use_2d=1`)

The `MOD128_*` lines: fixed period 24 + Inception + 2D reshape.

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id MOD128_96_96 --model TimesNetModified --use_inception 1 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id MOD128_96_192 --model TimesNetModified --use_inception 1 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id MOD128_96_336 --model TimesNetModified --use_inception 1 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id MOD128_96_720 --model TimesNetModified --use_inception 1 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

### 5.7 Ablation — `ABL_simpleblock` (`use_inception=0`, `use_2d=1`)

Single 3×3 convolution instead of the Inception block: how much does multi-scale
matter?

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simpleblock_96_96 --model TimesNetModified --use_inception 0 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simpleblock_96_192 --model TimesNetModified --use_inception 0 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simpleblock_96_336 --model TimesNetModified --use_inception 0 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simpleblock_96_720 --model TimesNetModified --use_inception 0 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

### 5.8 Ablation — `ABL_no2d` (`use_inception=1`, `use_2d=0`)

1D convolution, no 2D reshape: is the core idea of the paper doing the work?

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_no2d_96_96 --model TimesNetModified --use_inception 1 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_no2d_96_192 --model TimesNetModified --use_inception 1 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_no2d_96_336 --model TimesNetModified --use_inception 1 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_no2d_96_720 --model TimesNetModified --use_inception 1 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

### 5.9 Ablation — `ABL_simple1d` (`use_inception=0`, `use_2d=0`)

Both ingredients removed: TimesNet stripped to the bone.

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simple1d_96_96 --model TimesNetModified --use_inception 0 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simple1d_96_192 --model TimesNetModified --use_inception 0 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simple1d_96_336 --model TimesNetModified --use_inception 0 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

```bash
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ABL_simple1d_96_720 --model TimesNetModified --use_inception 0 --use_2d 0 --data custom --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --train_epochs 10 --num_workers 0 --use_amp --eval_every 100 --track_batch_error
```

### 5.10 Testing only, reusing a trained checkpoint

With `--is_training 0` training is skipped and `checkpoints/<setting>/checkpoint.pth`
is loaded instead. Every other argument must match the training run **exactly**, since
the checkpoint folder name is built from them:

```bash
python -u run.py --task_name long_term_forecast --is_training 0 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id MOD128_96_96 --model TimesNetModified --use_inception 1 --use_2d 1 --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --d_model 128 --d_ff 128 --top_k 5 --des Exp --itr 1 --num_workers 0
```

### 5.11 Arguments we added to `run.py`

| Flag | Default | Effect |
|---|---|---|
| `--use_inception {0,1}` | `0` | Multi-kernel Inception block vs single 3×3 convolution (`TimesNetModified` only) |
| `--use_2d {0,1}` | `1` | 1D→2D reshape by period vs 1D convolution along time (`TimesNetModified` only) |
| `--track_batch_error` | on | Evaluate val/test every `--eval_every` batches, save a PNG curve and a CSV |
| `--no_track_batch_error` | — | Disable the tracking (faster runs) |
| `--eval_every N` | `100` | How many training batches between evaluations |
| `--no_use_gpu` | — | Force CPU |

Everything else is a standard TSLib argument. Relevant defaults left untouched:
Adam with learning rate `1e-4`, batch size 32, 10 epochs, early stopping with
`patience=3`, `type1` learning-rate schedule, MSE loss.

## 6. Outputs

| Path | Content |
|---|---|
| `result_long_term_forecast.txt` | **Final MSE/MAE of every run**, appended. This is the file the report's tables come from |
| `result/analysis/result_long_term_forecast.txt` | Metrics of the training-free baselines (seasonal-naive) |
| `result/<VARIANT>/*_batch_errors.csv` | Train/val/test error every 100 batches |
| `result/<VARIANT>/*_batch_error_curve.png` | The same data as a plot |
| `result/analysis/*.png` | Analysis figures used in the report |
| `checkpoints/<setting>/checkpoint.pth` | Weights of the best model on validation |
| `test_results/<setting>/*.pdf` | Prediction vs ground-truth plots per window (auto-generated by TSLib) |

`checkpoints/`, `test_results/` and `dataset/` are **not versioned**: they are
regenerated by the commands above.

## 7. Repository structure

```
.
├── README.md                    <- this file
├── LICENSE                      <- MIT, THUML @ Tsinghua University
├── requirements.txt
├── run.py                       <- entry point (TSLib + our flags)
├── models/
│   ├── TimesNetModified.py      <- THE CONTRIBUTION OF THIS PROJECT
│   ├── TimesNet.py              <- TSLib reference, unmodified
│   └── DLinear.py               <- TSLib baseline, unmodified
├── analysis/
│   ├── period_study.py          <- period selection (training split only, no leakage)
│   └── seasonal_naive.py        <- training-free baseline
├── layers/                      <- embedding + 2D Inception block (TSLib)
├── data_provider/               <- Dataset_Custom, splits and scaler (TSLib)
├── exp/                         <- training/testing loop (TSLib + our patch)
├── utils/                       <- metrics, early stopping, time features (TSLib)
├── result/                      <- error curves, CSVs and figures
└── result_long_term_forecast.txt
```

## 8. Credits

The code base is [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
(THUML @ Tsinghua University), released under the MIT license — see `LICENSE`. The
reference model is described in *TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis* (Wu et al., ICLR 2023).
