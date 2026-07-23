"""
Authors: Filippo Facco, Francesco Pizzato

period_study.py

the goal is to find the dominant period of the Electricity training set, so the first 70%. no leakage from validation/test
The result of this study is used to choose the fixed period that we
hard-code into our modified TimesNet model (TimesNetModified.py).

What the script does, step by step:
  1. Load electricity.csv
  2. Keep only the first 70% of the rows, only the data the model will see NO LEAKAGE
  3. Standardize each channel
  4. Compute the FFT of each channel, take the amplitude, then average the amplitudes over all channels
  5. Print the best 5 periods 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#path to the dataset and output directory
CSV_PATH = Path("../dataset/electricity/electricity.csv")
OUT_DIR = Path("../result/analysis/")

TRAIN_RATIO = 0.7   #same split as the framework: first 70% of rows = train
TOP_K = 5           #how many dominant periods to report
MIN_PERIOD = 2      #ignore periods shorter than 2 hours (noise)
MAX_PERIOD = 336    #ignore periods longer than 336 hours (2 weeks) (slow trend)

#load the dataset
print(f"Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)


#drop the date column: for values only
data = df.drop(columns=["date"]).to_numpy(dtype=np.float64)


#keep only the first 70% of the rows 
n_train = int(len(data) * TRAIN_RATIO)
train = data[:n_train]

#standardizing each users column (channel) to mean=0, std=1
mean = train.mean(axis=0)               #one mean per user
std = train.std(axis=0) + 1e-8          #+1e-8 avoids division by zero
train_std = (train - mean) / std

#FFT transform of each channel (user) in the training set
spectrum = np.fft.rfft(train_std, axis=0)

#abs() of a complex number for taking the module and then average over all channels
module = np.abs(spectrum)
amplitude = module.mean(1) 

#convert each frequency index k to a period in hours:
#period = number_of_samples / k
freq_index = np.arange(1, len(amplitude))          #1, 2, 3, ...
periods = n_train / freq_index                     #period in hours
amps = amplitude[1:]                               #matching amplitudes skipping the DC component (k=0)

#keeping only good periods betwen the thresholds
keep = (periods >= MIN_PERIOD) & (periods <= MAX_PERIOD)
periods = periods[keep]
amps = amps[keep]

#keep the top k periods
order = np.argsort(amps)[::-1]          #indices sorted by amplitude, descending
top_periods = []                        #list of (period, amplitude)

for i in order:
    p = periods[i]
    #skip if too close to a period already in the list
    if any(abs(p - q) / q < 0.10 for q, _ in top_periods):
        continue
    top_periods.append((p, amps[i]))
    if len(top_periods) == TOP_K:
        break

#some printout for the user
print("")
print("------Dominant periods-------")

rank = 1
for p, a in top_periods:
    # p = period in hours, a = amplitude
    print("Rank", rank, "-> period:", round(p, 1), "hours, amplitude:", round(a, 1))
    rank = rank + 1

best_period = round(top_periods[0][0])
print("")
print("Chosen fixed period for TimesNetModified:", best_period, "hours")

#plots the rusults and saves them to the output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

#plot 1: FFT amplitude spectrum
plt.figure(figsize=(12, 5))
plt.plot(periods, amps, linewidth=0.8)
plt.xscale("log")   #log scale: small and large periods both visible

# use the top periods as x-axis labels (instead of the default 10, 100)
tick_values = []
for p, _ in top_periods:
    tick_values.append(round(p))
tick_values.sort()

plt.xticks(tick_values, tick_values)   # put ticks exactly on the found periods
plt.minorticks_off()                   # remove the small automatic log ticks

plt.xlabel("Period (hours, log scale)")
plt.ylabel("Mean FFT amplitude over channels")
plt.title("FFT spectrum of the TRAIN set only (no test leakage)")
plt.grid(alpha=0.3)
plt.tight_layout()
fft_path = OUT_DIR / "period_study.png"
plt.savefig(fft_path, dpi=150)
plt.close()