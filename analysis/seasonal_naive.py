"""
Authors: Filippo Facco, Francecso Pizzato

seasonal_naive.py

the goal is to replicate the simplest baseline for long-term forecasting, the seasonal naive model,
which predicts the next horizon steps by repeating the last observed daily cycle (24 hours) over the whole horizon.

To make the numbers directly comparable with result of TimesNet and TimesNetModified:
  - the test split and the per-channel standardization replicate EXACTLY
    data_provider/data_loader.py (Dataset_Custom): train = first 70% of rows,
    test = last 20% (with seq_len extra rows of left context), scaler fitted
    on train only;
  - MSE/MAE are computed on the standardized values, mean over all test
    windows x horizon steps x 321 channels;
  - results are appended to result_long_term_forecast.txt with the same
    naming convention.
"""
import numpy as np
import pandas as pd

#path to the dataset and output file
DATA_PATH = '../dataset/electricity/electricity.csv'
RESULT_FILE = '../result/analysis/result_long_term_forecast.txt'

#same settings as the other models, to make the results comparable
SEQ_LEN = 96                     #input window length (hours)
PERIOD = 24                      #the daily cycle we repeat
PRED_LENS = [96, 192, 336, 720]  #the 4 forecasting horizons

#load the dataset and drop the date column: values only
df = pd.read_csv(DATA_PATH)
values = df.drop(columns=['date']).to_numpy(dtype=np.float32)

#same split sizes as the framework: first 70% train, last 20% test
n = len(values)
num_train = int(n * 0.7)
num_test = int(n * 0.2)

#standardize each channel using mean and std of the TRAIN rows only
mean = values[:num_train].mean(axis=0, keepdims=True)
std = values[:num_train].std(axis=0, keepdims=True) + 1e-8   #+1e-8 avoids division by zero
scaled = (values - mean) / std

#test region: last 20% of the rows, plus SEQ_LEN extra rows on the left
#(the first test window needs them as input), exactly like the framework does
test = scaled[n - num_test - SEQ_LEN:]
print('total rows:', n, '| test region rows:', len(test))

#evaluation, one horizon at a time
for pred_len in PRED_LENS:
    #how many sliding windows fit in the test region (same count as the framework)
    n_windows = len(test) - SEQ_LEN - pred_len + 1

    se_sum = 0.0   #sum of squared errors
    ae_sum = 0.0   #sum of absolute errors
    count = 0      #how many values we predicted in total

    for s in range(n_windows):
        x = test[s : s + SEQ_LEN]                        #input window
        y = test[s + SEQ_LEN : s + SEQ_LEN + pred_len]   #ground truth (the real future)

        #take the last 24 hours of the input...
        last_day = x[SEQ_LEN - PERIOD:]
        #...and repeat them until the horizon is filled
        reps = int(np.ceil(pred_len / PERIOD))           #how many copies of the day we need
        pred = np.tile(last_day, (reps, 1))[:pred_len]   #stack the copies, cut the extra part

        #accumulate the errors of this window
        err = pred - y
        se_sum = se_sum + float((err ** 2).sum())
        ae_sum = ae_sum + float(np.abs(err).sum())
        count = count + err.size

    mse = se_sum / count
    mae = ae_sum / count
    print('pred_len:', pred_len, '| windows:', n_windows, '| mse:', round(mse, 6), '| mae:', round(mae, 6))

    #append the result to the result file, same naming style as the framework
    setting = ('long_term_forecast_ECL_96_' + str(pred_len) + '_SeasonalNaive' + str(PERIOD) +
               '_custom_ftM_sl' + str(SEQ_LEN) + '_ll48_pl' + str(pred_len) + '_notrain_Exp_0')
    f = open(RESULT_FILE, 'a')
    f.write(setting + '  \n')
    f.write('mse:' + str(mse) + ', mae:' + str(mae) + ', dtw:Not calculated')
    f.write('\n\n')
    f.close()

print('')
print('results appended to', RESULT_FILE)