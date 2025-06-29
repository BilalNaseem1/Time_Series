import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.io as pio
pio.templates.default = "plotly_white"
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
# %load_ext autoreload
# %autoreload 2
np.random.seed()
tqdm.pandas()

DATA_FOLDER_PATH = "/home/bilal326/Time_Series/data/london_smart_meters"

train_df = pd.read_parquet(f"{DATA_FOLDER_PATH}/preprocessed/selected_blocks_train.parquet")
val_df = pd.read_parquet(f"{DATA_FOLDER_PATH}/preprocessed/selected_blocks_val.parquet")
test_df = pd.read_parquet(f"{DATA_FOLDER_PATH}/preprocessed/selected_blocks_test.parquet")

from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd

def seasonal_interpolate(df, period=48, model="additive", method="linear", fill_border=0):
    original = df.values
    missing_mask = df.isna().values
    observed_mask = ~missing_mask

    # Initial interpolation to avoid NaNs for decomposition
    filled = df.interpolate(method=method, axis=0).fillna(fill_border).values

    trend_list, seasonal_list, resid_list = [], [], []

    for col in range(filled.shape[1]):
        dec = seasonal_decompose(
            filled[:, col],
            model=model,
            period=period,
            extrapolate_trend="freq"
        )
        trend_list.append(dec.trend)
        seasonal_list.append(dec.seasonal)
        resid_list.append(dec.resid)

    trend = np.vstack(trend_list).T
    seasonal = np.vstack(seasonal_list).T
    resid = np.vstack(resid_list).T

    # Deseasonalize
    if model == "additive":
        deseasonalized = trend + resid
    else:
        deseasonalized = trend * resid

    deseasonalized[missing_mask] = np.nan

    # Interpolate deseasonalized signal
    deseasonalized = pd.DataFrame(deseasonalized, index=df.index).interpolate(method=method, axis=0).fillna(fill_border).values

    # Reconstruct signal
    if model == "additive":
        reconstructed = deseasonalized + seasonal
    else:
        reconstructed = deseasonalized * seasonal

    # Restore original observed values
    reconstructed[observed_mask] = original[observed_mask]

    return pd.DataFrame(reconstructed, index=df.index, columns=df.columns)


imputed_train = seasonal_interpolate(train_df, period=48)
imputed_val = seasonal_interpolate(val_df, period=48)
imputed_test = seasonal_interpolate(test_df, period=48)

imputed_train.to_parquet(f"imputed_train.parquet")
imputed_val.to_parquet(f"imputed_val.parquet")
imputed_test.to_parquet(f"imputed_test.parquet")