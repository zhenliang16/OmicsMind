import numpy as np
import pandas as pd
import os, random, warnings


def load_omics_table(path):
    """
    Goal: return a float DataFrame with shape (samples × features).
    Auto-fixes common formats:
      - First column is sample_id -> set as index
      - If features are in rows and samples in columns -> transpose
      - Non-numeric -> NaN
    """
    df = pd.read_csv(path, sep=None, engine="python")

    # If first column looks like sample ids (strings), set it as index
    if df.shape[1] > 1 and df.iloc[:, 0].dtype == object:
        df = df.set_index(df.columns[0])

    # Heuristic for transpose:
    # If index mostly strings (feature names) but columns mostly numeric/sample-like
    # and rows > cols, treat as (features × samples) and transpose.
    idx_obj_ratio = np.mean([isinstance(x, str) for x in df.index])
    col_obj_ratio = np.mean([isinstance(x, str) for x in df.columns])
    if idx_obj_ratio > 0.8 and col_obj_ratio < 0.2 and df.shape[0] > df.shape[1]:
        df = df.T

    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


class MultiOmicsDataset(Dataset):
    def __init__(self, X_dict, mask_prob=0.3):
        self.mods = list(X_dict.keys())
        self.X = X_dict
        self.n = next(iter(X_dict.values())).shape[0]
        self.mask_prob = mask_prob

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample, observed, train_mask = {}, {}, {}
        # observed vs true-missing (original NaN)
        for m in self.mods:
            x_m = self.X[m][idx]
            obs_m = not np.all(np.isnan(x_m))
            observed[m] = obs_m
            sample[m] = np.nan_to_num(x_m, nan=0.0)

        # training-time random modality masking
        for m in self.mods:
            if observed[m] and random.random() < self.mask_prob:
                train_mask[m] = 0
            else:
                train_mask[m] = 1

        return sample, observed, train_mask
