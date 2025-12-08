import numpy as np
import pandas as pd
import os, random, warnings
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def compute_nrmse_numpy(gt_df, pred_df, norm="std"):
    gt_df = gt_df.loc[pred_df.index, pred_df.columns]

    gt_values = gt_df.values.astype("float64")
    pred_values = pred_df.values.astype("float64")

    mask = ~np.isnan(gt_values)
    if mask.sum() == 0:
        raise ValueError("No non-NaN ground truth values to evaluate.")

    gt_sel = gt_values[mask]
    pred_sel = pred_values[mask]

    diff = gt_sel - pred_sel
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)

    if norm == "std":
        denom = np.std(gt_sel)
    elif norm == "mean":
        denom = np.mean(gt_sel)
    elif norm == "range":
        denom = np.max(gt_sel) - np.min(gt_sel)
    else:
        raise ValueError("Unknown norm method")

    nrmse = rmse / denom
    return rmse, nrmse


def calculate_rmse_nrmse(gt_data_dir):
    gt_modalities = {}
    for k, fn in files.items():
        p = os.path.join(gt_data_dir, fn)
        df = load_omics_table(p)
        gt_modalities[k] = df.loc[impute_samples, impute_aligned[k].columns]
    results = {}
    for m in omics_keys:
        gt_df = gt_modalities[m]
        pred_df = imputed_original_scale[m]

        rmse_val, nrmse_val = compute_nrmse_numpy(gt_df, pred_df)
        results[m] = {"RMSE": rmse_val, "NRMSE": nrmse_val}
        print(f"{m}: RMSE={rmse_val:.4f}, NRMSE={nrmse_val:.4f}")


def plot_rmse_nrmse_pies(results, colors=None, title="RMSE & NRMSE Pie Charts"):
    """
    Plot pie charts showing RMSE and NRMSE for each modality.
    Each pie chart contains:
        - one colored slice representing the metric value
        - one white slice representing (1 - value)

    Parameters
    ----------
    results : dict
        Dictionary formatted as:
        {
            "CyTOF": {"RMSE": 0.23, "NRMSE": 0.087},
            "Metabolomics": {"RMSE": 0.50, "NRMSE": 0.17},
            "Proteomics": {"RMSE": 0.25, "NRMSE": 0.10},
        }

    colors : list, optional
        List of colors for each modality. If None, a default color palette is used.

    title : str
        Title of the figure.
    """
    modalities = list(results.keys())
    rmse_vals = [results[m]["RMSE"] for m in modalities]
    nrmse_vals = [results[m]["NRMSE"] for m in modalities]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
            [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
        ],
        subplot_titles=[f"{m} RMSE" for m in modalities]
        + [f"{m} NRMSE" for m in modalities],
    )

    for i, m in enumerate(modalities):
        fig.add_trace(
            go.Pie(
                labels=["Value", "1 - Value"],
                values=[rmse_vals[i], 1 - rmse_vals[i]],
                marker=dict(
                    colors=[colors[i], "white"], line=dict(color="black", width=1)
                ),
                textinfo="percent",
            ),
            row=1,
            col=i + 1,
        )

    for i, m in enumerate(modalities):
        fig.add_trace(
            go.Pie(
                labels=["Value", "1 - Value"],
                values=[nrmse_vals[i], 1 - nrmse_vals[i]],
                marker=dict(
                    colors=[colors[i], "white"], line=dict(color="black", width=1)
                ),
                textinfo="percent",
            ),
            row=2,
            col=i + 1,
        )

    fig.update_layout(
        height=550, width=880, showlegend=False, margin=dict(t=80, l=20, r=20, b=20)
    )

    fig.show()
