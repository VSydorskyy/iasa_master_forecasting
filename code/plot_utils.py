from typing import List, Tuple, Callable, Dict, Any, Optional, Union

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def plot_store_charts(df: pd.DataFrame, y_col: str = "close"):
    # Time series
    plt.figure(figsize=(30, 5))
    plt.title(f"time series")
    plt.plot(df["date"], df[y_col])
    plt.xlabel("date")
    plt.ylabel(y_col)
    plt.show()

    # Sales histogram
    plt.title(f"distribution")
    plt.hist(df[y_col], bins=30)
    plt.show()


def plot_folds(
    df: pd.DataFrame,
    train_val_ids: List[Tuple[np.ndarray, np.ndarray]],
    y_col: str = "Nclose",
):

    plt.figure(figsize=(30, 5))
    plt.title(f"Time validation")
    for fold_n, (train_ids, val_ids) in enumerate(train_val_ids):
        train_sub_df = df.iloc[train_ids]
        val_sub_df = df.iloc[val_ids]

        if fold_n == len(train_val_ids) - 1:
            plt.plot(
                train_sub_df["date"],
                train_sub_df[y_col] + float(fold_n),
                "b",
                label="train part",
            )
            plt.plot(
                val_sub_df["date"],
                val_sub_df[y_col] + float(fold_n),
                "r",
                label="validation part",
            )
        else:
            plt.plot(
                train_sub_df["date"], train_sub_df[y_col] + float(fold_n), "b",
            )
            plt.plot(
                val_sub_df["date"], val_sub_df[y_col] + float(fold_n), "r"
            )

    plt.xlabel("date")
    plt.ylabel(f"Normalized {y_col}")
    plt.legend()
    plt.show()
