from typing import List, Tuple, Callable, Dict, Any, Optional, Union

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def preprocess_df(input: pd.DataFrame):
    input["Date"] = pd.to_datetime(input["Date"])
    input = input.sort_values("Date")
    if "Sales" in input.columns:
        input["NSales"] = input["Sales"] / np.abs(input["Sales"]).max()
    return input


def plot_store_charts(
    df: pd.DataFrame, store_id: int, drop_close: bool = True
):

    store_df = df[df["Store"] == store_id]
    if drop_close:
        store_df = store_df[store_df["Open"] == 1]

    # Time series
    plt.figure(figsize=(30, 5))
    plt.title(f"Store {store_id} time series")
    plt.plot(store_df["Date"], store_df["Sales"])
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()

    # Sales histogram
    plt.title(f"Store {store_id} Sales distribution")
    plt.hist(store_df["Sales"], bins=30)
    plt.show()


def plot_folds(
    df: pd.DataFrame, train_val_ids: List[Tuple[np.ndarray, np.ndarray]]
):

    plt.figure(figsize=(30, 5))
    plt.title(f"Time validation")
    for fold_n, (train_ids, val_ids) in enumerate(train_val_ids):
        train_sub_df = df.iloc[train_ids]
        val_sub_df = df.iloc[val_ids]

        if fold_n == len(train_val_ids) - 1:
            plt.plot(
                train_sub_df["Date"],
                train_sub_df["NSales"] + float(fold_n),
                "b",
                label="train part",
            )
            plt.plot(
                val_sub_df["Date"],
                val_sub_df["NSales"] + float(fold_n),
                "r",
                label="validation part",
            )
        else:
            plt.plot(
                train_sub_df["Date"],
                train_sub_df["NSales"] + float(fold_n),
                "b",
            )
            plt.plot(
                val_sub_df["Date"], val_sub_df["NSales"] + float(fold_n), "r"
            )

    plt.xlabel("Date")
    plt.ylabel("Normalized Sales")
    plt.legend()
    plt.show()
