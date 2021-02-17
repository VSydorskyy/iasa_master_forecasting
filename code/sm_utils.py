from typing import List, Tuple, Callable, Dict, Any, Optional, Union
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf, acf
from matplotlib import pyplot as plt

PLOT_LEN = 30
PLOT_HIGHT = 5


def add_diffs(
    init_value: float, diffs: np.ndarray, include_first: bool = True
):
    accum = init_value
    if include_first:
        result = [accum]
    else:
        result = []

    for diff in diffs:
        accum += diff
        result.append(accum)
    return np.array(result)


def compute_pacf_acf(input: np.ndarray, verbose: bool = True):
    acf_r = acf(input)
    pacf_r = pacf(input)

    if verbose:
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(input, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(input, ax=ax2)

    return acf_r, pacf_r


class StatmodelsWrapper(object):
    def __init__(self, stat_model: object, stat_model_config):
        self.model_class = stat_model
        self.model = None
        self.model_config = stat_model_config

        self.X_len = None
        self.X = None

    def fit(self, X: Union[np.ndarray, List, pd.Series]):
        X = np.array(X)

        self.model = self.model_class(X, **self.model_config).fit()

        self.X_len = len(X)
        self.X = X

        return self

    def predict(self, n_steps: int):
        return self.model.predict(start=self.X_len, end=self.X_len + n_steps)

    def dynamic_predict(self, n_steps: int):
        all_pred = []
        dump_len = self.X_len
        dump_x = self.X.copy()
        for step in tqdm(range(n_steps)):
            pred = self.model.forecast()[0]
            self.fit(list(self.X) + [pred])
            all_pred.append(pred)

        return np.array(all_pred)


def run_k_fold_tain_val_statmodels(
    input_df: pd.DataFrame,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    orders: List[Tuple[int, int, int]],
    metric_funcs: List[Tuple[str, Callable]],
    y_col: str = "close",
):
    results = dict(summaries=[])
    # Add fields for metric
    for metric_name, _ in metric_funcs:
        results[metric_name + "_train"] = []
        results[metric_name + "_valid"] = []
    # Run CV
    for f_idx, (train_id, val_id) in enumerate(fold_indices):
        print(f"==== Fold {f_idx} Starting")
        # Split data
        fold_train, fold_val = input_df.iloc[train_id], input_df.iloc[val_id]
        # Normalize data
        y_normed_train = (
            np.log1p(fold_train[y_col])
            .diff(periods=1)
            .dropna()
            .reset_index(drop=True)
        )
        print("Residual ACF/PACF")
        compute_pacf_acf(y_normed_train)
        plt.show()
        # Compute AR model
        temp_oders = (orders[f_idx][0], 0, 0)
        arma = StatmodelsWrapper(
            stat_model=ARIMA, stat_model_config={"order": temp_oders}
        ).fit(y_normed_train)
        # Compute acf/pacf residuals
        print("Residual ACF/PACF")
        compute_pacf_acf(y_normed_train - arma.model.predict())
        plt.show()
        # Compute ARMA model
        arma = StatmodelsWrapper(
            stat_model=ARIMA, stat_model_config={"order": orders[f_idx]}
        ).fit(y_normed_train)
        arma_train_pred = arma.model.predict()
        arma_test_pred = arma.dynamic_predict(len(fold_val))
        print(arma.model.summary())
        # Denormalize
        arma_train_pred_absolute = add_diffs(
            init_value=np.log1p(fold_train[y_col].iloc[0]),
            diffs=arma_train_pred,
        )
        arma_train_pred_absolute = np.expm1(arma_train_pred_absolute)

        arma_test_pred_absolute = add_diffs(
            init_value=np.log1p(fold_train[y_col].iloc[-1]),
            diffs=arma_test_pred,
            include_first=False,
        )
        arma_test_pred_absolute = np.expm1(arma_test_pred_absolute)

        non_rec_train_pred = arma_train_pred + np.log1p(
            fold_train[y_col].iloc[:-1]
        )
        non_rec_train_pred = np.expm1(non_rec_train_pred)
        # Plot Train/Test
        plt.figure(figsize=(PLOT_LEN, PLOT_HIGHT))
        plt.title("Train diff(log)")
        plt.plot(arma_train_pred, label="ARMA")
        plt.plot(y_normed_train, label="Real")
        plt.legend()
        plt.show()

        plt.figure(figsize=(PLOT_LEN, PLOT_HIGHT))
        plt.title("Train Absolute NonReccurent")
        plt.plot(non_rec_train_pred, label="ARMA")
        plt.plot(fold_train[y_col].iloc[:-1], label="Real")
        plt.legend()
        plt.show()

        plt.figure(figsize=(PLOT_LEN, PLOT_HIGHT))
        plt.title("Train Absolute Reccurent")
        plt.plot(arma_train_pred_absolute, label="ARMA")
        plt.plot(fold_train[y_col].tolist(), label="Real")
        plt.legend()
        plt.show()

        plt.figure(figsize=(PLOT_LEN, PLOT_HIGHT))
        plt.title("Validation Absolute Reccurent")
        plt.plot(arma_test_pred_absolute, label="ARMA")
        plt.plot(fold_val[y_col].tolist(), label="Real")
        plt.legend()
        plt.show()
        # Save results
        results["summaries"].append(deepcopy(arma.model.summary))
        for metric_name, metric_func in metric_funcs:
            m_value = metric_func(arma_train_pred_absolute, fold_train[y_col])
            print(metric_name + f"_train: {m_value}")
            results[metric_name + "_train"].append(m_value)

            m_value = metric_func(arma_test_pred_absolute, fold_val[y_col])
            print(metric_name + f"_valid: {m_value}")
            results[metric_name + "_valid"].append(m_value)
        print(f"==== Fold {f_idx} Completed")

    return results
