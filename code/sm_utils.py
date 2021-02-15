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
    metric_func: Callable,
):
    results = dict(summaries=[], train_scores=[], val_scores=[],)
    for f_idx, (train_id, val_id) in enumerate(fold_indices):
        print(f"==== Fold {f_idx} Starting")
        # Split data
        fold_train, fold_val = input_df.iloc[train_id], input_df.iloc[val_id]
        # Normalize data
        standart_norm = StandardScaler().fit(
            fold_train["Sales"].values[:, None]
        )
        fold_train["Sales"] = standart_norm.transform(
            fold_train["Sales"].values[:, None]
        )[:, 0]
        fold_val["Sales"] = standart_norm.transform(
            fold_val["Sales"].values[:, None]
        )[:, 0]
        # Plot inital data
        plt.title("Train/Val data")
        plt.plot(fold_train["Date"], fold_train["Sales"], label="Train")
        plt.plot(fold_val["Date"], fold_val["Sales"], label="Val")
        plt.legend()
        plt.show()
        # Plot acf/pacf of inital data
        print("Residual ACF/PACF")
        compute_pacf_acf(fold_train["Sales"])
        plt.show()
        # Compute AR model
        temp_oders = (orders[f_idx][0], 0, 0)
        arma = StatmodelsWrapper(
            stat_model=ARIMA, stat_model_config={"order": temp_oders}
        ).fit(fold_train["Sales"])
        # Compute acf/pacf residuals
        print("Residual ACF/PACF")
        compute_pacf_acf(fold_train["Sales"] - arma.model.predict())
        plt.show()
        # Compute ARMA model
        arma = StatmodelsWrapper(
            stat_model=ARIMA, stat_model_config={"order": orders[f_idx]}
        ).fit(fold_train["Sales"])
        arma_train_pred = arma.model.predict()
        arma_test_pred = arma.dynamic_predict(len(fold_val))
        print(arma.model.summary())
        # Denormalize
        arma_train_pred = standart_norm.inverse_transform(
            arma_train_pred[:, None]
        )[:, 0]
        arma_test_pred = standart_norm.inverse_transform(
            arma_test_pred[:, None]
        )[:, 0]
        fold_train["Sales"] = standart_norm.inverse_transform(
            fold_train["Sales"].values[:, None]
        )[:, 0]
        fold_val["Sales"] = standart_norm.inverse_transform(
            fold_val["Sales"].values[:, None]
        )[:, 0]
        # Plot Train/Test
        plt.title("Train")
        plt.plot(arma_train_pred, label="ARMA")
        plt.plot(fold_train["Sales"].tolist(), label="Real")
        plt.legend()
        plt.show()

        plt.title("Validation")
        plt.plot(arma_test_pred, label="ARMA")
        plt.plot(fold_val["Sales"].tolist(), label="Real")
        plt.legend()
        plt.show()
        # Save results
        results["summaries"].append(deepcopy(arma.model.summary))
        results["train_scores"].append(
            metric_func(arma_train_pred, fold_train["Sales"])
        )
        results["val_scores"].append(
            metric_func(arma_test_pred, fold_val["Sales"])
        )
        print(f"==== Fold {f_idx} Completed")

    return results
