import numpy as np

from sklearn.metrics import *
from statsmodels.stats.stattools import durbin_watson


def convert_to_np_array(*args):
    return [np.array(el) for el in args]


def rmse(y_true, y_pred):
    y_true, y_pred = convert_to_np_array(y_true, y_pred)
    return np.sqrt(((y_true - y_pred) ** 2).sum())


def rss(y_true, y_pred):
    y_true, y_pred = convert_to_np_array(y_true, y_pred)
    return ((y_true - y_pred) ** 2).sum()
