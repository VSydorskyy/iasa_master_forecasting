from typing import List

import pandas as pd
import numpy as np


def preprocess_df(input: pd.DataFrame, cols_to_norm: List[str] = ["close"]):
    # Process column names
    input = input.rename(
        columns={col_n: col_n[1:-1].lower() for col_n in input.columns}
    )
    # Process date
    input["date"] = pd.to_datetime(
        input["date"].apply(
            lambda x: str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:]
        )
    )
    # Sort by date
    input = input.sort_values("date")
    # Norm cols
    for col_n in cols_to_norm:
        input["N" + col_n] = input[col_n] / np.abs(input[col_n]).max()
    return input
