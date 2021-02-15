from typing import List, Tuple, Callable, Dict, Any, Optional, Union

import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss


def run_stat_test(
    input: pd.Series, stat_test: Callable, index_of_critical_values: int = 4
):
    result = stat_test(input)
    test_name = stat_test.__name__
    parsed_result = {
        test_name + "_statistics": result[0],
        test_name + "_p_value": result[1],
    }
    parsed_result.update(
        {
            test_name + "_critical_value_" + k: v
            for k, v in result[index_of_critical_values].items()
        }
    )
    return parsed_result
