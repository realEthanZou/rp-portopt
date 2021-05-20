import pandas as pd

from .utils import gen_crsp_subset
from .weights import get_weights


def run_backtest(codename, lookback, seed=42):
    all_weights = []

    for df_before, df_after in gen_crsp_subset('ret', year=2001, month=3, day='last', before=lookback, after=1,
                                               rolling_freq=1, seed=seed):
        weights = get_weights(codename, df_ret=df_before)
        all_weights.append(weights)

    df_weights = pd.concat(all_weights).fillna(0)
    turnover = calc_turnover(df_weights)

    return turnover


def calc_turnover(df_weights):
    df_diff = df_weights.shift(1) - df_weights
    df_diff.iloc[0] = df_weights.iloc[0]
    turnover = df_diff.abs().mean().sum()

    return turnover
