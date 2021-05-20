import pandas as pd
from joblib import Parallel, delayed

from .utils import gen_crsp_subset
from .weights import get_weights


def run_backtest(codename, lookback, n_sample=500, seed=42):
    ret_gen = gen_crsp_subset('ret', year=2001, month=3, day='last', before=lookback, n_sample=n_sample, seed=seed)
    cap_gen = gen_crsp_subset('cap', year=2001, month=3, day='last', n_sample=n_sample, seed=seed)

    if codename != 'vw':
        weights = Parallel(n_jobs=16, prefer="threads", verbose=5)(delayed(get_weights)(
            codename, df_ret=df_before, label=df_after.index[0][:7]) for df_before, df_after in ret_gen)

    else:
        weights = Parallel(n_jobs=16, prefer="threads", verbose=5)(delayed(get_weights)(
            codename, df_ret=None, df_cap=df_before, label=df_after.index[0][:7]) for df_before, df_after in cap_gen)

    df_weights = pd.concat(weights).fillna(0)
    turnover = calc_turnover(df_weights)

    return turnover


def calc_variance():
    pass


def calc_sharpe():
    pass


def calc_turnover(df_weights):
    df_diff = df_weights.shift(1) - df_weights
    df_diff.iloc[0] = df_weights.iloc[0]
    turnover = df_diff.abs().mean().sum()

    return turnover
