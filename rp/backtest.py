import numpy as np

from .utils import gen_crsp_subset, BACKTEST_START_YEAR, BACKTEST_START_MONTH
from .weights import get_weights


def run_backtest(codename, lookback=120, holding=1, freq='m', n_sample=500, seed=42, verbose=True):
    df_weights = get_weights(codename=codename, lookback=lookback, holding=holding, freq=freq, n_sample=n_sample,
                             seed=seed, verbose=verbose)

    returns = _calc_returns(df_weights, lookback=lookback, holding=holding, freq=freq, n_sample=n_sample, seed=seed)
    variance = np.var(returns, ddof=1)
    sharpe = np.mean(returns) / np.std(returns, ddof=1)
    turnover = calc_turnover(df_weights)
    print(f"variance: {variance:f}, sharpe: {sharpe:f}, turnover: {turnover:f}")

    return


def _calc_returns(df_weights, lookback, holding, freq, n_sample, seed):
    if freq == 'm':
        day = 'last'
    elif freq == 'd':
        day = 30
    else:
        raise ValueError

    ret_gen = gen_crsp_subset('ret', year=BACKTEST_START_YEAR, month=BACKTEST_START_MONTH, day=day, before=lookback,
                              after=holding, rolling_freq=holding, n_sample=n_sample, seed=seed)

    idx = 0
    returns = []
    for _, df_after in ret_gen:
        df_return = (df_after.fillna(0) + 1).prod() - 1
        ret = (df_weights.iloc[idx][df_return.index] * df_return).sum()
        returns.append(ret)
        idx += 1

    return returns


def calc_turnover(df_weights):
    df_diff = df_weights.shift(1) - df_weights
    df_diff.iloc[0] = df_weights.iloc[0]
    turnover = df_diff.abs().mean().sum()

    return turnover
