import numpy as np

from .optimise import get_results
from .utils import BACKTEST_START_YEAR, BACKTEST_START_MONTH


def run_backtest(codename, lookback=120, holding=1, freq='m', n_sample=500, seed=42, verbose=True):
    print(f"codename:{codename}, lookback:{lookback}, holding:{holding}, freq:{freq}, n_sample:{n_sample}, seed:{seed}")
    df_weights, df_returns = get_results(
        codename=codename, year=BACKTEST_START_YEAR, month=BACKTEST_START_MONTH, lookback=lookback, holding=holding,
        freq=freq, n_sample=n_sample, seed=seed, verbose=verbose)

    returns = df_returns.returns.to_list()
    variance = np.var(returns, ddof=1)
    sharpe = np.mean(returns) / np.std(returns, ddof=1)
    turnover = calc_turnover(df_weights)
    print(f"  > variance:{variance:f}, sharpe:{sharpe:f}, turnover:{turnover:f}")

    return variance, sharpe, turnover


def calc_turnover(df_weights):
    df_diff = df_weights.shift(1) - df_weights
    df_diff.iloc[0] = df_weights.iloc[0]
    turnover = df_diff.abs().mean().sum()

    return turnover
