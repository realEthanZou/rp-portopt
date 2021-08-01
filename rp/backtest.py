import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .optimise import get_optimise_results
from .utils import BACKTEST_START_YEAR, BACKTEST_START_MONTH


def get_backtest_results(codename, lookbacks, holding, freq, n_sample, seeds, verbose=True):
    if codename in ['ew', 'vw']:
        lookbacks = [1]

    if verbose:
        parallel_verbose = 5
    else:
        parallel_verbose = False

    results = Parallel(n_jobs=-2, verbose=parallel_verbose)(
        delayed(run_backtest)(codename, lookback, holding, freq, n_sample, seed, verbose)
        for lookback in lookbacks for seed in seeds
    )

    returns = [x[0] for x in results]
    variances = [x[1] for x in results]
    sharpes = [x[2] for x in results]
    turnovers = [x[3] for x in results]

    idx = 0
    dfs = []
    for lookback in lookbacks:
        if codename in ['ew', 'vw']:
            key = f"holding{holding}{freq}_sample{n_sample}"
        else:
            key = f"lookback{lookback}{freq}_holding{holding}{freq}_sample{n_sample}"

        df_results = pd.DataFrame()
        df_results.name = key
        for seed in seeds:
            df_results.loc['return', f"seed{seed}"] = returns[idx]
            df_results.loc['variance', f"seed{seed}"] = variances[idx]
            df_results.loc['sharpe', f"seed{seed}"] = sharpes[idx]
            df_results.loc['turnover', f"seed{seed}"] = turnovers[idx]
            idx += 1

        df_results['avg'] = df_results.mean(axis=1)
        dfs.append(df_results)

    df_results = pd.DataFrame({df.name: df.avg for df in dfs})
    df_results.name = codename

    return df_results


def run_backtest(codename, lookback, holding, freq, n_sample, seed, verbose=True):
    if codename in ['ew', 'vw']:
        lookback = 1

    df_weights, df_returns = get_optimise_results(codename, BACKTEST_START_YEAR, BACKTEST_START_MONTH, lookback,
                                                  holding, freq, n_sample, seed, verbose)

    if freq == 'm':
        af = 12 / holding
    elif freq == 'd':
        af = 252 / holding
    else:
        raise ValueError

    returns = df_returns.returns.to_list()
    mean_return = np.mean(returns) * af
    variance = np.var(returns, ddof=1) * af
    sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(af)

    df_diff = df_weights.shift(1) - df_weights
    df_diff = df_diff.iloc[1:]
    turnover = df_diff.abs().mean().sum() * af

    return mean_return, variance, sharpe, turnover
