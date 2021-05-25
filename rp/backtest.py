from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .optimise import get_optimise_results
from .utils import BACKTEST_START_YEAR, BACKTEST_START_MONTH


def get_backtest_results(codename, lookbacks, holding, freq, n_sample, seeds, verbose=True):
    if codename in ['ew', 'vw']:
        lookbacks = [1]

    results = Parallel(n_jobs=-1, verbose=5)(delayed(run_backtest)(
        codename=codename, lookback=lookback, seed=seed, verbose=verbose) for lookback in lookbacks for seed in seeds)

    variances = [x[0] for x in results]
    sharpes = [x[1] for x in results]
    turnovers = [x[2] for x in results]

    idx = 0
    dfs = []
    for lookback in lookbacks:
        key = f"lookback{lookback}{freq}_holding{holding}{freq}_sample{n_sample}"
        for seed in seeds:
            _save_results(variances[idx], codename=codename, label='variance', key=key, seed=seed, verbose=verbose)
            _save_results(sharpes[idx], codename=codename, label='sharpe', key=key, seed=seed, verbose=verbose)
            _save_results(turnovers[idx], codename=codename, label='turnover', key=key, seed=seed, verbose=verbose)
            idx += 1

        df_results = pd.read_hdf(f"results/{codename}.h5", key=key)
        df_results.name = key
        dfs.append(df_results)

    return dfs


def run_backtest(codename, lookback=120, holding=1, freq='m', n_sample=500, seed=42, verbose=True):
    if codename in ['ew', 'vw']:
        lookback = 1

    df_weights, df_returns = get_optimise_results(
        codename=codename, year=BACKTEST_START_YEAR, month=BACKTEST_START_MONTH, lookback=lookback, holding=holding,
        freq=freq, n_sample=n_sample, seed=seed, verbose=verbose)

    returns = df_returns.returns.to_list()
    variance = np.var(returns, ddof=1)
    sharpe = np.mean(returns) / np.std(returns, ddof=1)
    turnover = calc_turnover(df_weights)

    return variance, sharpe, turnover


def calc_turnover(df_weights):
    df_diff = df_weights.shift(1) - df_weights
    df_diff.iloc[0] = df_weights.iloc[0]
    turnover = df_diff.abs().mean().sum()

    return turnover


def _save_results(data, codename, label, key, seed, verbose=True):
    assert label in ['variance', 'sharpe', 'turnover']

    results_path = f"results/{codename}.h5"
    df_results = None
    if Path(results_path).is_file():
        try:
            df_results = pd.read_hdf(results_path, key=key)
        except KeyError:
            pass

    if df_results is None:
        df_results = pd.DataFrame(data, index=[label], columns=[f"seed{seed}"])
    else:
        df_results.loc[label, f"seed{seed}"] = data

    if verbose:
        print(f"Saving {label} to {results_path} with key='{key}'")
    df_results.to_hdf(results_path, key=key)
