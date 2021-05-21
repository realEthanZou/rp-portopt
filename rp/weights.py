from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .models import get_risk_matrix
from .utils import gen_crsp_subset, BACKTEST_START_YEAR, BACKTEST_START_MONTH


def get_weights(codename, lookback, holding, freq, n_sample, seed, verbose=True):
    key = f"lookback{lookback}{freq}_holding{holding}{freq}_sample{n_sample}_seed{seed}"

    if freq == 'm':
        day = 'last'
    elif freq == 'd':
        day = 30
    else:
        raise ValueError

    if Path(f"weights/{codename}.h5").is_file():
        try:
            if verbose:
                print(f"Loading cache from weights/{codename}.h5 with key='{key}'")
            return pd.read_hdf(f"weights/{codename}.h5", key=key)

        except KeyError:
            if verbose:
                print(f"No cache found at weights/{codename}.h5 with key='{key}'")
            pass

    else:
        if verbose:
            print(f"No cache found at weights/{codename}.h5")

    ret_gen = gen_crsp_subset('ret', year=BACKTEST_START_YEAR, month=BACKTEST_START_MONTH, day=day, before=lookback,
                              after=holding, rolling_freq=holding, n_sample=n_sample, seed=seed)
    cap_gen = gen_crsp_subset('cap', year=BACKTEST_START_YEAR, month=BACKTEST_START_MONTH, day=day,
                              rolling_freq=holding, n_sample=n_sample, seed=seed)

    if codename == 'vw':
        weights = Parallel(n_jobs=-1, verbose=verbose)(delayed(calc_weights)(
            codename, df_ret=None, df_cap=df_before, label=df_after.index[0]) for df_before, df_after in cap_gen)

    else:
        results = Parallel(n_jobs=-1, verbose=verbose)(delayed(calc_weights)(
            codename, df_ret=df_before, label=df_after.index[0]) for df_before, df_after in ret_gen)

        if codename[:2] == 'ls':
            weights = [x[0] for x in results]
            deltas = [x[1] for x in results]

        else:
            weights = results

    df_weights = pd.concat(weights).fillna(0)
    if freq == 'm':
        df_weights = df_weights.set_index(df_weights.reset_index().iloc[:, 0].apply(lambda x: x[:7]))
    if verbose:
        print(f"Saving cache to weights/{codename}.h5 with key='{key}'")
    df_weights.to_hdf(f"weights/{codename}.h5", key=key)

    if codename[:2] == 'ls':
        df_deltas = pd.DataFrame(deltas, index=df_weights.index, columns=['delta'])
        if verbose:
            print(f"Saving cache to weights/{codename}_deltas.h5 with key='{key}'")
            df_deltas.to_hdf(f"weights/{codename}_deltas.h5", key=key)

    return df_weights


def calc_weights(codename, df_ret, label='weights', df_cap=None):
    if codename == 'ew':
        n = df_ret.shape[1]
        weights = (np.ones((1, n)) / n)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns)

    elif codename == 'vw':
        cap = df_cap.tail(1).values
        weights = cap / np.sum(cap)

        return pd.DataFrame(weights, index=[label], columns=df_cap.columns)

    elif codename == 'mvu':
        df_cov = get_risk_matrix(df_ret, method='sample')
        weights = min_var_unconstrained(df_cov.values)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns)

    elif codename == 'lssi':
        df_cov, delta = get_risk_matrix(df_ret, method='ls_scaled_identity')
        weights = min_var_unconstrained(df_cov.values)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns), delta

    elif codename == 'lssf':
        df_cov, delta = get_risk_matrix(df_ret, method='ls_single_factor')
        weights = min_var_unconstrained(df_cov.values)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns), delta

    elif codename == 'lscc':
        df_cov, delta = get_risk_matrix(df_ret, method='ls_constant_corr')
        weights = min_var_unconstrained(df_cov.values)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns), delta

    else:
        raise ValueError


def min_var_unconstrained(cov):
    n = cov.shape[1]
    weights = np.linalg.solve(cov, np.ones(n))
    weights = weights / np.sum(weights)

    return weights.reshape(1, n)
