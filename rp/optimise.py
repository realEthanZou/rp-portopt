from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from cvxpy.atoms.affine.wraps import psd_wrap

from .models import get_risk_matrix
from .utils import gen_crsp_subset, gen_ff_subset


def get_optimise_results(codename, year, month, lookback, holding, freq, n_sample, seed, verbose=True):
    key = f"lookback{lookback}{freq}_holding{holding}{freq}_sample{n_sample}_seed{seed}"

    if codename in ['ew', 'vw']:
        lookback = 1
        key = f"holding{holding}{freq}_sample{n_sample}_seed{seed}"

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
            return pd.read_hdf(f"weights/{codename}.h5", key=key), \
                   pd.read_hdf(f"results/{codename}_returns.h5", key=key)

        except KeyError:
            if verbose:
                print(f"No cache found at weights/{codename}.h5 with key='{key}'")
            pass

    else:
        if verbose:
            print(f"No cache found at weights/{codename}.h5")

    ret_gen = gen_crsp_subset('ret', year=year, month=month, day=day, before=lookback, after=holding,
                              rolling_freq=holding, n_sample=n_sample, seed=seed)

    if codename == 'vw':
        cap_gen = gen_crsp_subset('cap', year=year, month=month, day=day, rolling_freq=holding, n_sample=n_sample,
                                  seed=seed)
        results = [
            calc_weights_and_returns(codename, df_before=None, df_after=tuple_ret[1], df_cap=tuple_cap[0])
            for tuple_ret, tuple_cap in zip(ret_gen, cap_gen)
        ]

    elif codename.split('_')[0] in ['ff3', 'ca4']:
        ff_gen = gen_ff_subset(codename.split('_')[0], year=year, month=month, day=day, before=lookback, after=holding,
                               rolling_freq=holding, n_sample=n_sample, seed=seed)
        results = [
            calc_weights_and_returns(codename, df_before=tuple_ret[0], df_after=tuple_ret[1], df_ff=df_ff)
            for tuple_ret, df_ff in zip(ret_gen, ff_gen)
        ]

    else:
        results = [
            calc_weights_and_returns(codename, df_before=df_before, df_after=df_after)
            for df_before, df_after in ret_gen
        ]

    weights = [x[0] for x in results]
    df_weights = pd.concat(weights).fillna(0)
    if freq == 'm':
        df_weights = df_weights.set_index(df_weights.reset_index().iloc[:, 0].apply(lambda x: x[:7]))
    if verbose:
        print(f"Saving cache to weights/{codename}.h5 with key='{key}'")
    df_weights.to_hdf(f"weights/{codename}.h5", key=key)

    returns = [x[1] for x in results]
    df_returns = pd.DataFrame(returns, index=df_weights.index, columns=['returns'])
    if verbose:
        print(f"Saving cache to results/{codename}_returns.h5 with key='{key}'")
    df_returns.to_hdf(f"results/{codename}_returns.h5", key=key)

    if codename[:2] == 'ls':
        deltas = [x[2] for x in results]
        df_deltas = pd.DataFrame(deltas, index=df_weights.index, columns=['delta'])
        if verbose:
            print(f"Saving cache to weights/{codename}_deltas.h5 with key='{key}'")
        df_deltas.to_hdf(f"weights/{codename}_deltas.h5", key=key)

    return df_weights, df_returns


def calc_weights_and_returns(codename, df_before, df_after, **kwargs):
    codename = codename.split('_')
    if len(codename) == 1:
        constraint = None
    elif len(codename) == 2:
        constraint = codename[1]
    else:
        raise ValueError
    codename = codename[0]

    delta = np.nan

    if codename == 'ew':
        n = df_before.shape[1]
        weights = (np.ones((1, n)) / n)

    elif codename == 'vw':
        cap = kwargs.get('df_cap').fillna(0).tail(1).values
        weights = cap / np.sum(cap)

    else:
        if codename == 'mv':
            df_cov = get_risk_matrix(df_before, method='sample')
        elif codename == 'sf':
            df_cov = get_risk_matrix(df_before, method='single_factor')
        elif codename in ['ff3', 'ca4']:
            df_cov = get_risk_matrix(df_before, df_ff=kwargs.get('df_ff'), method='ff_3_factors')
        elif codename[:3] == 'pca':
            df_cov = get_risk_matrix(df_before, method=f"pca_{codename[3:]}_factors")
        elif codename == 'lssi':
            df_cov, delta = get_risk_matrix(df_before, method='ls_scaled_identity')
        elif codename == 'lssf':
            df_cov, delta = get_risk_matrix(df_before, method='ls_single_factor')
        elif codename == 'lscc':
            df_cov, delta = get_risk_matrix(df_before, method='ls_constant_corr')
        else:
            raise ValueError

        if constraint is None:
            weights = min_var_unconstrained(df_cov.values)
        elif constraint == 'long':
            weights = min_var_long(df_cov.values)
        else:
            raise ValueError

    df_weights = pd.DataFrame(weights, index=[df_after.index[0]], columns=df_after.columns)
    df_return = (df_after.fillna(0) + 1).prod() - 1
    ret = np.sum((df_weights * df_return).values)

    if codename[:2] == 'ls':
        return df_weights, ret, delta
    else:
        return df_weights, ret


def min_var_unconstrained(cov):
    n = cov.shape[1]
    weights = np.linalg.solve(cov, np.ones(n))
    weights = weights / np.sum(weights)

    return weights.reshape(1, n)


def min_var_long(cov):
    n = cov.shape[1]
    weights = cp.Variable(n)
    var = cp.quad_form(weights, psd_wrap(cov))  # cvxpy issue #1424 workaround
    prob = cp.Problem(cp.Minimize(var), [cp.sum(weights) == 1, weights >= 0])
    prob.solve()
    weights = weights.value

    return weights.reshape(1, n)
