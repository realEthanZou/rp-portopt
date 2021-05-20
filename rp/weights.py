import numpy as np
import pandas as pd

from .models import get_risk_matrix


def get_weights(codename, df_ret, label='weights', df_cap=None):
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

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns)

    elif codename == 'lssf':
        df_cov, delta = get_risk_matrix(df_ret, method='ls_single_factor')
        weights = min_var_unconstrained(df_cov.values)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns)

    elif codename == 'lscc':
        df_cov, delta = get_risk_matrix(df_ret, method='ls_constant_corr')
        weights = min_var_unconstrained(df_cov.values)

        return pd.DataFrame(weights, index=[label], columns=df_ret.columns)

    else:
        raise ValueError


def min_var_unconstrained(cov):
    n = cov.shape[1]
    weights = np.linalg.solve(cov, np.ones(n))
    weights = weights / np.sum(weights)

    return weights.reshape(1, n)
