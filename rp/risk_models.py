import warnings

import numpy as np
import pandas as pd


def _is_positive_semidefinite(df_cov):
    try:
        np.linalg.cholesky(df_cov + 1e-16 * np.eye(len(df_cov)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(df_cov):
    if _is_positive_semidefinite(df_cov):
        return df_cov

    warnings.warn("The covariance matrix is non positive semidefinite. Amending eigenvalues.")
    permnos = df_cov.index

    q, V = np.linalg.eigh(df_cov)
    q = np.where(q > 0, q, 0)
    fixed_matrix = V @ np.diag(q) @ V.T

    if not _is_positive_semidefinite(fixed_matrix):
        warnings.warn('Could not fix matrix. Please try a different risk model.')

    return pd.DataFrame(fixed_matrix, index=permnos, columns=permnos)


def sample_cov(df_ret, freq=250):
    return fix_nonpositive_semidefinite(df_ret.cov() * freq)


def get_risk_matrix(df_ret, method, **kwargs):
    if method == 'sample_cov':
        return sample_cov(df_ret, **kwargs)

    else:
        return ValueError
