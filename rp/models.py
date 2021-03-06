import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_risk_matrix(df_ret, method, **kwargs):
    if method == 'sample':
        return sample_covariance(df_ret)

    elif method == 'single_factor':
        return single_factor(df_ret)

    elif method == 'ff_3_factors':
        return multi_factor(df_ret, kwargs.get('df_ff'))

    elif method.split('_')[0] == 'pca':
        return pca_k_factors(df_ret, k=int(method.split('_')[1]))

    elif method == 'ls_scaled_identity':
        return linear_shrinkage(df_ret, target='scaled_identity')

    elif method == 'ls_single_factor':
        return linear_shrinkage(df_ret, target='single_factor')

    elif method == 'ls_constant_corr':
        return linear_shrinkage(df_ret, target='constant_corr')

    else:
        raise ValueError


def _is_psd(df_cov):
    try:
        np.linalg.cholesky(df_cov + 1e-16 * np.eye(len(df_cov)))
        return True

    except np.linalg.LinAlgError:
        return False


def fix_non_psd(df_cov, label, verbose=True):
    if _is_psd(df_cov):
        return df_cov

    if verbose:
        print(f"Fixing non-PSD covariance matrix @ {label}")
    q, v = np.linalg.eigh(df_cov)
    q = np.where(q > 0, q, 0)
    fixed_matrix = v @ np.diag(q) @ v.T

    if not _is_psd(fixed_matrix):
        msg = f"FAILED to fix non-PSD covariance matrix @ {label}"
        warnings.warn(msg)

    return pd.DataFrame(fixed_matrix, index=df_cov.index, columns=df_cov.columns)


def sample_covariance(df_ret):
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)
    cov_sample = x.T @ x / t

    return fix_non_psd(pd.DataFrame(cov_sample, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1])


def single_factor(df_ret):
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)
    x_mkt = np.mean(x, axis=1).reshape(t, 1)
    x_ext = np.append(x, x_mkt, axis=1)

    cov_sample = x_ext.T @ x_ext / t
    cov_mkt = cov_sample[:n, n].reshape(n, 1)
    var_mkt = cov_sample[n, n]
    cov_sample = cov_sample[:n, :n]
    cov_sf = cov_mkt @ cov_mkt.T / var_mkt
    cov_sf[np.eye(n) == 1] = np.diag(cov_sample)

    return fix_non_psd(pd.DataFrame(cov_sf, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1])


def multi_factor(df_ret, df_factor):
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)
    fac = df_factor.values

    cov_fac = fac.T @ fac / t
    regs = [LinearRegression().fit(fac, x[:, idx]) for idx in range(n)]
    coefs = np.array([reg.coef_ for reg in regs]).T
    var_res = np.var([x[:, idx] - reg.predict(fac) for idx, reg in enumerate(regs)], axis=1)
    cov_mf = coefs.T @ cov_fac @ coefs + np.diag(var_res)

    return fix_non_psd(pd.DataFrame(cov_mf, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1])


def pca_k_factors(df_ret, k):
    x = np.nan_to_num(df_ret)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)
    cov = x.T @ x / t
    var = np.diag(cov)

    s, u = np.linalg.eigh(cov)
    s = s[::-1]
    u = u[:, ::-1]

    bfb = u[:, :k] @ np.diag(s[:k]) @ u[:, :k].T
    d = var - np.diag(bfb)
    d = np.maximum(d, np.percentile(d, 5))
    cov_pca = bfb + np.diag(d)

    for epoch in range(200):
        s_prev = s[:k]
        d_invsq = np.diag(1 / np.sqrt(d))
        d_sq = np.diag(np.sqrt(d))

        s, u = np.linalg.eigh(d_invsq @ cov @ d_invsq)
        s = s[::-1]
        u = u[:, ::-1]

        bfb = d_sq @ u[:, :k] @ np.diag(s[:k] - 1) @ u[:, :k].T @ d_sq
        d = var - np.diag(bfb)
        d = np.maximum(d, np.percentile(d, 5))
        cov_pca = bfb + np.diag(d)

        err = np.max(np.abs(s_prev - s[:k]))
        if err < 1e-4:
            break

        if epoch == 199:
            warnings.warn('Joreskog did not converge after 200 iterations!')

    return fix_non_psd(pd.DataFrame(cov_pca, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1])


def linear_shrinkage(df_ret, target):
    if target == 'scaled_identity':
        return _linear_shrinkage_scaled_identity(df_ret)

    elif target == 'single_factor':
        return _linear_shrinkage_single_factor(df_ret)

    elif target == 'constant_corr':
        return _linear_shrinkage_constant_corr(df_ret)

    else:
        raise ValueError


def _linear_shrinkage_scaled_identity(df_ret):
    # Ledoit, O. and Wolf, M. (2004).
    # A well-conditioned estimator for large-dimensional covariance matrices.
    # Journal of Multivariate Analysis, 88:365-411.
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)

    cov_sample = x.T @ x / t
    cov_tgt = np.mean(np.diag(cov_sample)) * np.eye(n)

    y = x ** 2
    p_arr = y.T @ y / t - cov_sample ** 2
    p = np.sum(p_arr)
    c = np.linalg.norm(cov_sample - cov_tgt, "fro") ** 2

    k = p / c
    delta = max(0., min(1., k / t))
    shrunk_cov = delta * cov_tgt + (1 - delta) * cov_sample

    return fix_non_psd(pd.DataFrame(shrunk_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1]), delta


def _linear_shrinkage_single_factor(df_ret):
    # Ledoit, O. and Wolf, M. (2003).
    # Improved estimation of the covariance matrix of stock returns with an application to portfolio selection.
    # Journal of Empirical Finance, 10:603-621.
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)
    x_mkt = np.mean(x, axis=1).reshape(t, 1)
    x_ext = np.append(x, x_mkt, axis=1)

    cov_sample = x_ext.T @ x_ext / t
    cov_mkt = cov_sample[:n, n].reshape(n, 1)
    var_mkt = cov_sample[n, n]
    cov_sample = cov_sample[:n, :n]
    cov_tgt = cov_mkt @ cov_mkt.T / var_mkt
    cov_tgt[np.eye(n) == 1] = np.diag(cov_sample)

    y = x ** 2
    p_arr = y.T @ y / t - cov_sample ** 2
    p = np.sum(p_arr)
    c = np.linalg.norm(cov_sample - cov_tgt, "fro") ** 2

    r_diag = np.sum(y ** 2 / t) - np.sum(np.diag(cov_sample) ** 2)
    z = x * np.tile(x_mkt, n)
    v1 = 1 / t * y.T @ z - np.tile(cov_mkt, n) * cov_sample
    r_off1 = (np.sum(v1 * np.tile(cov_mkt, n).T) - np.sum(np.diag(v1) * cov_mkt.T)) / var_mkt
    v3 = 1 / t * z.T @ z - var_mkt * cov_sample
    r_off3 = (np.sum(v3 * (cov_mkt @ cov_mkt.T)) - np.sum(np.diag(v3).reshape(n, 1) * cov_mkt ** 2)) / var_mkt ** 2
    r_off = 2 * r_off1 - r_off3
    r = r_diag + r_off

    k = (p - r) / c
    delta = max(0., min(1., k / t))
    shrunk_cov = delta * cov_tgt + (1 - delta) * cov_sample

    return fix_non_psd(pd.DataFrame(shrunk_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1]), delta


def _linear_shrinkage_constant_corr(df_ret):
    # Ledoit, O. and Wolf, M. (2004).
    # Honey, I shrunk the sample covariance matrix.
    # Journal of Portfolio Management, 30(4):110-119.
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)

    cov_sample = x.T @ x / t
    var = np.diag(cov_sample).reshape(n, 1)
    std = np.sqrt(var)
    std_arr = np.tile(std, n)
    r_bar = (np.sum(cov_sample / (std_arr * std_arr.T)) - n) / (n * (n - 1))
    cov_tgt = r_bar * std_arr * std_arr.T
    cov_tgt[np.eye(n) == 1] = var.reshape(n)

    y = x ** 2
    p_arr = y.T @ y / t - cov_sample ** 2
    p = np.sum(p_arr)
    c = np.linalg.norm(cov_sample - cov_tgt, "fro") ** 2

    theta_arr = (x ** 3).T @ x / t - np.tile(np.diag(cov_sample).reshape(n, 1), n) * cov_sample
    theta_arr[np.eye(n) == 1] = np.zeros(n)
    r = np.trace(p_arr) + r_bar * np.sum((1. / std) @ std.T * theta_arr)

    k = (p - r) / c
    delta = max(0., min(1., k / t))
    shrunk_cov = delta * cov_tgt + (1 - delta) * cov_sample

    return fix_non_psd(pd.DataFrame(shrunk_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1]), delta
