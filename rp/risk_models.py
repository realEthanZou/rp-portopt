import numpy as np
import pandas as pd

from .utils import fix_non_psd


def get_risk_matrix(df_ret, method):
    if method == 'sample':
        return sample_covariance(df_ret)

    elif method == 'ls_scaled_identity':
        return linear_shrinkage(df_ret, target='scaled_identity')

    elif method == 'ls_single_factor':
        return linear_shrinkage(df_ret, target='single_factor')

    elif method == 'ls_constant_corr':
        return linear_shrinkage(df_ret, target='constant_corr')

    else:
        raise ValueError


def sample_covariance(df_ret):
    x = np.nan_to_num(df_ret.values)
    sample_cov = np.cov(x, rowvar=False)
    return fix_non_psd(pd.DataFrame(sample_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1])


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

    sample = np.cov(x, ddof=0, rowvar=False)
    target = np.mean(np.diag(sample)) * np.eye(n)

    y = x ** 2
    p_arr = y.T.dot(y) / t - sample ** 2
    p = np.sum(p_arr)
    c = np.linalg.norm(sample - target, "fro") ** 2

    k = p / c
    delta = max(0., min(1., k / t))
    shrunk_cov = delta * target + (1 - delta) * sample

    return fix_non_psd(pd.DataFrame(shrunk_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1]), delta


def _linear_shrinkage_single_factor(df_ret):
    # Ledoit, O. and Wolf, M. (2003).
    # Improved estimation of the covariance matrix of stock returns with an application to portfolio selection.
    # Journal of Empirical Finance, 10:603-621.
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)
    xmkt = np.mean(x, axis=1).reshape(t, 1)

    sample = np.cov(np.append(x, xmkt, axis=1), ddof=0, rowvar=False)
    covmkt = sample[:n, n].reshape(n, 1)
    varmkt = sample[n, n]
    sample = sample[:n, :n]
    target = covmkt.dot(covmkt.T) / varmkt
    target[np.eye(n) == 1] = np.diag(sample)

    y = x ** 2
    p_arr = y.T.dot(y) / t - sample ** 2
    p = np.sum(p_arr)
    c = np.linalg.norm(sample - target, "fro") ** 2

    rdiag = np.sum(y ** 2 / t) - np.sum(np.diag(sample) ** 2)
    z = x * np.tile(xmkt, n)
    v1 = 1 / t * y.T.dot(z) - np.tile(covmkt, n) * sample
    roff1 = (np.sum(v1 * np.tile(covmkt, n).T) - np.sum(np.diag(v1) * covmkt.T)) / varmkt
    v3 = 1 / t * z.T.dot(z) - varmkt * sample
    roff3 = (np.sum(v3 * covmkt.dot(covmkt.T)) - np.sum(np.diag(v3).reshape(n, 1) * covmkt ** 2)) / varmkt ** 2
    roff = 2 * roff1 - roff3
    r = rdiag + roff

    k = (p - r) / c
    delta = max(0., min(1., k / t))
    shrunk_cov = delta * target + (1 - delta) * sample

    return fix_non_psd(pd.DataFrame(shrunk_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1]), delta


def _linear_shrinkage_constant_corr(df_ret):
    # Ledoit, O. and Wolf, M. (2004).
    # Honey, I shrunk the sample covariance matrix.
    # Journal of Portfolio Management, 30(4):110-119.
    x = np.nan_to_num(df_ret.values)
    t, n = np.shape(x)
    x = x - x.mean(axis=0)

    sample = np.cov(x, ddof=0, rowvar=False)
    var = np.diag(sample).reshape(n, 1)
    std = np.sqrt(var)
    var_arr = np.tile(var, n)
    std_arr = np.tile(std, n)
    r_bar = (np.sum(sample / (std_arr * std_arr.T)) - n) / (n * (n - 1))
    target = r_bar * std_arr * std_arr.T
    target[np.eye(n) == 1] = var.reshape(n)

    y = x ** 2
    p_arr = y.T.dot(y) / t - sample ** 2
    p = np.sum(p_arr)
    c = np.linalg.norm(sample - target, "fro") ** 2

    term1 = (x ** 3).T.dot(x) / t
    help_ = x.T.dot(x) / t
    help_diag = np.diag(help_).reshape(n, 1)
    term2 = np.tile(help_diag, n) * sample
    term3 = help_ * var_arr
    term4 = var_arr * sample
    theta_arr = term1 - term2 - term3 + term4
    theta_arr[np.eye(n) == 1] = np.zeros(n)
    r = np.trace(p_arr) + r_bar * np.sum((1. / std).dot(std.T) * theta_arr)

    k = (p - r) / c
    delta = max(0., min(1., k / t))
    shrunk_cov = delta * target + (1 - delta) * sample

    return fix_non_psd(pd.DataFrame(shrunk_cov, index=df_ret.columns, columns=df_ret.columns), df_ret.index[-1]), delta
