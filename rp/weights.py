import numpy as np
import pandas as pd


def get_weights(codename, **kwargs):
    if codename == 'ew':
        df_ret = kwargs.get('df_ret')
        n = df_ret.shape[1]
        weights = np.ones(n) / n

        return pd.DataFrame(weights, index=df_ret.columns, columns=[df_ret.index[-1]]).T

    else:
        raise ValueError
