from .weights import get_weights


def run_backtest(codename, lookback=120, holding=1, freq='m', n_sample=500, seed=42, verbose=True):
    df_weights = get_weights(codename=codename, lookback=lookback, holding=holding, freq=freq, n_sample=n_sample,
                             seed=seed, verbose=verbose)

    turnover = calc_turnover(df_weights)
    print(f"turnover: {turnover:f}")

    return df_weights


def calc_variance():
    pass


def calc_sharpe():
    pass


def calc_turnover(df_weights):
    df_diff = df_weights.shift(1) - df_weights
    df_diff.iloc[0] = df_weights.iloc[0]
    turnover = df_diff.abs().mean().sum()

    return turnover
