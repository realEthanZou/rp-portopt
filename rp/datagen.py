from .utils import get_data, get_universe, gen_trading_dates


def gen_crsp_subset(key, year, month, day, before, after, rolling_freq):
    if day is None:
        freq = 'm'
    else:
        freq = 'd'

    df_master = get_data('crsp', freq, key, verbose=False)
    universe_master = get_universe()

    for period_before, period_after in gen_trading_dates(year, month, day, before, after, rolling_freq):
        universe = universe_master[f"{year}-04"]
        try:
            universe = universe_master[period_after[-1][:7]].to_list()
        except KeyError:
            pass

        df = df_master[universe]

        if freq == 'm':
            period_before = list(dict.fromkeys([x[:7] for x in period_before]))
            period_after = list(dict.fromkeys([x[:7] for x in period_after]))

        yield df.query("date in @period_before"), df.query("date in @period_after")
