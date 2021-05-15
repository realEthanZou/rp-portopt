from .utils import get_data, get_universe, gen_trading_dates


def gen_ret_data(year, month, day, before, after, rolling_freq):
    if day is None:
        freq = 'm'
    else:
        freq = 'd'

    ret_master = get_data('crsp', freq, 'ret', verbose=False)
    universe_master = get_universe()

    for period_before, period_after in gen_trading_dates(year, month, day, before, after, rolling_freq):
        universe = universe_master[f"{year}-04"]
        try:
            universe = universe_master[period_after[-1][:7]].to_list()
        except KeyError:
            pass

        ret = ret_master[universe]

        if freq == 'm':
            months_before = list(dict.fromkeys([x[:7] for x in period_before]))
            months_after = list(dict.fromkeys([x[:7] for x in period_after]))

            yield ret.query("date in @months_before"), ret.query("date in @months_after")

        else:
            yield ret.query("date in @period_before"), ret.query("date in @period_after")
