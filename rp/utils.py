from pathlib import Path

import numpy as np
import pandas as pd
import wrds

WRDS_USERNAME = 'realethanzou'
HISTORY_START_DATE = '1990-01-01'
HISTORY_END_DATE = '2021-03-31'
BACKTEST_START_DATE = '2001-03-30'  # first out-of-sample actually 1 day after this


def get_raw_data(source, freq, verbose=True):
    """
    Helper function to get data from WRDS and handle cache
    :param source: wrds library name
    :param freq: 'm': monthly or 'd': daily
    :param verbose: whether print information
    :return: dataframe
    """
    if source == 'crsp':
        if not Path(f"data/crsp_{freq}.h5").is_file():
            if verbose:
                print(f"No cache found at data/crsp_{freq}.h5")
                print('Loading from WRDS...')
            db = wrds.Connection(wrds_username=WRDS_USERNAME)
            df = db.raw_sql(
                f"select permno, date, prc, vol, ret, shrout, cfacpr, cfacshr from crspq.{freq}sf where date between '{HISTORY_START_DATE}' and '{HISTORY_END_DATE}'")
            db.close()

            df.permno = df.permno.astype(int)
            if freq == 'm':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m'))
            elif freq == 'd':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
            df['acprc'] = df.prc / df.cfacpr
            df['dolvol'] = df.acprc * df.vol * df.cfacshr  # in hundreds for monthly data
            df['cap'] = df.acprc * df.shrout * df.cfacshr  # in thousands
            df = df.drop(['shrout', 'cfacpr', 'cfacshr'], axis=1)

            if verbose:
                print(f"Saving cache to data/crsp_{freq}.h5 with key='raw'")
            df.to_hdf(f"data/crsp_{freq}.h5", key='raw')

            return df

        else:
            if verbose:
                print(f"Loading cache from data/crsp_{freq}.h5 with key='raw'")
            return pd.read_hdf(f'data/crsp_{freq}.h5', key='raw')

    elif source == 'ff':
        if not Path(f"data/ff_{freq}.h5").is_file():
            if freq == 'm':
                label = 'monthly'
            elif freq == 'd':
                label = 'daily'
            else:
                raise ValueError

            if verbose:
                print(f"No cache found at data/ff_{freq}.h5")
                print('Loading from WRDS...')
            db = wrds.Connection(wrds_username=WRDS_USERNAME)
            df = db.raw_sql(
                f"select date, mktrf, smb, hml, rf, umd from ff.factors_{label} where date between '{HISTORY_START_DATE}' and '{HISTORY_END_DATE}'")
            db.close()

            if freq == 'm':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m'))
            elif freq == 'd':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))

            if verbose:
                print(f"Saving cache to data/ff_{freq}.h5 with key='raw'")
            df.to_hdf(f"data/ff_{freq}.h5", key='raw')

            return df

        else:
            if verbose:
                print(f"Loading cache from data/ff_{freq}.h5 with key='raw'")
            return pd.read_hdf(f"data/ff_{freq}.h5", key='raw')

    else:
        raise ValueError


def get_data(source, freq, key='raw', verbose=True):
    """
    Helper function to get data and handle cache
    :param source: wrds library name
    :param freq: 'm': monthly or 'd': daily
    :param key: key for cached data
    :param verbose: whether print information
    :return: dataframe
    """
    if source == 'crsp':
        if not Path(f"data/crsp_{freq}.h5").is_file():
            get_raw_data(source, freq)

        try:
            if verbose:
                print(f"Loading cache from data/crsp_{freq}.h5 with key='{key}'")
            return pd.read_hdf(f"data/crsp_{freq}.h5", key=key)

        except KeyError:
            if verbose:
                print(f"No cache found at data/crsp_{freq}.h5 with key='{key}'")

            if key in ['ret', 'acprc', 'dolvol', 'cap']:
                df = pd.read_hdf(f"data/crsp_{freq}.h5", 'raw')[['permno', 'date', key]]
                df = df.pivot('date', 'permno', key)

                if verbose:
                    print(f"Saving cache to data/crsp_{freq}.h5 with key='{key}'")
                df.to_hdf(f"data/crsp_{freq}.h5", key=key)

                return df

            else:
                raise ValueError

    elif source == 'ff':
        if not Path(f"data/ff_{freq}.h5").is_file():
            get_raw_data(source, freq)

        if verbose:
            print(f"Loading cache from data/ff_{freq}.h5 with key='{key}'")
        return pd.read_hdf(f"data/ff_{freq}.h5", key=key)

    else:
        raise ValueError


def get_raw_trading_dates():
    if not Path('data/cal.csv').is_file():
        cal = pd.DataFrame({'full': get_data('ff', 'd', verbose=False).date})
        cal['year'] = cal.full.apply(lambda x: int(x[:4]))
        cal['month'] = cal.full.apply(lambda x: int(x[5:7]))
        cal['day'] = cal.full.apply(lambda x: int(x[8:]))
        cal.to_csv('data/cal.csv')

        return cal

    else:
        return pd.read_csv('data/cal.csv', index_col=0)


def _is_date_valid(cal, year, month, day):
    if year is not None:
        assert isinstance(year, (int, str))
        year = int(year)
        assert int(HISTORY_START_DATE[:4]) <= int(year) <= int(HISTORY_END_DATE[:4])

    if month is not None:
        assert isinstance(month, (int, str))
        month = int(month)
        assert 1 <= month <= 12

    if day is not None:
        assert isinstance(day, (int, str)) and year is not None and month is not None
        day = int(day)
        assert len(cal.query("year == @year and month == @month and day == @day")) == 1

    return year, month, day


def get_trading_dates(by='all', year=None, month=None, day=None, before=0, after=0):
    cal = get_raw_trading_dates()
    year, month, day = _is_date_valid(cal, year, month, day)
    assert before >= 0 and after >= 0

    if by == 'all':
        pass
    elif by == 'first':
        cal = cal.groupby(['year', 'month']).min()
    elif by == 'last':
        cal = cal.groupby(['year', 'month']).max()
    else:
        raise ValueError

    if year is None and month is None:
        assert before == 0 and after == 0

        return cal.full.to_list()

    elif year is not None and month is None:
        if before == 0 and after == 0:
            return cal.query("year == @year").full.to_list()

        else:
            return cal.query("@year - @before + 1 <= year <= @year").full.to_list(), \
                   cal.query("@year < year <= @year + @after").full.to_list()

    elif year is None and month is not None:
        assert before == 0 and after == 0

        return cal.query("month == @month").full.to_list()

    else:
        if day is None:
            if before == 0 and after == 0:
                return cal.query("year == @year and month == @month").full.to_list()

            else:
                assert by == 'all'
                cal_orig = cal.copy()
                cal = cal.groupby(['year', 'month']).max().reset_index()
                base_idx = cal.query("year == @year and month == @month").index[0]
                from_idx = base_idx - before + 1
                to_idx = base_idx + after
                assert from_idx >= 0 and to_idx < len(cal)

                from_idx = cal_orig.query(
                    "year == @cal.iloc[@from_idx].year and month == @cal.iloc[@from_idx].month").index[0]
                base_idx = cal_orig.query(
                    "year == @cal.iloc[@base_idx].year and month == @cal.iloc[@base_idx].month").index[-1]
                to_idx = cal_orig.query(
                    "year == @cal.iloc[@to_idx].year and month == @cal.iloc[@to_idx].month").index[-1]

                return cal_orig[from_idx: base_idx + 1].full.to_list(), \
                       cal_orig[base_idx + 1: to_idx + 1].full.to_list()

        else:
            if before == 0 and after == 0:
                return cal.query("year == @year and month == @month and day == @day").full.to_list()

            else:
                assert by == 'all'
                base_idx = cal.query("year == @year and month == @month and day == @day").index[0]
                from_idx = base_idx - before + 1
                to_idx = base_idx + after
                assert from_idx >= 0 and to_idx < len(cal)

                return cal[from_idx: base_idx + 1].full.to_list(), cal[base_idx + 1: to_idx + 1].full.to_list()


def gen_trading_dates(year, month, day, before, after, rolling_freq):
    cal = get_raw_trading_dates()
    year, month, day = _is_date_valid(cal, year, month, day)
    assert type(before) is int and type(after) is int and type(rolling_freq) is int
    assert before > 0 and after > 0 and rolling_freq > 0

    if day is None:
        cal_orig = cal.copy()
        cal = cal.groupby(['year', 'month']).max().reset_index()
        base_idx = cal.query("year == @year and month == @month").index[0]
        from_idx = base_idx - before + 1
        to_idx = base_idx + after
        assert from_idx >= 0 and to_idx < len(cal)

        while from_idx >= 0 and to_idx < len(cal):
            from_idx = cal_orig.query(
                "year == @cal.iloc[@from_idx].year and month == @cal.iloc[@from_idx].month").index[0]
            split_idx = cal_orig.query(
                "year == @cal.iloc[@base_idx].year and month == @cal.iloc[@base_idx].month").index[-1]
            to_idx = cal_orig.query(
                "year == @cal.iloc[@to_idx].year and month == @cal.iloc[@to_idx].month").index[-1]

            yield cal_orig[from_idx: split_idx + 1].full.to_list(), cal_orig[split_idx + 1: to_idx + 1].full.to_list()

            base_idx += rolling_freq
            from_idx = base_idx - before + 1
            to_idx = base_idx + after

    else:
        base_idx = cal.query("year == @year and month == @month and day == @day").index[0]
        from_idx = base_idx - before + 1
        to_idx = base_idx + after
        assert from_idx >= 0 and to_idx < len(cal)

        while from_idx >= 0 and to_idx < len(cal):
            yield cal[from_idx: base_idx + 1].full.to_list(), cal[base_idx + 1: to_idx + 1].full.to_list()

            base_idx += rolling_freq
            from_idx = base_idx - before + 1
            to_idx = base_idx + after


def get_universe(n_sample=500, seed=42, verbose=True):
    key = f"sample{n_sample}_seed{seed}"

    if Path('data/universe.h5').is_file():
        try:
            if verbose:
                print(f"Loading cache from data/universe.h5 with key='{key}'")
            return pd.read_hdf('data/universe.h5', key=key)

        except KeyError:
            if verbose:
                print(f"No cache found at data/universe.h5 with key='{key}'")
            pass

    else:
        if verbose:
            print(f"No cache found at data/universe.h5")

    acprcm = get_data('crsp', 'm', 'acprc', verbose=False)
    dolvolm = get_data('crsp', 'm', 'dolvol', verbose=False)
    capm = get_data('crsp', 'm', 'cap', verbose=False)

    universe = []
    for dates_before, dates_after in gen_trading_dates(year=BACKTEST_START_DATE[:4], month=BACKTEST_START_DATE[5:7],
                                                       day=None, before=120, after=2, rolling_freq=12):
        dates = list(dict.fromkeys([x[:7] for x in dates_before + dates_after]))

        acprc = acprcm.query("date in @dates")
        acprc = acprc[acprc > 0].dropna(axis=1)
        mask_acprc = acprc.ge(acprc.quantile(0.2, axis=1), axis=0).all()

        dolvol = dolvolm.query("date in @dates")
        dolvol = dolvol[dolvol > 0].dropna(axis=1)
        mask_dolvol = dolvol.ge(dolvol.quantile(0.2, axis=1), axis=0).all()

        cap = capm.query("date in @dates")
        cap = cap[cap > 0].dropna(axis=1)
        mask_cap = cap.ge(cap.quantile(0.2, axis=1), axis=0).all()

        mask = mask_acprc & (mask_dolvol | mask_cap)

        universe.append(
            pd.Series(np.sort(mask[mask].sample(n_sample, random_state=seed).index.to_list()), name=dates_after[0][:7]))

    universe = pd.concat(universe, axis=1)
    if verbose:
        print(f"Saving cache to data/universe.h5 with key='{key}'")
    universe.to_hdf('data/universe.h5', key=key)

    return universe
