from pathlib import Path

import pandas as pd
import wrds

START_DATE = '1990-01-01'
END_DATE = '2021-03-31'


def get_raw_data(source, freq, verbose=True):
    if source == 'crsp':
        if not Path(f'data/crsp_{freq}.h5').is_file():
            if verbose:
                print(f'No cache found at data/crsp_{freq}.h5')
                print(f'Loading from WRDS...')
            db = wrds.Connection(wrds_username='realethanzou')
            df = db.raw_sql(
                f"select permno, date, prc, vol, ret, shrout, cfacpr from crspq.{freq}sf where date between {START_DATE} and {END_DATE}")
            db.close()

            df.permno = df.permno.astype(int)
            if freq == 'm':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m'))
            elif freq == 'd':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
            df['acprc'] = df.prc / df.cfacpr
            df['dolvol'] = df.prc * df.vol
            df['cap'] = df.prc * df.shrout
            df = df.drop(['shrout', 'cfacpr'], axis=1)
            if verbose:
                print(f"Saving cache to data/crsp_{freq}.h5 for key='raw'")
            df.to_hdf(f'data/crsp_{freq}.h5', key='raw')

            return df

        else:
            if verbose:
                print(f"Loading cache from data/crsp_{freq}.h5 for key='raw'")
            return pd.read_hdf(f'data/crsp_{freq}.h5', key='raw')

    elif source == 'ff':
        if not Path(f'data/ff_{freq}.h5').is_file():
            if verbose:
                print(f'No cache found at data/ff_{freq}.h5')
                print(f'Loading from WRDS...')
            if freq == 'm':
                label = 'monthly'
            elif freq == 'd':
                label = 'daily'
            else:
                raise ValueError
            db = wrds.Connection(wrds_username='realethanzou')
            df = db.raw_sql(
                f"select date, mktrf, smb, hml, rf, umd from ff.factors_{label} where date between {START_DATE} and {END_DATE}")
            db.close()

            if freq == 'm':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m'))
            elif freq == 'd':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
            if verbose:
                print(f"Saving cache to data/ff_{freq}.h5 for key='raw'")
            df.to_hdf(f'data/ff_{freq}.h5', key='raw')

            return df

        else:
            if verbose:
                print(f"Loading cache from data/ff_{freq}.h5 for key='raw'")

            return pd.read_hdf(f'data/ff_{freq}.h5', key='raw')

    else:
        raise ValueError


def get_data(source, freq, key='raw', verbose=True):
    if source == 'crsp':
        if not Path(f'data/crsp_{freq}.h5').is_file():
            get_raw_data(source, freq)

        try:
            if verbose:
                print(f"Loading cache from data/crsp_{freq}.h5 for key='{key}'")

            return pd.read_hdf(f'data/crsp_{freq}.h5', key=key)

        except KeyError:
            if verbose:
                print(f"No cache found at data/crsp_{freq}.h5 for key='{key}'")

            if key in ['ret', 'acprc', 'dolvol', 'cap']:
                df = pd.read_hdf(f'data/crsp_{freq}.h5', 'raw')[['permno', 'date', key]]
                df = df.pivot('date', 'permno', key)
                if verbose:
                    print(f"Saving cache to data/crsp_{freq}.h5 for key='{key}'")
                df.to_hdf(f'data/crsp_{freq}.h5', key=key)

                return df

            else:
                raise ValueError

    elif source == 'ff':
        if not Path(f'data/ff_{freq}.h5').is_file():
            get_raw_data(source, freq)

        if verbose:
            print(f"Loading cache from data/ff_{freq}.h5 for key='{key}'")

        return pd.read_hdf(f'data/ff_{freq}.h5', key=key)

    else:
        raise ValueError


def get_subset(df, base, before, after):
    base_idx = df.index.get_loc(base)
    from_idx = base_idx - before + 1
    to_idx = base_idx + after + 1
    assert from_idx >= 0 and to_idx <= len(df)

    return df.iloc[from_idx: to_idx]


def get_valid_subset(df_ret, df_acprc, df_dolvol, df_cap, base, before, after):
    mask = pd.Series(True, df_ret.columns)
    mask = mask[(get_subset(df_acprc, base, before, after) >= 5).all()]

    dolvol = get_subset(df_dolvol, base, before, after)
    dolvol = dolvol[dolvol > 0].dropna(axis=1)
    mask = mask[dolvol.ge(dolvol.quantile(0.2, axis=1), axis=0).all()]

    cap = get_subset(df_cap, base, before, after)
    cap = cap[cap > 0].dropna(axis=1)
    mask = mask[cap.ge(cap.quantile(0.2, axis=1), axis=0).all()]

    valid_permno = mask[mask].index.to_list()

    return get_subset(df_ret, base, before, after)[valid_permno].dropna(axis=1)


def get_sampled_subset(df_ret, n):
    return df_ret.sample(n, random_state=42, axis=1).sort_index(axis=1)


def get_trading_dates(by='all', month=None):
    cal = pd.DataFrame(get_data('ff', 'd', verbose=False).date)
    if by == 'all':
        return cal.date.to_list()

    cal['year-month'] = cal.date.apply(lambda x: x[:7])
    cal['date'] = cal.date.apply(lambda x: x[8:])
    if by == 'first':
        cal = cal.groupby('year-month').min()
    elif by == 'last':
        cal = cal.groupby('year-month').max()
    else:
        raise ValueError
    cal = list(cal.index.values + '-' + cal.date.values)

    if month is not None:
        return [_ for _ in cal if _[5:7] == f'{month:02}']

    return cal
