from pathlib import Path

import pandas as pd
import wrds

START_DATE = '1990-01-01'
END_DATE = '2021-03-31'


def get_raw_data(source, freq):
    if source == 'crsp':
        if not Path(f'data/crsp_{freq}.h5').is_file():
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
            print(f'Saving cache to data/crsp_{freq}.h5')
            df.to_hdf(f'data/crsp_{freq}.h5', key='raw')

            return df

        else:
            print(f'Loading cache from data/crsp_{freq}.h5')
            return pd.read_hdf(f'data/crsp_{freq}.h5', key='raw')

    elif source == 'ff':
        if not Path(f'data/ff_{freq}.h5').is_file():
            print(f'No cache found at data/ff_{freq}.h5')
            print(f'Loading from WRDS...')
            if freq == 'm':
                label = 'monthly'
            elif freq == 'd':
                label = 'daily'
            else:
                raise KeyError
            db = wrds.Connection(wrds_username='realethanzou')
            df = db.raw_sql(
                f"select date, mktrf, smb, hml, rf, umd from ff.factors_{label} where date between {START_DATE} and {END_DATE}")
            db.close()

            if freq == 'm':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m'))
            elif freq == 'd':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
            print(f'Saving cache to data/ff_{freq}.h5')
            df.to_hdf(f'data/ff_{freq}.h5', key='raw')

            return df

        else:
            print(f'Loading cache from data/ff_{freq}.h5')
            return pd.read_hdf(f'data/ff_{freq}.h5', key='raw')

    else:
        raise KeyError


def get_data(source, freq, key='raw'):
    if source == 'crsp':
        if not Path(f'data/crsp_{freq}.h5').is_file():
            get_raw_data(source, freq)

        try:
            print(f'Loading cache from data/crsp_{freq}.h5')
            return pd.read_hdf(f'data/crsp_{freq}.h5', key=key)

        except KeyError:
            print(f'No cache found at data/crsp_{freq}.h5')
            if key in ['ret', 'acprc', 'dolvol', 'cap']:
                df = pd.read_hdf(f'data/crsp_{freq}.h5', 'raw')[['permno', 'date', key]]
                df = df.pivot('date', 'permno', key)
                print(f'Saving cache to data/crsp_{freq}.h5')
                df.to_hdf(f'data/crsp_{freq}.h5', key=key)

                return df

            else:
                raise KeyError

    elif source == 'ff':
        if not Path(f'data/ff_{freq}.h5').is_file():
            get_raw_data(source, freq)

        print(f'Loading cache from data/crsp_{freq}.h5')
        return pd.read_hdf(f'data/ff_{freq}.h5', key=key)

    else:
        raise KeyError
