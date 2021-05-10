from pathlib import Path

import pandas as pd
import wrds


def get_raw_data(source, freq):
    if source == 'crsp':
        if not Path(f'data/crsp_{freq}.h5').is_file():
            print(f'No cache found at data/crsp_{freq}.h5')
            print(f'Loading from WRDS...')
            db = wrds.Connection(wrds_username='realethanzou')
            df = db.raw_sql(
                f"select permno, date, prc, vol, ret, shrout, cfacpr from crsp.{freq}sf where date between '1991-01-01' and '2020-12-31'")
            db.close()

            df.permno = df.permno.astype(int)
            if freq == 'd':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
            elif freq == 'm':
                df.date = df.date.apply(lambda x: x.strftime('%Y-%m'))
            df['acprc'] = df.prc / df.cfacpr
            df['cap'] = df.prc * df.shrout
            df = df.drop(['shrout', 'cfacpr'], axis=1)
            print(f'Saving cache to data/crsp_{freq}.h5')
            df.to_hdf(f'data/crsp_{freq}.h5', key='raw')

            return df

        else:
            print(f'Loading cache from data/crsp_{freq}.h5')
            return pd.read_hdf(f'data/crsp_{freq}.h5', key='raw')


def get_data(source, freq, key='raw'):
    if source == 'crsp':
        if not Path(f'data/crsp_{freq}.h5').is_file():
            get_raw_data(source, freq)

        try:
            print(f'Loading cache from data/crsp_{freq}.h5')
            return pd.read_hdf(f'data/crsp_{freq}.h5', key)
        except KeyError:
            print(f'No cache found at data/crsp_{freq}.h5')
            if key == 'ret':
                df = pd.read_hdf(f'data/crsp_{freq}.h5', 'raw')[['permno', 'date', 'ret']]
                df = df.pivot('date', 'permno', 'ret')
                df.to_hdf(f'data/crsp_{freq}.h5', key=key)

                return df
