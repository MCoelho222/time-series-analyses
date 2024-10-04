from __future__ import annotations

from os.path import abspath, dirname, join

import pandas as pd

from rhis_ts.evol.rhis import Rhis


def main():
    ROOTPATH = dirname(abspath(__file__))
    EXAMPLE_DIR_PATH = join(ROOTPATH, "examples/")

    df = pd.read_excel('./data/dataset.xlsx')
    print(df.info())
    df[df.select_dtypes(include=['object']).columns.drop('PONTO')] = \
        df.select_dtypes(include=['object']).drop(columns='PONTO').apply(pd.to_numeric, errors='coerce')
    df = df.loc[df['PONTO'] == 'IG5',
                ['DATA', 'NH4 (mg/L)', 'NT (mg/L)', 'T (°C)']]
    df['DATA'] = pd.to_datetime(df['DATA'])
    df.set_index('DATA', inplace=True)
    df.dropna(inplace=True)

    rhis = Rhis(df)
    # print(rhis.orig_df.info())
    # print(rhis.orig_df.head())
    rhis.evol(stat='min', rhis=True)
    rhis.add_repr_cols_to_df(direction='ba')
    title = "RHIS time series from backward evolution of p-values"
    rhis.plot(save_dir_path=EXAMPLE_DIR_PATH, rhis=False, show_repr=True, figtitle=title, figsize=(12, 6))

if __name__ == "__main__":
    main()
