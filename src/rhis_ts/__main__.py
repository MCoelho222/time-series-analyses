from __future__ import annotations

from os.path import abspath, dirname, join

import pandas as pd

from rhis_ts.evol.rhis import Rhis


def main():
    ROOTPATH = dirname(abspath(__file__))
    EXAMPLE_PATH = join(ROOTPATH, "examples/")

    # UFPR dataset
    df = pd.read_excel('./data/dataset.xlsx')
    df[df.select_dtypes(include=['object']).columns.drop('PONTO')] = \
        df.select_dtypes(include=['object']).drop(columns='PONTO').apply(pd.to_numeric, errors='coerce')
    df = df.loc[df['PONTO'] == 'IG5',
                ['DATA', 'NH4 (mg/L)']]
    df['DATA'] = pd.to_datetime(df['DATA'])
    df.set_index('DATA', inplace=True)
    df.dropna(inplace=True)

    savefig_path = f"{EXAMPLE_PATH}rhis_evol.png"
    rhis = Rhis(df)
    rhis.evol(stat='min', rhis=True)
    rhis.add_repr_cols_to_df(direction='ba')
    rhis.plot(savefig_path, rhis=True, show_repr=True)

if __name__ == "__main__":
    main()
