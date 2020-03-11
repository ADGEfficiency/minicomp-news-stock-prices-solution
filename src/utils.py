from pathlib import Path

backend = 'pandas'
if backend == 'dask':
    import dask.dataframe as pd
else:
    import pandas as pd

home = Path.home() / 'minicomp-news-stock-prices-solution'


def load_dataset(path):
    raw = [p for p in path.iterdir() if (p.is_file() and p.suffix == '.csv')]
    raw = {p.stem: pd.read_csv(p, parse_dates=True, encoding='utf-8', index_col=0) for p in raw}
    for k, v in raw.items():
        print(k, v.shape)
        try:
            v = v.set_index('Date', drop=True)
        except KeyError:
            pass
    return raw
