from pathlib import Path
import pandas as pd

home = Path.home() / 'minicomp-news-stock-prices-solution'


def load_dataset(path):
    raw = [p for p in path.iterdir() if (p.is_file() and p.suffix == '.csv')]
    raw = {p.stem: pd.read_csv(p, parse_dates=True, encoding='utf-8') for p in raw}
    for k, v in raw.items():
        print(k, v.shape)
        v.set_index('Date', drop=True, inplace=True)
    return raw