import click
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.utils import load_dataset, home


rf_params = {'n_estimators': 500, 'max_depth': 5}
lr_params = {'C': 5}


def fit(mdl, x_tr, y_tr, x_te, y_te, vec=None):
    if vec:
        x_tr = vec.fit_transform(x_tr.loc[:, 'news'])
        x_te = vec.transform(x_te.loc[:, 'news'])

    y_tr = y_tr.values.flatten()
    y_te = y_te.values.flatten()

    mdl.fit(x_tr, y_tr)

    res = {
        'tr-score': mdl.score(x_tr, y_tr),
        'te-score': mdl.score(x_te, y_te),
        'avg-te-pred': np.mean(mdl.predict(x_te))
    }

    for k, v in res.items():
        print(k, v)

    return mdl, res


@click.command()
def main():

    f = load_dataset(home / 'data' / 'processed')
    x_tr, y_tr, x_te, y_te = f['x_tr'], f['y_tr'], f['x_te'], f['y_te']

    mdl = RandomForestClassifier(**rf_params)
    mdl, res = fit(mdl, x_tr, y_tr, x_te, y_te)

    print('fitting final model')
    x = pd.concat([x_tr, x_te], axis=0)
    y = pd.concat([y_tr, y_te], axis=0)

    mdl, res = fit(mdl, x, y, x, y)
    from joblib import dump
    dump(mdl, home / 'models' / 'final.joblib')


if __name__ == '__main__':
    main()
