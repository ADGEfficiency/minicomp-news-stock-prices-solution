from joblib import load

from src.utils import home, load_dataset

from src.data.features import process_combined

import pandas as pd


def main():

    mdl = load(home / 'models' / 'final.joblib')
    dataset = load_dataset(home / 'data' / 'holdout')

    comb = dataset['Combined_News_DJIA_test']
    f = process_combined(comb, 'holdout')

    x_tr, y_tr, x_te, y_te = f['x_tr'], f['y_tr'], f['x_te'], f['y_te']

    x = pd.concat([x_tr, x_te], axis=0)
    y = pd.concat([y_tr, y_te], axis=0)

    print('holdout x shape {}'.format(x.shape))

    acc = mdl.score(x, y)
    print('final model accuracy on holdout {}'.format(acc))



if __name__ == '__main__':
    main()
