# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import spacy

from src.utils import home, load_dataset

import pandas as pd

nlp = spacy.load('en_core_web_sm')

def inspect_df(df):
    print('shape {}'.format(df.shape))
    print(' ')
    
    print('nulls')
    print('---')
    for col in df.columns:
        print(col, sum(df.loc[:, col].isnull()))

def generate_labels(data):
    data = data.copy()
    #  small bug with the first day - should be nan
    data.loc[:, 'close-day-before'] = data.loc[:, 'Close'].shift(1)
    data.loc[:, 'mask'] = data['Close'] >= data['close-day-before']
    data.loc[:, 'label'] = data['mask'].astype('int64')
    return data
    
def test_generate_labels():
    data = pd.DataFrame({
        'Close': [0, 1, 0, 1, 2]
    })
    
    expected_labels = [0, 1, 0, 1, 1]
    labels = generate_labels(data)
    assert all(expected_labels == labels.loc[:, 'label'].values)
    return labels


def clean_news_headline(sample):
    sample = sample.lower()
    sample = sample.replace('/n', '')
    sample = sample.replace("\'", '')
    doc = nlp(sample)
    lemmas = [token.lemma_ for token in doc if (not token.is_stop and not token.is_punct and not token.is_space)]
    return ' '.join(lemmas)


@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # logger.info(input_filepath, output_filepath)
    raw = home / 'data' / 'raw'
    raw = load_dataset(raw)

    dija = raw['DJIA_table_train'].copy()
    expected_labels = generate_labels(dija)
    dija_labels = expected_labels.loc[:, 'label'].to_frame()
    dija_labels.to_csv(home / 'data' / 'interim' / 'dija-labels.csv')

    comb = raw['Combined_News_DJIA_train'].copy()

    news_cols = [c for c in comb.columns if 'Top' in c]

    for name in news_cols:
        col = comb.loc[:, name]
        col = col.fillna(' ')
        col = col.apply(lambda x: x.strip('b'))
        col = col.apply(lambda x: x.strip('"'))
        col = col.apply(lambda x: x.strip("'"))
        comb.loc[:, name] = col

    comb.loc[:, 'Label'] = comb.loc[:, 'Label'].fillna(dija_labels.loc[:, 'label'])
    assert sum(comb.loc[:, 'Label'].isnull()) == 0
    comb.iloc[:, 1:] = comb.iloc[:, 1:].fillna(" ")
    print('saving combined to data/interim')
    comb.to_csv(home / 'data' / 'interim' / 'combined.csv', index=True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
