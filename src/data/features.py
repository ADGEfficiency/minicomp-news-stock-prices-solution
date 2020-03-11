# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from textblob import TextBlob

import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en_core_web_sm')

from sklearn.model_selection import TimeSeriesSplit
from src.data.cleaning import clean_news_headline
from src.utils import backend, home

def split_train_test(comb):
    split = TimeSeriesSplit(n_splits=2)

    for tr_idx, te_idx in split.split(comb):
        if backend == 'pandas':
            tr = comb.iloc[tr_idx, :]
            te = comb.iloc[te_idx, :]
        else:
            tr = comb.loc[tr_idx.tolist()]
            te = comb.iloc[te_idx, :]

    assert tr.shape[1] == te.shape[1]
    assert tr.shape[0] + te.shape[0] == comb.shape[0]
    return tr, te


def split_features_target(combined, name, load=False):
    print('splitting {} into x, y'.format(name))
    if load:
        print('loading from data/interim')
        corpus = pd.read_csv(home / 'data' / 'interim' / '{}-features.csv'.format(name), index_col=0)
        target = pd.read_csv(home / 'data' / 'interim' / '{}-target.csv'.format(name), index_col=0)
        return corpus, target

    target = combined.loc[:, 'Label'].to_frame()
    target.columns = ['target']

    corpus = combined.drop(['Label'], axis=1)
    corpus = corpus.agg(' '.join, axis=1)
    print('cleaning news headlines')
    corpus = corpus.apply(clean_news_headline)
    corpus = corpus.to_frame()
    corpus.columns = ['news']
    print('target shape {} distribution - {}'.format(target.shape, np.mean(target.values)))

    print('saving to data/interim')
    corpus.to_csv(home / 'data' / 'interim' / '{}-features.csv'.format(name))
    target.to_csv(home / 'data' / 'interim' / '{}-target.csv'.format(name))
    return corpus, target


def gensim_tokenize(docs):
    tokens = []
    for doc in docs:
        doc = remove_stopwords(doc)
        tokens.append(list(tokenize(doc, lower=True)))
    return tokens


def get_doc_vecs(docs, model):
    vecs = []
    for sample in docs:
        vecs.append(model.infer_vector(sample))
    return np.array(vecs)


def make_document_vectors(x_tr, x_te):
    tr_tokens = gensim_tokenize(x_tr.loc[:, 'news'].values)
    te_tokens = gensim_tokenize(x_te.loc[:, 'news'].values)

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tr_tokens)]
    model = Doc2Vec(documents, vector_size=32, window=3, min_count=1, workers=4, verbose=1)

    tr_vecs = get_doc_vecs(tr_tokens, model)
    te_vecs = get_doc_vecs(te_tokens, model)

    cols = ['doc2vec-{}'.format(i) for i in range(tr_vecs.shape[1])]
    tr_vecs = pd.DataFrame(tr_vecs, index=x_tr.index, columns=cols)
    te_vecs = pd.DataFrame(te_vecs, index=x_te.index, columns=cols)

    return tr_vecs, te_vecs


def make_day_feature(x_tr, x_te):
    tr_dates = pd.to_datetime(x_tr.index).dayofweek.to_frame()
    tr_dates.columns = ['dayofweek']
    te_dates = pd.to_datetime(x_te.index).dayofweek.to_frame()
    te_dates.columns = ['dayofweek']
    return tr_dates, te_dates


def sentiment(row):
    return TextBlob(row).sentiment.polarity


def subjectivity(row):
    return TextBlob(row).sentiment.subjectivity


def sentiment_subjectivity(df):
    df = df.copy()
    df.loc[:, 'sent'] = df.loc[:, 'news'].apply(sentiment)
    df.loc[:, 'subj'] = df.loc[:, 'news'].apply(subjectivity)
    return df.drop('news', axis=1)


def find_entities(sample):
    doc = nlp(sample)

    ents = []
    for token in doc:
        if token.pos_ == 'PROPN' and token.tag_ == 'NNP':
            ents.append(token.text)

    return np.array(ents).reshape(1, -1)


def extract_ents_as_str(ents):
    only_str = []
    for row in ents:
        only_str.append(" ".join(row.flatten().tolist()))
    return only_str


def generate_ents(df):
    tokens = []
    print('generating tokens')
    for row in range(df.shape[0]):
        sample = df.iloc[row, :].loc['news']
        tokens.append(find_entities(sample))

    assert len(tokens) == df.shape[0]
    print('converting ents to str')
    return extract_ents_as_str(tokens)


def process_combined(comb, name):

    tr, te = split_train_test(comb)

    x_tr, y_tr = split_features_target(tr, 'train')
    x_te, y_te = split_features_target(te, 'test')

    doc2vec = True
    time = True
    subj = True
    ents = True

    tr_features, te_features = [], []

    if doc2vec:
        tr_vecs, te_vecs = make_document_vectors(x_tr, x_te)
        tr_features.append(tr_vecs)
        te_features.append(te_vecs)

    if time:
        tr_dates, te_dates = make_day_feature(x_tr, x_te)
        tr_features.append(tr_dates)
        te_features.append(te_dates)

    if subj:
        x_s_tr = sentiment_subjectivity(x_tr)
        x_s_te = sentiment_subjectivity(x_te)
        tr_features.append(x_s_tr)
        te_features.append(x_s_te)

    if ents:
        tr_ents = generate_ents(x_tr)
        te_ents = generate_ents(x_te)

        enc = CountVectorizer()
        tr_ents = enc.fit_transform(tr_ents).todense()
        te_ents = enc.transform(te_ents).todense()
        tr_ents = pd.DataFrame(tr_ents, index=x_tr.index)
        te_ents = pd.DataFrame(te_ents, index=x_te.index)
        tr_features.append(tr_ents)
        te_features.append(te_ents)

    assert len(tr_features) == len(te_features)

    for tr, te in zip(tr_features, te_features):
        print(tr.shape, te.shape)
        tr.set_index(x_tr.index, drop=True, inplace=True)
        te.set_index(x_te.index, drop=True, inplace=True)

    print('starting the final concats')
    tr_features = pd.concat(tr_features, axis=1)
    te_features = pd.concat(te_features, axis=1)

    dataset = {
        'x_tr': tr_features,
        'x_te': te_features,
        'y_tr': y_tr,
        'y_te': y_te,
    }

    print('saving features & target into data/processed')
    for k, v in dataset.items():
        v.to_csv(home / 'data' / 'processed' / '{}.csv'.format(k))

    return dataset


@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # logger.info(input_filepath, output_filepath)

    from src.utils import load_dataset, home

    raw = home / 'data' / 'interim'
    raw = load_dataset(raw)
    comb = raw['combined']
    process_combined(comb, 'processed')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
