import pandas as pd

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from pandas import DataFrame
from collections import OrderedDict
from sklearn.externals import joblib
import os.path
import joblib
import nltk

nltk.data.path.append('../nltk_data')
from nltk import pos_tag, word_tokenize

all_headlines = None
vectorizer = None
lengths = []
avg_word_len = []
dot_count = []
tags = OrderedDict()
i = 0


def read_files():
    return pd.read_csv("../Training set/Headlines/haaretz.csv", names=['headlines']), \
           pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['headlines'])


def extract_features(headlines):
    global i

    for h in headlines:
        lengths.append(len(h))
        word_lengths = [len(w) for w in h.split()]
        avg_word_len.append(sum(word_lengths) / len(word_lengths))
        dot_count.append(h.count('.'))
        pt = pos_tag(word_tokenize(h))

        for tok_tag in pt:
            tag = tok_tag[1]

            if tag in tags:
                tags[tag][i] += 1
            else:
                tags[tag] = [0] * len(all_headlines)
                tags[tag][i] = 1
        i += 1


def process(headlines):
    global i
    x = vectorizer.transform(headlines)
    df = DataFrame(x.A, columns=vectorizer.get_feature_names())

    for h in headlines:
        lengths.append(len(h))
        word_lengths = [len(w) for w in h.split()]
        avg_word_len.append(sum(word_lengths) / len(word_lengths))
        dot_count.append(h.count('.'))
        pt = pos_tag(word_tokenize(h))

        for tok_tag in pt:
            tag = tok_tag[1]

            if tag in tags:
                tags[tag][i] += 1

        i += 1

    df['_lengths'] = lengths
    df['_avg_word_len'] = avg_word_len
    df['_dot'] = dot_count

    for (k, v) in tags.items():
        df[k] = v[:i]

    return df


if __name__ == '__main__':
    mlp = None

    if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl') and os.path.exists('tags.pkl')):
        print('Fetching data...')
        haaretz_headlines, israel_hayom_headlines = read_files()
        haaretz_headlines['label'] = 0
        israel_hayom_headlines['label'] = 1
        all_headlines = np.concatenate((haaretz_headlines['headlines'], israel_hayom_headlines['headlines']))

        print('Vectorizing data...')
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        x = vectorizer.fit_transform(all_headlines)
        df = DataFrame(x.A, columns=vectorizer.get_feature_names())

        print('Processing Haaretz headlines...')
        extract_features(haaretz_headlines)
        print('Processing Israel Hayom headlines...')
        extract_features(israel_hayom_headlines)

        print('Aggregating features...')
        df['_lengths'] = lengths
        df['_avg_word_len'] = avg_word_len
        df['_dot'] = dot_count

        for (k, v) in tags.items():
            df[k] = v

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(df.values,
                                                            np.append(haaretz_headlines['label'],
                                                                      israel_hayom_headlines['label']),
                                                            test_size=0.2,
                                                            random_state=42)

        print('Training model...')
        mlp = MLPClassifier(verbose=True)
        mlp.fit(x_train, y_train)
        joblib.dump(mlp, 'model.pkl')
        for t in tags:
            tags[t] = [0] * len(tags[t])
        joblib.dump(tags, 'tags.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        print(mlp.score(x_test, y_test))
    else:
        mlp = joblib.load('model.pkl')
        tags = joblib.load('tags.pkl')
        vectorizer = joblib.load('vectorizer.pkl')

        p = process(["Israel, India pledge to fight evils of terrorism together"])
        print(mlp.predict(sparse.csr_matrix(p.values)))
