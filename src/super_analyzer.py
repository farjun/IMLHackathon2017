import pandas as pd

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame
from collections import OrderedDict

import nltk

nltk.data.path.append('../nltk_data')
from nltk import pos_tag, word_tokenize

all_headlines = None
lengths = []
avg_word_len = []
dot = []
tags = OrderedDict()


def read_files():
    return pd.read_csv("../Training set/Headlines/haaretz.csv", names=['headlines']), \
           pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['headlines'])


def process(headlines):
    i = 0

    for h in headlines:
        lengths.append(len(h))
        word_lengths = [len(w) for w in h]
        avg_word_len.append(sum(word_lengths) / len(word_lengths))
        dot.append(h.count('.'))
        pt = pos_tag(word_tokenize(h))

        for tok_tag in pt:
            tag = tok_tag[1]

            if tag in tags:
                tags[tag][i] += 1
            else:
                tags[tag] = [0] * len(all_headlines)
                tags[tag][i] = 1

        i += 1


if __name__ == '__main__':
    print('Fetching data...')
    haaretz_headlines, israel_hayom_headlines = read_files()
    haaretz_headlines['label'] = 0
    israel_hayom_headlines['label'] = 1
    all_headlines = np.concatenate((haaretz_headlines['headlines'], israel_hayom_headlines['headlines']))

    print('Vectorizing data...')
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    x = vectorizer.fit_transform(all_headlines)
    df = DataFrame(x.A, columns=vectorizer.get_feature_names())

    import pickle

    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(list(df), f)

    print('Processing Haaretz headlines...')
    process(haaretz_headlines['headlines'])
    print('Processing Israel Hayom headlines...')
    process(israel_hayom_headlines['headlines'])

    print('Aggregating features...')
    df['lengths'] = lengths
    df['avg_word_len'] = avg_word_len
    df['dot'] = dot

    with open('tags.pkl', 'wb') as f:
        pickle.dump(tags, f)

    for (k, v) in tags.items():
        df[k] = v

    # Split into train and test sets
    print("df shape - trainer")
    print(df.shape)
    df.reindex_axis(sorted(df.columns), axis=1)

    with open('all_titles.pkl', 'wb') as f:
        pickle.dump(list(df), f)

    x_train, x_test, y_train, y_test = train_test_split(sparse.csr_matrix(df.values),
                                                        np.append(haaretz_headlines['label'],
                                                                  israel_hayom_headlines['label']),
                                                        test_size=0.5, random_state=42)

    print('Training model...')
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    with open('mlp.pkl', 'wb') as f:
        pickle.dump(mlp, f)
    # Print accuracy results
    print(f"Train Accuracy: {mlp.score(x_train, y_train)}")
    print(f"Test Accuracy: {mlp.score(x_test, y_test)}")

    import classifier

    clf = classifier.Classifier()
    prd = clf.classify(df.values)
    from sklearn.metrics import accuracy_score

    score = accuracy_score(y_test, prd)
    print(score)
