import pandas as pd

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame
from nltk import pos_tag, word_tokenize
from collections import OrderedDict


def readFiles():
    haaretz_headlines = pd.read_csv("../Training set/Headlines/haaretz.csv", names=['Headers'])
    israel_hayom_headlines = pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['Headers'])
    return haaretz_headlines, israel_hayom_headlines


haaretzHeadlines, israelHayomHeadlines = readFiles()
haaretzHeadlines['label'] = 0
israelHayomHeadlines['label'] = 1

base_df = pd.concat([haaretzHeadlines, israelHayomHeadlines])

all_headlines = []
lengths = []
avg_word_len = []
dot = []
tags = OrderedDict()

i = 0
for headline in haaretzHeadlines['Headers']:
    lengths.append(len(headline))
    word_lengths = [len(w) for w in headline]
    avg_word_len.append(sum(word_lengths) / len(word_lengths))
    dot.append(headline.count('.'))
    pt = pos_tag(word_tokenize(headline))
    for t in pt:
        if t[1] in tags:
            tags[t[1]][i] += 1
        else:
            tags[t[1]] = [1] + [0] * (len(haaretzHeadlines) + len(israelHayomHeadlines) - 1)
    all_headlines.append(headline)
    i += 1

i = 0
for headline in israelHayomHeadlines['Headers']:
    lengths.append(len(headline))
    word_lengths = [len(w) for w in headline]
    avg_word_len.append(sum(word_lengths) / len(word_lengths))
    dot.append(headline.count('.'))
    pt = pos_tag(word_tokenize(headline))
    for t in pt:
        if t[1] in tags:
            tags[t[1]][i] += 1
        else:
            tags[t[1]] = [1] + [0] * (len(haaretzHeadlines) + len(israelHayomHeadlines) - 1)
    all_headlines.append(headline)
    i += 1

# pull the data into vectors
vectorizer = CountVectorizer(ngram_range=(1, 2))
x = vectorizer.fit_transform(all_headlines)
print(x.shape)
df = DataFrame(x.A, columns=vectorizer.get_feature_names())
df['lengths'] = lengths
df['avg_word_len'] = avg_word_len
df['dot'] = dot
for (k, v) in tags.items():
    df[k] = v

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(sparse.csr_matrix(df.values),
                                                    np.append(haaretzHeadlines['label'],
                                                              israelHayomHeadlines['label']),
                                                    test_size=0.2, random_state=42)

vc = MLPClassifier()
vc.fit(x_train, y_train)

# Print the accuracy

print(f"Train Accuracy: {vc.score(x_train, y_train)}")
print(f"Test Accuracy: {vc.score(x_test, y_test)}")
