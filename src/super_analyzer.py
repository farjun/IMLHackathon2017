import pandas as pd

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame


def readFiles():
    haaretz_headlines = pd.read_csv("../Training set/Headlines/haaretz.csv", names=['Headers'])
    israel_hayom_headlines = pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['Headers'])
    return haaretz_headlines, israel_hayom_headlines


haaretzHeadlines, israelHayomHeadlines = readFiles()
haaretzHeadlines['label'] = 0
israelHayomHeadlines['label'] = 1

base_df = pd.concat([haaretzHeadlines, israelHayomHeadlines])

list_them_all = []
lengths = []
avg_word_len = []
dot = []

for headline in haaretzHeadlines['Headers']:
    lengths.append(len(headline))
    word_lengths = [len(w) for w in headline]
    avg_word_len.append(sum(word_lengths) / len(word_lengths))
    dot.append(headline.count('.'))
    list_them_all.append(headline)
for headline in israelHayomHeadlines['Headers']:
    lengths.append(len(headline))
    word_lengths = [len(w) for w in headline]
    avg_word_len.append(sum(word_lengths) / len(word_lengths))
    dot.append(headline.count('.'))
    list_them_all.append(headline)

# pull the data into vectors
vectorizer = CountVectorizer(ngram_range=(1, 2))
x = vectorizer.fit_transform(list_them_all)
print(x.shape)
df = DataFrame(x.A, columns=vectorizer.get_feature_names())
np.sum(df, 1)
df['lengths'] = lengths
df['avg_word_len'] = avg_word_len
df['dot'] = dot

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(sparse.csr_matrix(df.values),
                                                    np.append(haaretzHeadlines['label'],
                                                              israelHayomHeadlines['label']),
                                                    test_size=0.2, random_state=0)

vc = VotingClassifier(estimators=[('logReg', LogisticRegression(random_state=0)),
                                  ('rndfc', RandomForestClassifier(random_state=0))],
                      n_jobs=4)

vc.fit(x_train, y_train)

# Print the accuracy
print(f"Accuracy: {vc.score(x_test, y_test)}")
