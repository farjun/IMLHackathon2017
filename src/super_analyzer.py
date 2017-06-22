import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler


def readFiles():
    haaretz_headlines = pd.read_csv("../Training set/Headlines/haaretz.csv", names=['Headers'])
    israel_hayom_headlines = pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['Headers'])
    return haaretz_headlines, israel_hayom_headlines


haaretzHeadlings, israelHayomHeadlines = readFiles()
haaretzHeadlings['label'] = 0
israelHayomHeadlines['label'] = 1

base_df = pd.concat([haaretzHeadlings, israelHayomHeadlines])

list_them_all = []

for headline in haaretzHeadlings['Headers']:
    list_them_all.append(headline)
for headline in israelHayomHeadlines['Headers']:
    list_them_all.append(headline)

# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(list_them_all)
# split into train and test sets
x_train, x_test, y_train, y_test = \
    train_test_split(x, np.append(haaretzHeadlings['label'], israelHayomHeadlines['label']),
                     test_size=0.1, random_state=0)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression(n_jobs=2, random_state=0), n_jobs=2)

# Fit the classifier to the training data
clf.fit(x_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(x_test, y_test)))

clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(x_train, y_train)
print("Accuracy: {}".format(clf.score(x_test, y_test)))

rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(x_train)
clf = SGDClassifier()
clf.fit(X_features, y_train)
print("Accuracy: {}".format(clf.score(x_test, y_test)))
