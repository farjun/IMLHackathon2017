import numpy as np
import pandas as pd
import re
from nltk import bigrams
from scipy.sparse import coo_matrix, hstack, scipy
import nltk

nltk.data.path.append('../nltk_data')
from nltk import pos_tag, word_tokenize

# preproccessibng
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

from matplotlib import pyplot as plt


# other

class Classifier(object):
    def __init__(self):
        self.train()

        def process(self, headlines):
            i = 0

            for h in headlines:
                self.lengths.append(len(h))
                word_lengths = [len(w) for w in h]
                self.avg_word_len.append(sum(word_lengths) / len(word_lengths))
                self.dot.append(h.count('.'))
                pt = pos_tag(word_tokenize(h))

                for tok_tag in pt:
                    tag = tok_tag[1]

                    if tag in self.tags:
                        self.tags[tag][i] += 1
                    else:
                        self.tags[tag] = [0] * (len(headlines))
                        self.tags[tag][i] += 1

                i += 1

    def readFiles(self):
        haaretzHeadlings = pd.read_csv("../Training set/Headlines/haaretz.csv", names=['Headers'])
        israelHayomHeadlines = pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['Headers'])
        haaretzHeadlings['length'] = haaretzHeadlings['Headers'].str.len()
        israelHayomHeadlines['length'] = israelHayomHeadlines['Headers'].str.len()
        haaretzHeadlings['tag'] = pd.Series('H', index=haaretzHeadlings.index)
        israelHayomHeadlines['tag'] = pd.Series('I', index=israelHayomHeadlines.index)

        res = pd.DataFrame(pd.concat([haaretzHeadlings, israelHayomHeadlines]))
        return res

    def normelizeText(self, s):
        s = s.lower()
        s = re.sub('\s\W', ' ', s)
        s = re.sub('\W\s', ' ', s)
        s = re.sub('\s+', ' ', s)
        return s

    def getVectores(self, res):
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x = vectorizer.fit_transform(list(res['Headers']))
        df = pd.DataFrame(x.A, columns=vectorizer.get_feature_names())
        lengh = np.sum(df, 1)
        df['length'] = lengh
        z2 = scipy.sparse.csr_matrix(df.values)

        encoder = LabelEncoder()
        y = encoder.fit_transform(res['tag'])
        return z2, y

    def getTrainSplit(self, res):
        x, y = self.getVectores(res)
        return train_test_split(x, y, test_size=0.5, random_state=0)

    def classify(self, x):
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform(x)
        df = pd.DataFrame(X.A, columns=vectorizer.get_feature_names())
        lengh = np.sum(df, 1)
        df['length'] = lengh
        z2 = scipy.sparse.csr_matrix(df.values)
        return self.ovrc.predict(z2)

    def train(self):
        x_train, x_test, y_train, y_test = self.getTrainSplit(self.readFiles())
        self.nb = MultinomialNB()
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.svc = SVC(kernel='linear')
        self.lsvc = LinearSVC()
        self.sig = SGDClassifier()
        self.ocs = OneClassSVM()
        self.linReg = LinearRegression(n_jobs=1)
        self.logReg = LogisticRegression(n_jobs=1)
        self.ovrc = OneVsRestClassifier(estimator=self.logReg, n_jobs=2)
        self.vot = VotingClassifier(estimators=[('nb', self.nb), ('knn', self.knn), ('svc', self.svc),
                                                ('lsvc', self.lsvc)])
        # self.mlp = MLPClassifier(random_state=1, warm_start=True)

        self.nb.fit(x_train, y_train)
        self.knn.fit(x_train, y_train)
        self.svc.fit(x_train, y_train)
        self.lsvc.fit(x_train, y_train)
        self.sig.fit(x_train, y_train)
        self.ocs.fit(x_train, y_train)
        self.ovrc.fit(x_train, y_train)
        self.linReg.fit(x_train, y_train)
        self.logReg.fit(x_train, y_train)
        self.vot.fit(x_train, y_train)
        # self.mlp.fit(x_train, y_train)

        print("MultinomialNB: ", self.nb.score(x_test, y_test))
        print("KNeighborsClassifier: ", self.knn.score(x_test, y_test))
        print("SVC: ", self.svc.score(x_test, y_test))
        print("OneVsRestClassifier: ", self.ovrc.score(x_test, y_test))
        print("Lin SVC: ", self.lsvc.score(x_test, y_test))
        print("Sig SVC: ", self.sig.score(x_test, y_test))
        print("LinearRegression: ", self.linReg.score(x_test, y_test))
        print("LogisticRegression: ", self.logReg.score(x_test, y_test))
        print("VotingClassifier: ", self.vot.score(x_test, y_test))

        # print("MLPClassifier: ", self.mlp.score(x_test, y_test))


import reduced_classifier
clf = reduced_classifier.Classifier()
c = clf
