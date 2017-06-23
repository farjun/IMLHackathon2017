"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Auther(s):

===================================================
"""
import pandas as pd

import numpy as np
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pandas import DataFrame
from nltk import pos_tag, word_tokenize
from collections import OrderedDict


class Classifier(object):
    all_headlines = None
    lengths = []
    avg_word_len = []
    dot = []
    tags = OrderedDict()

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
                    self.tags[tag] = [0] * (len(headlines) )
                    self.tags[tag][i] += 1

            i += 1

    def classify(self, X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """
        self.all_headlines = None
        self.lengths = []
        self.avg_word_len = []
        self.dot = []
        self.tags = OrderedDict()

        print('Fetching data...')
        self.all_headlines = X

        print('Vectorizing data...')
        with open('vocabulary.pkl', 'rb') as f:
            vocabulary = pickle.load(f)

        #print(vocabulary)
        vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=vocabulary)
        x = vectorizer.fit_transform(self.all_headlines)
        df = DataFrame(x.A, columns=vectorizer.get_feature_names())
        print(df.head())
        print(df.shape)
        print(df[vocabulary].shape)
        print(len(vocabulary))
        print(vocabulary)
        #print(vocabulary.shape)
        #df = df[vocabulary]

        print('Processing all headlines...')
        self.process(self.all_headlines)

        print('Aggregating features...')
        df['lengths'] = self.lengths
        df['avg_word_len'] = self.avg_word_len
        df['dot'] = self.dot

        with open('tags.pkl', 'rb') as f:
            saved_tags = pickle.load(f)


        for (k, v) in saved_tags.items():
            if k in self.tags:
                df[k] = self.tags[k]
            else:
                df[k] = 0

        with open('all_titles.pkl', 'rb') as f:
            all_titles = pickle.load(f)

        df.reindex_axis(sorted(df.columns), axis=1)
        #
        #
        #
        # print(len(list(set(list(df)) & set(all_titles))))
        # print(len(all_titles))
        # print(len(list(df)))
        # for i in range(0, (len(list(df)))):
        #     if all_titles[i] != list(df)[i]:
        #         print("error")

        print("df shape")
        print(df.shape)
        with open('mlp.pkl', 'rb') as f:
            mlp = pickle.load(f)
        return mlp.predict(df)


# clf = Classifier()
# a = clf.classify(
#     ["Hundreds of Thousands of Muslims Celebrate Islam's Holiest Night in Jerusalem's Old City"])
# print(a)

def read_files():
    return pd.read_csv("../Training set/Headlines/haaretz.csv", names=['headlines']), \
           pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['headlines'])


print('Fetching data...')
haaretz_headlines, israel_hayom_headlines = read_files()
haaretz_headlines['label'] = 0
israel_hayom_headlines['label'] = 1
all_headlines = np.concatenate((haaretz_headlines['headlines'], israel_hayom_headlines['headlines']))

clf = Classifier()
prd = clf.classify(all_headlines)
from sklearn.metrics import accuracy_score
score = accuracy_score(np.append(haaretz_headlines['label'], israel_hayom_headlines['label']), prd)
print(score)