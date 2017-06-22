import numpy as np
import pandas as pd
import re as re

import sklearn
from sklearn.neighbors import NearestNeighbors
# import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors.ball_tree import BallTree
BallTree.valid_metrics

import sklearn as sk
#preproccessibng

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# #algorithems
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

def readFiles():
    haaretzHeadlings = pd.read_csv("/cs/usr/omribloch/PycharmProjects/IMLHackathon2017/Training set/Headlines/haaretz.csv", names = ['Headers'])
    israelHayomHeadlines = pd.read_csv("/cs/usr/omribloch/PycharmProjects/IMLHackathon2017/Training set/Headlines/israelhayom.csv", names=['Headers'])
    return haaretzHeadlings, israelHayomHeadlines

haaretzHeadlings ,israelHayomHeadlines = readFiles()

haaretzHeadlings_list = []
isreal_head_list = []
x = []


for headline in haaretzHeadlings['Headers']:
    wordList = re.sub("[^\w]", " ",  headline).split()
    haaretzHeadlings_list.append(wordList)
    x.append(wordList)

for headline in israelHayomHeadlines['Headers']:
    wordList = re.sub("[^\w]", " ",  headline).split()
    isreal_head_list.append(wordList)
    x.append(wordList)


# x.append(haaretzHeadlings_list)
# x.append(isreal_head_list)
x = pd.DataFrame(haaretzHeadlings_list)
print(x.head())

x_array = x.as_matrix()
print(x_array)



vectorizer = CountVectorizer()
# y = vectorizer.fit_transform(x)
# print(y)
# print(x)

def string_simpe_dist_metric(sentence1, sentece2):
    distance = 10.0
    for word1 in sentence1:
        if word1!=None:
            if word1 in sentece2:
                distance = distance / 2;

    return distance;

nbrs = NearestNeighbors(func=string_simpe_dist_metric)
print(len(x))
x.reshape(-1, 1)
nbrs.fit(x, np.zeros(shape=(len(x),1)))


























# _array = np.array(haaretzHeadlings_list)
# israel_array = np.array(isreal_head_list)
#
# for headline in haaretzHeadlings['Headers']:
#     wordList = re.sub("[^\w]", " ",  headline).split()
#     x.append(wordList)
#
# for headline in israelHayomHeadlines['Headers']:
#     wordList = re.sub("[^\w]", " ",  headline).split()
#     x.append(wordList)
#
#
# XX = np.array(x)
#
# print(haaretz_array.shape)
# print(haaretz_array)
# # print(x.shape)
# # print(x.head())
