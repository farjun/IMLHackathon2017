import numpy as np
import pandas as pd
import re as re

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

def readFiles():
    haaretzHeadlings = pd.read_csv("/cs/usr/omribloch/PycharmProjects/IMLHackathon2017/Training set/Headlines/haaretz.csv", names = ['Headers'])
    israelHayomHeadlines = pd.read_csv("/cs/usr/omribloch/PycharmProjects/IMLHackathon2017/Training set/Headlines/israelhayom.csv", names=['Headers'])
    return haaretzHeadlings, israelHayomHeadlines

haaretzHeadlings ,israelHayomHeadlines = readFiles()
haaretzHeadlings['label'] = 0;
israelHayomHeadlines['label'] = 1;

frames = [haaretzHeadlings, israelHayomHeadlines]
base_df = pd.concat(frames)

# print(base_df)

list_them_all = []
for headline in haaretzHeadlings['Headers']:
    list_them_all.append(headline)
for headline in israelHayomHeadlines['Headers']:
    list_them_all.append(headline)



vect = CountVectorizer(min_df=0., max_df=1.0)
X = vect.fit_transform(list_them_all)
# print(type(X))
# print(DataFrame(X.A, columns=vect.get_feature_names()).to_string())
the_final_df = DataFrame(X.A, columns=vect.get_feature_names());
the_final_df['the_label__'] = list(base_df['label'])
# print(the_final_df.head())


def word_matrix_dataframe():
    return the_final_df

def word_matrix():
    return X
def base_df():
    return base_df


print(word_matrix_dataframe().head())