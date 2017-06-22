import numpy as np
import pandas as pd
import re
# preproccessibng

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer


def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv", names=['Headers'])
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv", names=['Headers'])
    return haaretzHeadlings, israelHayomHeadlines


def normelizeText(s):
    s = s.lower()
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s


def main():
    haaretzHeadlings, israelHayomHeadlines = readFiles()
    print(haaretzHeadlings)
    print(israelHayomHeadlines)

    # init vectorizer
    vectorizer1 = CountVectorizer()
    Vheaders1 = vectorizer1.fit_transform(haaretzHeadlings['Headers'])
    vectorizer2 = CountVectorizer()
    Vheaders2 = vectorizer2.fit_transform(israelHayomHeadlines['Headers'])
    print('---------------')
    print(Vheaders2)
    print('---------------')
    print(Vheaders2.shape)
    print('---------------')
    print(Vheaders1)
    print('---------------')
    print(vectorizer1.get_feature_names())
    print(vectorizer2.get_feature_names())




if __name__ == '__main__':
    main()
