import numpy as np
import pandas as pd
import re
#preproccessibng

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv",names = ['Headers'])
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv",names=['Headers'])
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


    #init vectorizer
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(haaretzHeadlings['Headers'])
    y = vectorizer.fit_transform(israelHayomHeadlines['Headers'])

if __name__ == '__main__':
    main()


