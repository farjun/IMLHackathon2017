import numpy as np
import pandas as pd
import re
#preproccessibng
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv",names = ['Headers'])
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv",names=['Headers'])
    haaretzHeadlings['tag'] = pd.Series('H', index=haaretzHeadlings.index)
    israelHayomHeadlines['tag'] = pd.Series('I',index=israelHayomHeadlines.index)
    res = pd.DataFrame(pd.concat([haaretzHeadlings,israelHayomHeadlines]))
    return res


def normelizeText(s):
    s = s.lower()
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s



if __name__ == '__main__':
    res = readFiles()


    #init vectorizer
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(res['Headers'])
    encoder = LabelEncoder()
    y = encoder.fit_transform(res['tag'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    print(nb.score(x_test, y_test))