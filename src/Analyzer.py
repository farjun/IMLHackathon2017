import numpy as np
import pandas as pd
import re
#preproccessibng
pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv",names = ['Headers'])
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv",names=['Headers'])
    haaretzHeadlings['tag'] = pd.Series(0, index=haaretzHeadlings.index)
    israelHayomHeadlines['tag'] = pd.Series(1,index=israelHayomHeadlines.index)
    res = pd.concat([haaretzHeadlings,israelHayomHeadlines])
    return res



def normelizeText(s):
    s = s.lower()
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s


def main():
    all = readFiles()
    print(all)


    #init vectorizer
    vectorizer = CountVectorizer()
    Vheaders1 = vectorizer.fit_transform(all['Headers'])
    print('---------------')
    print(Vheaders1)




if __name__ == '__main__':
    main()


