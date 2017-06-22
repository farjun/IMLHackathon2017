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
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv",names = ['Headres'])
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv",names=['Headres'])
    return haaretzHeadlings, israelHayomHeadlines

haaretzHeadlings ,israelHayomHeadlines = readFiles()


def normalize_text(s):
    s = s.lower()
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s

print(haaretzHeadlings)
haaretzHeadlings = [normalize_text(s) for s in haaretzHeadlings]
print("sdkjhfs",haaretzHeadlings)

