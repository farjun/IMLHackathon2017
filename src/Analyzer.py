import numpy as np
import pandas as pd
#preproccessibng

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv")
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv")
    return haaretzHeadlings, israelHayomHeadlines

haaretzHeadlings ,israelHayomHeadlines = readFiles()


