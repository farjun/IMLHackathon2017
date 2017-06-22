# get some libraries that will be useful
#import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder



def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv")
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv")
    return haaretzHeadlings, israelHayomHeadlines

haaretzHeadlings ,israelHayomHeadlines = readFiles()



# grab the data
news = pd.read_csv("uci-news-aggregator.csv")
# let's take a look at our data
print(news.head())
# 对新闻标题的处理
def normalize_text(s):
    s = s.lower()
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)
    return s
news['TEXT'] = [normalize_text(s) for s in news['TITLE']]
print(news['TITLE'].head())
# 准备训练集和测试集
# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(news['TEXT'])
encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])
# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 开始预测
nb = MultinomialNB()
nb.fit(x_train, y_train)
print(nb.score(x_test, y_test))
