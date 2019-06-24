import pandas as pd

import nltk

nltk.data.path.append('../nltk_data')
from nltk import pos_tag, word_tokenize

def read_files():
    return pd.read_csv("../Training set/Headlines/haaretz.csv", names=['headlines']), \
           pd.read_csv("../Training set/Headlines/israelhayom.csv", names=['headlines'])