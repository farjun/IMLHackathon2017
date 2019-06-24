from collections import OrderedDict
import pandas as pd
import numpy as np

lengths = []
avg_word_len = []
dot_count = []
tags = OrderedDict()
from nltk import pos_tag, word_tokenize

def extract_features(headlines : pd.Series ):
    i = 0
    headlines['_length'] = headlines['headlines'].str.len()
    headlines['_avg_word_len'] =  headlines['headlines'].str.len() / headlines['headlines'].str.count('\s')
    headlines['_dot'] = headlines['headlines'].str.count('\.')
    return headlines