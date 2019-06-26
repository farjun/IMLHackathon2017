from collections import OrderedDict
import pandas as pd
import numpy as np
from nltk import pos_tag, word_tokenize
from textblob import TextBlob

tags = OrderedDict()
def get_tags():
    return tags



def extract_features(headlines):
    i = 0
    headlines['_length'] = headlines['headlines'].str.len()
    headlines['_avg_word_len'] =  headlines['headlines'].str.len() / headlines['headlines'].str.count('\s')
    headlines['_dot'] = headlines['headlines'].str.count('\.')

    for h in headlines['headlines']:
        blob = TextBlob(h)
        pt = blob.tags

        for tok_tag in pt:
            tag = tok_tag[1]

            if tag in tags:
                tags[tag][i] += 1
            else:
                tags[tag] = [0] * len(headlines)
                tags[tag][i] = 1
        i += 1

    for (k, v) in tags.items():
        headlines[k] = v

    return headlines