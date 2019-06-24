import deprecated.create_dataframe as data1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import scipy


df = data1.word_matrix_dataframe()
matrix = data1.word_matrix()


titles = df.columns

a = data1.get_final_df_no_labeles().as_matrix()
lengh = np.sum(a,1)
df['length'] = lengh

def get_dataframe_with_lengh_and_label():
    return df
def get_dataframe_with_lengh_no_label():
    df2 = df.copy(True)
    del df2['the_label__']
    return df2

def get_data_as_sparse_matrix():
    return scipy.sparse.csr_matrix(df.values)

def get_data_as_sparse_matrix_no_label():
    df2 = df.copy(True)
    del df2['the_label__']
    return scipy.sparse.csr_matrix(df2.values)

