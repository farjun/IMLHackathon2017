import numpy as np
import create_dataframe as data1
import scipy as scipy

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, ClassifierMixin


class SuperClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""
    multinomial_reg = None
    multinomial_poly2 = None
    gaussian_nb = None
    sgd_classifier = None
    one_vs_rest_clasifier = None
    linear_svc_classifier = None



    def __init__(self):
        """
        Called when initializing the classifier
        """
        self.sgd_classifier = SGDClassifier(loss="hinge", penalty="l2")
        self.one_vs_rest_clasifier = OneVsRestClassifier(LogisticRegression(n_jobs=2, random_state=0), n_jobs=2)
        self.linear_svc_classifier = LinearSVC()
        self.multinomial_reg = MultinomialNB()
        self.multinomial_poly2 = MultinomialNB()
        self.gaussian_nb = GaussianNB()


    def fit(self, X, y):
        """
        X must(!) be the DataFrame ~without~ labeles!!
        """
        df = X
        X_matrix = X.as_matrix()
        X_sparse_matrix = scipy.sparse.csr_matrix(matrix)

        self.multinomial_reg.fit(X_sparse_matrix, y)
        self.gaussian_nb.fit(X_matrix, y)
        self.sgd_classifier.fit(X_sparse_matrix, y)
        self.one_vs_rest_clasifier.fit(X_sparse_matrix, y)
        self.linear_svc_classifier.fit(X_sparse_matrix, y)




        titles = df.columns
        print(titles)
        print(len(titles))
        a = data1.get_final_df_no_labeles().as_matrix()
        a = np.sum(a,0)
        filtered = titles[a>50]
        print(len(filtered))
        print(filtered)
        df_reduced = df[filtered]

        X_reduced = df_reduced.as_matrix()
        poly = PolynomialFeatures(2)
        X_2 = poly.fit_transform(X_reduced)

        self.multinomial_poly2.fit(X_reduced, y)

        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True if x >= self.treshold_ else False )

    def predict(self, X, y=None):
        res1 = self.multinomial_reg.predict(X)
        res2 = self.gaussian_nb.predict(X)
        res3 = self.sgd_classifier.predict(X)
        res4 = self.one_vs_rest_clasifier.predict(X)
        res5 = self.linear_svc_classifier.predict(X)
        res6 = self.multinomial_poly2.predict(X)

        final = ((np.sign(((res1 + res2 + res3 + res4 + res5 + res6) / 6) - 0.5) + 1) / 2 )

        return(final)

    def score(self, X, y=None):
        # counts success
        prd = self.predict(X)
        sum1 = 0
        for pi, yi in prd,y:
            if pi!=yi:
                sum1 = sum1 + 1
        return(sum1/y.size)






import create_dataframe as data1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


df = data1.word_matrix_dataframe()
matrix = data1.word_matrix()

# X_train, X_test, y_train, y_test = train_test_split(df, df['the_label__'], test_size=0.25, random_state=42)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

print(train.head())
X_train = train.drop('the_label__')
y_train = train['the_label']
X_test = test.drop('the_label__')
y_test = train['the_label']

import multi_classifier as ms
supClas = ms.SuperClassifier()
supClas.fit(X_train, y_train)

print(supClas.score(X_test, y_test))
