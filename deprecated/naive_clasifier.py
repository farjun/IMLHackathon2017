import create_dataframe as data1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


df = data1.word_matrix_dataframe()
matrix = data1.word_matrix()

import multi_classifier as ms
supClas = ms.SuperClassifier()
supClas.fit(data1.get_final_df_no_labeles(), df['the_label__'])

titles = df.columns
print(titles)
print(len(titles))

#
a = data1.get_final_df_no_labeles().as_matrix()
a = np.sum(a,0)
# print(type(a))
# print(a.shape)

filtered = titles[a>50]
print(len(filtered))
print(filtered)

df_reduced = df[filtered]
# df_reduced.drop('the_label__')
# counter = np.sum(a,0)
#
# df_reduced = df[counter>1]
# print(df_reduced.head())
# print(df_reduced.shape)
#
#
# # plt.plot(a.T, '*')
# # plt.show()
#
# print(type(df['is']))
# print((matrix))
#
# print(df.head())
from sklearn.naive_bayes import GaussianNB




# X_train, X_test, y_train, y_test = train_test_split(matrix.toarray(), df['the_label__'], test_size=0.25, random_state=42)
#
# clf = GaussianNB()
# # clf.fit(X_train, y_train)
# from sklearn.model_selection import cross_val_score
# print(cross_val_score(clf, X_train, y_train))
#
#
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# print(cross_val_score(clf, X_train, y_train))
#
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=500)
# print(cross_val_score(neigh, matrix, df['the_label__']))
#
# from sklearn import svm
# clf = svm.SVC(decision_function_shape=None, degree=2)
# print(cross_val_score(clf, matrix, df['the_label__']))
#
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)
#
# X_2 = poly.fit_transform(X_train)
# clf = MultinomialNB()
# print(cross_val_score(clf, X_2, y_train))


# ---------------------------------------------------------- reduced data
print((df_reduced.head().as_matrix()))
X_train, X_test, y_train, y_test = train_test_split(df_reduced.as_matrix(), df['the_label__'], test_size=0.25, random_state=42)
#
# clf = GaussianNB()
# # clf.fit(X_train, y_train)
# from sklearn.model_selection import cross_val_score
# print(cross_val_score(clf, X_train, y_train))
#
#
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# print(cross_val_score(clf, X_train, y_train))
#
# import scipy
# X_sparse = scipy.sparse.csr_matrix(df_reduced.values)
#
#
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=500)
# print(cross_val_score(neigh, X_sparse, df['the_label__']))
#
# from sklearn import svm
# clf = svm.LinearSVC()
# print(cross_val_score(clf, X_sparse, df['the_label__']))
#
# # from sklearn.preprocessing import PolynomialFeatures
# # poly = PolynomialFeatures(2)
#
# # ---------------------------------------------------------------- 0.73 with reduced > 50
# from sklearn.model_selection import cross_val_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)
#
# X_2 = poly.fit_transform(X_train)
# clf = MultinomialNB()
# print(cross_val_score(clf, X_2, y_train))


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
import scipy as scipy
poly = PolynomialFeatures(2)

import poly as poly
print(type(matrix))
# X_2 = poly.polynomial_features(matrix.tocsc(), 2)
X_2 = poly.polynomial_features(scipy.sparse.csc_matrix(df_reduced.as_matrix()), 2)
# X_2 = poly.fit_transform(X_train)
print(type(X_2))
# print(X_2.shape)
# clf = MultinomialNB()
# print(cross_val_score(clf, X_2, y_train))