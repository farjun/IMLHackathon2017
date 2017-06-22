import create_dataframe as data1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


df = data1.word_matrix_dataframe()
matrix = data1.word_matrix()


a = np.array(matrix.sum(0))
print(a.shape)

# plt.plot(a.T, '*')
# plt.show()

print(type(df['is']))
# print((matrix))
#
# print(df.head())
from sklearn.naive_bayes import GaussianNB




X_train, X_test, y_train, y_test = train_test_split(matrix.toarray(), df['the_label__'], test_size=0.9, random_state=42)

clf = GaussianNB()
# clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
print(cross_val_score(clf, X_train, y_train))


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
print(cross_val_score(clf, X_train, y_train))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=500)
print(cross_val_score(neigh, matrix, df['the_label__']))

from sklearn import svm
clf = svm.SVC(decision_function_shape=None, degree=10)
print(cross_val_score(clf, matrix, df['the_label__']))

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)

X_2 = poly.fit_transform(X_train)
clf = MultinomialNB()
print(cross_val_score(clf, X_2, y_train))