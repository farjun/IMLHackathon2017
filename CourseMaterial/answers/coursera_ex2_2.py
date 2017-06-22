import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

y_mush = mush_df2.iloc[:, 1]
X_mush = mush_df2.iloc[:, 2:]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train2, y_train2)
    features = []

    for feature, importance in zip(X_train2.columns, tree.feature_importances_):
        features.append((importance, feature))

    features.sort(reverse=True)

    return [f[1] for f in features[:5]]


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    svc = SVC(random_state=0)
    gamma = np.logspace(-4, 1, 6)

    train_scores, test_scores = validation_curve(svc, X_subset, y_subset,
                                                 param_name='gamma',
                                                 param_range=gamma,
                                                 scoring='accuracy')
    train_scores = train_scores.mean(axis=1)
    test_scores = test_scores.mean(axis=1)

    return train_scores, test_scores


def answer_seven():
    """
    :return: (Underfitting, Overfitting, Good_Generalization)
    """
    return 0.0001, 10, 0.1
