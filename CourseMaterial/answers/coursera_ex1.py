import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()

def answer_zero():
    return len(cancer.feature_names)


def answer_one():
    np.append(cancer.feature_names, ['target'])
    data = np.vstack((cancer.data.T, cancer.target)).T
    return pd.DataFrame(columns=cancer.feature_names, data=data).to_csv()

def answer_two():
    cancerdf = answer_one()
    targets = cancerdf.target
    lengths = [len(cancerdf[targets == 0]), len(cancerdf[targets == 1])]
    return pd.Series(lengths, index=['malignant', 'benign'])


def answer_three():
    cancerdf = answer_one()
    return cancerdf[cancer.feature_names], cancerdf.target


def answer_four():
    X, y = answer_three()
    return train_test_split(X, y, random_state=0)


def answer_five():
    X_train, _, y_train, _ = answer_four()

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    return knn


def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()

    return knn.predict(means)


def answer_seven():
    _, X_test, _, _ = answer_four()
    knn = answer_five()
    return knn.predict(X_test)


def answer_eight():
    _, X_test, _, y_test = answer_four()
    knn = answer_five()

    return knn.score(X_test, y_test)


def accuracy_plot():
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train == 0]
    mal_train_y = y_train[y_train == 0]
    ben_train_X = X_train[y_train == 1]
    ben_train_y = y_train[y_train == 1]

    mal_test_X = X_test[y_test == 0]
    mal_test_y = y_test[y_test == 0]
    ben_test_X = X_test[y_test == 1]
    ben_test_y = y_test[y_test == 1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0', '#4c72b0', '#55a868', '#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width() / 2, height * .90, '{0:.{1}f}'.format(height, 2),
                       ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0, 1, 2, 3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8)
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()


print(answer_three())