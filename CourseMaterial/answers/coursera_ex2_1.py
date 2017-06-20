import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    result = np.zeros((4, 100))
    X_train_vec = X_train.reshape(-1, 1)

    for i, degree in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train_vec)
        linreg = LinearRegression().fit(X_poly, y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0, 10, 100).reshape(-1, 1)))
        result[i, :] = y

    return result


# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i, degree in enumerate([1, 3, 6, 9]):
        plt.plot(np.linspace(0, 10, 100), degree_predictions[i], alpha=0.8, lw=2, label=f'{degree}')
    plt.ylim(-1, 2.5)
    plt.legend(loc=4)


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    r2_train = np.zeros(10, )
    r2_test = np.zeros(10, )
    X_train_vec = X_train.reshape(-1, 1)
    X_test_vec = X_test.reshape(-1, 1)

    for degree in range(10):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train_vec)
        linreg = LinearRegression().fit(X_poly, y_train)
        r2_train[degree] = r2_score(y_train, linreg.predict(X_poly))
        X_test_poly = poly.fit_transform(X_test_vec)
        r2_test[degree] = r2_score(y_test, linreg.predict(X_test_poly))

    return r2_train, r2_test


def answer_three():
    """
    Clearly, the lowest R_2 scores indicate the lowest accuracy, hence underfitting.
    That's why 0 is the degree level that corresponds to a model that is underfitting.
    The 9th degree corresponds to overfitting because it scores a nearly perfect R_2
    with the training data but scores poorly with the test data.
    The 6th degree (could also be 7th) provides a good generalization performance on
    the entire dataset because it scores a close to 1.0 R_2 with both training and test data.
    """
    return 0, 9, 6


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    poly = PolynomialFeatures(degree=12)
    X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
    X_poly = poly.fit_transform(X_train.reshape(-1, 1))
    linreg = LinearRegression().fit(X_poly, y_train)
    linRegTestScore = r2_score(y_test, linreg.predict(X_test_poly))
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_poly, y_train)
    lassoTestScore = r2_score(y_test, lasso.predict(X_test_poly))

    return linRegTestScore, lassoTestScore
