import numpy as np
import pandas as pd
import re
#preproccessibng
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords

#algorithems
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def readFiles():
    haaretzHeadlings = pd.read_csv("./Training set/Headlines/haaretz.csv",names = ['Headers'])
    israelHayomHeadlines = pd.read_csv("./Training set/Headlines/israelhayom.csv",names=['Headers'])

    haaretzHeadlings['tag'] = pd.Series('H', index=haaretzHeadlings.index)
    israelHayomHeadlines['tag'] = pd.Series('I',index=israelHayomHeadlines.index)
    res = pd.DataFrame(pd.concat([haaretzHeadlings,israelHayomHeadlines]))
    return res


def normelizeText(s):
    s = s.lower()
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s

def getVectores():
    res = readFiles()

    vectorizer = CountVectorizer(stop_words='english')
    x = vectorizer.fit_transform(res['Headers'])
    encoder = LabelEncoder()
    y = encoder.fit_transform(res['tag'])
    return x,y

def getTrainSplit():
    x,y = getVectores()
    return train_test_split(x, y, test_size=0.2,random_state = 0)

######################## main ###################################
x_train, x_test, y_train, y_test = getTrainSplit()

nb = MultinomialNB(alpha=.11)
knn = KNeighborsClassifier(n_neighbors=1)
svc = SVC(random_state=0)

nb.fit(x_train, y_train)
knn.fit(x_train,y_train)
svc.fit(x_train,y_train)

print("MultinomialNB: ",nb.score(x_test, y_test))
print("KNeighborsClassifier: ",knn.score(x_test, y_test))
print("SVC: ", svc.score(x_test, y_test))


###################################################################3

#change if you want to see another algorithem Chart
def getAlgo():
    return nb

def accuracy_plot():
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = getTrainSplit()

    Algo = getAlgo()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train == 0]
    mal_train_y = y_train[y_train == 0]
    ben_train_X = X_train[y_train == 1]
    ben_train_y = y_train[y_train == 1]

    mal_test_X = X_test[y_test == 0]
    mal_test_y = y_test[y_test == 0]
    ben_test_X = X_test[y_test == 1]
    ben_test_y = y_test[y_test == 1]

    scores = [Algo.score(mal_train_X, mal_train_y), Algo.score(ben_train_X, ben_train_y),
              Algo.score(mal_test_X, mal_test_y), Algo.score(ben_test_X, ben_test_y)]

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

    plt.xticks([0, 1, 2, 3], ['Haaretz\nTraining', 'Israel\nTraining', 'Haaretz\nTest', 'Israel\nTest'], alpha=0.8)
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()


#accuracy_plot()
