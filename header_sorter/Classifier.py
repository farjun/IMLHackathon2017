from sklearn.neural_network import MLPClassifier


class HeadersClassifier(object):
    def __init__(self):
        self.mlp = MLPClassifier(verbose=True)


    def train(self,x_train,y_train):
        print('Training model...')
        self.mlp.fit(x_train, y_train)
        print("Done!")