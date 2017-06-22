"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Spelling Corrector  **

Auther(s):

===================================================
"""

class Classifier(object):

    def classify(self,X):
        """
        Recieves a list of m corrupted words, and predicts 3 most likely corrections.
        :param X: A list of length m containing the words (strings)
        :return: y_hat - a matrix of size mx3. The i'th row has the prediction for the
                 i'th test sample, containing word indices of the correction candidates.
                 Word indices are specified in the file dictionary_5000.pickle
        """
    raise NotImplementedError("TODO: Implement this method by 12pm tomorrow!")
