import math
import numpy as np


# helper function to compute sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# helper function to compute sigmoid(Wkt*xi)
def likelihood(x, weights):
    return sigmoid(np.dot(x, weights))


# helper function to calculate the cross entropy error
# def cross_entropy(like, y):
#     E = 0
#     for i in range(len(y.T)):
#         if np.array(y)[i] == 1:
#             E -= np.log(like[i])
#         else:
#             E -= np.log(1 - like[i])
#     return E


class LogisticRegression:

    # constructor, argument: model parameters, theta (initialized to zeros)
    def __init__(self, theta):
        self.weights = theta

    # fit, arguments: training data (X, Y), learning rate (alpha), # gradient descent iterations (num_iterations)
    def fit(self, X, y, alpha, num_iterations):
        # CEE_history = [] (used for calculating cross entropy error)
        for i in range(num_iterations):
            hypo = likelihood(X, self.weights)
            new_weights = np.add(self.weights, alpha * np.dot(X.T, np.subtract(y, hypo)))
            self.weights = new_weights
        #     CEE_history.append(cross_entropy(hypo, y))
        # return CEE_history

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            prediction = likelihood(X[i], self.weights)
            if prediction > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
