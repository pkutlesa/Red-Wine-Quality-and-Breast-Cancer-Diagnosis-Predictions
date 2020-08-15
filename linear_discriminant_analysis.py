import numpy as np


class LinearDiscriminantAnalysis:

    # constructor, argument: model parameters, theta (initialized to zeros)
    def __init__(self):
        self.p_y_0 = 0
        self.p_y_1 = 0
        self.mean0 = []
        self.mean1 = []
        self.covariance_m = []

    # fit, arguments: training data (X, Y)
    def fit(self, X, y):
        self.covariance_m = np.zeros((len(X[0]), len(X[0])))
        # find and set P(y=0) and P(y=1)
        N0, N1 = 0, 0
        for i in range(len(y)):
            if y[i] == 0:
                N0 += 1
            else:
                N1 += 1
        self.p_y_0 = N0 / (N0 + N1)
        self.p_y_1 = N1 / (N0 + N1)
        # calculate the mean vectors for class 1 and 2
        for column in X.T:
            sum0, sum1 = 0, 0
            for i in range(len(y)):
                if y[i] == 0:
                    sum0 += column[i]
                else:
                    sum1 += column[i]
            self.mean0.append(sum0/N0)
            self.mean1.append(sum1/N1)
        # find covariance matrix
        class_mt_0 = np.zeros((len(X[0]), len(X[0])))
        class_mt_1 = np.zeros((len(X[0]), len(X[0])))
        for i in range(len(y)):
            row = X[i].reshape(len(X[0]), 1)
            mn0, mn1 = np.array(self.mean0), np.array(self.mean1)
            mn0 = mn0.reshape(len(X[0]), 1)
            mn1 = mn1.reshape(len(X[0]), 1)
            if y[i] == 0:
                class_mt_0 += (row - mn0).dot((row - mn0).T) / (N0 + N1 - 2)
            else:
                class_mt_1 += (row - mn1).dot((row - mn1).T) / (N0 + N1 - 2)
        self.covariance_m += class_mt_0
        self.covariance_m += class_mt_1

    def predict(self, X):
        mn0, mn1, cov_mat = np.array(self.mean0), np.array(self.mean1), np.array(self.covariance_m)
        predictions = []
        for i in range(len(X)):
            first = np.log10(self.p_y_1 / self.p_y_0)
            second = 0.5 * np.matmul(np.matmul(mn1.T, np.linalg.inv(cov_mat)), mn1)
            third = 0.5 * np.matmul(np.matmul(mn0.T, np.linalg.inv(cov_mat)), mn0)
            fourth = np.matmul(np.matmul(X[i].T, np.linalg.inv(cov_mat)), (mn1 - mn0))
            log_odds = first - second + third + fourth
            if log_odds > 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
