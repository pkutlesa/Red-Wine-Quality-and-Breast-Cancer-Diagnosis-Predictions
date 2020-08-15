import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import stats
from linear_discriminant_analysis import LinearDiscriminantAnalysis
from logistic_regression import LogisticRegression

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# evaluate_acc, arguments: true labels (y_true), target labels (y_target)
def evaluate_acc(y_true, y_target):
    correct = 0
    total = 0
    for i in range(len(y_true)):
        if y_target[i] == y_true[i]:
            correct += 1
        total += 1
    return (correct / total) * 100


# 5-fold cross validation, returns average accuracy of the model
def cross_validate_logistic(X, y, alpha, num_iterations):
    num_data_points = len(y)
    one_fifth = math.ceil(num_data_points / 5)
    initial_theta = []
    sum_accuracy = 0
    for i in range(len(X.columns)):
        initial_theta.append(0)
    for i in range(5):
        x_valid = X.iloc[one_fifth * i:one_fifth * (i + 1)]
        y_valid = y.iloc[one_fifth * i:one_fifth * (i + 1)]
        x_train = X.drop(X.index[one_fifth * i:one_fifth * (i + 1)])
        y_train = y.drop(y.index[one_fifth * i:one_fifth * (i + 1)])
        x_train, y_train, x_valid, y_valid = np.array(x_train), np.array(y_train), np.array(x_valid), np.array(y_valid)
        LR = LogisticRegression(initial_theta)
        LR.fit(x_train, y_train, alpha, num_iterations)
        prediction = LR.predict(x_valid)
        sum_accuracy += evaluate_acc(y_valid, prediction)
    return sum_accuracy / 5


def cross_validate_lda(X, y):
    num_data_points = len(y)
    one_fifth = math.ceil(num_data_points / 5)
    sum_accuracy = 0
    for i in range(5):
        x_valid = X.iloc[one_fifth * i:one_fifth * (i + 1)]
        y_valid = y.iloc[one_fifth * i:one_fifth * (i + 1)]
        x_train = X.drop(X.index[one_fifth * i:one_fifth * (i + 1)])
        y_train = y.drop(y.index[one_fifth * i:one_fifth * (i + 1)])
        x_train, y_train, x_valid, y_valid = np.array(x_train), np.array(y_train), np.array(x_valid), np.array(y_valid)
        LDA = LinearDiscriminantAnalysis()
        LDA.fit(x_train, y_train)
        prediction = LDA.predict(x_valid)
        sum_accuracy += evaluate_acc(y_valid, prediction)
    return sum_accuracy / 5


# import and pre-process data wine data
def get_wine_data():
    df = pd.read_csv("winequality-red.csv", delimiter=";")
    df.columns.str.strip('\"')
    df['quality'] = np.where(df['quality'] > 5, 1, 0)
    df = df[(np.abs(stats.zscore(df)) < 4).all(axis=1)]

    # added features
    df['mSO2'] = df.apply(lambda row: row['free sulfur dioxide'] / (1 + pow(10, row['pH'] - 1.8)), axis=1)
    df['total acidity'] = df['fixed acidity'] + df['volatile acidity']
    df['alcohol to sulphates ratio'] = df['alcohol'] / (df['sulphates'])
    df['total SO2 x sulphates'] = df['total sulfur dioxide'] * df['sulphates']
    df['log chlorides'] = np.log10(df['chlorides'])
    df['log free sulfur dioxide'] = np.log10(df['free sulfur dioxide'])

    X = df.drop(columns=['quality'])
    y = df['quality']
    return X, y


# import and pre-process data cancer data
def get_cancer_data():
    names = ['sample code number', 'clump thickness', 'uniformity of cell size',
             'uniformity of cell shape', 'marginal adhesion', 'single epithelial cell size',
             'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses', 'class']
    df = pd.read_csv("breast-cancer-wisconsin.data", names=names)
    numdf = (df.drop(df.columns, axis=1)
             .join(df[df.columns].apply(pd.to_numeric, errors='coerce')))
    df = numdf[numdf[df.columns].notnull().all(axis=1)]
    df = df.drop(columns=['sample code number'])
    df['class'] = np.where(df['class'] == 4, 1, 0)
    df = df[(np.abs(stats.zscore(df)) < 4).all(axis=1)]

    # added feature
    df['cell uniformity'] = df['uniformity of cell size'] * df['uniformity of cell shape']

    X = df.drop(columns=['class', 'uniformity of cell size', 'uniformity of cell shape'])
    y = df['class']
    return X, y


# print correlation matrix
def print_corr_matrix(data_set):
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(data_set.corr(), cmap='coolwarm', annot=True, square=True)
    plt.show()


# main script
X_wine, y_wine = get_wine_data()
X_cancer, y_cancer = get_cancer_data()

acc1 = cross_validate_logistic(X_wine, y_wine, 0.000001, 100000)
print("Logistic Regression accuracy (wine):             ", acc1)

acc2 = cross_validate_lda(X_wine, y_wine)
print("Linear Discriminant Analysis accuracy (wine):    ", acc2)

acc3 = cross_validate_logistic(X_cancer, y_cancer, 0.000001, 100000)
print("Logistic Regression accuracy (cancer):           ", acc3)

acc4 = cross_validate_lda(X_cancer, y_cancer)
print("Linear Discriminant Analysis accuracy (cancer):  ", acc4)


# initial_theta = []
# for i in range(len(X_wine.columns)):
#     initial_theta.append(0)
# LR1 = LogisticRegression(initial_theta)
# cost1 = LR1.fit(X_wine, y_wine, 0.000001, 1000)
#
# LR2 = LogisticRegression(initial_theta)
# cost2 = LR2.fit(X_wine, y_wine, 0.0000001, 1000)
#
# LR3 = LogisticRegression(initial_theta)
# cost3 = LR3.fit(X_wine, y_wine, 0.00000001, 1000)
#
# fig,ax = plt.subplots(figsize=(12, 8))
# ax.set_ylabel('Cross Entropy Loss')
# ax.set_xlabel('Iterations')
# _=ax.plot(range(1000), cost1, 'r-', range(1000), cost2, 'b-', range(1000), cost3, 'g-')
# plt.show()
