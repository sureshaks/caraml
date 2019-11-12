from math import exp
from numpy import concatenate, dot, ones
from numpy.random import random

import numpy as np
from pandas import read_csv

def _softmax(z):
    denom = sum([exp(el) for el in z])
    return [exp(el)/denom for el in z]

def _oneHot(y):
    # get number of unique values
    unique_y = np.unique(y)
    if(len(unique_y) == 1):
        raise Exception("There should be atleast 2 distinct values")

    return np.eye(len(unique_y))[y.T.reshape(-1)]

def _predict(features, weights):
    return np.array([_softmax(el) for el in np.dot(features, weights)])

def _cost_function(features, weights, labels):
    observations = len(labels)
    predictions = _predict(features, weights)
    return -np.sum(labels * np.log(predictions))/observations

def _update_weights(features, weights, labels, lr):
    observations = len(features)
    predictions = _predict(features, weights)

    gradient = np.dot(features.T, predictions-labels)

    gradient /= observations

    gradient *= lr

    weights = weights - gradient

    return weights

class SoftMaxRegression(object):
    def __init__(self, fit_intercept=True, learning_rate = 0.05, n_iter=10000):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        unique_y = np.unique(y)
        if len(X.shape) != 2 or len(y.shape) != 2 or X.shape[0] != y.shape[0] or y.shape[1] != 1:
            raise Exception("X should be of shape (rows, features); y should be of shape: (rows, 1)")

        if self.fit_intercept:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)

        encoded_y = _oneHot(y)

        features = X
        labels = y
        weights = random((X.shape[1], len(unique_y)))

        self.cost_history = []
        for _ in range(self.n_iter):
            weights = _update_weights(features, weights, labels, self.learning_rate)
            cost = _cost_function(features, weights, labels)
            if len(self.cost_history) > 1 and self.cost_history[-1] - cost < self.learning_rate/5:
                break
            self.cost_history.append(cost)

        self.coef_ = weights
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)
        return np.array([np.argmax(_predict(X, self.coef_), axis=1)]).T

    def predict_proba(self, X):
        if self.fit_intercept:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)
        return _predict(X, self.coef_)

    def score(self, X, y):
        return len(np.where(self.predict(X) == y)[0])/len(y)