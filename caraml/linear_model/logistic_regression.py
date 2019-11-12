from math import exp
from numpy import concatenate, dot, ones
from numpy.random import random

import numpy as np
def _sigmoid(z):
    return 1/(1 + exp(-z))

def _predict(features, weights):
    z = dot(features, weights)
    return np.array([[_sigmoid(el) for el in z]]).T

def _cost_function(features, weights, labels):
    observations = len(labels)
    predictions = _predict(features, weights)

    # error when label is 1
    class1_cost = -labels * np.log(predictions)

    # error when label is 0
    class0_cost = -(1-labels) * np.log((1-predictions))

    # total cost
    cost = class0_cost + class1_cost

    return cost.sum()/observations

def _update_weights(features, weights, labels, lr):
    observations = len(features)
    predictions = _predict(features, weights)

    gradient = np.dot(features.T, predictions-labels)

    gradient /= observations

    gradient *= lr

    weights = weights - gradient

    return weights

class LogisticRegression(object):
    def __init__(self, fit_intercept=True, learning_rate = 0.005, n_iter=10000, tol=0.0001):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if len(X.shape) != 2 or len(y.shape) != 2 or X.shape[0] != y.shape[0] or y.shape[1] != 1:
            raise Exception("X should be of shape (rows, features); y should be of shape: (rows, 1)")

        if self.fit_intercept:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)

        features = X
        labels = y
        weights = random((X.shape[1], 1))
        self.cost_history = []
        for _ in range(self.n_iter):
            weights = _update_weights(features, weights, labels, self.learning_rate)
            cost = _cost_function(features, weights, labels)
            if len(self.cost_history) > 1 and self.cost_history[-1] - cost < self.tol:
                break
            self.cost_history.append(cost)

        self.coef_ = weights
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)
        predictions = _predict(X, self.coef_)
        return np.array([[1 if pr >= 0.5 else 0 for pr in predictions]]).T

    def predict_proba(self, X):
        if self.fit_intercept:
            X = concatenate((ones((X.shape[0], 1)), X), axis=1)
        predictions = _predict(X, self.coef_)
        return predictions

    def score(self, X, y):
        return len(np.where(self.predict(X) == y)[0])/len(y)


