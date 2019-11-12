from numpy import expand_dims, concatenate
from numpy.linalg import inv
import numpy as np

class LinearRegression(object):
    """
    LinearRegression - fits a linear model on X to predict y
    fit_intercept - whether to fit an intercept to the model or not
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # TODO: check the type of X
        # it should be compatible with numpy libraries
        X = np.array(X)
        y = np.array(y)

        if self.fit_intercept:
            X = concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

        self.coef_ = theta
        return self


    def predict(self, X):
        print(X)
