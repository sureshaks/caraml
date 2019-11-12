from pandas import read_csv
import numpy as np
from scipy import linalg
class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.array(X)
        row = X.shape[0]
        col = X.shape[1]
        stdX = np.vstack([(X[:, i] - np.mean(X[:, i])) for i in range(X.shape[1])]).T
        symmX = np.dot(stdX.T, stdX)/(row-1)
        w, V = np.linalg.eig(symmX)
        self.pc = V
        return self

    def transform(self, X):
        X = np.array(X)
        row = X.shape[0]
        col = X.shape[1]
        stdX = np.vstack([(X[:, i] - np.mean(X[:, i])) for i in range(X.shape[1])]).T
        return np.dot(stdX, self.pc)[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)