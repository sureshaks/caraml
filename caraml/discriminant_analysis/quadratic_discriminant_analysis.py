import numpy as np
import pandas as pd

class QuadraticDiscriminantAnalysis(object):
    def _discriminant_(self, X, k):
        # term 1
        term1 = np.diagonal(np.dot(X, np.dot(self.invcov[k], X.T)))
        term1 = -0.5 * np.expand_dims(term1, axis=1)

        # term2
        term2 = np.dot(X, np.dot(self.invcov[k], self.mu_k[k].T))
        term2 = np.expand_dims(term2, axis=1)

        # term 3
        term3 = np.dot(np.array([self.mu_k[k]]), np.dot(self.invcov[k], self.mu_k[k].T))
        term3 = -0.5 * np.repeat(np.array([term3]), X.shape[0], axis=0)

        # term 4
        term4 = -0.5 * np.log(np.linalg.det(self.cov[k]))
        term4 = np.repeat(np.array([[term4]]), X.shape[0], axis=0)

        # term 5
        term5 = np.log(np.repeat(self.pi_k[0][k], X.shape[0]))
        term5 = np.expand_dims(term5, axis=1)
        out = term1 + term2 + term3 + term4 + term5
        return out

    def fit(self, X, y):
        X = np.asarray(X)
        self.unique_y = np.unique(y)
        self.pi_k = np.expand_dims(np.bincount(y[:,0])/y.size, axis=0)

        # individual covariance
        self.cov = np.array([np.cov(X[np.where(y == el)[0], :].T) for el in self.unique_y])

        # lda assumes common covariance
        self.invcov = np.linalg.inv(self.cov)

        self.mu_k = np.vstack([np.mean(X[np.where(y == el),:][0], axis=0) for el in np.unique(y)])

        return self

    def predict(self, X):
        X = np.asarray(X)
        val = np.hstack([self._discriminant_(X, el) for el in self.unique_y])
        out = np.expand_dims(np.argmax(val, axis=1), axis=1)
        return out

    def score(self, X, y):
        return len(np.where(self.predict(X) == y)[0])/len(y)