import numpy as np
import pandas as pd

class LinearDiscriminantAnalysis(object):
    def _discriminant_(self, X):
        # term 1
        term1 = np.dot(X, np.dot(self.invcov, self.mu_k.T))

        # term 2
        row = np.array([np.diagonal(np.dot(self.mu_k, np.dot(self.invcov, self.mu_k.T)))])
        term2 = np.repeat(row, X.shape[0], axis=0)

        # term 3
        priors = np.repeat(np.array([self.pi_k[0]]), X.shape[0], axis=0)
        term3 = np.log(priors)
        
        out = term1 - 0.5 * term2 + term3

        return out

    def fit(self, X, y):
        X = np.asarray(X)
        self.unique_y = np.unique(y)
        self.pi_k = np.expand_dims(np.bincount(y[:,0])/y.size, axis=0)

        # pooled covariance
        pooled_cov = np.array(sum([np.cov(X[np.where(y == el)[0], :].T) * self.pi_k[0][el] for el in self.unique_y]))

        # lda assumes common covariance
        self.invcov = np.linalg.inv(pooled_cov)
        
        self.mu_k = np.vstack([np.mean(X[np.where(y == el),:][0], axis=0) for el in np.unique(y)])

        return self

    def predict(self, X):
        X = np.asarray(X)
        val = self._discriminant_(X)
        out = np.expand_dims(np.argmax(val, axis=1), axis=1)
        return out

    def score(self, X, y):
        return len(np.where(self.predict(X) == y)[0])/len(y)