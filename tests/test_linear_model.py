import pytest
import numpy as np
from caraml.linear_model import LinearRegression
def test_linear_regression_fit_intercept():
    X = [[1], [2], [3]]
    y = [[1], [2], [3]]
    lr = LinearRegression()
    lr.fit(X, y)
    assert lr.coef_ == pytest.approx(np.array([[0.],[1.]]))

def test_linear_regression_no_intercept():
    X = [[1], [2], [3]]
    y = [[1], [2], [3]]
    lr = LinearRegression(fit_intercept = False)
    lr.fit(X, y)
    assert lr.coef_ == pytest.approx(np.array([[1.]]))