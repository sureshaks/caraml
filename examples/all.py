# TODO:

# 1. add a score method for each element
# 2. re-implement logistic regression for multinomial y
# 3. set up a testing framework
# 4. add validation for data types
# 5. try the code on multiple examples

from numpy.random import random
from caraml.linear_model import LinearRegression
from caraml.linear_model import LogisticRegression
from caraml.discriminant_analysis import LinearDiscriminantAnalysis
from caraml.discriminant_analysis import QuadraticDiscriminantAnalysis
from pandas import read_csv
from numpy import array, expand_dims, unique, where

from caraml.metrics import confusion_matrix
from caraml.decomposition import PCA

# data
data = read_csv("iris.csv")
X = data.iloc[:,:4]
y = data.iloc[:,4]
unique_y = dict([(v,k) for k,v in enumerate(unique(y))])
y = expand_dims(array([unique_y[el] for el in y]), axis=1)

# linear discriminant analysis
lda = LinearDiscriminantAnalysis()
model = lda.fit(X, y)
print("LDA: ", model.score(X, y))

# quadratic discriminant analysis
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(X, y)
pr = model.predict(X)
print("QDA: ", model.score(X, y))

print(confusion_matrix(y, pr))

pca = PCA()
model = pca.fit(X)
print(model.transform(X))