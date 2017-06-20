
from src.pyspark_sc import *
import systemml as sml
import numpy as np
from sklearn import datasets


m1 = sml.matrix(np.ones((3, 3)) + 2)
m2 = sml.matrix(np.ones((3, 3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Split the data into training/testing sets
X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]
# Train Linear Regression model
X = sml.matrix(X_train)
y = sml.matrix(np.matrix(y_train).T)
A = X.transpose().dot(X)
b = X.transpose().dot(y)
beta = sml.solve(A, b).toNumPy()
y_predicted = X_test.dot(beta)
print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2))



sc.stop()