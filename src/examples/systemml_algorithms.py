from src.pyspark_sc import *

import numpy as np
from sklearn import datasets
from systemml.mllearn import LinearRegression
import systemml as sml

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
# Create linear regression object
regr = LinearRegression(spark, fit_intercept=True, C=float("inf"), solver='direct-solve')
# Train the model using the training sets
regr.fit(X_train, y_train)
y_predicted = regr.predict(X_test)
print('Residual sum of squares: %.2f' % np.mean((y_predicted - y_test) ** 2))


# Scikit-learn way
from sklearn import datasets, neighbors
from systemml.mllearn import LogisticRegression
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)
X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]
logistic = LogisticRegression(spark)
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))

logistic.save('logistic_model')
new_logistic = LogisticRegression(spark)
new_logistic.load('logistic_model')
print('LogisticRegression score: %f' % new_logistic.score(X_test, y_test))