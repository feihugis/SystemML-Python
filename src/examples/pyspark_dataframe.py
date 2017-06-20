from src.pyspark_sc import *

import pandas as pd
import systemml as sml
from sklearn import datasets
from sklearn.metrics import accuracy_score
from systemml.mllearn import LogisticRegression



digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
df_train = sml.convertToLabeledDF(sqlCtx, X_digits[:int(.9 * n_samples)], y_digits[:int(.9 * n_samples)])
X_test = spark.createDataFrame(pd.DataFrame(X_digits[int(.9 * n_samples):]))
logistic = LogisticRegression(spark)
logistic.fit(df_train)
y_predicted = logistic.predict(X_test)
y_predicted = y_predicted.select('prediction').toPandas().as_matrix().flatten()
y_test = y_digits[int(.9 * n_samples):]
print('LogisticRegression score: %f' % accuracy_score(y_test, y_predicted))