from src.pyspark_sc import *

import pandas as pd
import systemml as sml
from sklearn import datasets

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
X_df = sqlCtx.createDataFrame(pd.DataFrame(X_digits[:int(.9 * n_samples)]))
y_df = sqlCtx.createDataFrame(pd.DataFrame(y_digits[:int(.9 * n_samples)]))
ml = sml.MLContext(sc)
# Run the MultiLogReg.dml script at the given URL
scriptUrl = "https://raw.githubusercontent.com/apache/systemml/master/scripts/algorithms/MultiLogReg.dml"
script = sml.dml(scriptUrl).input(X=X_df, Y_vec=y_df).output("B_out")
beta = ml.execute(script).get('B_out').toNumPy()

