from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

# script = """
#          threshold = function(matrix[double] X, double thresh)
#             return (matrix[double] out) {
#             out = X > thresh
#           }
#           out = threshold(X, thresh)
#          """

script = """
         source("nn/util.dml") as util
         out = util::threshold(X, thresh)
         """

ml = MLContext(sc)

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

prog = dml(script).input(X=matrix, thresh=5).output("out")

out = ml.execute(prog).get("out").toNumPy()

print(out)