from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

# script = """
#          top_k_row = function(matrix[double] X, integer r, integer k)
#                 return (matrix[double] values, matrix[double] indices) {
#                 #TODO does k need to be checked in the valid range
#                 row = X[r, ]
#                 row_t = t(row)
#                 indices = order(target=row_t, by=1, decreasing=TRUE, index.return=TRUE)
#                 indices = t(indices)
#                 indices = indices[1, 1:k]
#
#                 values = matrix(0, rows=1, cols=k)
#                 for (i in 1:k) {
#                     values[1, i] = row[1, as.scalar(indices[1, i])]
#                 }
#           }
#
#           top_k = function(matrix[double] X, integer k)
#                 return (matrix[double] values, matrix[double] indices) {
#                 N = nrow(X)
#                 D = ncol(X)
#                 values = matrix(0, rows=N, cols=k)
#                 indices = matrix(0, rows=N, cols=k)
#
#                 parfor (r in 1:N) {
#                     [value, index] = top_k_row(X, r, k)
#                     values[r, ] = value
#                     indices[r, ] = index
#                 }
#           }
#
#           [values, indices] = top_k(X, k)
#          """
script = """
         source("nn/util.dml") as util
         [values, indices] = util::top_k(X, k)
         """

ml = MLContext(sc)

matrix = np.array([[0.1, 0.4, 0.4, 0.5], [0.4, 0.1, 0.6, 0.1], [0.7, 0.8, 0.3, 0.2]])

k = 4
out = ('values', 'indices')

prog = (dml(script).input(X=matrix, k=k).output(*out))

values, indices = ml.execute(prog).get(*out)

print("Input Matrix:")
print(matrix)

print("Matrix for Top %d: " % k)
print(values.toNumPy())

print("Indices for Top %d: " % k)
print(indices.toNumPy())