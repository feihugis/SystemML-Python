from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

script = """
         source("nn/util.dml") as util
         [values, indices] = util::top_k2d(X, k, C, Hin, Win)
         """

ml = MLContext(sc)
ml.setExplain("HOPS")

matrix = np.array([[[[0.1, 0.4, 0.4, 0.5],
                    [0.4, 0.1, 0.6, 0.1],
                    [0.7, 0.8, 0.3, 0.2]],

                   [[0.2, 0.5, 0.4, 0.5],
                    [0.4, 0.1, 0.6, 0.1],
                    [0.7, 0.8, 0.3, 0.2]],

                   [[0.3, 0.4, 0.4, 0.5],
                    [0.4, 0.1, 0.6, 0.1],
                    [0.7, 0.8, 0.3, 0.2]]],

                   [[[0.1, 0.4, 0.4, 0.5],
                     [0.4, 0.1, 0.6, 0.1],
                     [0.7, 0.8, 0.3, 0.2]],

                    [[0.2, 0.5, 0.4, 0.5],
                     [0.4, 0.1, 0.6, 0.1],
                     [0.7, 0.8, 0.3, 0.2]],

                    [[0.3, 0.4, 0.4, 0.5],
                     [0.4, 0.1, 0.6, 0.1],
                     [0.7, 0.8, 0.3, 0.2]]]])
print(matrix.shape)

matrix_reshape = matrix.reshape([2, 3*3*4])
print(matrix_reshape)

k = 2
out = ('values', 'indices')

prog = (dml(script).input(X=matrix_reshape, k=k, C=3, Hin=3, Win=4).output(*out))

values, indices = ml.execute(prog).get(*out)

print("Matrix for Top %d: " % k)
print(values.toNumPy().reshape([2, k, 3, 4]))

print("Indices for Top %d: " % k)
print(indices.toNumPy().reshape([2, k, 3, 4]))
