from src.pyspark_sc import *
import systemml as sml
from systemml import MLContext, dml
import numpy as np

script = """
         source("nn/util.dml") as util
         values = util::transpose_NCHW_to_CNHW(X, C)
         
         """

ml = MLContext(sc)

matrix2 = np.array([[[[0.1, 0.4, 0.4, 0.5],
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

matrix = np.array([[[range(0,4),
                      range(4, 8),
                      range(8, 12)],

                      [range(12,16),
                      range(16, 20),
                      range(20, 24)]],

                    [[range(0, 4),
                      range(4, 8),
                      range(8, 12)],

                     [range(12, 16),
                      range(16, 20),
                      range(20, 24)]]])
print(matrix.shape)
print(matrix)

C = 2



prog = dml(script).input(X=matrix.reshape([2, 2*3*4]), C=C).output("values")

values = ml.execute(prog).get("values")

print(values.toNumPy().reshape([2, 2, 3, 4]))
