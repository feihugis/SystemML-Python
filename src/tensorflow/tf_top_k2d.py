import tensorflow as tf
import numpy as np

matrix = np.array([[[[0.1, 0.4, 0.4, 0.5],
                    [0.4, 0.1, 0.6, 0.1],
                    [0.7, 0.8, 0.3, 0.2]],

                   [[0.2, 0.5, 0.4, 0.5],
                    [0.4, 0.1, 0.6, 0.1],
                    [0.7, 0.8, 0.3, 0.2]],

                   [[0.3, 0.4, 0.4, 0.5],
                    [0.4, 0.1, 0.6, 0.1],
                    [0.7, 0.8, 0.3, 0.2]]]])

print(matrix.shape)

matrix_reshape = matrix.reshape([1, 36])

print(matrix_reshape)

print(matrix_reshape)

sess = tf.InteractiveSession()
values, indices = tf.nn.top_k(matrix, 2)
print(values.eval().shape)
print(indices.eval())