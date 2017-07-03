from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import test
from tensorflow.python.training import momentum as momentum_lib


import tensorflow as tf

#tf.nn.softmax_cross_entropy_with_logits()
class SoftmaxCrossEntropyLossTest(test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=None)

  def testAllCorrect(self):
    with self.test_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEquals('softmax_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllWrong(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits,
                                          constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant((1.2, 3.4, 5.6))
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss.eval(), 3)

  def testAllWrongAllWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([0, 0, 0], shape=[3])
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testSomeWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([1.2, 0, 0], shape=[3])
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, loss.eval(), 3)

  def testSoftmaxWithMeasurementSpecificWeightsRaisesException(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=weights).eval()

  def testSoftmaxLabelSmoothing(self):
    with self.test_session():
      # Softmax Cross Entropy Loss is:
      #   -\sum_i p_i \log q_i
      # where for a softmax activation
      # \log q_i = x_i - \log \sum_j \exp x_j
      #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
      # For our activations, [100, -100, -100] the log partion function becomes
      # \log ( exp(0) + exp(-200) + exp(-200) ) = 0
      # so our log softmaxes become: [0, -200, -200]
      # so our cross entropy loss is:
      # -(1 - L + L/n) * 0 + 400 * L/n = 400 L/n
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = losses.softmax_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 3.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)

if __name__ == '__main__':
    #test.main()

    sess = tf.InteractiveSession()

    logits = constant_op.constant([[10.0, 0.0, 0.0],
                                   [10.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0],
                                   [10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0],
                                   [10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0],
                                   [10.0, 0.0, 0.0],
                                   [10.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0],
                                   [10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0],
                                   [10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])

    softmax_logits = tf.nn.softmax(logits=logits).eval()
    print(logits)

    print(softmax_logits)

    labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    loss = losses.softmax_cross_entropy(labels, logits)
    print(loss.eval())
    #0.0770996176470916