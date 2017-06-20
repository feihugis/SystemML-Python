from src.pyspark_sc import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split  # module deprecated in 0.18
# from sklearn.model_selection import train_test_split  # use this module for >=0.18
from sklearn import metrics
from systemml import MLContext, dml

ml = MLContext(sc)
print("Spark Version: {}".format(sc.version))
print("SystemML Version: {}".format(ml.version()))
print("SystemML Built-Time: {}".format(ml.buildTime()))

# download MNIST
mnist = datasets.fetch_mldata("MNIST Original")

print("MNIST data features: {}".format(mnist.data.shape))
print("MNIST data labels: {}".format(mnist.target.shape))

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target.astype(np.uint8).reshape(-1, 1),
    test_size=10000)

print("Training images, labels: {}, {}".format(X_train.shape, y_train.shape))
print("Testing images, labels: {}, {}".format(X_test.shape, y_test.shape))
print("Each image is: {0:d}x{0:d} pixels".format(int(np.sqrt(X_train.shape[1]))))


# Train a LeNet-like CNN model using SystemML
def train_LeNet():
    script = """
      source("nn/examples/mnist_lenet_distrib_sgd.dml") as mnist_lenet

      # Scale images to [-1,1], and one-hot encode the labels
      images = (images / 255) * 2 - 1
      n = nrow(images)
      labels = table(seq(1, n), labels+1, n, 10)

      # Split into training (55,000 examples) and validation (5,000 examples)
      X = images[5001:nrow(images),]
      X_val = images[1:5000,]
      y = labels[5001:nrow(images),]
      y_val = labels[1:5000,]

      # Train the model to produce weights & biases.
      [W1, b1, W2, b2, W3, b3, W4, b4] = mnist_lenet::train(X, y, X_val, y_val, C, Hin, Win, 3, 4, epochs)
    """
    out = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4')
    prog = (dml(script).input(images=X_train, labels=y_train, epochs=1, C=1, Hin=28, Win=28)
            .output(*out))

    W1, b1, W2, b2, W3, b3, W4, b4 = ml.execute(prog).get(*out)

    return list(W1, b1, W2, b2, W3, b3, W4, b4)


W1, b1, W2, b2, W3, b3, W4, b4 = train_LeNet()

# Use the trained model to make predictions for the test data, and evaluate the quality of the predictions.
script_predict = """
  source("nn/examples/mnist_lenet_distrib_sgd.dml") as mnist_lenet

  # Scale images to [-1,1]
  X_test = (X_test / 255) * 2 - 1

  # Predict
  y_prob = mnist_lenet::predict(X_test, C, Hin, Win, W1, b1, W2, b2, W3, b3, W4, b4)
  y_pred = rowIndexMax(y_prob) - 1
"""
prog = (dml(script_predict).input(X_test=X_test, C=1, Hin=28, Win=28, W1=W1, b1=b1,
                                  W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
        .output("y_pred"))

y_pred = ml.execute(prog).get("y_pred").toNumPy()

print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

#Define a function that randomly selects a test image, displays the image, and scores it.
img_size = int(np.sqrt(X_test.shape[1]))

def displayImage(i):
  image = (X_test[i]).reshape(img_size, img_size).astype(np.uint8)
  imgplot = plt.imshow(image, cmap='gray')

def predictImage(i):
  image = X_test[i].reshape(1, -1)
  out = "y_pred"
  prog = (dml(script_predict).input(X_test=image, C=1, Hin=28, Win=28, W1=W1, b1=b1,
                                    W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
                             .output(out))
  pred = int(ml.execute(prog).get(out).toNumPy())
  return pred


i = np.random.randint(len(X_test))
p = predictImage(i)

print("Image {}\nPredicted digit: {}\nActual digit: {}\nResult: {}".format(
    i, p, int(y_test[i]), p == int(y_test[i])))

displayImage(i)

pd.set_option('display.max_columns', 28)
pd.DataFrame((X_test[i]).reshape(img_size, img_size), dtype='uint')




