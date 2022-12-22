import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch

from ocnn import OneClassNeuralNetwork


def main():
    data = h5py.File('Data/http.mat', 'r')

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)

    num_features = 784
    num_hidden = 32
    r = 1.0
    epochs = 10
    nu = 0.001

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
    model, history = oc_nn.train_model(x_train, epochs=epochs, nu=nu)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="train_loss")
    plt.plot(history.epoch, history.history["quantile_loss"], label="quantile_loss")
    plt.plot(history.epoch, history.history["r"], label="r")

    plt.title("OCNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    y_pred = model.predict(x_test)

    row_sums = torch.sum(y_pred, 1) # normalization
    row_sums = row_sums.repeat(1, num_classes) # expand to same size as out
    y_pred = torch.div( y_pred , row_sums ) # these should be histograms

    roc_auc_score(y_test, y_pred, multi_class='ovr')

if __name__ == "__main__":
    main()
    exit()
