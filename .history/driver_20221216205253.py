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

    print(x_train[0])

    normal_idx = 1
    abnormal_idx = 7

    x_normal_train = x_train[np.where(y_train==normal_idx,True,False)]
    x_normal_test = x_test[np.where(y_test==normal_idx,True,False)]
    x_abnormal_train = x_train[np.where(y_train==abnormal_idx,True,False)]
    x_abnormal_test = x_test[np.where(y_test==abnormal_idx,True,False)]

    y_mixed_test = np.concatenate((np.zeros(x_normal_test.shape[0]),np.ones(x_abnormal_test.shape[0])))
    x_mixed_test = np.concatenate((x_normal_test,x_abnormal_test))

    x_normal_train = x_normal_train / 255
    x_mixed_test = x_mixed_test / 255

    x_normal_train = x_normal_train.reshape(x_normal_train.shape[0], 28*28)
    x_mixed_test = x_mixed_test.reshape(x_mixed_test.shape[0], 28*28)

    num_features = 784
    num_hidden = 32
    r = 1.0
    epochs = 50
    nu = 0.1

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
    model, history = oc_nn.train_model(x_normal_train, epochs=epochs, nu=nu)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="train_loss")
    plt.plot(history.epoch, history.history["quantile_loss"], label="quantile_loss")
    plt.plot(history.epoch, history.history["r"], label="r")

    plt.title("OCNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    y_pred = model.predict(x_mixed_test)

    print(y_pred)

    r = history.history['r'].pop()

    s_n = [y_pred[i, 0] - r >= 0.5 for i in range(len(y_pred))]

    print(roc_auc_score(y_mixed_test, s_n, multi_class='ovr'))

if __name__ == "__main__":
    main()
    exit()
