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

    normal_idx = 1
    abnormal_idx = 5

    x_normal_train = x_train[np.where(y_train==normal_idx,True,False)]
    x_normal_test = x_test[np.where(y_test==normal_idx,True,False)][:900]
    x_abnormal_train = x_train[np.where(y_train==abnormal_idx,True,False)]
    x_abnormal_test = x_test[np.where(y_test==abnormal_idx,True,False)][:100]

    x_mixed_train = np.concatenate((x_normal_train[:5400],x_abnormal_train[:600]))

    y_mixed_test = np.concatenate((np.zeros(x_normal_test.shape[0]),np.ones(x_abnormal_test.shape[0])))
    x_mixed_test = np.concatenate((x_normal_test,x_abnormal_test))

    x_normal_train = x_normal_train / 255
    x_mixed_test = x_mixed_test / 255

    x_mixed_train = x_mixed_train.reshape(x_mixed_train.astype('float32').shape[0], 28*28)
    x_mixed_test = x_mixed_test.reshape(x_mixed_test.astype('float32').shape[0], 28*28)

    num_features = 784
    num_hidden = 8
    r = 5.0
    epochs = 50
    nu = 0.1

    oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
    model, history = oc_nn.train_model(x_mixed_train, epochs=epochs, nu=nu)

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

    r = history.history['r'].pop()

    # y_score = []
    # for y in y_pred:
    #     if y - r >= 0:
    #         y_score.append([0])
    #     else:
    #         y_score.append([1])
    s_n = [y_pred[i, 0] - r >= 0 for i in range(len(y_pred))]

    print(y_pred)
    print(s_n)

    print(roc_auc_score(y_mixed_test, y_pred, multi_class='ovr'))

if __name__ == "__main__":
    main()
    exit()
