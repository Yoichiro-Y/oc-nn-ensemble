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
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    normal_idx = 0
    abnormal_idx = 1

    x_normal_train = []
    x_abnormal_train = []
    x_normal_test = []
    x_abnormal_test = []

    for i in range(len(y_train)):
        if y_train[i][0] == 5 :
            x_normal_train.append(x_train[i])
        elif y_train[i][0] == 8 :
        # else:
            x_abnormal_train.append(x_train[i])

    for i in range(len(y_test)):
        if y_test[i][0] == 5 :
            x_normal_test.append(x_test[i])
        elif y_test[i][0] == 8 :
        # else:
            x_abnormal_test.append(x_test[i])

    x_normal_train = np.array(x_normal_train)
    x_abnormal_train = np.array(x_abnormal_train)
    x_normal_test = np.array(x_normal_test)
    x_abnormal_test = np.array(x_abnormal_test)

    x_normal_test = x_normal_test[:900]
    x_abnormal_test = x_abnormal_test[:100]

    x_mixed_train = np.concatenate((x_normal_train[:4500],x_abnormal_train[:500]))

    y_mixed_test = np.concatenate((np.zeros(x_normal_test.shape[0]),np.ones(x_abnormal_test.shape[0])))
    x_mixed_test = np.concatenate((x_normal_test,x_abnormal_test))

    x_normal_train = x_normal_train.astype('float32') / 255
    x_mixed_test = x_mixed_test.astype('float32') / 255

    x_mixed_train = x_mixed_train.reshape(x_mixed_train.astype('float32').shape[0], 32*32*3)
    x_mixed_test = x_mixed_test.reshape(x_mixed_test.astype('float32').shape[0], 32*32*3)

    num_features = 3072
    num_hidden = 512
    r = 50.0
    epochs = 200
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

    print(r)

    y_score = []
    for y in y_pred:
        if y - r >= 0:
            y_score.append([0])
        else:
            y_score.append([1])

    print(y_pred)
    print(y_score)

    print(roc_auc_score(y_mixed_test, y_pred, multi_class='ovr'))

if __name__ == "__main__":
    main()
    exit()
