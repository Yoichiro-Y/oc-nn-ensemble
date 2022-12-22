import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch

from ocnn import OneClassNeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    normal_idx = 5
    abnormal_idx = 8

    x_normal_train = x_train[np.where(y_train==normal_idx,True,False)]
    x_normal_test = x_test[np.where(y_test==normal_idx,True,False)][:900]
    x_abnormal_train = x_train[np.where(y_train==abnormal_idx,True,False)]
    x_abnormal_test = x_test[np.where(y_test==abnormal_idx,True,False)][:100]

    train_anomaly_rate_list = [0.001, 0.01, 0.1]

    while len(train_anomaly_rate_list) > 0:
        train_anomaly_rate = train_anomaly_rate_list.pop(0)

        print('======= anomaly=', train_anomaly_rate, '=============')

        train_anomaly_amount = int(6000 * train_anomaly_rate)

        x_mixed_train = np.concatenate((x_normal_train[:6000 - train_anomaly_amount],x_abnormal_train[:train_anomaly_amount]), axis=0)

        y_mixed_test = np.concatenate((np.ones(x_normal_test.shape[0]),np.zeros(x_abnormal_test.shape[0])))
        x_mixed_test = np.concatenate((x_normal_test,x_abnormal_test))

        x_mixed_test = x_mixed_test / 255.0

        x_mixed_train = x_mixed_train.reshape(x_mixed_train.astype('float32').shape[0], 28*28)
        x_mixed_test = x_mixed_test.reshape(x_mixed_test.astype('float32').shape[0], 28*28)

        p = np.random.permutation(len(x_mixed_train))
        x_mixed_train = x_mixed_train[p]
        y_train = y_train[p]

        x_mixed_train  = x_mixed_train / 255.0

        num_features = 784
        num_hidden = 128
        r = 1.0
        epochs = 300

        nu_list = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

        while len(nu_list) > 0:
            nu = nu_list.pop(0)

            print('======= nu=', nu, '=============')
            oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
            model, history = oc_nn.train_model(x_mixed_train, epochs=epochs, nu=nu)
            y_pred = model.predict(x_mixed_test)
            r = history.history['r'].pop(-1)
            r = r + (r - history.history['r'].pop(-1))
            print(r)
            y_score = []

            for y in y_pred:
                if y - r >= 0:
                    y_score.append([1])
                else:
                    y_score.append([0])

            cm = confusion_matrix(y_mixed_test, y_score)
            print('Confusion matrix:')
            print(cm)
            print(roc_auc_score(y_mixed_test, y_pred, multi_class='ovr'))
            print('========================')
        print('========================')

if __name__ == "__main__":
    main()
    exit()
