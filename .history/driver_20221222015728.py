import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.metrics import classification_report

import torch

from ocnn import OneClassNeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    normal_idx = 1
    abnormal_idx = 7

    x_normal_train = x_train[np.where(y_train==normal_idx,True,False)]
    x_normal_test = x_test[np.where(y_test==normal_idx,True,False)][:900]
    x_abnormal_train = x_train[np.where(y_train==abnormal_idx,True,False)]
    x_abnormal_test = x_test[np.where(y_test==abnormal_idx,True,False)][:100]

    nu_list = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    auc_list_list = []
    sum_list = []

    while len(nu_list) > 0:

        train_anomaly_rate = 0.1

        nu = nu_list.pop(0)

        print('======= nu=', nu, '=============')

        train_anomaly_amount = int(6000 * train_anomaly_rate)

        x_mixed_train = np.concatenate((x_normal_train[:6000 - train_anomaly_amount],x_abnormal_train[:train_anomaly_amount]), axis=0)

        y_mixed_test = np.concatenate((np.ones(x_normal_test.shape[0]),np.zeros(x_abnormal_test.shape[0])))
        x_mixed_test = np.concatenate((x_normal_test,x_abnormal_test))

        x_mixed_test = x_mixed_test / 255.0

        x_mixed_train = x_mixed_train.reshape(x_mixed_train.astype('float32').shape[0], 28*28)
        x_mixed_test = x_mixed_test.reshape(x_mixed_test.astype('float32').shape[0], 28*28)

        x_mixed_train  = x_mixed_train / 255.0

        num_features = 784
        num_hidden = 128
        r = 1.0
        epochs = 200

        sum = 0

        auc_list = []
        auc_list_list = []

        for _ in range(10):
            y_pred_total = 0
            for _ in range(10):
                oc_nn = OneClassNeuralNetwork(num_features, num_hidden, r)
                model, history = oc_nn.train_model(x_mixed_train, epochs=epochs, nu=nu)

                y_pred = model.predict(x_mixed_test)

                y_pred_total += y_pred

                # r = history.history['r'].pop(-1)
                # r = r + (r - history.history['r'].pop(-1))
                # y_score = []
                # for y in y_pred:
                #     if y - r >= 0:
                #         y_score.append([1])
                #     else:
                #         y_score.append([0])
                # cm = confusion_matrix(y_mixed_test, y_score)
                # print('Confusion matrix:')
                # print(cm)

            auc_list.append(roc_auc_score(y_mixed_test, y_pred_total, multi_class='ovr'))
            print(roc_auc_score(y_mixed_test, y_pred_total, multi_class='ovr'))

            sum += roc_auc_score(y_mixed_test, y_pred_total, multi_class='ovr')

        sum_list.append(sum / 10.0)
        auc_list_list.append(auc_list)
        print(auc_list)

    print(auc_list_list)
    print(sum_list)



if __name__ == "__main__":
    main()
    exit()
