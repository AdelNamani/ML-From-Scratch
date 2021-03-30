#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

NBR_TRAIN = 10
NBR_TEST = 3


def generate(X, Y, nbr, name):
    count = [nbr] * 10
    idx = []

    for i, y in enumerate(Y):
        if count[y] > 0:
            idx.append(i)
            count[y] -= 1
        if np.sum(count) == 0:
            break

    X = np.reshape(X, [-1, 784])[idx,:]
    Y = np.array([Y[idx]])

    Data = np.concatenate((X, Y.T), axis=1)
    dataset = pd.DataFrame(data=Data)
    dataset.to_csv(name, index=False)


generate(x_train, y_train, NBR_TRAIN, "mnist_train.csv")
generate(x_test, y_test, NBR_TEST, "mnist_test.csv")

