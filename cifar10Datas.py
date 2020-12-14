from sklearn.preprocessing import OneHotEncoder
import tensorflow.contrib.keras as keras
import numpy as np


class DataManager:
    TOTAL_BATCH = "total_batch"
    SUPERVISED_BATCH = "super_batch"
    UNSUPERVISED_BATCH = "un_super_batch"

    def __init__(self, number_of_unsupervised_data=10000):

        cifar10 = keras.datasets.cifar10

        # for batch count.
        self._i = 0

        # set divide size.
        self.train_dataset_size = 50000
        self.test_dataset_size = 10000

        # load data using scikit-learn.
        (self.train_X, self.train_y), (self.test_X, self.test_y) = cifar10.load_data()

        rnd_index = np.random.permutation(len(self.train_X))
        self.train_X = self.train_X[rnd_index]
        self.train_y = self.train_y[rnd_index]
        rnd_index = np.random.permutation(len(self.test_X))
        self.test_X = self.test_X[rnd_index]
        self.test_y = self.test_y[rnd_index]

        # one hot
        zeros_train = np.zeros([len(self.train_y), 10])
        zeros_test = np.zeros([len(self.test_y), 10])

        for i, value in enumerate(self.train_y):
            zeros_train[i, value] = 1
        for i, value in enumerate(self.test_y):
            zeros_test[i, value] = 1
        self.train_y = zeros_train
        self.test_y = zeros_test

        # change range 0 to 1.
        self.train_X = self.train_X / 255
        self.test_X = self.test_X / 255

        self.supervised_train_X = self.train_X[:40000]
        self.supervised_train_y = self.train_y[:40000]

        # NOTICE : unsupervised data.
        self.unsupervised_train_X = self.train_X[40000:40000 + number_of_unsupervised_data]

        pass

    def next_batch(self, batch_size, batchType=TOTAL_BATCH):
        if batchType == self.TOTAL_BATCH:
            x, y = self.train_X[self._i:self._i + batch_size], self.train_y[self._i: self._i + batch_size]
            self._i = (self._i + batch_size) % len(self.train_X)
            return x, y

        elif batchType == self.SUPERVISED_BATCH:
            x, y = self.supervised_train_X[self._i:self._i + batch_size], self.supervised_train_y[
                                                                          self._i: self._i + batch_size]
            self._i = (self._i + batch_size) % len(self.supervised_train_X)
            return x, y

        elif batchType == self.UNSUPERVISED_BATCH:
            x = self.unsupervised_train_X[self._i:self._i + batch_size]
            self._i = (self._i + batch_size) % len(self.unsupervised_train_X)
            return x

    # def shake_data(self):
    #     rnd_index = np.random.permutation(len(self.train_X))
    #     self.train_X = self.train_X[rnd_index]
    #     self.train_y = self.train_y[rnd_index]
