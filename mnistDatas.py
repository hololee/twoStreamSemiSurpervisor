from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
import numpy as np


class DataManager:
    TOTAL_BATCH = "total_batch"
    SUPERVISED_BATCH = "super_batch"
    UNSUPERVISED_BATCH = "un_super_batch"

    def __init__(self, number_of_unsupervised_data=10000):
        # for batch count.
        self._i = 0

        # set divide size.
        self.train_dataset_size = 60000
        self.test_dataset_size = 10000

        # load data using scikit-learn.
        self.train_X, self.test_X, self.train_y, self.test_y = self.load_data(one_hot=True)

        # change range 0 to 1.
        self.train_X = self.train_X / 255
        self.test_X = self.test_X / 255

        self.supervised_train_X = self.train_X[:30000]
        self.supervised_train_y = self.train_y[:30000]

        # NOTICE : unsupervised data.
        self.unsupervised_train_X = self.train_X[30000:number_of_unsupervised_data]

        pass

    def load_data(self, one_hot=True):
        print("loading data...")

        # only use scikit-learn when load MNIST data for convenience
        mnist = fetch_openml('mnist_784')
        X, y = mnist["data"], mnist["target"]

        if one_hot:
            one_hot = OneHotEncoder()
            y = one_hot.fit_transform(y.reshape(-1, 1))
            y = y.toarray()
            print("y are one-hot encoded..")

        X = np.reshape(X, newshape=[len(X), 28, 28, 1])

        return X[:self.train_dataset_size], X[self.train_dataset_size:], y[:self.train_dataset_size], y[
                                                                                                      self.train_dataset_size:]

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
