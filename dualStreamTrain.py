import numpy as np
import tensorflow as tf
import os
import plot
# from mnistDatas import DataManager as dm
from cifar10Datas import DataManager as dm

# ALERT : FAIL PROJECT.

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dataManager = dm()

trainA_acc_array = []
trainA_loss_array = []
testA_acc_array = []
testA_loss_array = []
trainB_acc_array = []
trainB_loss_array = []
testB_acc_array = []
testB_loss_array = []


def getAccuracyAndLoss(output, label):
    accuracy = np.mean(np.equal(np.argmax(output, axis=1),
                                np.argmax(label, axis=1)))
    loss = -np.mean(label * np.log(output + 1e-6))

    return accuracy, loss


def conv_layer(input, outdim):
    w = tf.Variable(tf.random_normal(shape=[3, 3, input.get_shape().as_list()[-1], outdim]))
    b = tf.Variable(tf.random_normal(shape=[outdim]))

    out = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME") + b

    # activation.
    out = tf.nn.relu(out)

    return out


def layer(input, outdim, activation=True):
    # already flattend
    if len(input.get_shape().as_list()) == 2:
        w = tf.Variable(tf.random_normal(shape=[input.get_shape().as_list()[-1], outdim]))
        b = tf.Variable(tf.random_normal(shape=[outdim]))

        out = tf.matmul(input, w) + b
        # activation.
        if activation:
            out = tf.nn.relu(out)
    else:
        # NOTICE:flatten
        out = tf.reshape(input,
                         shape=[-1, input.get_shape().as_list()[1] * input.get_shape().as_list()[2] *
                                input.get_shape().as_list()[3]])

        w = tf.Variable(tf.random_normal(shape=[out.get_shape().as_list()[-1], outdim]))
        b = tf.Variable(tf.random_normal(shape=[outdim]))

        out = tf.matmul(out, w) + b
        # activation.
        if activation:
            out = tf.nn.relu(out)
    return out


# x_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
x_inputA = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_outputA = tf.placeholder(tf.float32, shape=[None, 10])

x_inputB = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_outputB = tf.placeholder(tf.float32, shape=[None, 10])

# NOTICE: model A.
modelA = conv_layer(x_inputA, 32)
modelA = tf.nn.max_pool(modelA, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
modelA = conv_layer(modelA, 64)
modelA = tf.nn.max_pool(modelA, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
modelA = conv_layer(modelA, 128)
modelA = tf.nn.max_pool(modelA, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

modelA = layer(modelA, 256)
outputA = layer(modelA, 10, False)
softmax_outputA = tf.nn.softmax(outputA)

# NOTICE: mdoel B.
modelB = conv_layer(x_inputB, 32)
modelB = tf.nn.max_pool(modelB, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
modelB = conv_layer(modelB, 64)
modelB = tf.nn.max_pool(modelB, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
modelB = conv_layer(modelB, 128)
modelB = tf.nn.max_pool(modelB, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

modelB = layer(modelB, 256)
outputB = layer(modelB, 10, False)
softmax_outputB = tf.nn.softmax(outputB)

learning_rate = tf.placeholder(tf.float32)
learning_ratio = tf.placeholder(tf.float32)
iter = 1000

learning_rate_val = 0.001
learning_ratio_vala = 1.0
learning_ratio_valb = 0.0

lossA = tf.nn.softmax_cross_entropy_with_logits(logits=outputA, labels=y_outputA)
lossB = tf.nn.softmax_cross_entropy_with_logits(logits=outputB, labels=y_outputB)

optimizingA = tf.train.AdamOptimizer(learning_rate=learning_rate * learning_ratio).minimize(lossA)
optimizingB = tf.train.AdamOptimizer(learning_rate=learning_rate * learning_ratio).minimize(lossB)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for iteration in range(iter):
        # NOTICE: STREAM A step1
        for i in range(40):
            super_batch_x, super_batch_y = dataManager.next_batch(1000, batchType=dataManager.SUPERVISED_BATCH)
            _ = sess.run(optimizingA,
                         feed_dict={x_inputA: super_batch_x, y_outputA: super_batch_y, learning_rate: learning_rate_val,
                                    learning_ratio: learning_ratio_vala})
        # # NOTICE: STREAM B step1
        # for i in range(10):
        #     unsuper_batch_x = dataManager.next_batch(1000, batchType=dataManager.UNSUPERVISED_BATCH)
        #     _ = sess.run(optimizingB, feed_dict={x_inputB: unsuper_batch_x,
        #                                          y_outputB: sess.run(outputA, feed_dict={x_inputA: unsuper_batch_x}),
        #                                                                                  learning_rate: learning_rate_val,
        #                                                                                  learning_ratio: learning_ratio_valb})
        # NOTICE: STREAM B step2
        for i in range(40):
            super_batch_x, super_batch_y = dataManager.next_batch(1000, batchType=dataManager.SUPERVISED_BATCH)
            _ = sess.run(optimizingB,
                         feed_dict={x_inputB: super_batch_x, y_outputB: super_batch_y, learning_rate: learning_rate_val,
                                    learning_ratio: learning_ratio_vala})
        # # NOTICE: STREAM A step2
        # for i in range(10):
        #     unsuper_batch_x = dataManager.next_batch(1000, batchType=dataManager.UNSUPERVISED_BATCH)
        #     _ = sess.run(optimizingA, feed_dict={x_inputA: unsuper_batch_x,
        #                                          y_outputA: sess.run(outputA, feed_dict={x_inputA: unsuper_batch_x}),
        #                                                                                  learning_rate: learning_rate_val,
        #                                                                                  learning_ratio: learning_ratio_valb})

        # using supervised Data
        index = np.random.permutation(40000)[1000]
        train_predictonA = sess.run(softmax_outputA, feed_dict={x_inputA: dataManager.supervised_train_X[:index]})
        train_predictonB = sess.run(softmax_outputB, feed_dict={x_inputB: dataManager.supervised_train_X[:index]})
        train_label = dataManager.supervised_train_y[:index]

        test_predictonA = sess.run(softmax_outputA, feed_dict={x_inputA: dataManager.test_X})
        test_predictonB = sess.run(softmax_outputB, feed_dict={x_inputB: dataManager.test_X})
        test_label = dataManager.test_y

        trainA_acc, trainA_loss = getAccuracyAndLoss(train_predictonA, train_label)
        trainB_acc, trainB_loss = getAccuracyAndLoss(train_predictonB, train_label)

        testA_acc, testA_loss = getAccuracyAndLoss(test_predictonA, test_label)
        testB_acc, testB_loss = getAccuracyAndLoss(test_predictonB, test_label)

        print("Stream A // iter {} - train accuracy : {:.3}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(
            iteration + 1,
            trainA_acc,
            trainA_loss,
            testA_acc,
            testA_loss))

        print("Stream B // iter {} - train accuracy : {:.3}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(
            iteration + 1,
            trainB_acc,
            trainB_loss,
            testB_acc,
            testB_loss))

        trainA_acc_array.append(trainA_acc)
        trainA_loss_array.append(trainA_loss)
        testA_acc_array.append(testA_acc)
        testA_loss_array.append(testA_loss)

        trainB_acc_array.append(trainB_acc)
        trainB_loss_array.append(trainB_loss)
        testB_acc_array.append(testB_acc)
        testB_loss_array.append(testB_loss)

        if iteration % 5 == 0:
            plot.plotting("StreamA", trainA_acc_array, trainA_loss_array, testA_acc_array, testA_loss_array)
            plot.plotting("StreamB", trainB_acc_array, trainB_loss_array, testB_acc_array, testB_loss_array)
