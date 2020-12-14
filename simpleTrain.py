import numpy as np
import tensorflow as tf
import os
import plot
# from mnistDatas import DataManager as dm
from cifar10Datas import DataManager as dm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dataManager = dm()

train_acc_array = []
train_loss_array = []
test_acc_array = []
test_loss_array = []


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
x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_output = tf.placeholder(tf.float32, shape=[None, 10])

# model.
model = conv_layer(x_input, 32)
model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
model = conv_layer(model, 64)
model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
model = conv_layer(model, 128)
model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

model = layer(model, 256)
output = layer(model, 10, False)
softmax_output = tf.nn.softmax(output)

learning_rate = 0.002
iter = 200

loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_output)

optimizing = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for iteration in range(iter):
        for i in range(40):
            # NOTICE: supervised batch.
            batch_x, batch_y = dataManager.next_batch(1000, batchType=dataManager.SUPERVISED_BATCH)
            _ = sess.run(optimizing, feed_dict={x_input: batch_x, y_output: batch_y})

            y_train_predict = sess.run(softmax_output, feed_dict={x_input: batch_x})
            y_test_predict = sess.run(softmax_output, feed_dict={x_input: dataManager.test_X})

            train_acc, train_loss = getAccuracyAndLoss(y_train_predict, batch_y)
            test_acc, test_loss = getAccuracyAndLoss(y_test_predict, dataManager.test_y)

            print("iter {} - train accuracy : {:.3}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(
                iteration + 1,
                train_acc,
                train_loss,
                test_acc,
                test_loss))

            train_acc_array.append(train_acc)
            train_loss_array.append(train_loss)
            test_acc_array.append(test_acc)
            test_loss_array.append(test_loss)

        if iteration % 10 == 0:
            plot.plotting("", train_acc_array, train_loss_array, test_acc_array, test_loss_array)
