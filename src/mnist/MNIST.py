from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name='x')

W = tf.Variable(tf.zeros([784, 10], name='W'))
b = tf.Variable(tf.zeros([10]), name='b')

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(dtype=tf.float32, name='yy', shape=[None, 10])

cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initializers.global_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf .argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Точность: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

