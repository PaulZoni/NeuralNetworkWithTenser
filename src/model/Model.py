from tensorflow.examples.tutorials.mnist import input_data
from src.layer.Layer import Layer
import tensorflow as tf


class ModelBuilder:

    def __init__(self):
        self.layers = None
        self.x = tf.placeholder(tf.float32, [None, 784], name='X')
        self.y = tf.placeholder(dtype=tf.float32, name='y', shape=[None, 10])
        self.train_op = None

    def add_convolution2d(self, input=None, reshape=False, batch=False):
        if self.layers is None:
            self.layers = Layer.convolution2d(self.x, input_size=input, reshape=reshape, batch=batch)
        else:
            self.layers = Layer.convolution2d(self.layers, input_size=input, reshape=reshape, batch=batch)

    def flatten(self, ):
        self.layers = Layer.flatten(self.layers)

    def add_fully_connected_layer(self, input, output, activation='relu'):
        if self.layers is None:
            self.layers = Layer.fully_connected_layer(self.x, input, output, activation=activation)
        else:
            self.layers = Layer.fully_connected_layer(self.layers, input, output, activation=activation)

    def add_batch_norm_layer(self, size_batch):
        self.layers = Layer.batch_norm_layer(self.layers, size_batch)

    def add_dropout(self, probability=0.5):
        self.layers = Layer.dropout_rate(self.layers, probability)

    def add_pooling(self,):
        self.layers = Layer.pooling(tenser=self.layers)

    def build(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers, labels=self.y)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001,
                                               beta1=0.8,
                                               beta2=0.999,
                                               epsilon=1e-08,
                                               use_locking=False,
                                               name='Adam').minimize(loss)

    def fit(self, mnist_data, epochs=1000, images_test=None, labels_test=None):
        init = tf.initializers.global_variables()
        sess = tf.Session()
        sess.run(init)
        accuracy = 0.0

        for epoch in range(epochs):
            batch_xs, batch_ys = mnist_data.train.next_batch(100)

            sess.run(self.train_op, feed_dict={self.x: batch_xs, self.y: batch_ys})

            correct_prediction = tf.equal(tf.argmax(self.layers, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Точность: %s epoch: {}".format(str(epoch)) % sess.run(accuracy, feed_dict={self.x: images_test,
                                                                                              self.y: labels_test,
                                                                                              }))
