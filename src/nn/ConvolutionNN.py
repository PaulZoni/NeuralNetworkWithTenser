import tensorflow as tf
from src.layer.Layer import Layer

''' best accuracy 0.9773'''


class ConvolutionNN:

    def __init__(self):
        self.probability = None
        self.logits_layer = None
        self.train_op = None
        self.y = tf.placeholder(dtype=tf.float32, name='y', shape=[None, 10])
        self.x = tf.placeholder(tf.float32, [None, 784], name='X')

    def build_nn(self):
        convolution2d = Layer.convolution2d(self.x, input_size=28)
        flatten = Layer.flatten(convolution2d)
        layer1 = Layer.fully_connected_layer(flatten, 980, 100)
        batch_norm = Layer.batch_norm_layer(layer1, 100)
        layer2 = Layer.fully_connected_layer(batch_norm, 100, 100)
        dropout, self.probability = Layer.dropout(layer2)
        self.logits_layer = Layer.fully_connected_layer(dropout, 100, 10)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_layer, labels=self.y)
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    def fit(self, mnist, epochs=1000):
        init = tf.initializers.global_variables()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(epochs):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(self.train_op, feed_dict={self.x: batch_xs, self.y: batch_ys, self.probability: 0.5})

            correct_prediction = tf.equal(tf.argmax(self.logits_layer, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Точность: %s epoch: {}".format(str(epoch)) % sess.run(accuracy, feed_dict={self.x: mnist.test.images,
                                                                                              self.y: mnist.test.labels,
                                                                                              self.probability: 1.}))
