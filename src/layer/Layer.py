import tensorflow as tf


class Layer:

    @staticmethod
    def fully_connected_layer(tensor, input_size, out_size, activation='relu'):
        W = tf.Variable(tf.truncated_normal([input_size, out_size], stddev=0.1))
        b = tf.Variable(tf.truncated_normal([out_size], stddev=0.1))
        if activation == 'relu':
            return tf.nn.relu(tf.matmul(tensor, W) + b)
        elif activation == 'softmax':
            return tf.nn.softmax(tf.matmul(tensor, W) + b)

    @staticmethod
    def batch_norm_layer(tensor, size):
        batch_mean, batch_var = tf.nn.moments(tensor, [0])
        beta = tf.Variable(tf.zeros([size]))
        scale = tf.Variable(tf.ones([size]))
        return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 0.001)

    @staticmethod
    def dropout(tensor):
        keep_probability = tf.placeholder(tf.float32)
        return tf.nn.dropout(tensor, keep_probability), keep_probability

    @staticmethod
    def dropout_rate(tensor, probability):
        return tf.layers.dropout(tensor, rate=probability)

    @staticmethod
    def convolution2d(tenser, reshape=False, input_size=None, batch=False):
        if reshape:
            tenser = tf.reshape(tenser, shape=[-1, input_size, input_size, 1])
        convolution = tf.layers.conv2d(inputs=tenser, filters=5, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        if batch:
            convolution = tf.layers.batch_normalization(convolution, training=True,)
        return tf.layers.max_pooling2d(inputs=convolution, pool_size=[2, 2], strides=2)

    @staticmethod
    def flatten(tenser):
        return tf.contrib.layers.flatten(tenser)

    @staticmethod
    def pooling(tenser):
        return tf.layers.max_pooling2d(tenser, (3, 3), strides=(2, 2), padding='same')

