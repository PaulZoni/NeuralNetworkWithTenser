import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def fully_connected_layer(tensor, input_size, out_size):
    W = tf.Variable(tf.truncated_normal([input_size, out_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([out_size], stddev=0.1))
    return tf.nn.tanh(tf.matmul(tensor, W) + b)


def batch_norm_layer(tensor, size):
    batch_mean, batch_var = tf.nn.moments(tensor, [0])
    beta = tf.Variable(tf.zeros([size]))
    scale = tf.Variable(tf.ones([size]))
    return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 0.001)


x = tf.placeholder(tf.float32, [None, 784], name='X')
h1 = fully_connected_layer(x, 784, 100)
hl_bn = batch_norm_layer(h1, 100)
h2 = fully_connected_layer(hl_bn, 100, 100)
y_logit = fully_connected_layer(h2, 100, 10)

y = tf.placeholder(dtype=tf.float32, name='yy', shape=[None, 10])

loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initializers.global_variables()

sess = tf.Session()
sess.run(init)

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Точность: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

