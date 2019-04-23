import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size, learning_rate = 10, 0.1

ae_weights = {
    'conv': tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
    'b_hidden': tf.Variable(tf.truncated_normal([4], stddev=0.1)),
    'de_conv': tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
    'b_visible': tf.Variable(tf.truncated_normal([1], stddev=0.1))
}

input_shape = tf.stack([batch_size, 28, 28, 1])

ae_input = tf.placeholder(tf.float32, [batch_size, 784])
images = tf.reshape(ae_input, [-1, 28, 28, 1])
hidden_logits = tf.nn.conv2d(images, ae_weights['conv'], strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_hidden']
hidden = tf.nn.sigmoid(hidden_logits)
visible_logits = tf.nn.conv2d_transpose(hidden, ae_weights['de_conv'], input_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_visible']
visible = tf.nn.sigmoid(visible_logits)

optimizer = tf.train.AdagradOptimizer(learning_rate)
conv_cost = tf.reduce_mean(tf.square(visible - images))
conv_op = optimizer.minimize(conv_cost)

num_epoch = 50
num_test_images = 10

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    for epoch in range(num_epoch):

        num_batches = mnist.train.num_examples // batch_size
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(conv_op, feed_dict={ae_input: X_batch})

        train_loss = conv_cost.eval(feed_dict={ae_input: X_batch})
        print("epoch {} loss {}".format(epoch, train_loss))

    results = visible.eval(feed_dict={ae_input: mnist.test.images[:num_test_images]})

    f, a = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(num_test_images):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(results[i], (28, 28)))

    plt.show()

