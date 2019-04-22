import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 10
latent_space = 128
learning_rate = 0.1

fc_v_logits = None

ae_weights = {
    'encoder_w': tf.Variable(tf.truncated_normal([784, latent_space], stddev=0.1)),
    'encoder_b': tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
    'decoder_w': tf.Variable(tf.truncated_normal([latent_space, 784], stddev=0.1)),
    'decoder_b': tf.Variable(tf.truncated_normal([784], stddev=0.1))
}

ae_input = tf.placeholder(tf.float32, [batch_size, 784])
hidden = tf.nn.sigmoid(tf.matmul(ae_input, ae_weights['encoder_w']) + ae_weights['encoder_b'])

noised_hidden = tf.nn.relu(hidden - 0.1) + 0.1
noised_visible = tf.nn.sigmoid(tf.matmul(noised_hidden,ae_weights['decoder_w']) + ae_weights['decoder_b'])

'''visible_logits = tf.matmul(noised_visible, ae_weights['decoder_w']) + ae_weights['decoder_b']
visible = tf.nn.sigmoid(visible_logits)'''

ae_cost = tf.reduce_mean(tf.square(noised_visible - ae_input))
'''----'''
rho = 0.05
beta = 1.0
data_rho = tf.reduce_mean(hidden, 0)
reg_cost = - tf.reduce_mean(tf.log(data_rho/rho) * rho + tf.log((1 - data_rho) / (1 - rho)) * (1-rho))
total_cost = ae_cost + beta * reg_cost
'''----'''
optimizer = tf.train.AdagradOptimizer(learning_rate)
ae_op = optimizer.minimize(total_cost)

num_epoch = 50
num_test_images = 10

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    for epoch in range(num_epoch):

        num_batches = mnist.train.num_examples // batch_size
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(ae_op, feed_dict={ae_input: X_batch})

        train_loss = ae_cost.eval(feed_dict={ae_input: X_batch})
        print("epoch {} loss {}".format(epoch, train_loss))

    results = noised_visible.eval(feed_dict={ae_input: mnist.test.images[:num_test_images]})

    f, a = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(num_test_images):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(results[i], (28, 28)))

    plt.show()

