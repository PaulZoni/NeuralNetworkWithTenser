import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import math

mnist = input_data.read_data_sets("/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/autocoder/MNIST_data")


def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z, units=100, activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=tf.layers.dropout(hidden1, rate=0.25), units=120, activation=tf.nn.leaky_relu)
        hidden3 = tf.layers.dense(inputs=tf.layers.dropout(hidden2, rate=0.25), units=200, activation=tf.nn.leaky_relu)
        output = tf.layers.dense(inputs=tf.layers.dropout(hidden3, rate=0.25), units=784, activation=tf.nn.tanh)
        return output


def discriminator(X, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        print(X)
        '''tenser_reshape = tf.reshape(X, shape=[-1, 28, 28, 1])
        conv = tf.layers.conv2d(inputs=tenser_reshape, padding="same", kernel_size=[5, 5], activation=tf.nn.relu, filters=32)
        flatten = tf.contrib.layers.flatten(conv)'''
        hidden1 = tf.layers.dense(inputs=X, units=200, activation=tf.nn.leaky_relu)
        dropout_hidden1 = tf.layers.dropout(hidden1, rate=0.25)
        hidden2 = tf.layers.dense(inputs=dropout_hidden1, units=100, activation=tf.nn.leaky_relu)
        dropout_hidden2 = tf.layers.dropout(hidden2, rate=0.25)
        hidden3 = tf.layers.dense(inputs=dropout_hidden2, units=50, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(tf.layers.dropout(hidden3, rate=0.25), units=1)
        output = tf.sigmoid(logits)
        return output, logits


tf.reset_default_graph()
real_images = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)


def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))
D_cost = D_real_loss + D_fake_loss
G_cost = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))


lr = 0.01
batch_size = 100
epochs = 40

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=lr,
                                           global_step=global_step,
                                           decay_steps=1000,
                                           decay_rate=0.96,
                                           staircase=True)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
D_trainer = optimizer.minimize(D_cost, var_list=d_vars, global_step=global_step)
G_trainer = optimizer.minimize(G_cost, var_list=g_vars, global_step=global_step)

init = tf.global_variables_initializer()
samples = []
current_image = None
for_plot = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        num_batches = mnist.train.num_examples // batch_size
        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images * 2 - 1
            current_image = batch_images
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            _ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
            _ = sess.run(G_trainer, feed_dict={z: batch_z})

        '''batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
        correct_prediction = tf.equal(tf.argmax(D_logits_real, 1), tf.argmax(batch_z, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Точность: %s epoch: {}".format(str(epoch)) % sess.run(accuracy, feed_dict={real_images: batch_images, z: batch_z}))'''
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
        val = np.square(current_image[0] - gen_sample[0])
        print(np.mean(val))
        for_plot.append(np.mean(val))
        samples.append(gen_sample)

plt.imshow(samples[0].reshape(28, 28))
plt.show()
plt.imshow(samples[20].reshape(28, 28))
plt.show()
plt.imshow(samples[39].reshape(28, 28))
plt.show()
plt.plot(for_plot)
plt.xlabel('Epoch')
plt.legend(['acc'], loc='upper left')
plt.title('Model accuracy')
plt.show()
