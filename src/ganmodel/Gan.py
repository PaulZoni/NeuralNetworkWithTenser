import numpy as np
import tensorflow as tf

'''create generator weight'''

gen_weight = dict()
gen_weight['w1'] = tf.Variable(tf.random_normal([1, 5]))
gen_weight['b1'] = tf.Variable(tf.random_normal([5]))
gen_weight['w2'] = tf.Variable(tf.random_normal([5, 1]))
gen_weight['b2'] = tf.Variable(tf.random_normal([1]))

dics_weight = dict()
dics_weight['w1'] = tf.Variable(tf.random_normal([1, 10]))
dics_weight['b1'] = tf.Variable(tf.random_normal([10]))
dics_weight['w2'] = tf.Variable(tf.random_normal([10, 10]))
dics_weight['b2'] = tf.Variable(tf.random_normal([10]))
dics_weight['w3'] = tf.Variable(tf.random_normal([10, 1]))
dics_weight['b3'] = tf.Variable(tf.random_normal([10]))

z_p = tf.placeholder('float', [None, 1])
x_d = tf.placeholder('float', [None, 1])
g_h = tf.nn.softplus(tf.add(tf.matmul(z_p, gen_weight['w1']), gen_weight['b1']))
x_g = tf.add(tf.matmul(g_h, gen_weight['w2']), gen_weight['b2'])


def discriminator(x):
    d_h1 = tf.nn.tanh(tf.add(tf.matmul(x, dics_weight['w1']), dics_weight['b1']))
    d_h2 = tf.nn.tanh(tf.add(tf.matmul(d_h1, dics_weight['w2']), dics_weight['b2']))
    score = tf.nn.sigmoid(tf.add(tf.matmul(d_h2, dics_weight['w3']), dics_weight['b3']))
    return score


x_data_score = discriminator(x=x_d)
x_gen_score = discriminator(x=x_g)

'''D_plus_cost = tf.reduce_mean(tf.nn.relu(x_data_score) - x_data_score + tf.log(1.0 + tf.exp(- tf.abs(x_data_score))))
D_minus_cost = tf.reduce_mean(tf.nn.relu(x_gen_score) + tf.log(1.0 + tf.exp(- tf.abs(x_gen_score))))
G_cost = tf.reduce_mean(tf.nn.relu(x_gen_score) - x_gen_score + tf.log(1.0 + tf.exp(- tf.abs(x_gen_score))))
D_cost = D_minus_cost + D_minus_cost'''

D_cost = - tf.reduce_mean(tf.log(x_data_score) + tf.log(1.0 - x_gen_score))
G_cost = tf.reduce_mean(tf.log(1.0 - x_gen_score))

batch_size = 64
updates = 40000
learning_rate =0.01
prior_mu = -2.5
prior_std =0.5
noise_range = 5.

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

D_optimizer = optimizer.minimize(D_cost, var_list=[x for x in dics_weight.values()])
G_optimizer = optimizer.minimize(G_cost, var_list=[x for x in gen_weight.values()])


def sample_z(size=batch_size):
    return np.random.uniform(-noise_range, noise_range, size=[size, 1])


def sample_x(size=batch_size, mu=prior_mu, std=prior_std):
    return np.random.normal(mu, std, size=[size, 1])


init = tf.initializers.global_variables()
with tf.Session() as session:
    session.run(init)
    for i in range(updates):
        z_batch = sample_z()
        x_batch = sample_x()
        _ = session.run(D_optimizer, feed_dict={z_p: z_batch, x_d: x_batch})

        z_batch = sample_z()
        session.run(G_optimizer, feed_dict={z_p: z_batch})

        gen_result = session.run(x_g, feed_dict={z_p: z_batch})
        val = np.square(x_batch - gen_result)
        print(np.mean(val))


