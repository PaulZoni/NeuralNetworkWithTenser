from src.model.Model import ModelBuilder
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

''' best accuracy 0.9773'''

'''------------------'''

model = ModelBuilder()
model.add_convolution2d(28, reshape=True, batch=True)
model.add_pooling()
model.flatten()
model.add_fully_connected_layer(245, 100)
model.add_batch_norm_layer(size_batch=100)
model.add_dropout(0.5)
model.add_fully_connected_layer(100, 100)
model.add_dropout(0.25)
model.add_fully_connected_layer(100, 10, activation='softmax')
model.build()
model.fit(mnist_data=mnist, epochs=1000, images_test=mnist.test.images, labels_test=mnist.test.labels)
