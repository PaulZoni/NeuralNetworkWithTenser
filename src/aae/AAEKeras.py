import numpy as np
import keras as ke
from keras.layers import Conv2D, Flatten, Dense, Reshape, UpSampling2D
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
batch_size, img_rows, img_cols = 64, 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train[: 1000]
print(len(X_train))


class AEE_Keras:

    def __init__(self):
        self.encoder = self._build_model_encoder()
        self.decoder = self._build_model_decoder()
        self.discriminator = self._build_model_discriminator()
        self.discriminator.summary()

        model_auto_encoder = Sequential()
        model_auto_encoder.add(self.encoder)
        model_auto_encoder.add(self.decoder)
        self.auto_encoder = model_auto_encoder
        self.auto_encoder.summary()

        model_encoder_discriminator = Sequential()
        model_encoder_discriminator.add(self.encoder)
        model_encoder_discriminator.add(self.discriminator)
        self.encoder_discriminator = model_encoder_discriminator
        self.encoder_discriminator.summary()

    def create_optimizer(self):
        self.discriminator.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")
        self.encoder_discriminator.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")
        self.auto_encoder.compile(optimizer=ke.optimizers.Adam(lr=1e-3), loss="binary_crossentropy")

    def _build_model_encoder(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), activation="relu", padding="same"))
        model.add(Conv2D(128, (5, 5), strides=(2, 2), activation="relu", padding="same"))
        model.add(Flatten())
        model.add(Dense(2, activation="linear"))
        return model

    def _build_model_decoder(self):
        model = Sequential()
        model.add(Dense(6272, input_shape=(2,)))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
        model.add(UpSampling2D())
        model.add(Conv2D(1, (5, 5), activation="sigmoid", padding="same"))
        return model

    def _build_model_discriminator(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(2,)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def _show_generate_image(self, epoch_number=None):
        fig = plt.figure(figsize=[20, 20])
        for i in range(-5, 5):
            for j in range(-5, 5):
                topred = np.array((i * 0.5, j * 0.5))
                topred = topred.reshape((1, 2))
                img = self.decoder.predict(topred)
                img = img.reshape((28, 28))
                ax = fig.add_subplot(10, 10, (i + 5) * 10 + j + 5 + 1)
                ax.set_axis_off()
                ax.imshow(img,)

        plt.show()
        plt.close(fig)

    def set_trainable(self, model, boolean):
        for layer in model.layers:
            layer.trainable = boolean
        model.trainable = boolean

    def train(self, batchsize=64, epoch=10):

        for epoch_number in range(epoch):
            print('---------------------------')
            print('Epoch: ' + str(epoch_number))
            print('---------------------------')
            for i in range(int(len(X_train) / batchsize)):
                print('batch: ' + str(i))
                self.set_trainable(self.auto_encoder, True)
                self.set_trainable(self.encoder, True)
                self.set_trainable(self.decoder, True)

                batch = X_train[i * batchsize:i * batchsize + batchsize]
                self.auto_encoder.train_on_batch(batch, batch)

                self.set_trainable(self.discriminator, True)
                batchpred = self.encoder.predict(batch)
                fakepred = np.random.standard_normal((batchsize, 2))
                discbatch_x = np.concatenate([batchpred, fakepred])
                discbatch_y = np.concatenate([np.zeros(batchsize), np.ones(batchsize)])
                self.discriminator.train_on_batch(discbatch_x, discbatch_y)

                self.set_trainable(self.encoder_discriminator, True)
                self.set_trainable(self.encoder, True)
                self.set_trainable(self.discriminator, False)
                self.encoder_discriminator.train_on_batch(batch, np.ones(batchsize))

            print("AutoEncoder Loss:", self.auto_encoder.evaluate(X_test, X_test, verbose=0))
            print("Encoder-Discriminator Loss:", self.encoder_discriminator.evaluate(X_test, np.ones(len(X_test)), verbose=0))
        self._show_generate_image()


aee_keras = AEE_Keras()
aee_keras.create_optimizer()
aee_keras.train(epoch=20)
