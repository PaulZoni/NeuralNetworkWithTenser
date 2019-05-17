from __future__ import print_function, division
import pandas as pd
from keras.datasets import fashion_mnist
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


class ImageHelper(object):

    def save_image(self, generated, epoch, directory):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(generated[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        fig.savefig("{}/{}.png".format(directory, epoch))
        plt.close()

    def makegif(self, directory):
        filenames = np.sort(os.listdir(directory))
        filenames = [fnm for fnm in filenames if ".png" in fnm]

        with imageio.get_writer(directory + '/image.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(directory + filename)
                writer.append_data(image)


class GANSimple:

    def __init__(self, image_shape, generator_input_dim, image_hepler):
        optimizer = Adam(0.0002, 0.5)

        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim

        self._build_generator_model()
        self._build_and_compile_discriminator_model(optimizer)
        self._build_and_compile_gan(optimizer)

    def train(self, epochs, train_data, batch_size):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []

        for epoch in range(epochs):
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            genenerated = self._predict_noise(batch_size)
            loss_real = self.discriminator_model.train_on_batch(batch, real)
            loss_fake = self.discriminator_model.train_on_batch(genenerated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))
            generator_loss = self.gan.train_on_batch(noise, real)

            print("Epoch {}".format(epoch))
            print("Discriminator loss: {}".format(discriminator_loss[0]))
            print("Generator loss: {}".format(generator_loss))
            print("-----------------")
            history.append({"D": discriminator_loss[0], "G": generator_loss})

            if epoch % 200 == 0:
                self._save_images(epoch)

        self._plot_loss(history)
        self._image_helper.makegif("generated/")

    def _build_generator_model(self):
        model_gen = Sequential()
        model_gen.add(Dense(256, input_dim=self.generator_input_dim))
        model_gen.add(LeakyReLU(alpha=0.2))
        model_gen.add(BatchNormalization(momentum=0.8))
        model_gen.add(Dense(512))
        model_gen.add(LeakyReLU(alpha=0.2))
        model_gen.add(BatchNormalization(momentum=0.8))
        model_gen.add(Dense(1024))
        model_gen.add(LeakyReLU(alpha=0.2))
        model_gen.add(BatchNormalization(momentum=0.8))
        model_gen.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model_gen.add(Reshape(self.img_shape))
        self.generator_model = model_gen

    def _build_and_compile_discriminator_model(self, optimizer):
        model_cics = Sequential()
        model_cics.add(Flatten(input_shape=self.img_shape))
        model_cics.add(Dense(512))
        model_cics.add(LeakyReLU(alpha=0.2))
        model_cics.add(Dense(256))
        model_cics.add(LeakyReLU(alpha=0.2))
        model_cics.add(Dense(1, activation='sigmoid'))
        self.discriminator_model = model_cics
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator_model.trainable = False

    def _build_and_compile_gan(self, optimizer):
        real_input = Input(shape=(self.generator_input_dim,))
        generator_output = self.generator_model(real_input)
        discriminator_output = self.discriminator_model(generator_output)

        self.gan = Model(real_input, discriminator_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    def _save_images(self, epoch):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, "/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/ganmodel/plot/")

    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        return self.generator_model.predict(noise)

    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20, 5))
        for colnm in hist.columns:
            plt.plot(hist[colnm], label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()


(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()
generative_advarsial_network = GANSimple(X_train[0].shape, 100, image_helper)
generative_advarsial_network.train(3000, X_train, batch_size=32)