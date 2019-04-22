from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import Model
from keras.callbacks import ModelCheckpoint


(X_train, y_train), (X_test, y_test) = mnist.load_data()
batch_size, img_rows, img_cols = 64, 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


class SimpleConvolutional:

    @staticmethod
    def create() -> Model:
        model = Sequential()
        model.add(Convolution2D(filters=32, kernel_size=[5, 5], padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Convolution2D(filters=64, kernel_size=[5, 5], padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(units=32))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        return model


conv_model = SimpleConvolutional.create()


conv_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
conv_model.fit(X_train, Y_train, callbacks=[
    ModelCheckpoint('model.hdf5', monitor='val_acc', save_best_only=True, save_weights_only=False, mode='auto')
], batch_size=batch_size, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))

score = conv_model.evaluate(X_test, Y_test, verbose=0)
print('Test score: %f' % score[0])
print('Test accuracy: %f' % score[1])
