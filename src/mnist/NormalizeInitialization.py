from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = x_train.reshape([-1, 28 * 28]) / 255.
X_test = x_test.reshape([-1, 28 * 28]) / 255.

'''uniform = loss 0.1242 acc 0.9630'''


def create_model(init):
    model = Sequential()
    model.add(Dense(100, input_shape=(28 * 28,), kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(10, kernel_initializer=init, activation='softmax'))
    return model


uniform_model = create_model('uniform')
uniform_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
uniform_model.fit(X_train, Y_train, batch_size=64, epochs=30, verbose=1, validation_data=(X_test, Y_test))


