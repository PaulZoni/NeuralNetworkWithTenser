from keras.models import Sequential
from keras.layers import Dense

plot_path = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/wordrecognition/plot/FullyConnected_model.png'


class FullyConnected:

    @staticmethod
    def build(input_size):
        model = Sequential()
        model.add(Dense(units=50, input_shape=(input_size,), activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
