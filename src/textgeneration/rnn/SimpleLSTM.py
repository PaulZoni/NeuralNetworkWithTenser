from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation


class SimpleLSTM:

    @staticmethod
    def build(num_chars):
        model = Sequential()
        model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(None, num_chars)))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(num_chars)))
        model.add(Activation('softmax'))
        return model
