from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation, Input, Add, merge, Multiply, concatenate


class LSTMThreeLayers:

    @staticmethod
    def build(num_chars):
        # one layers LSTM
        vec = Input(shape=(None, num_chars))
        l1 = LSTM(128, activation='tanh', return_sequences=True)(vec)
        l1_d = Dropout(0.2)(l1)
        # two layers LSTM
        input2 = concatenate([vec, l1_d])
        l2 = LSTM(128, activation='tanh', return_sequences=True)(input2)
        l2_d = Dropout(0.2)(l2)
        # layers three
        input3 = concatenate([vec, l2_d])
        l3 = LSTM(128, activation='tanh', return_sequences=True)(input3)
        l3_d = Dropout(0.2)(l3)
        # dense layer
        input_d = concatenate([l1_d, l2_d, l3_d])
        dense3 = TimeDistributed(Dense(num_chars))(input_d)
        output_res = Activation('softmax')(dense3)
        model = Model(inputs=vec, outputs=output_res)
        return model

