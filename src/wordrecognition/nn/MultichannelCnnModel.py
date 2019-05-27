from keras.layers import Input, Embedding, MaxPooling1D, Conv1D, Dense, Concatenate, Dropout, Flatten, concatenate
from keras.models import Model


class MultichannelCnnModel:

    @staticmethod
    def build(length, vocab_size):
        # channel 1
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(input_dim=vocab_size, output_dim=100)(inputs1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(input_dim=vocab_size, output_dim=100)(inputs2)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(input_dim=vocab_size, output_dim=100)(inputs3)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge channels
        merge = concatenate([flat1, flat2, flat3])
        dense1 = Dense(units=10, activation='relu')(merge)
        outputs = Dense(units=1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


