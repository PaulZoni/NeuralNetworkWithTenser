from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Embedding


class ModelOneCnnLayer:

    @staticmethod
    def build(vocab_size, max_length):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
