from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, concatenate, TimeDistributed, Activation, \
     BatchNormalization, GRU, Conv1D, MaxPooling1D, Flatten, RepeatVector
from keras.layers.embeddings import Embedding


class RnnForWord:

    @staticmethod
    def build(vocab_size, embedding_vector_length, max_review_length):

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_length))

        model.add(Dropout(0.2))
        model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Flatten())
        model.add(RepeatVector(max_review_length))

        model.add(Dropout(0.2))
        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(100))
        model.add(Dropout(0.2))

        model.add(Dense(units=50, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=25, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=10, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        return model
