from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding


class ConvForWord:

    @staticmethod
    def build(vocab_size, embedding_vector_length, max_review_length):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_review_length))

        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Dropout(0.2))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Dropout(0.2))
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(activation='relu', units=20))
        model.add(Dropout(0.3))
        model.add(Dense(activation='relu', units=20))
        model.add(Dropout(0.3))
        model.add(Dense(activation='relu', units=10))
        model.add(Dense(units=1, activation='sigmoid'))
        return model
