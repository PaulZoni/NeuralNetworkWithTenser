from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten


class EmbeddingWithFullyConnected:

    @staticmethod
    def build(vocab_size, out, input_length, weight_matrix, train=True):

        model = Sequential()
        if weight_matrix is not None:
            model.add(Embedding(input_dim=vocab_size, output_dim=out, input_length=input_length, weights=[weight_matrix], trainable=train))
        else:
            model.add(Embedding(input_dim=vocab_size, output_dim=out, input_length=input_length))
        model.add(Flatten())
        model.add(Dense(units=1, activation='sigmoid'))
        return model
