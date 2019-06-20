from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


def define_model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=input_length))
    model.add(LSTM(50))
    model.add(Dense(units=vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def define_model_with_two_LSTM(vocab_size, output, seq_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=output, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

